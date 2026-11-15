"""Gate S1: the C++ SlotCache (Policy::LRU) must reproduce the Python R2 gate
hit-for-hit on the same trace and pool.

    py runtime/tools/gate_s1.py --replay-bin <path to trace_replay[.exe]> [--bpw 1.66]

Builds a pack manifest + 3 traces (agent/chat/uniform) with the engine, then
compares hits/misses at several pools. Any mismatch = a bug in one of the two
cache models; fix it before trusting either. Also prints CLOCK vs LRU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import manifest, replay, trace  # noqa: E402
from trace_to_txt import convert  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-bin", required=True)
    ap.add_argument("--bpw", type=float, default=1.66)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--pools", default="1,4,8,10,11.6")
    args = ap.parse_args()

    cfg = manifest.ModelConfig()
    man = manifest.build_manifest(cfg, args.bpw)
    slot = man["expert_slot_bytes"]
    pools = [float(p) for p in args.pools.split(",")]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        print(f"pack bpw={args.bpw} slot={slot} B   pools(GiB)={pools}")
        print(f"{'profile':8} {'pool':>5} {'py hit':>8} {'c++ lru':>8} {'c++ clock':>9} {'miss/layer':>10}  match")
        for profile, seed in (("agent", 1), ("chat", 2), ("uniform", 3)):
            t = trace.generate(args.tokens, cfg.moe_layers, cfg.n_experts, cfg.top_k, profile=profile, seed=seed)
            tj, tt = Path(td) / f"{profile}.json", Path(td) / f"{profile}.txt"
            tj.write_text(json.dumps(t))
            convert(str(tj), str(tt))
            for pool in pools:
                py = replay.replay(man, t, cache_gib=pool, bw_gib_s=1.0, tok_s_target=1.0)
                out = {}
                for pol in ("lru", "clock"):
                    r = subprocess.run([args.replay_bin, "--trace", str(tt), "--slot-bytes", str(slot),
                                        "--pool-gib", str(pool), "--policy", pol, "--json"],
                                       capture_output=True, text=True, check=True)
                    out[pol] = json.loads(r.stdout)
                # exact miss count from the gate's own cache model (the report rounds hit to 4 dp)
                sc = replay.SlotCache(py["cache_slots_total"])
                for tok in t["routing"]:
                    for layer, ids in enumerate(tok):
                        for e in ids:
                            sc.touch((layer, e))
                exact = (out["lru"]["slots"] == py["cache_slots_total"]
                         and out["lru"]["reads"] == py["expert_reads_total"]
                         and out["lru"]["misses"] == sc.misses
                         and out["lru"]["hits"] == sc.hits)
                ok &= exact
                print(f"{profile:8} {pool:5.1f} {py['hit_rate']:8.4f} {out['lru']['hit_rate']:8.4f} "
                      f"{out['clock']['hit_rate']:9.4f} {out['lru']['misses_per_layer_visit']:10.3f}  "
                      f"{'OK' if exact else 'MISMATCH'}")
    print("GATE S1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
