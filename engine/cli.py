"""PS5LM engine CLI.

    python -m engine ledger --g 1.24 --bw 5.1584
    python -m engine pack build --bpw 2.05 --out pack.json
    python -m engine pack verify pack.json
    python -m engine trace gen --tokens 2000 --profile agent --out trace.json
    python -m engine gate replay --pack pack.json --trace trace.json --cache-gib 8 --report report.md
"""

from __future__ import annotations

import argparse
import json
import sys

from . import decode, ledger, manifest, replay, trace


def _cmd_ledger(args: argparse.Namespace) -> int:
    g, bw = args.g, args.bw
    print(f"ledger: G={g} GiB/token, store={bw} GiB/s")
    print(f"{'hit':>6} | {'tok/s':>8}")
    for band in ledger.band_table(g, bw):
        print(f"{band.hit:>6.2f} | {band.tok_s:>8.2f}")
    checks = ledger.headline_check()
    print("\ninvariants:", " ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in checks.items()))
    return 0 if all(checks.values()) else 1


def _cmd_pack_build(args: argparse.Namespace) -> int:
    cfg = manifest.ModelConfig()
    hot = None
    if args.hot_experts:
        hot = json.loads(args.hot_experts)
    m = manifest.build_manifest(cfg, args.bpw, hot_experts=hot)
    manifest.save(m, args.out)
    print(
        f"pack: {m['pack_bytes'] / 1024**3:.1f} GiB, slot={m['expert_slot_bytes'] / 1024:.0f} KiB, "
        f"G={m['g_gib_per_token']:.3f} GiB/token -> {args.out}"
    )
    return 0


def _cmd_pack_verify(args: argparse.Namespace) -> int:
    m = manifest.load(args.path)
    problems = manifest.verify_manifest(m)
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print(f"OK: {args.path} verified ({m['model']}, {m['bpw']} bpw, G={m['g_gib_per_token']:.3f} GiB/token)")
    return 0


def _cmd_trace_gen(args: argparse.Namespace) -> int:
    cfg = manifest.ModelConfig()
    t = trace.generate(
        tokens=args.tokens,
        moe_layers=cfg.moe_layers,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        profile=args.profile,
        seed=args.seed,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(t, f)
    print(f"trace: {t['tokens']} tokens, profile={t['profile']}, seed={t['seed']} -> {args.out}")
    return 0


def _cmd_gate_replay(args: argparse.Namespace) -> int:
    m = manifest.load(args.pack)
    problems = manifest.verify_manifest(m)
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    t = trace.load(args.trace) if args.trace.endswith(".json") else None
    if t is None:
        print("trace must be a .json file", file=sys.stderr)
        return 1
    report = replay.replay(
        m, t, cache_gib=args.cache_gib, bw_gib_s=args.bw, tok_s_target=args.target, gate_hit=args.gate_hit
    )
    md = replay.report_markdown(report)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md)
    replay.save(report, args.report.replace(".md", ".json"))
    print(md)
    return 0 if report["verdict"] == "PASS" else 2


def _cmd_reprice(args: argparse.Namespace) -> int:
    g_values = [
        ("UD-Q4_K_XL (1.24)", ledger.G_Q4_GIB),
        ("EXL3 2.18bpw (0.581)", ledger.G_EXL3_2_18_GIB),
        ("EXL3 2.05bpw (0.558)", ledger.G_EXL3_2_05_GIB),
        ("EXL3 1.66bpw (0.443)", ledger.G_EXL3_1_66_GIB),
    ]
    hit_points: list[tuple[str, float]] = [("cold (h=0)", 0.0)]
    if args.curve:
        for part in args.curve.split(","):
            cap, hit_pct = part.split(":")
            hit_points.append((f"{cap} MB", float(hit_pct) / 100.0))
    # S10 unit fix: fio MB/s are decimal; the old ÷1000 "GiB/s" inflated
    # every tok/s by ~7.4% (user-caught 2026-10-27). 4920 MB/s -> 4.582 GiB/s.
    bw = ledger.mb_s_to_gib_s(args.bw_mb)
    if args.label:
        print(f"reprice: {args.label} — bw={args.bw_mb} MB/s (={bw:.3f} GiB/s)")
    print(ledger.reprice_table(g_values, hit_points, bw))
    return 0


DECODE_TARGETS = (10.0, 20.0, 30.0, 40.0, 43.7)


def _cmd_decode(args: argparse.Namespace) -> int:
    """S9-queue item 4 (decoder-gap spike): required decode rate vs anchors."""
    targets = [args.target] if args.target else list(DECODE_TARGETS)
    print("decoder requirement -- hit-rate-INDEPENDENT: the N7 cache saves expert")
    print("bytes from the SSD, not dequant compute; every token dequantizes all")
    print(f"{decode.reads_per_token(manifest.ModelConfig())} routed reads regardless of hit.")
    print(
        f"anchor [D-PS5, F28]: PS5 GPU fused dequant floor = "
        f"{decode.ANCHOR_PS5_GPU_GPARAMS_S:.1f} G params/s "
        f"(PS5LM-Plus dense-27B @ {decode.ANCHOR_PS5_GPU_DENSE_27B_TOK_S} tok/s, this console)"
    )
    print(f"\n{'tok/s':>6} | {'experts/s':>10} | {'G params/s':>10} | {'headroom':>9} | verdict")
    print("-" * 62)
    for t in targets:
        r = decode.requirement(t)
        print(
            f"{t:>6.1f} | {r.experts_per_s:>10,.0f} | {r.gparams_s:>10.1f} | "
            f"{r.headroom_vs_ps5_gpu:>8.1f}x | {r.verdict}"
        )
    checks = decode.headline_check()
    print("\ninvariants:", " ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in checks.items()))
    return 0 if all(checks.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engine", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("ledger", help="tok/s band table for a G and bandwidth")
    lg.add_argument("--g", type=float, default=ledger.G_Q4_GIB)
    lg.add_argument("--bw", type=float, default=ledger.BW_REFERENCE_GIB_S)
    lg.set_defaults(fn=_cmd_ledger)

    rp = sub.add_parser("reprice", help="Alpha deliverable: measured-constants tok/s table (N7 curve x G values)")
    rp.add_argument("--bw-mb", type=float, required=True, help="measured bandwidth MB/s (2625 seq / 3047 engine-shaped / 1400 N7 miss-stream)")
    rp.add_argument("--curve", default="", help="N7 hit points 'cap_mb:hitpct,...' e.g. 3072:20.7,8192:51.4")
    rp.add_argument("--label", default="", help="provenance label printed above the table")
    rp.set_defaults(fn=_cmd_reprice)

    dc = sub.add_parser("decode", help="decoder-gap spike: required decode rate vs the measured PS5-GPU anchor")
    dc.add_argument("--target", type=float, default=None, help="check a single tok/s target instead of the default ladder")
    dc.set_defaults(fn=_cmd_decode)

    pack = sub.add_parser("pack", help="pack manifest tool")
    psub = pack.add_subparsers(dest="pack_cmd", required=True)
    pb = psub.add_parser("build")
    pb.add_argument("--bpw", type=float, required=True, help="bits per weight of the pack")
    pb.add_argument("--hot-experts", help='optional JSON {"layer": [expert ids]} pinned to arena front')
    pb.add_argument("--out", default="pack.json")
    pb.set_defaults(fn=_cmd_pack_build)
    pv = psub.add_parser("verify")
    pv.add_argument("path")
    pv.set_defaults(fn=_cmd_pack_verify)

    tr = sub.add_parser("trace", help="routing-trace generator")
    tsub = tr.add_subparsers(dest="trace_cmd", required=True)
    tg = tsub.add_parser("gen")
    tg.add_argument("--tokens", type=int, default=2000)
    tg.add_argument("--profile", default="agent", choices=["agent", "chat", "uniform"])
    tg.add_argument("--seed", type=int, default=7)
    tg.add_argument("--out", default="trace.json")
    tg.set_defaults(fn=_cmd_trace_gen)

    gt = sub.add_parser("gate", help="R2 acceptance gate")
    gsub = gt.add_subparsers(dest="gate_cmd", required=True)
    gr = gsub.add_parser("replay")
    gr.add_argument("--pack", required=True)
    gr.add_argument("--trace", required=True)
    gr.add_argument("--cache-gib", type=float, default=8.0, help="Linux-track ~8 GiB; PS5-OS title ~2-3 GiB")
    gr.add_argument("--bw", type=float, default=ledger.BW_REFERENCE_GIB_S, help="measured store GiB/s (G1/N6)")
    gr.add_argument("--target", type=float, default=15.0, help="tok/s target (REAP band floor)")
    gr.add_argument("--gate-hit", type=float, default=0.8, help="R2 hit-rate bar (N5/N7: >=80%%)")
    gr.add_argument("--report", default="gate-report.md")
    gr.set_defaults(fn=_cmd_gate_replay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
