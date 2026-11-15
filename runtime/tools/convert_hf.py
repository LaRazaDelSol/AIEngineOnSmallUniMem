"""Stream Qwen3.8-Flash-Next from HuggingFace into the PS5LM on-disk model, shard by shard (resumable).

    python3 convert_hf.py --out ~/ps5lm/model --bin ~/ps5lm/build/pack_convert [--threads 7]

Per shard: download (the next shard downloads while this one converts) -> per tensor:
  layers.N.mlp.experts.{gate_up_proj,down_proj} -> pack_convert -> experts_q4.pack + experts_q2.pack (layer N)
  mtp.layers.0.mlp.experts.*                   -> same packs, layer 48
  ...ngram_embedding.shard_K.weight            -> ngram_bf16.bin at row K * rows_per_part (raw BF16 [rows,160])
  model.visual.*                               -> skipped (text-only engine)
  everything else (dense)                      -> dense/<name>.bin raw bytes + manifest.json {dtype, shape}
then the shard is deleted. Progress lives in done.txt; a re-run skips finished shards.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time

BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
EXPERT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
MTP_EXPERT_RE = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(gate_up_proj|down_proj)$")
NGRAM_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")


def header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def download(shard: str, dst: str) -> None:
    tmp = dst + ".part"
    for attempt in range(8):
        r = subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", tmp, BASE + shard])
        if r.returncode == 0:
            os.replace(tmp, dst)
            return
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"download failed: {shard}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, "dense"), exist_ok=True)
    os.makedirs(os.path.join(out, "dl"), exist_ok=True)
    idx = json.load(open(os.path.join(out, "index.json")))["weight_map"]
    shards = sorted(set(idx.values()))
    done_path = os.path.join(out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    todo = [s for s in shards if s not in done]
    man_path = os.path.join(out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    q4, q2 = os.path.join(out, "experts_q4.pack"), os.path.join(out, "experts_q2.pack")
    ngram = os.path.join(out, "ngram_bf16.bin")
    print(f"{len(todo)}/{len(shards)} shards to go", flush=True)

    pending: dict[str, threading.Thread] = {}

    def prefetch(i: int) -> None:
        if i < len(todo) and todo[i] not in pending:
            dst = os.path.join(out, "dl", todo[i])
            t = threading.Thread(target=lambda: os.path.exists(dst) or download(todo[i], dst), daemon=True)
            t.start()
            pending[todo[i]] = t

    prefetch(0)
    t_start = time.time()
    ahead = int(os.environ.get("PS5LM_DL_AHEAD", "3"))  # ~46 MB/s per HF connection; 3 fill a gigabit link
    for i, shard in enumerate(todo):
        for j in range(1, ahead + 1):
            prefetch(i + j)
        pending[shard].join()
        path = os.path.join(out, "dl", shard)
        t0 = time.time()
        data0, h = header(path)
        for name, info in h.items():
            off = data0 + info["data_offsets"][0]
            nbytes = info["data_offsets"][1] - info["data_offsets"][0]
            m = EXPERT_RE.match(name)
            mm = MTP_EXPERT_RE.match(name)
            if m or mm:
                layer, kind = (int(m.group(1)), m.group(2)) if m else (48, mm.group(1))
                subprocess.run([a.bin, "--src", path, "--offset", str(off), "--kind", "gate_up" if kind == "gate_up_proj" else "down",
                                "--layer", str(layer), "--q4", q4, "--q2", q2, "--threads", str(a.threads), "--steps", str(a.steps)],
                               check=True)
                manifest[name] = {"dtype": info["dtype"], "shape": info["shape"], "where": "pack", "layer": layer}
            elif NGRAM_RE.search(name):
                k = int(NGRAM_RE.search(name).group(1))
                rows, width = info["shape"]
                with open(path, "rb") as src, open(ngram, "r+b" if os.path.exists(ngram) else "wb") as dst:
                    src.seek(off)
                    dst.seek(k * rows * width * 2)
                    left = nbytes
                    while left:
                        buf = src.read(min(left, 64 << 20))
                        dst.write(buf)
                        left -= len(buf)
                manifest[name] = {"dtype": info["dtype"], "shape": info["shape"], "where": "ngram", "part": k}
            elif name.startswith("model.visual."):
                continue
            else:
                fn = os.path.join(out, "dense", name + ".bin")
                with open(path, "rb") as src, open(fn, "wb") as dst:
                    src.seek(off)
                    dst.write(src.read(nbytes))
                manifest[name] = {"dtype": info["dtype"], "shape": info["shape"], "where": "dense"}
        os.remove(path)
        with open(done_path, "a") as f:
            f.write(shard + "\n")
        json.dump(manifest, open(man_path + ".tmp", "w"), indent=0)
        os.replace(man_path + ".tmp", man_path)
        el = time.time() - t_start
        print(f"[{i + 1}/{len(todo)}] {shard} converted in {time.time() - t0:.0f}s; elapsed {el / 60:.1f} min, "
              f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
