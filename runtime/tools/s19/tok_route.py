"""S19: how token-determined is early-layer routing? (pred17 data: trace_i.txt routing + out_i.txt generated ids +
the pred_data.sh prompts). For every position whose token id occurs elsewhere in the corpus, predict layer-l experts
as the top-N most frequent experts of that id's OTHER occurrences (leave-one-out within the corpus) and report
precision@N against the true top-10. Also: the same with a corpus-wide prior (most frequent experts of the layer).
  python3 tok_route.py PRED17_DIR [--layers 0,1,2,3,4,8] [--n 10]
"""
import argparse, os, re, subprocess, sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--layers", default="0,1,2,3,4,8,20,47")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--script", default=os.path.expanduser("~/ps5lm/src/runtime/tools/s17/pred_data.sh"))
a = ap.parse_args()
sys.path.insert(0, os.path.expanduser("~/ps5lm/bench"))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.expanduser("~/ps5lm/model/tokenizer.json"))
src = open(a.script).read()
prompts = re.findall(r'^"(.*)"$', src.split("P=(")[1].split("\n)")[0], re.M)
layers = list(map(int, a.layers.split(",")))
seqs, routes = [], []
for i, p in enumerate(prompts):
    tp, ttr = os.path.join(a.dir, f"out_{i}.txt"), os.path.join(a.dir, f"trace_{i}.txt")
    if not (os.path.exists(tp) and os.path.exists(ttr)): continue
    text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(text).ids + [int(x) for x in open(tp).read().strip().split(",") if x]
    with open(ttr) as f:
        hdr = f.readline().split()
        nt, nl = int(hdr[1]), int(hdr[2])
        r = [[list(map(int, f.readline().split()[1:])) for _ in range(nl)] for _ in range(nt)]
    n = min(nt, len(ids))
    seqs.append(ids[:n]); routes.append(r[:n])
    print(f"prompt {i}: {n} positions (trace {nt}, ids {len(ids)})", flush=True)
for L in layers:
    cnt = defaultdict(Counter)   # token id -> expert counts at layer L
    occ = Counter()
    glob = Counter()
    for s, r in zip(seqs, routes):
        for t, tid in enumerate(s):
            cnt[tid].update(r[t][L]); occ[tid] += 1; glob.update(r[t][L])
    gtop = [e for e, _ in glob.most_common(a.n)]
    hit = tot = ghit = covered = 0
    for s, r in zip(seqs, routes):
        for t, tid in enumerate(s):
            true = set(r[t][L])
            ghit += len(true & set(gtop)); tot += 10
            if occ[tid] < 2: continue
            c = cnt[tid].copy(); c.subtract(r[t][L])
            pred = [e for e, v in c.most_common(a.n) if v > 0]
            covered += 10
            hit += len(true & set(pred))
    print(f"layer {L:2d}: token-id table precision@{a.n} {hit / max(covered, 1):.3f} on {covered / tot:.0%} of positions "
          f"(ids seen elsewhere); global top-{a.n} {ghit / tot:.3f}", flush=True)
