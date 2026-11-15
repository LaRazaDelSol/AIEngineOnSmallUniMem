"""S19: score_report by POSITION bucket (long-context check of the QSA indexer). top-1 = engine argmax == reference
argmax; ppl = exp(mean NLL of the actual next token); KL20 only if the score file carries the reference's 20 ids.
    python3 score_buckets.py ref_long.txt score_a.txt:name [score_b.txt:name ...] [--edges 0,2048,2052,4000,6000]
"""
import math, sys
from collections import defaultdict

args = [x for x in sys.argv[1:] if not x.startswith("--")]
edges = [0, 1024, 2048, 2052, 3000, 4000, 5000, 6000]
for i, x in enumerate(sys.argv):
    if x == "--edges": edges = list(map(int, sys.argv[i + 1].split(","))); args.remove(sys.argv[i + 1])
ref = {}
for l in open(args[0]):
    d, p, ids, lps, act = l.split()
    ref[(int(d), int(p))] = (list(map(int, ids.split(","))), list(map(float, lps.split(","))), float(act))
def bucket(p):
    for a, b in zip(edges, edges[1:]):
        if a <= p < b: return f"[{a},{b})"
    return f"[{edges[-1]},)"
print(f"{'run':14s} {'bucket':14s} {'n':>5s} {'top-1':>7s} {'KL20':>7s} {'ppl':>8s} {'ref ppl':>8s}")
for spec in args[1:]:
    path, name = spec.split(":")
    agg = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0])
    for l in open(path):
        f = l.split()
        if len(f) < 4: continue
        d, p, am, lpa = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        r = ref.get((d, p))
        if not r: continue
        ids, rlp, ract = r
        a = agg[bucket(p)]
        a[0] += 1; a[1] += int(am == ids[0]); a[3] += -lpa; a[4] += -ract if not math.isnan(ract) else 0.0
        if len(f) >= 5:
            elp = list(map(float, f[4].split(",")))
            if len(elp) == len(rlp):
                w = [math.exp(x) for x in rlp]; z = sum(w)
                a[2] += sum(wi / z * (math.log(wi / z) - e) for wi, e in zip(w, elp)); a[5] += 1
    for b in sorted(agg, key=lambda s: int(s[1:].split(",")[0])):
        n, t1, kl, nll, nllr, nk = agg[b]
        print(f"{name:14s} {b:14s} {n:5d} {t1 / n:7.4f} {(kl / nk if nk else float('nan')):7.4f} {math.exp(nll / n):8.4f} {math.exp(nllr / n):8.4f}")
