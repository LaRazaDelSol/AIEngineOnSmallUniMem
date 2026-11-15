"""S18 quality instrument, step 3: compare engine score files against the reference.
    python3 score_report.py ref.txt docs.ids.labels score_268.txt:2.68 score_318.txt:3.18
Metrics per pack (all documents and per document):
  top-1   = engine argmax == reference argmax
  KL20    = sum over the reference's top-20 of p_ref * (log p_ref - log p_eng)  (reference mass renormalised on its top-20)
  ppl     = exp(mean NLL of the actual next token); ref ppl from the reference's own logprob
"""
import math, sys
from collections import defaultdict

ref = {}
for l in open(sys.argv[1]):
    d, p, ids, lps, act = l.split()
    ref[(int(d), int(p))] = (list(map(int, ids.split(","))), list(map(float, lps.split(","))), float(act))
labels = [l.split("\t")[0] for l in open(sys.argv[2]) if l.strip()]
rows = []
for spec in sys.argv[3:]:
    path, name = spec.split(":")
    agg = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, 0])  # n, top1, kl, nll_eng, nll_ref, n_ref_ok
    for l in open(path):
        f = l.split()
        if len(f) < 5:
            continue
        d, p, am, lpa = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        r = ref.get((d, p))
        if not r:
            continue
        ids, rlp, ract = r
        elp = list(map(float, f[4].split(",")))
        w = [math.exp(x) for x in rlp]
        z = sum(w)
        kl = sum(wi / z * (math.log(wi / z) - e) for wi, e in zip(w, elp) if wi > 0)
        for key in ("all", labels[d] if d < len(labels) else str(d)):
            a = agg[key]
            a[0] += 1; a[1] += am == ids[0]; a[2] += kl; a[3] -= lpa
            if not math.isnan(ract):
                a[4] -= ract; a[5] += 1
    rows.append((name, agg))
keys = ["all"] + labels
print(f"{'pack':6} {'doc':18} {'n':>5} {'top-1':>7} {'KL20':>8} {'ppl':>8} {'ref ppl':>8}")
for k in keys:
    for name, agg in rows:
        a = agg.get(k)
        if not a or not a[0]:
            continue
        print(f"{name:6} {k:18} {a[0]:5d} {a[1] / a[0]:7.4f} {a[2] / a[0]:8.4f} {math.exp(a[3] / a[0]):8.3f} {math.exp(a[4] / max(a[5], 1)):8.3f}")
