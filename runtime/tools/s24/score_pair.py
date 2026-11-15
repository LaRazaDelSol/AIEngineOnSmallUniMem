"""S24: direct comparison of engine score files against the EXACT-routing run of the same engine (no reference model
in between, so the 2.68 bpw quantisation noise cancels and only the routing change is measured).
    python3 score_pair.py sc_exact.txt sc_b05.txt[:name] ...
Each score file line: "doc pos argmax lp_actual lp_1,...,lp_20" (log-probabilities of the reference's top-20 ids).
Per run:
  agree   = argmax equal to the exact run's argmax
  KL20    = KL(exact || run) over those 20 ids, both renormalised on them (nats)
  dNLL    = mean (NLL_run - NLL_exact) of the actual next token (nats; > 0 = the run is worse)
  p95 / max of the per-position KL20, and the share of positions with KL20 > 0.1
"""
import math, sys


def load(path):
    out = {}
    for l in open(path):
        f = l.split()
        if len(f) < 5:
            continue
        out[(int(f[0]), int(f[1]))] = (int(f[2]), float(f[3]), list(map(float, f[4].split(","))))
    return out


def norm(lps):
    m = max(lps)
    w = [math.exp(x - m) for x in lps]
    z = sum(w)
    return [x / z for x in w]


base = load(sys.argv[1])
print(f"{'run':10} {'n':>5} {'agree':>7} {'KL20':>8} {'dNLL':>8} {'p95 KL':>8} {'max KL':>8} {'KL>0.1':>7}")
for spec in sys.argv[2:]:
    path, _, name = spec.partition(":")
    run = load(path)
    n = agree = 0
    kls = []
    dnll = 0.0
    for key, (am, lpa, lps) in base.items():
        r = run.get(key)
        if not r:
            continue
        p, q = norm(lps), norm(r[2])
        kls.append(sum(a * (math.log(a) - math.log(max(b, 1e-30))) for a, b in zip(p, q) if a > 0))
        agree += am == r[0]
        dnll += lpa - r[1]
        n += 1
    kls.sort()
    print(f"{(name or path)[:10]:10} {n:5d} {agree / n:7.4f} {sum(kls) / n:8.4f} {dnll / n:8.4f} {kls[int(0.95 * n)]:8.4f} {kls[-1]:8.4f} "
          f"{sum(k > 0.1 for k in kls) / n:7.4f}")
