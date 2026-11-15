"""S19: eviction scores that are NOT a single time-invariant key (mixtures of decay rates, recency terms), evaluated
exactly with numpy at every eviction (argmin over the residents of one class). Same trace format / counting as
policy_sim2.py. A policy is "name:expr" where expr uses numpy arrays over the residents:
  f8 f36 f100 f300  decayed frequency (half-life in tokens), evaluated at the current token
  age               tokens since the last use, n = total uses
  python policy_sim3.py TRACE [...] --pol "lfu36:f36" --pol "mix:f8+0.5*f100"
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--n0", type=int, default=5922)
ap.add_argument("--n1", type=int, default=1333)
ap.add_argument("--edge", default="0,1,2,3,44,45,46,47")
ap.add_argument("--warm-tokens", type=int, default=56)
ap.add_argument("--pol", action="append", default=[])
ap.add_argument("--max-tokens", type=int, default=0)
ap.add_argument("--prior", nargs="*", default=[], help="traces for a static per-expert usage prior (variable pri, mean 1)")
a = ap.parse_args()
HLS = (4, 8, 16, 36, 100, 300)
edge = set(map(int, a.edge.split(",")))
seqs = []  # per token: list of (layer, ids)
for path in a.traces:
    with open(path) as f:
        hdr = f.readline().split()
        ntok, nl = int(hdr[1]), int(hdr[2])
        for t in range(ntok):
            seqs.append([np.array(list(map(int, f.readline().split()[1:])), dtype=np.int64) + l * 512 for l in range(nl)])
if a.max_tokens: seqs = seqs[:a.max_tokens]
NT = len(seqs)
NK = 48 * 512
PRI = np.ones(NK)
if a.prior:
    cnt_p = np.zeros(NK)
    for path in a.prior:
        with open(path) as f:
            hdr = f.readline().split()
            for t in range(int(hdr[1])):
                for l in range(int(hdr[2])):
                    for e in map(int, f.readline().split()[1:]): cnt_p[l * 512 + e] += 1
    PRI = (cnt_p + 0.5) / (cnt_p + 0.5).reshape(48, 512).mean(axis=1).repeat(512)   # mean 1 per layer


def sim(expr):
    lam = {h: np.log(2) / h for h in HLS}
    # decayed counts stored as value at reference time `ts` per key: F_h(t) = F_h(ts) * exp(-lam (t - ts))
    F = {h: np.zeros(NK) for h in HLS}
    ts = np.zeros(NK)
    last = np.full(NK, -1e9)
    cnt = np.zeros(NK)
    cls = np.array([1 if (k // 512) in edge else 0 for k in range(NK)])
    cap = [a.n0, a.n1]
    res = [np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)]
    inres = np.zeros(NK, dtype=bool)
    hit = tot = 0
    for t in range(NT):
        for ids in seqs[t]:
            c = cls[ids[0]]
            for k in ids:
                h = inres[k]
                if not h:
                    if len(res[c]) >= cap[c]:
                        r = res[c]
                        env = {"age": t - last[r], "n": cnt[r], "pri": PRI[r]}
                        for hh in HLS: env[f"f{hh}"] = F[hh][r] * np.exp(-lam[hh] * (t - ts[r]))
                        i = int(np.argmin(eval(expr, {"np": np}, env)))
                        inres[r[i]] = False
                        r[i] = k
                    else:
                        res[c] = np.append(res[c], k)
                    inres[k] = True
                # update the key's counters (after the eviction decision, like the engine)
                for hh in HLS: F[hh][k] = F[hh][k] * np.exp(-lam[hh] * (t - ts[k])) + 1.0
                ts[k] = t; last[k] = t; cnt[k] += 1
                if t >= a.warm_tokens: tot += 1; hit += h
    return hit / tot, (tot - hit) / (NT - a.warm_tokens)


print(f"{len(a.traces)} trace(s), {NT} tokens, slots {a.n0}/{a.n1}")
for p in a.pol:
    name, expr = p.split(":", 1)
    h, m = sim(expr)
    print(f"  {name:14s} hit {h:.4f}  misses/token {m:.2f}   [{expr}]", flush=True)
