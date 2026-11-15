"""S15 cross-layer expert predictor study on REAL activations (flash_decode --dump-moe, Q4, 2,000 tokens).

Question: given layer l's MoE input x_l (the moment layer l's router runs), how well can we name layer l+d's top-10
experts, so their misses can be read while layers l..l+d-1 compute (I/O hidden behind GPU)?
  R0  early routing (what the engine does): top-k of router_{l+d}(x_l)
  R1  R0 + ridge correction: z_{l+d} ~ router_{l+d}(x_l) + x_l M, M fitted on the first 70% of tokens (dual form)
  R2  R1 + temporal prior: + beta * onehot(routing of token t-1 at layer l+d)
Scored on the last 30% (different text): precision@10 vs the true top-10, and the miss-coverage that matters for
I/O: with an LRU pool (Q2 slots at 11.34 GiB), the fraction of DEMAND MISSES at layer l+d whose expert was in the
predictor's top-N (and not cached) d layers earlier, and the wasted reads per token.
"""
import os
from collections import OrderedDict
import numpy as np

D = __import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15")
L, HID, E = 48, 2560, 512
X = np.memmap(os.path.join(D, "calib_moe_q4.f32"), dtype=np.float32, mode="r")
T = X.size // (L * HID)
X = np.asarray(X[: T * L * HID]).reshape(T, L, HID)
route = []
with open(os.path.join(D, "calib_trace_q4.txt")) as f:
    hdr = f.readline().split()
    for t in range(int(hdr[1])):
        route.append([list(map(int, f.readline().split()[1:])) for _ in range(int(hdr[2]))])
T = min(T, len(route))
R = np.array(route[:T])  # T x L x 10, sorted by weight
split = int(T * 0.7)
W = []
for l in range(L):
    u = np.fromfile(os.path.join(D, "router", f"model.language_model.layers.{l}.mlp.gate.weight.bin"), dtype=np.uint16).astype(np.uint32) << 16
    W.append(u.view(np.float32).reshape(E, HID))
print(f"tokens {T}, calib {split}")

# sanity: router(x_l) top-10 must reproduce the recorded routing at the same layer
z = X[split:, 5] @ W[5].T
top = np.argsort(-z, axis=1)[:, :10]
print("sanity same-layer top-10 match:", np.mean([len(set(top[i]) & set(R[split + i, 5])) / 10 for i in range(T - split)]))


def ridge_dual(Xc, Zc, lam):
    K = Xc @ Xc.T
    A = np.linalg.solve(K + lam * np.trace(K) / K.shape[0] * np.eye(K.shape[0]), Zc)
    return Xc.T @ A  # HID x E


def prec(scores, truth, k=10):
    top = np.argsort(-scores, axis=1)[:, :k]
    return np.mean([len(set(top[i]) & set(truth[i])) / 10 for i in range(len(truth))])


preds = {}  # (d, method) -> T x L x E scores (eval rows only filled)
res = {}
for d in (1, 2, 3):
    for name in ("R0", "R1", "R2"):
        preds[(d, name)] = np.full((T, L, 10), -1, dtype=np.int32)
    p = {k: [] for k in ("R0", "R1", "R2")}
    for l in range(L - d):
        tgt = l + d
        xl = X[:, l].astype(np.float64)
        z0 = xl @ W[tgt].T.astype(np.float64)
        ztrue = X[:, tgt].astype(np.float64) @ W[tgt].T.astype(np.float64)
        M = ridge_dual(xl[:split], (ztrue - z0)[:split], 0.3)
        z1 = z0 + xl @ M
        prev = np.zeros((T, E))
        for t in range(1, T):
            prev[t, R[t - 1, tgt]] = 1.0
        sd = z1[:split].std()
        z2 = z1 + 0.5 * sd * prev
        ev = slice(split, T)
        for name, zz in (("R0", z0), ("R1", z1), ("R2", z2)):
            p[name].append(prec(zz[ev], R[ev, tgt]))
            preds[(d, name)][ev, tgt] = np.argsort(-zz[ev], axis=1)[:, :10]
    res[d] = {k: float(np.mean(v)) for k, v in p.items()}
    print(f"depth {d}: precision@10  R0 {res[d]['R0']:.3f}  R1 {res[d]['R1']:.3f}  R2 {res[d]['R2']:.3f}", flush=True)

# ---- miss coverage under LRU (eval stretch, cache warmed by the calibration stretch)
n_slots = int(11.34 * (1 << 30) // 1556480)
for d, name, N in [(1, "R0", 5), (1, "R1", 5), (1, "R2", 5), (1, "R1", 10), (2, "R1", 5), (2, "R2", 5), (2, "R1", 10), (3, "R2", 5)]:
    c = OrderedDict()
    demand_miss = covered = wasted_reads = pref_reads = 0
    pending = {}  # key -> token index when prefetched
    for t in range(T):
        for l in range(L):
            for e in R[t, l]:
                k = l * E + e
                if k in c:
                    if k in pending:
                        covered += t >= split
                        del pending[k]
                    c.move_to_end(k)
                else:
                    demand_miss += t >= split
                    if len(c) >= n_slots:
                        old, _ = c.popitem(last=False)
                        pending.pop(old, None)
                    c[k] = 1
            # after layer l's routing: prefetch top-N predicted for layer l+d (eval stretch only)
            if t >= split and l + d < L:
                for e in preds[(d, name)][t, l + d][:N]:
                    k = (l + d) * E + int(e)
                    if k not in c:
                        pref_reads += 1
                        if len(c) >= n_slots:
                            old, _ = c.popitem(last=False)
                            pending.pop(old, None)
                        c[k] = 1
                        pending[k] = t
    ntok = T - split
    tot_miss = demand_miss + covered
    print(f"  d={d} {name} top{N}: demand misses left {demand_miss / ntok:5.1f}/tok, covered {covered / ntok:5.1f}/tok "
          f"({covered / max(tot_miss, 1):.0%} of misses), prefetch reads {pref_reads / ntok:5.1f}/tok, "
          f"wasted {(pref_reads - covered) / ntok:5.1f}/tok", flush=True)
