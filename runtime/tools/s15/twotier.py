"""S15 two-tier store: quality and bytes when misses run at a lower-precision base code.

Real routing weights (softmax(router(x)) top-10 on the --dump-moe activations, all 48 layers, 2,000 tokens) +
LRU over the same routing. Routed-MoE output error, assuming independent per-expert errors of similar output norm:
    err^2 = sum_r w_r^2 eps_r^2   (eps = measured per-expert relative error of the code that expert ran with)
reported as a multiple of the all-PS5Q4 error (1.00 = PS5Q4 quality). Policies: base on every miss, or only on
misses at rank >= R (HOBBIT-style: strong experts always load full).
"""
import os
from collections import OrderedDict
import numpy as np

D = __import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15")
L, HID, E = 48, 2560, 512
X = np.memmap(os.path.join(D, "calib_moe_q4.f32"), dtype=np.float32, mode="r")
T = X.size // (L * HID)
X = np.asarray(X[: T * L * HID]).reshape(T, L, HID)
Wt = np.zeros((T, L, 10))
Ids = np.zeros((T, L, 10), dtype=np.int64)
for l in range(L):
    u = np.fromfile(os.path.join(D, "router", f"model.language_model.layers.{l}.mlp.gate.weight.bin"), dtype=np.uint16).astype(np.uint32) << 16
    W = u.view(np.float32).reshape(E, HID)
    z = X[:, l] @ W.T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    idx = np.argsort(-p, axis=1)[:, :10]
    Ids[:, l] = idx
    w = np.take_along_axis(p, idx, 1)
    Wt[:, l] = w / w.sum(1, keepdims=True)
print("mean routing weight by rank:", np.round(Wt.mean((0, 1)), 3))
print("share of sum w^2 by rank:   ", np.round((Wt ** 2).mean((0, 1)) / (Wt ** 2).mean((0, 1)).sum(), 3))

EPS = {"PS5Q4": 0.126, "nf3_3.25": 0.239, "PS5Q2": 0.500}
GIB = 1 << 30


def run(full_bytes, base_bytes, pool_gib, base_eps, min_rank, full_eps=0.126):
    n = int(pool_gib * GIB // full_bytes)
    c = OrderedDict()
    num = den = 0.0
    crit = 0.0
    nm = 0
    for t in range(T):
        for l in range(L):
            for r in range(10):
                k = l * E + int(Ids[t, l, r])
                w2 = Wt[t, l, r] ** 2
                if k in c:
                    c.move_to_end(k)
                    eps = full_eps
                else:
                    nm += 1
                    low = r >= min_rank
                    eps = base_eps if low else full_eps
                    crit += base_bytes if low else full_bytes
                    if len(c) >= n:
                        c.popitem(last=False)
                    c[k] = 1
                num += w2 * eps * eps
                den += w2 * full_eps * full_eps
    return np.sqrt(num / den), nm / T, crit / T / 1e6


print("\nfull code resident | base on miss | pool | policy -> MoE error x PS5Q4, misses/tok, critical-path MB/tok, ms @6.5GB/s")
for full_name, fb, feps in (("PS5Q4 4.5", 2785280, 0.126), ("RHT-nf4 4.25", 2617344, 0.123)):
    for base_name, bb, beps in (("PS5Q2 2.5", 1556480, 0.500), ("nf3 3.25", 1998848, 0.239)):
        for mr in (0, 3, 5, 10):
            e, m, mb = run(fb, bb, 11.34, beps, mr, feps)
            pol = "full only (no base)" if mr == 10 else f"base for rank>={mr}"
            print(f"  {full_name:12s} | {base_name:9s} | 11.34 | {pol:20s} -> x{e:.2f}  {m:5.1f}  {mb:5.0f} MB  {mb / 6.5:5.1f} ms", flush=True)
