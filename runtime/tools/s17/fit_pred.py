"""S17: refit the cross-layer expert predictor (W'_{l} for depth 1: router_{l+1}(x_l) + x_l M) on many more tokens of
the CURRENT engine's activations (tools/s17/pred_data.sh: per prompt moe_i.f32 [T][48][2560] f32 + trace_i.txt).

Primal ridge (N >> 2560 now): M = (X^T X + lam' I)^-1 X^T (Z - X W^T), Z = router_{l+1}(x_{l+1}) (the true logits).
Prompt-level held-out split. Reports precision@10 (true top-10 of layer l+1 from the trace) for: R0 = router_{l+1}(x_l),
OLD = the deployed S15 matrices (if given), NEW at several lambdas. Exports the best lambda's matrices with --export.

  python fit_pred.py DATA_DIR ROUTER_DIR [--old OLD_PRED_DIR] [--test 3] [--export OUT_DIR] [--lams 0.003,0.01,0.03,0.1]
"""
import argparse, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("router")
ap.add_argument("--old")
ap.add_argument("--test", type=int, default=3, help="number of prompts held out (the last ones)")
ap.add_argument("--export")
ap.add_argument("--lams", default="0.003,0.01,0.03,0.1,0.3")
ap.add_argument("--depth", type=int, default=1)
a = ap.parse_args()
L, HID, E, K = 48, 2560, 512, 10
D = a.depth

ids = sorted(int(re.search(r"moe_(\d+)\.f32", f).group(1)) for f in glob.glob(os.path.join(a.data, "moe_*.f32")))
X, R = [], []
for i in ids:
    m = np.memmap(os.path.join(a.data, f"moe_{i}.f32"), dtype=np.float32, mode="r")
    with open(os.path.join(a.data, f"trace_{i}.txt")) as f:
        hdr = f.readline().split()
        nt = int(hdr[1])
        tr = np.array([list(map(int, f.readline().split()[1:])) for _ in range(nt * L)], dtype=np.int32).reshape(nt, L, K)
    T = min(m.size // (L * HID), nt)
    X.append(m[: T * L * HID].reshape(T, L, HID))
    R.append(tr[:T])
    print(f"prompt {i}: {T} tokens", flush=True)
ntest = a.test
train = list(range(len(ids) - ntest))
test = list(range(len(ids) - ntest, len(ids)))
print(f"train tokens {sum(X[j].shape[0] for j in train)}, test tokens {sum(X[j].shape[0] for j in test)}", flush=True)


def router(l):
    u = np.fromfile(os.path.join(a.router, f"model.language_model.layers.{l}.mlp.gate.weight.bin"), dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(E, HID).astype(np.float64)


def bf16_load(p):
    u = np.fromfile(p, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(E, HID).astype(np.float64)


def to_bf16(arr):
    u = arr.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def prec(Wt, Xs, truth):  # Wt: HID x E
    s = Xs @ Wt
    top = np.argpartition(-s, K, axis=1)[:, :K]
    return np.mean([len(set(top[i].tolist()) & set(truth[i].tolist())) / K for i in range(len(truth))])


lams = [float(x) for x in a.lams.split(",")]
tot = {"R0": [], "OLD": [], **{lam: [] for lam in lams}}
best_lam_count = {lam: 0 for lam in lams}
if a.export:
    os.makedirs(a.export, exist_ok=True)
for l in range(L - D):
    W = router(l + D)
    Xtr = np.concatenate([X[j][:, l] for j in train]).astype(np.float64)
    Ztr = np.concatenate([X[j][:, l + D] for j in train]).astype(np.float64) @ W.T
    Xte = np.concatenate([X[j][:, l] for j in test]).astype(np.float64)
    Rte = np.concatenate([R[j][:, l + D] for j in test])
    G = Xtr.T @ Xtr
    B = Xtr.T @ (Ztr - Xtr @ W.T)
    tr_ = np.trace(G) / HID
    res = {"R0": prec(W.T, Xte, Rte)}
    if a.old and os.path.exists(os.path.join(a.old, f"pred_d{D}_l{l}.bf16")):
        res["OLD"] = prec(bf16_load(os.path.join(a.old, f"pred_d{D}_l{l}.bf16")).T, Xte, Rte)
    Ms = {}
    for lam in lams:
        M = np.linalg.solve(G + lam * tr_ * np.eye(HID), B)
        Ms[lam] = M
        res[lam] = prec(W.T + M, Xte, Rte)
    for k, v in res.items():
        tot[k].append(v)
    bl = max(lams, key=lambda x: res[x])
    best_lam_count[bl] += 1
    print(f"layer {l:2d}->{l + D:2d}: R0 {res['R0']:.3f} OLD {res.get('OLD', float('nan')):.3f} " +
          " ".join(f"{lam:g}:{res[lam]:.3f}" for lam in lams), flush=True)
    if a.export:
        # final fit on ALL prompts at the globally chosen lambda is done in a second pass (see below); here: train-only
        pass
print("mean precision@10: " + " ".join(f"{k}: {np.mean(v):.4f}" for k, v in tot.items() if v))
print("best lambda per layer:", best_lam_count)
if a.export:
    lam = max(lams, key=lambda x: np.mean(tot[x]))
    print(f"export: lambda {lam} fitted on ALL prompts -> {a.export}", flush=True)
    for l in range(L - D):
        W = router(l + D)
        Xa = np.concatenate([X[j][:, l] for j in range(len(ids))]).astype(np.float64)
        Za = np.concatenate([X[j][:, l + D] for j in range(len(ids))]).astype(np.float64) @ W.T
        G = Xa.T @ Xa
        M = np.linalg.solve(G + lam * np.trace(G) / HID * np.eye(HID), Xa.T @ (Za - Xa @ W.T))
        to_bf16((W.T + M).T).tofile(os.path.join(a.export, f"pred_d{D}_l{l}.bf16"))
    print("exported", flush=True)
