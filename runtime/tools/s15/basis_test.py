"""S15b: do Flash-Next's routed experts share structure (the premise of basis + delta streaming)?
On the 18 real BF16 experts per layer (layers 2/10/30/45):
 (1) mean-expert residual: ||W_e - mean||^2 / ||W_e||^2 (1.0 = nothing shared; D2-MoE/ResMoE need << 1)
 (2) shared input subspace: stack all experts' gate+up rows (18*1280 x 2560); energy captured by the top-k right
     singular vectors vs. the same k for ONE expert's own rows, and vs. a random Gaussian matrix of the same shape.
 (3) best possible rank-r delta on top of a shared rank-R basis (Sub-MoE form W_e ~ W_e V V^T, V shared): residual."""
import json, os
import numpy as np

EXP = __import__("os").path.join(__import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15"), "experts")
plan = json.load(open(os.path.join(EXP, "plan.json")))
HID, FFN = 2560, 640


def bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape).astype(np.float64)


for l in (2, 10, 30, 45):
    G = []
    for e in plan[str(l)]["pick"]:
        gu = bf16(os.path.join(EXP, f"L{l}_E{e}_gate_up_proj.bf16"), (2 * FFN, HID))
        G.append(gu)
    G = np.stack(G)  # 18 x 1280 x 2560
    mean = G.mean(0)
    res = np.mean([((g - mean) ** 2).sum() / (g ** 2).sum() for g in G])
    S = G.reshape(-1, HID)
    sv = np.linalg.svd(S, compute_uv=False) ** 2
    own = np.linalg.svd(G[0], compute_uv=False) ** 2
    rnd = np.linalg.svd(np.random.default_rng(0).standard_normal(S.shape), compute_uv=False) ** 2
    out = []
    for k in (256, 640, 1280):
        out.append(f"k={k}: stack {sv[:k].sum() / sv.sum():.3f} / one expert {own[:k].sum() / own.sum():.3f} / random {rnd[:k].sum() / rnd.sum():.3f}")
    # shared input basis V (top-R right singular vectors of the stack): residual of each expert outside span(V)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    shared = []
    for R in (640, 1280, 1920):
        V = Vt[:R].T
        shared.append(f"R={R}: {np.mean([(((g - g @ V @ V.T) ** 2).sum() / (g ** 2).sum()) for g in G]):.3f}")
    print(f"L{l}: mean-expert residual {res:.3f} (1.0 = no shared expert)")
    print("      energy in top-k input directions: " + " | ".join(out))
    print("      residual outside a shared rank-R input basis: " + " | ".join(shared), flush=True)
