"""Export the S15 trained cross-layer predictor as router-shaped bf16 matrices: W'_{l,d} = W_{l+d} + M_{l,d}^T,
so the engine's early-routing dispatch (router.comp on x_l) computes router_{l+d}(x_l) + x_l M with no extra cost.
Fitted on ALL calibration tokens (the held-out score is in predictor.log). Out: <out>/pred_d{d}_l{l}.bf16 [512][2560]."""
import os, sys
import numpy as np

D = __import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15")
OUT = sys.argv[1]
DEPTHS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2]
LAM = 0.3
L, HID, E = 48, 2560, 512
os.makedirs(OUT, exist_ok=True)
X = np.memmap(os.path.join(D, "calib_moe_q4.f32"), dtype=np.float32, mode="r")
T = X.size // (L * HID)
X = np.asarray(X[: T * L * HID]).reshape(T, L, HID)
W = []
for l in range(L):
    u = np.fromfile(os.path.join(D, "router", f"model.language_model.layers.{l}.mlp.gate.weight.bin"), dtype=np.uint16).astype(np.uint32) << 16
    W.append(u.view(np.float32).reshape(E, HID).astype(np.float64))


def to_bf16(a):  # round to nearest even
    u = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r


for d in DEPTHS:
    for l in range(L - d):
        xl = X[:, l].astype(np.float64)
        z0 = xl @ W[l + d].T
        zt = X[:, l + d].astype(np.float64) @ W[l + d].T
        K = xl @ xl.T
        A = np.linalg.solve(K + LAM * np.trace(K) / T * np.eye(T), zt - z0)
        M = xl.T @ A  # HID x E
        Wp = W[l + d] + M.T
        to_bf16(Wp).tofile(os.path.join(OUT, f"pred_d{d}_l{l}.bf16"))
    print("depth", d, "done", flush=True)
