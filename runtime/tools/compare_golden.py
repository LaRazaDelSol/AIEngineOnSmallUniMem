"""Compare flash_decode --dump stream states against ref_qwen4.py golden hidden states, per layer and token.

    python3 compare_golden.py golden.npz engine.dump [--layers N]
"""
import sys
import numpy as np

g = np.load(sys.argv[1])
hid = g["hidden"]  # [NL+1, T, 10240]
NL1, T, D = hid.shape
eng = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, NL1, D)[:T]  # [T, NL+1, D]
print(f"{'layer':>5} " + " ".join(f"tok{t:<6}" for t in range(T)))
worst = 0.0
for l in range(NL1):
    errs = []
    for t in range(T):
        r = hid[l, t]
        e = eng[t, l]
        errs.append(float(np.linalg.norm(e - r) / (np.linalg.norm(r) + 1e-12)))
    worst = max(worst, max(errs))
    print(f"{l:>5} " + " ".join(f"{x:.2e}" for x in errs))
print(f"worst rel err {worst:.3e} -> {'PASS' if worst < 2e-3 else 'CHECK'}")
