"""cmp_dump — compare two flash_decode --dump files (per token: (layers+1) x 10240 f32 hyper-connection state).

    python3 cmp_dump.py A.bin B.bin [--layers 48]
Prints the worst relative error max|a-b| / max|b| over all (token, layer-step) snapshots and where it occurs.
Bit-identical refactors must report 0; reordered reductions (fusions, split attention) ~1e-6..1e-5.
"""
import sys
import numpy as np

a, b = sys.argv[1], sys.argv[2]
L = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 48
x = np.fromfile(a, dtype=np.float32)
y = np.fromfile(b, dtype=np.float32)
step = (L + 1) * 10240
n = min(x.size, y.size) // step
x = x[: n * step].reshape(n, L + 1, 10240)
y = y[: n * step].reshape(n, L + 1, 10240)
err = np.abs(x - y).max(axis=2) / np.maximum(np.abs(y).max(axis=2), 1e-30)
t, l = np.unravel_index(np.argmax(err), err.shape)
print(f"{n} tokens: worst rel err {err.max():.3e} at token {t} layer-step {l}; median {np.median(err):.3e}; "
      f"final-step worst {err[:, -1].max():.3e}; finite={np.isfinite(x).all()}")
bad = err > 1e-4
if bad.any():
    tok_bad = np.where(bad.any(axis=1))[0]
    first = [int(np.argmax(bad[t])) for t in tok_bad]
    print(f"snapshots > 1e-4: {int(bad.sum())} in {tok_bad.size} tokens; tokens {tok_bad[:12].tolist()} first bad layer-step {first[:12]}")
