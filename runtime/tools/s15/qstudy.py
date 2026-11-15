"""S15 quantizer study: which code gives PS5Q4-level expert-output error at the fewest bits?

Inputs: BF16 experts (fetch_experts.py), the engine's real MoE inputs x per layer (flash_decode --dump-moe, Q4 run)
and that run's routing trace. Metric: relative error of the full expert output down(silu(gate x) * up x) on
held-out tokens (last 30% of the run, i.e. a different stretch of text), aggregated over the sampled experts.
"""
import json, os, sys, time
import numpy as np
from scipy.linalg import hadamard

D = __import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15")
EXP = os.path.join(D, "experts")
HID, FFN, L = 2560, 640, 48
LAYERS = [int(x) for x in os.environ.get("QS_LAYERS", "2,10,30,45").split(",")]
plan = json.load(open(os.path.join(EXP, "plan.json")))

# ---------------------------------------------------------------- data
X_all = np.memmap(os.path.join(D, "calib_moe_q4.f32"), dtype=np.float32, mode="r")
T = X_all.size // (L * HID)
X_all = X_all[: T * L * HID].reshape(T, L, HID)
route = []
with open(os.path.join(D, "calib_trace_q4.txt")) as f:
    hdr = f.readline().split()
    for t in range(int(hdr[1])):
        route.append([list(map(int, f.readline().split()[1:])) for _ in range(int(hdr[2]))])
T = min(T, len(route))
split = int(T * 0.7)
print(f"tokens {T} (calib {split}, eval {T - split})")


def bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape).astype(np.float64)


def f16(a):
    return a.astype(np.float16).astype(np.float64)


# ---------------------------------------------------------------- level sets
def lloyd(n, iters=200):
    rng = np.random.default_rng(0)
    s = np.sort(rng.standard_normal(400000))
    c = np.quantile(s, (np.arange(n) + 0.5) / n)
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(b, s)
        c = np.array([s[idx == k].mean() for k in range(n)])
    return c / np.abs(c).max()


LEVELS = {}
for b in (2, 3, 4):
    LEVELS[f"u{b}"] = np.arange(-(1 << (b - 1)), 1 << (b - 1), dtype=np.float64)  # PS5Q-style integer set
    LEVELS[f"nf{b}"] = lloyd(1 << b)


def quant_levels(w, s, lev):
    """w (rows, g), s (rows,1) -> nearest level * s"""
    z = w / np.where(s == 0, 1, s)
    b = (lev[1:] + lev[:-1]) / 2
    return lev[np.searchsorted(b, z)] * s


def fit_scale(w, lev, grid=np.linspace(0.55, 1.25, 29), refit=True):
    """per-row MSE-optimal scale over a grid (both signs for asymmetric sets), fp16-rounded"""
    amax_i = np.argmax(np.abs(w), axis=1)
    m = w[np.arange(w.shape[0]), amax_i][:, None]
    best_e = np.full((w.shape[0], 1), np.inf)
    best_s = np.zeros((w.shape[0], 1))
    cands = []
    for g in grid:
        cands += [m / lev.min() * g, m / lev.max() * g] if lev.min() != -lev.max() else [np.abs(m) / lev.max() * g]
    for s in cands:
        s = f16(s)
        e = ((w - quant_levels(w, s, lev)) ** 2).sum(1, keepdims=True)
        better = e < best_e
        best_e = np.where(better, e, best_e)
        best_s = np.where(better, s, best_s)
    if refit:  # least-squares scale for the chosen codes (PS5Q does the same)
        z = quant_levels(w, best_s, lev) / np.where(best_s == 0, 1, best_s)
        num, den = (w * z).sum(1, keepdims=True), (z * z).sum(1, keepdims=True)
        s2 = f16(np.where(den > 0, num / np.where(den == 0, 1, den), best_s))
        e2 = ((w - z * s2) ** 2).sum(1, keepdims=True)
        best_s = np.where(e2 < best_e, s2, best_s)
    return best_s


def rtn(W, lev, g):
    out = np.empty_like(W)
    for c in range(0, W.shape[1], g):
        w = W[:, c:c + g]
        out[:, c:c + g] = quant_levels(w, fit_scale(w, lev), lev)
    return out


def gptq(W, H, lev, g, blk=128, damp=0.01):
    W = W.copy()
    H = H.copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * np.mean(np.diag(H)) * np.eye(H.shape[0])
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T  # upper
    Q = np.zeros_like(W)
    cols = W.shape[1]
    for i1 in range(0, cols, blk):
        i2 = min(i1 + blk, cols)
        W1 = W[:, i1:i2].copy()
        Q1 = np.zeros_like(W1)
        E1 = np.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        s = None
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                s = fit_scale(W1[:, i:i + g], lev)
            w = W1[:, i]
            q = quant_levels(w[:, None], s, lev)[:, 0]
            Q1[:, i] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= np.outer(e, Hi[i, i:])
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def rht(n, blk, seed):
    rng = np.random.default_rng(seed)
    Hb = hadamard(blk) / np.sqrt(blk)
    R = np.kron(np.eye(n // blk), Hb)
    return R * rng.choice([-1.0, 1.0], n)[None, :]  # R = Hblk . diag(s): orthogonal


R_HID = rht(HID, 512, 1)
R_FFN = rht(FFN, 128, 2)


def silu(a):
    return a / (1 + np.exp(-a))


# ---------------------------------------------------------------- codes: name -> (bpw, fn(Wg,Wu,Wd,Hx,Hh) -> (Wg,Wu,Wd))
def code_ps5q(b):
    lev = LEVELS[f"u{b}"]
    return b + 0.5, lambda Wg, Wu, Wd, Hx, Hh: (rtn(Wg, lev, 32), rtn(Wu, lev, 32), rtn(Wd, lev, 32))


def code_gptq(levname, g, rot):
    lev = LEVELS[levname]
    b = int(levname[-1])

    def fn(Wg, Wu, Wd, Hx, Hh):
        if rot:
            Rg, Rd = R_HID, R_FFN
            Hx2, Hh2 = Rg @ Hx @ Rg.T, Rd @ Hh @ Rd.T
            qg = gptq(Wg @ Rg.T, Hx2, lev, g) @ Rg
            qu = gptq(Wu @ Rg.T, Hx2, lev, g) @ Rg
            qd = gptq(Wd @ Rd.T, Hh2, lev, g) @ Rd
            return qg, qu, qd
        return gptq(Wg, Hx, lev, g), gptq(Wu, Hx, lev, g), gptq(Wd, Hh, lev, g)
    return b + 16 / g, fn


def code_rtn_rot(levname, g):
    lev = LEVELS[levname]
    b = int(levname[-1])
    return b + 16 / g, lambda Wg, Wu, Wd, Hx, Hh: (rtn(Wg @ R_HID.T, lev, g) @ R_HID, rtn(Wu @ R_HID.T, lev, g) @ R_HID,
                                                  rtn(Wd @ R_FFN.T, lev, g) @ R_FFN)


CODES = {
    "PS5Q4 (baseline)": code_ps5q(4),
    "PS5Q2": code_ps5q(2),
    "RHT+RTN nf4 g64": code_rtn_rot("nf4", 64),
    "GPTQ u4 g128 (no rot)": code_gptq("u4", 128, False),
    "RHT+GPTQ u4 g128": code_gptq("u4", 128, True),
    "RHT+GPTQ nf4 g128": code_gptq("nf4", 128, True),
    "RHT+GPTQ nf3 g64": code_gptq("nf3", 64, True),
    "RHT+GPTQ u3 g64": code_gptq("u3", 64, True),
    "RHT+GPTQ nf3 g128": code_gptq("nf3", 128, True),
    "RHT+GPTQ nf2 g64": code_gptq("nf2", 64, True),
}
only = sys.argv[1:] if len(sys.argv) > 1 else None
if only:
    CODES = {k: v for k, v in CODES.items() if any(o in k for o in only)}

res = {k: {"num": 0.0, "den": 0.0, "num_r": 0.0, "den_r": 0.0, "per": []} for k in CODES}
t0 = time.time()
for l in LAYERS:
    Xl = X_all[:, l, :].astype(np.float64)
    Xc, Xe = Xl[:split], Xl[split:]
    Hx = Xc.T @ Xc / split
    for e in plan[str(l)]["pick"]:
        gu = bf16(os.path.join(EXP, f"L{l}_E{e}_gate_up_proj.bf16"), (2 * FFN, HID))
        Wg, Wu = gu[:FFN], gu[FFN:]
        Wd = bf16(os.path.join(EXP, f"L{l}_E{e}_down_proj.bf16"), (HID, FFN))
        routed_c = [t for t in range(split) if e in route[t][l]]
        routed_e = [t for t in range(split, T) if e in route[t][l]]
        hc = silu(Xc @ Wg.T) * (Xc @ Wu.T)
        Hh = hc.T @ hc / split
        if routed_c:  # routed tokens are the true input distribution: upweight them
            hr = hc[routed_c]
            Hh = Hh + 10 * hr.T @ hr / len(routed_c)
            Hx_e = Hx + 10 * Xc[routed_c].T @ Xc[routed_c] / len(routed_c)
        else:
            Hx_e = Hx
        y = (silu(Xe @ Wg.T) * (Xe @ Wu.T)) @ Wd.T
        idx_r = [t - split for t in routed_e]
        for k, (bpw, fn) in CODES.items():
            qg, qu, qd = fn(Wg, Wu, Wd, Hx_e, Hh)
            yq = (silu(Xe @ qg.T) * (Xe @ qu.T)) @ qd.T
            n, d = ((yq - y) ** 2).sum(), (y ** 2).sum()
            r = res[k]
            r["num"] += n; r["den"] += d
            if idx_r:
                r["num_r"] += ((yq[idx_r] - y[idx_r]) ** 2).sum(); r["den_r"] += (y[idx_r] ** 2).sum()
            r["per"].append((l, e, float(np.sqrt(n / d))))
        print(f"L{l} E{e} routed calib {len(routed_c)} eval {len(routed_e)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"--- after layer {l}")
    for k, (bpw, _) in CODES.items():
        r = res[k]
        print(f"  {k:24s} {bpw:5.3f} bpw  rel-err all {np.sqrt(r['num'] / r['den']):.4f}  routed {np.sqrt(r['num_r'] / max(r['den_r'], 1e-30)):.4f}", flush=True)
json.dump({k: {"bpw": CODES[k][0], "all": float(np.sqrt(v["num"] / v["den"])), "routed": float(np.sqrt(v["num_r"] / max(v["den_r"], 1e-30))), "per": v["per"]}
           for k, v in res.items()}, open(os.path.join(D, "qstudy_L%s.json" % "_".join(map(str, LAYERS))), "w"), indent=1)
