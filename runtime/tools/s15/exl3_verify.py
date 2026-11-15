"""S15b: verify our reading of the EXL3 format on REAL weights before any engine work.
Range-fetches gate/up/down (trellis, suh, svh, mul1) of experts we hold in BF16 ($PS5LM_STUDY_DIR/experts, layers 10/30 = K=3)
from turboderp/Qwen3.8-Flash-Next-exl3 @ 3.18bpw_h6_ng8, decodes them with the same window/codebook math as the GPU
kernel (exl3_mv2.comp), applies W = diag(suh) H128 W_inner H128 diag(svh), and compares with the BF16 original.
Pass = relative weight error in the range expected for ~3-bit LDLQ (a format misread gives ~1.0+)."""
import json, os, struct, sys, urllib.request
import numpy as np

REPO = "https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/3.18bpw_h6_ng8/"
EXP = __import__("os").path.join(__import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15"), "experts")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
HID, FFN = 2560, 640


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


idx_path = os.path.join(CACHE, "idx318.json")
if not os.path.exists(idx_path):
    open(idx_path, "wb").write(get(REPO + "model.safetensors.index.json"))
wmap = json.load(open(idx_path))["weight_map"]
headers = {}


def tensor(name):
    shard = wmap[name]
    if shard not in headers:
        n = struct.unpack("<Q", get(REPO + shard, (0, 7)))[0]
        headers[shard] = (8 + n, json.loads(get(REPO + shard, (8, 8 + n - 1))))
    base, h = headers[shard]
    m = h[name]
    a, b = m["data_offsets"]
    raw = get(REPO + shard, (base + a, base + b - 1))
    dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "F32": np.float32, "BF16": np.uint16}[m["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(m["shape"])


def bytesum(p):
    return (p & 0xFF) + ((p >> 8) & 0xFF) + ((p >> 16) & 0xFF) + (p >> 24)


# tile geometry: 16-bit window of position p ENDS at bit end(p) (+ one tile for the tail-biting wrap), MSB-first
# funnel over 32-bit words. Integer K: end = K*(p+1) (exllamav3 dq: b1 = p*K + K + 256*K). Half-integer K = KA + 0.5
# (dq8_half): positions alternate KA and KA+1 bits, odd p end at (2KA+1)*(p+1)/2, even p 3 bits... = (2KA+1)*p/2 + KA.
pos = np.arange(256)
t = pos // 8
j = pos % 8
r0 = (t % 4) * 2
row = r0 + np.array([0, 1, 8, 9, 0, 1, 8, 9])[j]
col = t // 4 + np.where(j >= 4, 8, 0)


def geometry(K2):  # K2 = 2*K
    bits = K2 * 128                      # bits per tile
    if K2 % 2 == 0:
        end = (K2 // 2) * (pos + 1)
    else:
        KA = K2 // 2
        end = np.where(pos % 2 == 1, (2 * KA + 1) * (pos + 1) // 2, (2 * KA + 1) * pos // 2 + KA)
    end = end + bits
    words = bits // 32
    hi = (end - 1) // 32
    lo = (end - 16) // 32
    s = (hi + 1) * 32 - end
    return lo % words, hi % words, s, words


def decode_inner(trellis, mul):
    kt, nt, w16 = trellis.shape
    lo, hi, sh, words = geometry(w16 // 8)          # w16 = 16*K uint16 per tile -> 2K = w16/8
    wd = trellis.astype(np.uint16).view(np.uint32).reshape(kt, nt, w16 // 2).astype(np.uint64)  # little-endian pairs
    assert wd.shape[2] == words
    merged = (wd[:, :, lo] << np.uint64(32)) | wd[:, :, hi]
    w = ((merged >> sh.astype(np.uint64)) & np.uint64(0xFFFF)).astype(np.uint64)
    p = (w * np.uint64(mul)) & np.uint64(0xFFFFFFFF)
    s = bytesum(p).astype(np.float64)
    # fp16 path of the CUDA decoder: h = half(1024 + s); v = hfma(h, k_inv, k_bias)
    k_inv = np.frombuffer(struct.pack("<H", 0x1EEE), np.float16)[0]
    k_bias = np.frombuffer(struct.pack("<H", 0xC931), np.float16)[0]
    v = (np.float16(1024.0) + s.astype(np.float16)) * k_inv + k_bias   # float16 arithmetic (numpy rounds per op)
    out = np.zeros((kt * 16, nt * 16), np.float64)
    for q in range(256):
        out[row[q]::16, col[q]::16][:kt, :nt] = v[:, :, q]
    return out


def had128():
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(128)


H = had128()


def full_weight(name):
    tr = tensor(name + ".trellis")
    suh = tensor(name + ".suh").astype(np.float64)
    svh = tensor(name + ".svh").astype(np.float64)
    mul = int(tensor(name + ".mul1").astype(np.uint32).reshape(-1)[0])
    w = decode_inner(tr, mul)                               # [in, out]
    k, n = w.shape
    w = np.einsum("ab,ibn->ian", H, w.reshape(k // 128, 128, n)).reshape(k, n)   # had_l (row blocks)
    w *= suh[:, None]
    w = np.einsum("kib,bc->kic", w.reshape(k, n // 128, 128), H).reshape(k, n)   # had_r (col blocks)
    w *= svh[None, :]
    return w, mul


def bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape).astype(np.float64)



# ---------------------------------------------------------------- PS5Q4 reference code (copied from runtime/tools/s15/qstudy.py)
U4 = np.arange(-8, 8, dtype=np.float64)


def f16(a):
    return a.astype(np.float16).astype(np.float64)


def quant_levels(w, s, lev):
    z = w / np.where(s == 0, 1, s)
    b = (lev[1:] + lev[:-1]) / 2
    return lev[np.searchsorted(b, z)] * s


def fit_scale(w, lev, grid=np.linspace(0.55, 1.25, 29)):
    amax_i = np.argmax(np.abs(w), axis=1)
    m = w[np.arange(w.shape[0]), amax_i][:, None]
    best_e = np.full((w.shape[0], 1), np.inf)
    best_s = np.zeros((w.shape[0], 1))
    for g in grid:
        for s_ in (m / lev.min() * g, m / lev.max() * g):
            s_ = f16(s_)
            e = ((w - quant_levels(w, s_, lev)) ** 2).sum(1, keepdims=True)
            better = e < best_e
            best_e = np.where(better, e, best_e)
            best_s = np.where(better, s_, best_s)
    z = quant_levels(w, best_s, lev) / np.where(best_s == 0, 1, best_s)
    num, den = (w * z).sum(1, keepdims=True), (z * z).sum(1, keepdims=True)
    s2 = f16(np.where(den > 0, num / np.where(den == 0, 1, den), best_s))
    e2 = ((w - z * s2) ** 2).sum(1, keepdims=True)
    return np.where(e2 < best_e, s2, best_s)


def ps5q4(W):
    out = np.empty_like(W)
    for c in range(0, W.shape[1], 32):
        w = W[:, c:c + 32]
        out[:, c:c + 32] = quant_levels(w, fit_scale(w, U4), U4)
    return out


def silu(a):
    return a / (1 + np.exp(-a))


if __name__ == "__main__":
    import time
    D = __import__("os").environ.get("PS5LM_STUDY_DIR", "ps5lm_s15")
    L = 48
    X_all = np.memmap(os.path.join(D, "calib_moe_q4.f32"), dtype=np.float32, mode="r")
    T = X_all.size // (L * HID)
    X_all = X_all[: T * L * HID].reshape(T, L, HID)
    route = []
    with open(os.path.join(D, "calib_trace_q4.txt")) as f:
        hdr = f.readline().split()
        for _ in range(int(hdr[1])):
            route.append([list(map(int, f.readline().split()[1:])) for _ in range(int(hdr[2]))])
    T = min(T, len(route))
    split = int(T * 0.7)
    plan = json.load(open(os.path.join(EXP, "plan.json")))
    layers = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2,10,30,45").split(",")]
    tot = {k: [0.0, 0.0, 0.0, 0.0] for k in ("PS5Q4", "EXL3-3.18")}
    t0 = time.time()
    for l in layers:
        Xe = X_all[split:T, l, :].astype(np.float64)
        per = {k: [0.0, 0.0, 0.0, 0.0] for k in tot}
        werr = []
        for e in plan[str(l)]["pick"]:
            P = f"model.language_model.layers.{l}.mlp.experts.{e}."
            gu = bf16(os.path.join(EXP, f"L{l}_E{e}_gate_up_proj.bf16"), (2 * FFN, HID))
            Wg, Wu = gu[:FFN], gu[FFN:]
            Wd = bf16(os.path.join(EXP, f"L{l}_E{e}_down_proj.bf16"), (HID, FFN))
            xg, _ = full_weight(P + "gate_proj")
            xu, _ = full_weight(P + "up_proj")
            xd, _ = full_weight(P + "down_proj")
            werr.append(np.linalg.norm(xg.T - Wg) / np.linalg.norm(Wg))
            y = (silu(Xe @ Wg.T) * (Xe @ Wu.T)) @ Wd.T
            idx_r = [t - split for t in range(split, T) if e in route[t][l]]
            for k, (qg, qu, qd) in (("PS5Q4", (ps5q4(Wg), ps5q4(Wu), ps5q4(Wd))), ("EXL3-3.18", (xg.T, xu.T, xd.T))):
                yq = (silu(Xe @ qg.T) * (Xe @ qu.T)) @ qd.T
                a = [((yq - y) ** 2).sum(), (y ** 2).sum(), 0.0, 0.0]
                if idx_r:
                    a[2] = ((yq[idx_r] - y[idx_r]) ** 2).sum(); a[3] = (y[idx_r] ** 2).sum()
                for i in range(4):
                    per[k][i] += a[i]; tot[k][i] += a[i]
        K = tensor(f"model.language_model.layers.{l}.mlp.experts.0.gate_proj.trellis").shape[2] / 16
        print(f"L{l} (EXL3 K={K:g}, weight err {np.mean(werr):.3f}, {time.time() - t0:.0f}s): " + " | ".join(
            f"{k} all {np.sqrt(v[0] / v[1]):.4f} routed {np.sqrt(v[2] / max(v[3], 1e-30)):.4f}" for k, v in per.items()), flush=True)
    print("ALL: " + " | ".join(f"{k} all {np.sqrt(v[0] / v[1]):.4f} routed {np.sqrt(v[2] / max(v[3], 1e-30)):.4f}" for k, v in tot.items()))
