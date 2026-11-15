"""S15 ledger: what the real routing traces say about the Q4-quality @ 40 tok/s target.

A. LRU misses/token per pack geometry (slot bytes) at the measured pool (11.34 GiB) and +dense savings.
B. Speculative verify windows: union of experts per layer over w consecutive tokens (GPU expert work) and
   misses per token when w tokens are processed together (I/O), accepted-only vs with rejected drafts.
C. Miss rank: at which routing-weight rank (0 = strongest) misses land (HOBBIT-style low-precision-on-miss).
D. Two-tier (progressive) store: resident = full precision, a miss loads only the base layer.
"""
import sys
from collections import OrderedDict, Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from policy_sweep import load  # noqa: E402

TR = [__import__("os").path.join(__import__("os").environ.get("PS5LM_TRACES", "traces"), "real2_%d.txt") % i for i in (1, 2, 3)]
toks = []
for p in TR:
    t, L, E = load(p)
    toks += t
T = len(toks)
print(f"tokens {T}, layers {L}, experts {E}")
GIB = 1 << 30


def lru_run(n_slots, window=1, extra=None):
    """Process tokens in windows; within a window, layer by layer, the union of the window's experts.
    extra: optional fn(t0) -> list of extra (rejected-draft) tokens' routing appended to the window.
    Returns misses per *accepted* token, union size per layer per window, and miss-rank histogram."""
    c = OrderedDict()
    miss = 0
    union_sum = 0
    rank_miss = Counter()
    rank_all = Counter()
    nwin = 0
    for t0 in range(0, T, window):
        win = toks[t0:t0 + window]
        ext = extra(t0) if extra else []
        nwin += 1
        for l in range(L):
            seen = {}
            for tk in win + ext:
                for r, e in enumerate(tk[l]):
                    if e not in seen or r < seen[e]:
                        seen[e] = r
            union_sum += len(seen)
            for e, r in seen.items():
                k = l * E + e
                rank_all[r] += 1
                if k in c:
                    c.move_to_end(k)
                else:
                    miss += 1
                    rank_miss[r] += 1
                    if len(c) >= n_slots:
                        c.popitem(last=False)
                    c[k] = 1
    return miss / T, union_sum / (nwin * L), rank_miss, rank_all


SLOTS = {"PS5Q2 2.5bpw": 1556480, "3.0bpw": 1843200, "3.25bpw": 1998848, "3.5bpw": 2150400, "PS5Q4 4.5bpw": 2785280}
POOLS = [11.34, 11.8, 12.3]
print("\nA. LRU misses/token and I/O ms/token at 6.5 GB/s (decimal), real traces back to back")
print(f"{'pack':14s} " + " ".join(f"{p:>7.2f}GiB:hit  miss  MB  ms " for p in POOLS))
for name, sb in SLOTS.items():
    row = []
    for p in POOLS:
        n = int(p * GIB // sb)
        m, _, _, _ = lru_run(n)
        hit = 1 - m / (L * 10)
        mb = m * sb / 1e6
        row.append(f"{hit:.3f} {m:5.1f} {mb:5.0f} {mb / 6.5:5.1f}")
    print(f"{name:14s} " + " | ".join(row))

print("\nB. verify windows (accepted tokens only; 3.5 bpw slots at 11.8 GiB)")
n35 = int(11.8 * GIB // 2150400)
base_m, _, _, _ = lru_run(n35, 1)
for w in (1, 2, 3, 4, 6, 8):
    m, u, _, _ = lru_run(n35, w)
    print(f"  w={w}: misses/accepted token {m:5.1f} (x{m / base_m:.3f} of w=1), union experts/layer {u:5.1f} "
          f"(per token {u / w:4.1f}; GPU expert reads x{u / (10 * w):.2f} of serial)")


# rejected drafts: model a rejected draft token's routing as a token from far away in the trace (pessimistic:
# unrelated routing) vs a neighbour (optimistic: a wrong token in the same context routes like its neighbours)
def far(t0):
    return [toks[(t0 + T // 2) % T]]


def near(t0):
    return [toks[min(T - 1, t0 + 4)]]


for w in (2, 3, 4):
    mf, _, _, _ = lru_run(n35, w, far)
    mn, _, _, _ = lru_run(n35, w, near)
    print(f"  w={w} + 1 rejected draft/step: misses/accepted token far {mf:5.1f} (x{mf / base_m:.2f}), near {mn:5.1f} (x{mn / base_m:.2f})")

print("\nC. miss rank (0 = highest routing weight), 3.5 bpw @ 11.8 GiB, serial decode")
_, _, rm, ra = lru_run(n35, 1)
tot = sum(rm.values())
print("  rank:      " + " ".join(f"{r:5d}" for r in range(10)))
print("  miss share " + " ".join(f"{rm[r] / tot:5.3f}" for r in range(10)))
print("  miss rate  " + " ".join(f"{rm[r] / ra[r]:5.3f}" for r in range(10)))

print("\nD. two-tier store: resident experts full precision (F), a miss loads the base layer (B) only;")
print("   the enhancement is fetched later in idle SSD time (not counted on the critical path)")
for F, B, nm in ((2150400, 1228800, "3.5 full / 2.0 base"), (2150400, 1536000, "3.5 / 2.5"), (2785280, 1556480, "Q4 full / Q2 base"),
                 (1843200, 1228800, "3.0 / 2.0")):
    for p in (11.34, 11.8):
        n = int(p * GIB // F)
        m, _, _, _ = lru_run(n)
        print(f"  {nm:20s} pool {p:5.2f}: {m:5.1f} misses/token x base {B / 1e6:.2f} MB = {m * B / 1e6:5.0f} MB = {m * B / 6.5e6:5.1f} ms; "
              f"enhancement backlog {m * (F - B) / 1e6:4.0f} MB/token")
