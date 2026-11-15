"""S17: fast eviction-policy simulator on a real routing trace (flash_decode --trace-out), per slot class.
Policies: LRU, LRU-2 (evict the oldest 2nd-last access), LFU-aged (decayed frequency, O(log n) via the time-invariant key
log(score) + lam * stamp), and the class-split sweep (how to divide a fixed byte budget between the two slot classes).

  python policy_sim.py TRACE --bytes0 1862656 --bytes1 2170880 --total-gib 10.6 [--edge 0,1,2,3,44,45,46,47]
"""
import argparse, heapq, math
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--bytes0", type=int, default=1863680)
ap.add_argument("--bytes1", type=int, default=2170880)
ap.add_argument("--total-gib", type=float, default=10.6)
ap.add_argument("--edge", default="0,1,2,3,44,45,46,47")
ap.add_argument("--warm-tokens", type=int, default=56)
ap.add_argument("--policies", default="lru,lru2,lfu8,lfu32,lfu128")
a = ap.parse_args()

with open(a.trace) as f:
    hdr = f.readline().split()
    ntok, nl = int(hdr[1]), int(hdr[2])
    rows = [list(map(int, f.readline().split()[1:])) for _ in range(ntok * nl)]
edge = set(map(int, a.edge.split(",")))
streams = [[], []]
for t in range(ntok):
    for l in range(nl):
        for e in rows[t * nl + l]:
            streams[1 if l in edge else 0].append((l * 512 + e, t))


def lru(st, n):
    od, hit, tot = OrderedDict(), 0, 0
    for k, t in st:
        h = k in od
        if h: od.move_to_end(k)
        else:
            if len(od) >= n: od.popitem(last=False)
            od[k] = 1
        if t >= a.warm_tokens: tot += 1; hit += h
    return hit / tot


def heap_policy(st, n, keyfn):
    """generic: evict min key; key(k, i, state) recomputed on every access (lazy heap)"""
    res, heap, hit, tot, state = {}, [], 0, 0, {}
    for i, (k, t) in enumerate(st):
        h = k in res
        kv = keyfn(k, i, t, state)
        if not h and len(res) >= n:
            while True:
                v, kk = heapq.heappop(heap)
                if res.get(kk) == v:
                    del res[kk]; break
        res[k] = kv
        heapq.heappush(heap, (kv, k))
        if t >= a.warm_tokens: tot += 1; hit += h
    return hit / tot


def lru2_key(k, i, t, s):
    prev = s.get(k, -1 << 40)
    s[k] = i
    return (prev, i)          # evict the oldest second-to-last access (ties: oldest last access)


def lfu_key(hl):
    lam = math.log(2) / hl
    def f(k, i, t, s):
        ls, st = s.get(k, (None, t))
        # log-score update: score' = score * exp(-lam (t - st)) + 1 ; stored as invariant key log(score) + lam * t
        if ls is None: sc = 1.0
        else: sc = math.exp(ls - lam * st) * math.exp(-lam * (t - st)) + 1.0
        key = math.log(sc) + lam * t
        s[k] = (key, t)
        return (key, i)
    return f


def run(name, st, n):
    if name == "lru": return lru(st, n)
    if name == "lru2": return heap_policy(st, n, lru2_key)
    if name.startswith("lfu"): return heap_policy(st, n, lfu_key(float(name[3:])))
    raise ValueError(name)


tot_bytes = a.total_gib * (1 << 30)
n_lay = [nl - len(edge), len(edge)]
w = [len(streams[0]), len(streams[1])]
# default split = share of layers
f1 = n_lay[1] / nl
n = tot_bytes / (f1 * a.bytes1 + (1 - f1) * a.bytes0)
n1d = round(f1 * n); n0d = int((tot_bytes - n1d * a.bytes1) // a.bytes0)
print(f"default split: class0 {n0d} slots, class1 {n1d} slots ({a.total_gib} GiB)")
for pol in a.policies.split(","):
    h0, h1 = run(pol, streams[0], n0d), run(pol, streams[1], n1d)
    print(f"  {pol:7s} class0 {h0:.4f} class1 {h1:.4f}  overall {(h0 * w[0] + h1 * w[1]) / sum(w):.4f}")
# split sweep (LRU and the best other policy): move slots between classes at equal bytes
for pol in ("lru", "lfu32"):
    best = None
    for d in range(-400, 1201, 100):
        n1 = n1d + d
        n0 = int((tot_bytes - n1 * a.bytes1) // a.bytes0)
        h0, h1 = run(pol, streams[0], n0), run(pol, streams[1], n1)
        ov = (h0 * w[0] + h1 * w[1]) / sum(w)
        print(f"  split {pol} class1 {n1:5d} class0 {n0:5d}: {h0:.4f} / {h1:.4f} overall {ov:.4f}")
        if best is None or ov > best[0]: best = (ov, n0, n1)
    print(f"  best {pol}: overall {best[0]:.4f} at class0 {best[1]} class1 {best[2]}")
