"""S17: cache-policy headroom on a real routing trace (flash_decode --trace-out): LRU vs Belady (OPT) vs frequency-aware
variants, per slot class (EXL3 packs: one LRU per slot size, class = layer set). Demand-only model (no prefetch): the
hit rate a policy reaches when every expert of a layer is looked up at the layer's bind.

  python policy_bound.py TRACE --class0 N0 --class1 N1 [--edge 0,1,2,3,44,45,46,47] [--warm-tokens 56]
"""
import argparse, heapq
from collections import OrderedDict, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--class0", type=int, required=True)
ap.add_argument("--class1", type=int, required=True)
ap.add_argument("--edge", default="0,1,2,3,44,45,46,47")
ap.add_argument("--warm-tokens", type=int, default=56, help="tokens excluded from the hit statistic (prompt)")
a = ap.parse_args()

with open(a.trace) as f:
    hdr = f.readline().split()
    ntok, nl = int(hdr[1]), int(hdr[2])
    rows = [list(map(int, f.readline().split()[1:])) for _ in range(ntok * nl)]
edge = set(map(int, a.edge.split(",")))
cls_of = [1 if l in edge else 0 for l in range(nl)]
cap = [a.class0, a.class1]

# request stream per class: (time, key, token)
streams = [[], []]
for t in range(ntok):
    for l in range(nl):
        c = cls_of[l]
        for e in rows[t * nl + l]:
            streams[c].append((l * 512 + e, t))


def run_lru(st, n):
    od, hit, tot = OrderedDict(), 0, 0
    for k, t in st:
        h = k in od
        if h:
            od.move_to_end(k)
        else:
            if len(od) >= n:
                od.popitem(last=False)
            od[k] = 1
        if t >= a.warm_tokens:
            tot += 1; hit += h
    return hit / tot


def run_opt(st, n):
    nxt = [0] * len(st)
    last = {}
    for i in range(len(st) - 1, -1, -1):
        k = st[i][0]
        nxt[i] = last.get(k, 1 << 60)
        last[k] = i
    res, heap, hit, tot = {}, [], 0, 0
    for i, (k, t) in enumerate(st):
        h = k in res
        if not h and len(res) >= n:
            while True:
                negn, kk = heapq.heappop(heap)
                if res.get(kk) == -negn:
                    del res[kk]
                    break
        res[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
        if t >= a.warm_tokens:
            tot += 1; hit += h
    return hit / tot


def run_lfu_aged(st, n, half_life_tok):
    """evict min decayed-frequency score (score = sum 2^(-age/half_life)); ties -> LRU"""
    import math
    res, score, stamp, hit, tot = set(), defaultdict(float), {}, 0, 0
    lam = math.log(2) / half_life_tok
    for k, t in st:
        # lazily decayed score: s * exp(-lam (t - stamp))
        s = score[k] * math.exp(-lam * (t - stamp.get(k, t))) + 1.0
        score[k], stamp[k] = s, t
        h = k in res
        if not h:
            if len(res) >= n:
                v = min(res, key=lambda kk: (score[kk] * math.exp(-lam * (t - stamp[kk])), stamp[kk]))
                res.discard(v)
            res.add(k)
        if t >= a.warm_tokens:
            tot += 1; hit += h
    return hit / tot


for c in (0, 1):
    st = streams[c]
    print(f"class {c}: {len(st)} lookups, {cap[c]} slots, keys {len({k for k, _ in st})}")
    print(f"  LRU {run_lru(st, cap[c]):.4f}  OPT(Belady) {run_opt(st, cap[c]):.4f}")
    if c == 1 or "--lfu" in __import__("sys").argv:
        for hl in (8, 32, 128):
            print(f"  LFU-aged half-life {hl} tok: {run_lfu_aged(st, cap[c], hl):.4f}")
w = [len(streams[0]), len(streams[1])]
