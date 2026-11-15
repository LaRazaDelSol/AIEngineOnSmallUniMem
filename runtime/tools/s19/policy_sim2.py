"""S19: eviction + ADMISSION policies on real routing traces (ps5lm-trace-txt/1), per slot class, vs Belady.

The engine's LFU (S17) overwrites the minimum-score resident on every miss, so every one-off expert evicts something
that has been used. Window policies (W-TinyLFU shape) keep new arrivals in a small LRU window; on a miss the slot
overwritten is EITHER the window's LRU victim (if its score <= the main area's minimum) OR the main minimum (and the
window victim is promoted, no copy: a slot just changes membership). Scores = decayed frequency (half-life in tokens),
kept across evictions (like slot_cache.hpp Policy::LFU).

  python policy_sim2.py TRACE [TRACE ...] --n0 5922 --n1 1333 [--edge 0,1,2,3,44,45,46,47] [--warm-tokens 56]
Several traces are concatenated (a long session with topic changes).
"""
import argparse, heapq, math
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--n0", type=int, default=5922)
ap.add_argument("--n1", type=int, default=1333)
ap.add_argument("--edge", default="0,1,2,3,44,45,46,47")
ap.add_argument("--warm-tokens", type=int, default=56)
ap.add_argument("--policies", default="lru,lfu36,w1lfu36,w2lfu36,w5lfu36,w10lfu36,w2lfu72,w2lfu18,opt")
a = ap.parse_args()

edge = set(map(int, a.edge.split(",")))
streams = [[], []]
T0 = 0
first_counted = []
for path in a.traces:
    with open(path) as f:
        hdr = f.readline().split()
        ntok, nl = int(hdr[1]), int(hdr[2])
        for t in range(ntok):
            for l in range(nl):
                ids = list(map(int, f.readline().split()[1:]))
                for e in ids:
                    streams[1 if l in edge else 0].append((l * 512 + e, T0 + t))
    first_counted.append(T0 + a.warm_tokens)
    T0 += ntok
count_from = a.warm_tokens  # only the first trace's prompt is excluded (later prompts are part of the session)


def counted(t):
    return t >= count_from


class Score:
    def __init__(self, hl):
        self.lam = math.log(2) / hl
        self.s = {}

    def bump(self, k, t):  # returns the time-invariant key log(score) + lam * t
        ls = self.s.get(k)
        if ls is None: sc = 1.0
        else: sc = math.exp(ls - self.lam * t) + 1.0
        key = math.log(sc) + self.lam * t
        self.s[k] = key
        return key

    def key(self, k):
        return self.s.get(k, -1e30)


def lru(st, n):
    od, hit, tot = OrderedDict(), 0, 0
    for k, t in st:
        h = k in od
        if h: od.move_to_end(k)
        else:
            if len(od) >= n: od.popitem(last=False)
            od[k] = 1
        if counted(t): tot += 1; hit += h
    return hit, tot


def lfu(st, n, hl):
    sc, res, heap, hit, tot = Score(hl), set(), [], 0, 0
    for k, t in st:
        h = k in res
        kv = sc.bump(k, t)
        if not h:
            if len(res) >= n:
                while True:
                    v, kk = heapq.heappop(heap)
                    if kk in res and sc.key(kk) == v:
                        res.discard(kk); break
            res.add(k)
        heapq.heappush(heap, (kv, k))
        if counted(t): tot += 1; hit += h
    return hit, tot


def wlfu(st, n, hl, wfrac):
    W = max(1, int(round(n * wfrac / 100.0)))
    M = n - W
    sc, win, main, heap, hit, tot = Score(hl), OrderedDict(), set(), [], 0, 0

    def main_min():
        while heap:
            v, kk = heap[0]
            if kk in main and sc.key(kk) == v: return v, kk
            heapq.heappop(heap)
        return None

    for k, t in st:
        kv = sc.bump(k, t)
        if k in main:
            hit_ = True
            heapq.heappush(heap, (kv, k))
        elif k in win:
            hit_ = True
            win.move_to_end(k)
        else:
            hit_ = False
            if len(win) + len(main) < n:  # filling: window first, overflow promotes the window LRU into main
                if len(win) >= W:
                    v, _ = win.popitem(last=False)
                    main.add(v); heapq.heappush(heap, (sc.key(v), v))
            else:
                if len(win) == 0:  # (W >= 1 always holds after filling)
                    pass
                else:
                    wv = next(iter(win))
                    mm = main_min()
                    if mm is not None and sc.key(wv) > mm[0]:  # window victim beats main's minimum: promote it
                        heapq.heappop(heap); main.discard(mm[1])
                        win.pop(wv); main.add(wv); heapq.heappush(heap, (sc.key(wv), wv))
                    else:
                        win.pop(wv)
            win[k] = 1
        if counted(t): tot += 1; hit += hit_
    return hit, tot


def opt(st, n):
    # Belady: evict the resident whose next use is furthest
    INF = 1 << 60
    nxt = [INF] * len(st)
    last = {}
    for i in range(len(st) - 1, -1, -1):
        k = st[i][0]
        nxt[i] = last.get(k, INF)
        last[k] = i
    res, heap, hit, tot = {}, [], 0, 0
    for i, (k, t) in enumerate(st):
        h = k in res
        if not h and len(res) >= n:
            while True:
                nu, kk = heapq.heappop(heap)
                if res.get(kk) == -nu:
                    del res[kk]; break
        res[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
        if counted(t): tot += 1; hit += h
    return hit, tot


def run(pol, st, n):
    if pol == "lru": return lru(st, n)
    if pol == "opt": return opt(st, n)
    if pol.startswith("lfu"): return lfu(st, n, float(pol[3:]))
    if pol.startswith("w"):  # w<window %>lfu<hl>
        w, hl = pol[1:].split("lfu")
        return wlfu(st, n, float(hl), float(w))
    raise ValueError(pol)


print(f"{len(a.traces)} trace(s), {T0} tokens; slots class0 {a.n0} class1 {a.n1}; accesses {len(streams[0])} / {len(streams[1])}")
for pol in a.policies.split(","):
    h0, t0 = run(pol, streams[0], a.n0)
    h1, t1 = run(pol, streams[1], a.n1)
    print(f"  {pol:10s} class0 {h0 / t0:.4f} class1 {h1 / t1:.4f} overall {(h0 + h1) / (t0 + t1):.4f}  misses/token {(t0 + t1 - h0 - h1) / (T0 - count_from):.2f}", flush=True)
