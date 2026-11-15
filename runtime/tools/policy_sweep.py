"""policy_sweep — expert-cache replacement policies replayed on REAL Flash-Next routing traces (S14).

    py runtime/tools/policy_sweep.py --trace T1.txt [--trace T2.txt ...] [--calib C.txt] [--slot-bytes 1556480]
                                     [--pool-gib 7 8.5 10 11.5] [--json out.json]

Traces are ps5lm-trace-txt/1 (flash_decode --trace-out). Several traces are replayed back to back through
ONE cache (a server that keeps its pool across requests). Policies:
  lru         global LRU (what the engine runs; must equal trace_replay --policy lru hit-for-hit)
  clock       second-chance CLOCK
  slru        segmented LRU, protected = 50% (llama.cpp #30637 shape)
  lfu         LFU with exponential decay (half-life H accesses per slot-key), evict lowest score
  lfu_adm     LFU-decay + admission: a miss only replaces the victim if its score beats the victim's,
              else it is read into a bypass slot and dropped (homeofe shape; needs K bypass slots)
  perlayer    static partition: pool/48 slots per layer, LRU inside each layer
  pin_lru     the top-F most frequent (layer, expert) of the CALIBRATION trace pinned (F = 50% of pool),
              LRU on the rest (text-calibrated pinning; calibration must be a different text)
  belady      Belady/OPT with bypass: the ceiling for any online policy at this pool size
Reports hit rate, misses/token and misses/layer (the layer-serial queue depth).
"""
import argparse
import heapq
import json
import math
from collections import Counter, OrderedDict


def load(path):
    with open(path) as f:
        hdr = f.readline().split()
        assert hdr[0] == "ps5lm-trace-txt/1", path
        T, Lyr, E = int(hdr[1]), int(hdr[2]), int(hdr[3])
        toks = []
        for _ in range(T):
            lay = []
            for _ in range(Lyr):
                p = f.readline().split()
                lay.append([int(x) for x in p[1:1 + int(p[0])]])
            toks.append(lay)
    return toks, Lyr, E


def flatten(tokens, E):
    seq = []  # (token_index, layer, key)
    for t, lay in enumerate(tokens):
        for l, ex in enumerate(lay):
            for e in ex:
                seq.append((t, l, l * E + e))
    return seq


def run_lru(seq, n):
    c, h = OrderedDict(), 0
    miss = []
    for _, _, k in seq:
        if k in c:
            c.move_to_end(k)
            h += 1
            miss.append(0)
        else:
            if len(c) >= n:
                c.popitem(last=False)
            c[k] = 1
            miss.append(1)
    return miss


def run_clock(seq, n):
    keys, ref, idx, hand = [None] * n, [0] * n, {}, 0
    fill = 0
    miss = []
    for _, _, k in seq:
        s = idx.get(k)
        if s is not None:
            ref[s] = 1
            miss.append(0)
            continue
        miss.append(1)
        if fill < n:
            s, fill = fill, fill + 1
        else:
            while ref[hand]:
                ref[hand] = 0
                hand = (hand + 1) % n
            s, hand = hand, (hand + 1) % n
            del idx[keys[s]]
        keys[s], ref[s], idx[k] = k, 1, s
    return miss


def run_slru(seq, n, prot_frac=0.5):
    npro = int(n * prot_frac)
    prob, pro = OrderedDict(), OrderedDict()
    miss = []
    for _, _, k in seq:
        if k in pro:
            pro.move_to_end(k)
            miss.append(0)
        elif k in prob:
            del prob[k]
            pro[k] = 1
            if len(pro) > npro:
                dk, _ = pro.popitem(last=False)
                prob[dk] = 1  # demoted to probation MRU
            miss.append(0)
        else:
            miss.append(1)
            prob[k] = 1
            while len(prob) + len(pro) > n:
                if prob:
                    prob.popitem(last=False)
                else:
                    pro.popitem(last=False)
    return miss


def run_lfu(seq, n, half=2000.0, admission=False):
    """Score = sum of 2^(-(now - t_access)/half) over accesses (decayed frequency), kept per key (ghost too)."""
    lam = math.log(2) / half
    score, stamp = {}, {}
    cached = set()
    heap = []  # (score_at_ref_time normalised to time 0, key) lazy
    miss = []

    def cur(k, now):
        return score.get(k, 0.0) * math.exp(-lam * (now - stamp.get(k, now)))

    for now, (_, _, k) in enumerate(seq):
        s = cur(k, now) + 1.0
        score[k], stamp[k] = s, now
        norm = math.log(s) + lam * now  # monotone in the decayed score at any common time
        if k in cached:
            miss.append(0)
            heapq.heappush(heap, (norm, k))
            continue
        miss.append(1)
        if len(cached) < n:
            cached.add(k)
            heapq.heappush(heap, (norm, k))
            continue
        while True:  # pop the true minimum among cached keys (lazy deletion)
            vn, vk = heap[0]
            if vk not in cached or abs(vn - (math.log(score[vk]) + lam * stamp[vk])) > 1e-9:
                heapq.heappop(heap)
                continue
            break
        if admission and vn >= norm:
            continue  # bypass: read for this use only, keep the victim
        heapq.heappop(heap)
        cached.discard(vk)
        cached.add(k)
        heapq.heappush(heap, (norm, k))
    return miss


def run_perlayer(seq, n, layers):
    q = max(1, n // layers)
    cs = [OrderedDict() for _ in range(layers)]
    miss = []
    for _, l, k in seq:
        c = cs[l]
        if k in c:
            c.move_to_end(k)
            miss.append(0)
        else:
            miss.append(1)
            if len(c) >= q:
                c.popitem(last=False)
            c[k] = 1
    return miss


def run_pin_lru(seq, n, calib_counts, frac=0.5):
    pins = {k for k, _ in calib_counts.most_common(int(n * frac))}
    rest = n - len(pins)
    c = OrderedDict()
    miss = []
    seen_pin = set()
    for _, _, k in seq:
        if k in pins:
            # pinned keys are loaded once (counted as a miss the first time, like a warm-up read)
            miss.append(0 if k in seen_pin else 1)
            seen_pin.add(k)
            continue
        if k in c:
            c.move_to_end(k)
            miss.append(0)
        else:
            miss.append(1)
            if len(c) >= rest:
                c.popitem(last=False)
            c[k] = 1
    return miss


def run_belady(seq, n):
    nxt = [0] * len(seq)
    last = {}
    for i in range(len(seq) - 1, -1, -1):
        k = seq[i][2]
        nxt[i] = last.get(k, math.inf)
        last[k] = i
    cached = {}  # key -> next use
    heap = []  # (-next_use, key)
    miss = []
    for i, (_, _, k) in enumerate(seq):
        if k in cached:
            miss.append(0)
            cached[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        miss.append(1)
        if len(cached) < n:
            cached[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        while True:
            fn, fk = heap[0]
            if fk not in cached or cached[fk] != -fn:
                heapq.heappop(heap)
                continue
            break
        if -fn <= nxt[i]:
            continue  # bypass: the new key is needed later than everything cached
        heapq.heappop(heap)
        del cached[fk]
        cached[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return miss


def summarise(seq, miss, tokens, visits, skip_tokens=0):
    """Hit rate over accesses whose token index >= skip_tokens (steady state if > 0)."""
    m = sum(x for (t, _, _), x in zip(seq, miss) if t >= skip_tokens)
    a = sum(1 for (t, _, _) in seq if t >= skip_tokens)
    nt = tokens - skip_tokens
    return {"hit": round(1 - m / a, 4), "misses_per_token": round(m / nt, 1), "misses_per_layer": round(m / (nt * visits), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--calib", action="append", default=[])
    ap.add_argument("--slot-bytes", type=int, default=1556480)
    ap.add_argument("--pool-gib", type=float, nargs="+", default=[7, 8.5, 10, 11.5])
    ap.add_argument("--policies", nargs="+", default=["lru", "clock", "slru", "lfu", "lfu_adm", "perlayer", "pin_lru", "belady"])
    ap.add_argument("--json")
    a = ap.parse_args()
    tokens, layers, E = [], 0, 512
    for p in a.trace:
        t, layers, E = load(p)
        tokens += t
    seq = flatten(tokens, E)
    calib = Counter()
    for p in a.calib:
        t, _, _ = load(p)
        calib.update(k for _, _, k in flatten(t, E))
    distinct = len({k for _, _, k in seq})
    print(f"{len(tokens)} tokens x {layers} layers, {len(seq)} accesses, {distinct} distinct experts "
          f"({distinct / (layers * E):.1%} of {layers * E}); compulsory-miss floor {distinct / len(seq):.3f}")
    out = []
    warm = len(tokens) // 3
    for g in a.pool_gib:
        n = int(g * (1 << 30) // a.slot_bytes)
        print(f"\npool {g} GiB = {n} slots ({n / (layers * E):.1%} of all experts)   [all tokens | steady state: tokens >= {warm}]")
        for pol in a.policies:
            if pol == "pin_lru" and not calib:
                continue
            f = {"lru": lambda: run_lru(seq, n), "clock": lambda: run_clock(seq, n), "slru": lambda: run_slru(seq, n),
                 "lfu": lambda: run_lfu(seq, n), "lfu_adm": lambda: run_lfu(seq, n, admission=True),
                 "perlayer": lambda: run_perlayer(seq, n, layers), "pin_lru": lambda: run_pin_lru(seq, n, calib),
                 "belady": lambda: run_belady(seq, n)}[pol]
            miss = f()
            s_all, s_ss = summarise(seq, miss, len(tokens), layers), summarise(seq, miss, len(tokens), layers, warm)
            print(f"  {pol:9s} hit {s_all['hit']:.4f}  miss/tok {s_all['misses_per_token']:6.1f}   | steady hit {s_ss['hit']:.4f}"
                  f"  miss/tok {s_ss['misses_per_token']:6.1f}  miss/layer {s_ss['misses_per_layer']:.2f}")
            out.append({"pool_gib": g, "slots": n, "policy": pol, "all": s_all, "steady": s_ss})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
