import sys
from collections import OrderedDict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from policy_sweep import load
tr = {i: load(__import__("os").path.join(__import__("os").environ.get("PS5LM_TRACES", "traces"), "real2_%d.txt") % i)[0] for i in (1, 2, 3)}
def run(seqs, n, count_from):
    c = OrderedDict(); m = 0; t = 0
    for j, toks in enumerate(seqs):
        for tk in toks:
            for l, ex in enumerate(tk):
                for e in ex:
                    k = l * 512 + e
                    if k in c: c.move_to_end(k)
                    else:
                        if j >= count_from: m += 1
                        if len(c) >= n: c.popitem(last=False)
                        c[k] = 1
            if j >= count_from: t += 1
    return m / t
for name, sb in (("Q2", 1556480), ("Q4", 2785280)):
    n = int(11.34 * (1 << 30) // sb)
    cold = run([tr[1][:335]], n, 0)
    warm_other = run([tr[2], tr[3], tr[1][:335]], n, 2)   # pool warmed by OTHER conversations (server)
    cont = run([tr[1][:1000]], n, 0)
    print(f"{name} pool 11.34 GiB: cold 335-token request {cold:5.1f} misses/tok | same request after 2 other convs {warm_other:5.1f} | one 756-token conv from cold {cont:5.1f}")
