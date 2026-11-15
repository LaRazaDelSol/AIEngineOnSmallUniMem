"""S18 server smoke test: the same multi-turn conversation through `flash_decode --serve` with and without --spec.
Greedy answers must be token-identical (speculative decoding is exact); sampled requests must run and stay coherent.
    sudo systemctl stop ps5lm-server; python3 serve_spec_smoke.py [--spec 1] [--pack experts_exl3_268.pack]
"""
import argparse, subprocess, sys, time
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--spec", type=int, default=0)
ap.add_argument("--pack", default="experts_exl3_268.pack")
ap.add_argument("--out", default="/tmp/spec_smoke.txt")
a = ap.parse_args()
M = "/home/steam/ps5lm/model"
B = "/home/steam/ps5lm/build14"
tok = Tokenizer.from_file(M + "/tokenizer.json")
cmd = ["sudo", B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pack", a.pack,
       "--pool-gib", "auto", "--predict", "1", "--snapshots", "2"]
if a.spec:
    cmd += ["--spec", str(a.spec)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
print(p.stdout.readline().strip(), flush=True)
log = open(a.out, "w")


def chat(msgs):
    t = ""
    for r, c in msgs:
        t += f"<|im_start|>{r}\n" + (f"<think>\n\n</think>\n\n{c}<|im_end|>\n" if r == "assistant" else f"{c}<|im_end|>\n")
    return t + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def gen(text, n=120, extra=""):
    ids = tok.encode(text).ids
    p.stdin.write(f"G n={n} stop=248046,248044 snap={len(ids)} {extra} ids={','.join(map(str, ids))}\n")
    p.stdin.flush()
    out = []
    while True:
        l = p.stdout.readline().split()
        if l[0] == "T":
            out.append(int(l[1]))
        else:
            print("  ", " ".join(l), flush=True)
            break
    log.write(",".join(map(str, out)) + "\n")
    s = tok.decode([i for i in out if i not in (248046, 248044)])
    print("  ->", repr(s[:200]), flush=True)
    return s


q1 = "Explain in three sentences why a B-tree suits disk-based databases."
a1 = gen(chat([("user", q1)]))
a2 = gen(chat([("user", q1), ("assistant", a1), ("user", "Now give a short Python sketch of a B-tree node split.")]), 200)
gen(chat([("user", "List four prime numbers above 100, comma separated.")]), 40)
gen(chat([("user", q1), ("assistant", a1), ("user", "Now give a short Python sketch of a B-tree node split."), ("assistant", a2),
          ("user", "What is its time complexity?")]), 80)
gen(chat([("user", "Write a haiku about SSDs.")]), 40, "t=0.7 p=0.9 k=40 s=7")
gen(chat([("user", "Describe a sunset over the ocean in two sentences.")]), 80, "t=1.0 p=0.95 k=20 s=3 r=1.05")
p.stdin.write("Q\n")
p.stdin.flush()
p.wait()
log.close()
