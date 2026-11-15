"""S30 rk-ring exactness at long context (console, ps5lm-server stopped, run as root).
  sudo python3 ring_ab.py [N=20002] [BASE_BUILD=/home/steam/ps5lm/build14-s30a]
1. base binary vs new binary, same prompt, exact routing: generated tokens must be identical.
2. new binary: answer stored with --kv-store (end point inside a 4-token block), then a fresh engine continues it from
   the drive vs a fresh engine without the store computing the same request: tokens must be identical.
"""
import glob, hashlib, os, shutil, subprocess, sys
from tokenizers import Tokenizer
H = "/home/steam/ps5lm"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20002
BASE = sys.argv[2] if len(sys.argv) > 2 else H + "/build14-s30a"
tok = Tokenizer.from_file(H + "/model/tokenizer.json")
text = ""
for f in sorted(glob.glob("/usr/lib/python3.14/*.py")):
    text += open(f, errors="replace").read()
    if len(text) > 6 * N: break
ids = tok.encode(text).ids
P, X = ids[:N], ids[N + 5000:N + 5040]
flags = []
for l in open("/etc/ps5lm/server.env"):
    if l.startswith("PS5LM_ARGS="): flags = l.strip().split("=", 1)[1].split()
drop = {"--route-bias": 1, "--prefetch-max": 1, "--route-pf": 0, "--kv-store": 1}
f2, i = [], 0
while i < len(flags):
    if flags[i] in drop: i += 1 + drop[flags[i]]; continue
    f2.append(flags[i]); i += 1
M = H + "/model"
def run(build, req_ids, n, store=None, tag=""):
    cmd = [build + "/flash_decode", "--serve", "--model", M, "--shaders", build + "/shaders", "--bits", "3", "--pool-gib", "auto",
           "--predict", "1", "--snapshots", "2", "--residency", M + "/residency_q3.txt", "--warm", M + "/residency_q3.txt"] + f2
    if store: cmd += ["--kv-store", store]
    log = open(H + "/ring_ab.%s.log" % tag, "w")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
    while True:
        l = p.stdout.readline()
        if not l: sys.exit(f"{tag}: engine exited before READY (see {log.name})")
        if l.startswith("READY"): break
    p.stdin.write("G n=%d t=0 ids=%s\n" % (n, ",".join(map(str, req_ids)))); p.stdin.flush()
    out, d = [], ""
    while True:
        l = p.stdout.readline()
        if not l: sys.exit(f"{tag}: engine died (see {log.name})")
        if l.startswith("T "): out.append(int(l.split()[1]))
        elif l.startswith("D "): d = l.strip(); break
    p.stdin.write("Q\n"); p.stdin.flush(); p.wait(timeout=180)
    md = hashlib.md5(" ".join(map(str, out)).encode()).hexdigest()[:12]
    keys = {k: v for k, v in (x.split("=", 1) for x in d.split()[1:])}
    print(f"{tag:14s} md5 {md} resume={keys['resume']} reused={keys['reused']} tps={keys['tps']} spec_acc={keys['spec_acc']} hit={keys['hit']} prefill_s={keys['prefill_s']}", flush=True)
    return out, md
NEW = H + "/build14"
print(f"prompt {len(P)} tokens", flush=True)
_, a = run(BASE, P, 64, tag="base")
g, b = run(NEW, P, 64, tag="new")
print("1. base vs new:", "PASS" if a == b else "DIFF", flush=True)
S = H + "/ring-ab-store"
shutil.rmtree(S, ignore_errors=True)
g2, c = run(NEW, P, 63, store=S, tag="new+store")
Q = P + g2 + X
_, d1 = run(NEW, Q, 32, store=S, tag="resume-disk")
_, d2 = run(NEW, Q, 32, tag="from-scratch")
print("2. resume from drive vs from scratch:", "PASS" if d1 == d2 else "DIFF", flush=True)
shutil.rmtree(S, ignore_errors=True)
