"""S30 long-context decode A/B on a stored conversation (console side; stop ps5lm-server first).
    python3 longctx_ab.py SESSION_DIR [N_GEN=300] [-- extra flash_decode args]
Copies nothing: pass a COPY of a session directory's parent as STORE (env), default ~/ps5lm/sess-ab. For each variant
(PS5LM_IDX_OLD=1, then the S30 path) it starts `flash_decode --serve` with the server.env flags minus the biased
routing (exact, deterministic), sends the conversation's stored tokens greedily and prints the engine's D line.
"""
import os, struct, subprocess, sys, time, hashlib
args = sys.argv[1:]
extra = []
if "--" in args:
    i = args.index("--"); args, extra = args[:i], args[i + 1:]
sd = args[0]; NG = int(args[1]) if len(args) > 1 else 300
b = open(os.path.join(sd, "meta"), "rb").read()
fl = struct.unpack_from("<I", b, 8)[0]; q = 12 + fl + 8
nt, nc = struct.unpack_from("<II", b, q); q += 8
ck = struct.unpack_from(f"<{nc}I", b, q); q += 4 * nc
toks = struct.unpack_from(f"<{nt}i", b, q)
print(f"conversation {sd}: {nt} tokens, resume points {list(ck)}", flush=True)
flags = []
for l in open("/etc/ps5lm/server.env"):
    if l.startswith("PS5LM_ARGS="): flags = l.strip().split("=", 1)[1].split()
drop = {"--route-bias": 1, "--prefetch-max": 1, "--route-pf": 0, "--kv-store": 1}
f2, i = [], 0
while i < len(flags):
    if flags[i] in drop: i += 1 + drop[flags[i]]; continue
    f2.append(flags[i]); i += 1
B = ("/home/steam/ps5lm/build14"); M = ("/home/steam/ps5lm/model")
store = os.path.dirname(os.path.abspath(sd))
cmd = [B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto",
       "--predict", "1", "--snapshots", "2", "--residency", M + "/residency_q3.txt", "--warm", M + "/residency_q3.txt",
       "--kv-store", store] + f2 + extra
req = "G n=%d t=0 ids=%s\n" % (NG, ",".join(map(str, toks)))
for name, env in (("old (PS5LM_IDX_OLD=1)", {"PS5LM_IDX_OLD": "1"}), ("S30", {})):
    e = dict(os.environ); e.update(env)
    log = open(("/home/steam/ps5lm/longctx_ab.%s.log" % ("old" if env else "s30")), "w")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=e)
    while True:
        l = p.stdout.readline()
        if not l: sys.exit(f"{name}: engine exited before READY (see {log.name})")
        if l.startswith("READY"): break
    t0 = time.time(); p.stdin.write(req); p.stdin.flush()
    out = []
    while True:
        l = p.stdout.readline()
        if not l: print("engine died"); break
        if l.startswith("T "): out.append(l.split()[1])
        elif l.startswith("D "): print(f"{name}: tokens md5 {hashlib.md5(' '.join(out).encode()).hexdigest()[:12]} | {l.strip()}", flush=True); break
    p.stdin.write("Q\n"); p.stdin.flush(); p.wait(timeout=120)
