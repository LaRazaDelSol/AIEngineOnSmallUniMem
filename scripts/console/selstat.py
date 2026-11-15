"""S32 tiering probe: QSA selection churn on a long real conversation (console side; stop ps5lm-server first).
    python3 selstat.py SESSION_DIR [N_TOK=all] [N_GEN=512] [-- extra flash_decode args]
Reads the token ids of a stored conversation (meta; any KV format, only the ids are used), prefills the first N_TOK
of them with exact routing and no speculation (selection is a property of positions), decodes N_GEN tokens greedily
with PS5LM_SELSTAT=1 and prints the engine's selstat lines: for each attention layer, the share of selected blocks by
how long ago they were last selected (never / previous token / <=4 / ... / inside the newest 4096 positions).
"""
import os, struct, subprocess, sys
args = sys.argv[1:]
extra = []
if "--" in args:
    i = args.index("--"); args, extra = args[:i], args[i + 1:]
sd = args[0]
b = open(os.path.join(sd, "meta"), "rb").read()
fl = struct.unpack_from("<I", b, 8)[0]; q = 12 + fl + 8
nt, nc = struct.unpack_from("<II", b, q); q += 8 + 4 * nc
toks = list(struct.unpack_from(f"<{nt}i", b, q))
NT = int(args[1]) if len(args) > 1 and args[1] != "all" else nt
NG = int(args[2]) if len(args) > 2 else 512
toks = toks[:NT]
print(f"conversation {sd}: {nt} tokens stored, prefilling {len(toks)}, decoding {NG}", flush=True)
B = "/home/steam/ps5lm/build14"; M = "/home/steam/ps5lm/model"
cmd = [B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto",
       "--predict", "1", "--snapshots", "2", "--residency", M + "/residency_q3.txt", "--warm", M + "/residency_q3.txt",
       "--max-ctx", "262144", "--kv", "k4v4", "--kv-base", "8192", "--prefill-chunk", "8192", "--wide-slice", "512",
       "--pool-reserve-mib", "256", "--tier", "/dev/ps5lowmem"] + extra
env = dict(os.environ); env["PS5LM_SELSTAT"] = "1"
log = "/home/steam/ps5lm/selstat.log"
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, env=env)
while True:
    l = p.stdout.readline()
    if not l: sys.exit(f"engine exited before READY (see {log})")
    if l.startswith("READY"): break
p.stdin.write("G n=%d t=0 ids=%s\n" % (NG, ",".join(map(str, toks)))); p.stdin.flush()
while True:
    l = p.stdout.readline()
    if not l: sys.exit(f"engine exited (see {log})")
    if l.startswith("D "): print(l.strip()); break
p.stdin.close(); p.wait(timeout=120)
for l in open(log, errors="replace"):
    if l.startswith("selstat"): print(l.rstrip())
