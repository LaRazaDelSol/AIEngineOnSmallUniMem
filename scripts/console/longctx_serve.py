"""S30: rebuild a stored conversation under the current KV format and measure long-context decode (console, root,
ps5lm-server stopped). Reads the token ids from an old session's meta (any format), then with the SERVED flags and the
served store: run 1 prefills it and generates N tokens (decode right after a prefill), run 2 is a fresh engine
resuming it from the drive (decode right after a restore, with the S30 residency warm-up).
  sudo python3 longctx_serve.py OLD_SESSION_DIR [N=300]
"""
import struct, subprocess, sys, time
H = "/home/steam/ps5lm"; B = H + "/build14"; M = H + "/model"
sd = sys.argv[1]; NG = int(sys.argv[2]) if len(sys.argv) > 2 else 300
b = open(sd + "/meta", "rb").read()
fl = struct.unpack_from("<I", b, 8)[0]; q = 12 + fl + 8
nt, nc = struct.unpack_from("<II", b, q); q += 8 + 4 * nc
toks = struct.unpack_from(f"<{nt}i", b, q)
flags = []
for l in open("/etc/ps5lm/server.env"):
    if l.startswith("PS5LM_ARGS="): flags = l.strip().split("=", 1)[1].split()
cmd = [B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto", "--predict", "1",
       "--snapshots", "2", "--residency", M + "/residency_q3.txt", "--warm", M + "/residency_q3.txt"] + flags
print(f"{nt} tokens from {sd}; flags {' '.join(flags)}", flush=True)
req = "G n=%d t=0 ids=%s\n" % (NG, ",".join(map(str, toks)))
for run in ("after-prefill", "after-restore"):
    log = open(f"{H}/longctx_serve.{run}.log", "w")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
    while True:
        l = p.stdout.readline()
        if not l: sys.exit(f"{run}: engine exited before READY")
        if l.startswith("READY"): break
    t0 = time.time(); p.stdin.write(req); p.stdin.flush()
    while True:
        l = p.stdout.readline()
        if not l: sys.exit(f"{run}: engine died")
        if l.startswith("D "): print(f"{run} ({time.time() - t0:.0f} s): {l.strip()}", flush=True); break
    p.stdin.write("Q\n"); p.stdin.flush(); p.wait(timeout=180)
