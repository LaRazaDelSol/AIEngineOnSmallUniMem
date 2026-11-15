"""S31 long-context tier A/B on a stored conversation (console; stops ps5lm-server + dash, restarts BOTH at the end).
    sudo python3 longctx_tier.py SESSION_COPY_DIR [N_GEN=200]
SESSION_COPY_DIR must be a COPY of a stored conversation (its parent is used as the store; the engine writes to it).
Variants: exact routing (served flags minus the bias: tokens must be identical with and without the tier), then the
served flags as they are, each without / with --tier /dev/ps5lowmem. Prints the engine's D line + its tier/stat lines.
"""
import os, struct, subprocess, sys, hashlib, time
sd = os.path.abspath(sys.argv[1]); NG = int(sys.argv[2]) if len(sys.argv) > 2 else 200
b = open(os.path.join(sd, "meta"), "rb").read()
fl = struct.unpack_from("<I", b, 8)[0]; q = 12 + fl + 8
nt, nc = struct.unpack_from("<II", b, q); q += 8 + 4 * nc
toks = struct.unpack_from(f"<{nt}i", b, q)
print(f"conversation {sd}: {nt} tokens", flush=True)
flags = []
for l in open("/etc/ps5lm/server.env"):
    if l.startswith("PS5LM_ARGS="): flags = l.strip().split("=", 1)[1].split()
def without(fl, drop):
    out, i = [], 0
    while i < len(fl):
        if fl[i] in drop: i += 1 + drop[fl[i]]; continue
        out.append(fl[i]); i += 1
    return out
served = without(flags, {"--kv-store": 1})
exact = without(served, {"--route-bias": 1, "--prefetch-max": 1, "--route-pf": 0})
B = "/home/steam/ps5lm/build-tier"; M = "/home/steam/ps5lm/model"
base = [B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto",
        "--predict", "1", "--snapshots", "2", "--residency", M + "/residency_q3.txt", "--warm", M + "/residency_q3.txt",
        "--kv-store", os.path.dirname(sd)]
req = "G n=%d t=0 ids=%s\n" % (NG, ",".join(map(str, toks)))
subprocess.run(["systemctl", "stop", "ps5lm-server", "ps5lm-dash"])
for _ in range(60):
    if subprocess.run(["pgrep", "-x", "flash_decode"], capture_output=True).returncode: break
    time.sleep(1)
try:
    for name, fl in (("exact", exact), ("exact+tier", exact + ["--tier", "/dev/ps5lowmem"]),
                     ("served", served), ("served+tier", served + ["--tier", "/dev/ps5lowmem"])):
        logp = "/home/steam/ps5lm/s31/lct_%s.log" % name.replace("+", "_")
        log = open(logp, "w")
        p = subprocess.Popen(base + fl, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)
        while True:
            l = p.stdout.readline()
            if not l: sys.exit(f"{name}: engine exited before READY (see {logp})")
            if l.startswith("READY"): break
        p.stdin.write(req); p.stdin.flush()
        out = []
        while True:
            l = p.stdout.readline()
            if not l: print(f"{name}: engine died", flush=True); break
            if l.startswith("T "): out.append(l.split()[1])
            elif l.startswith("D "):
                d = dict(kv.split("=", 1) for kv in l.split()[1:] if "=" in kv)
                print(f"{name}: md5 {hashlib.md5(' '.join(out).encode()).hexdigest()[:12]} resume={d.get('resume')} load_s={d.get('load_s')} "
                      f"tps={d.get('tps')} hit={d.get('hit')} reads/tok={d.get('reads_per_tok')} gpu_ms={d.get('gpu_ms')} io_ms={d.get('io_ms')} "
                      f"tier/tok={d.get('tier_per_tok')} spec_acc={d.get('spec_acc')}", flush=True)
                break
        p.stdin.write("Q\n"); p.stdin.flush(); p.wait(timeout=120)
        log.close()
        for l in open(logp, errors="replace"):
            if "lowmem tier" in l or "displaced weight" in l: print("   " + l.strip(), flush=True)
finally:
    subprocess.run(["systemctl", "reset-failed", "ps5lm-server", "ps5lm-dash"])
    subprocess.run(["systemctl", "start", "ps5lm-server"]); subprocess.run(["systemctl", "start", "ps5lm-dash"])
    print("server", subprocess.run(["systemctl", "is-active", "ps5lm-server"], capture_output=True, text=True).stdout.strip(),
          "dash", subprocess.run(["systemctl", "is-active", "ps5lm-dash"], capture_output=True, text=True).stdout.strip(), flush=True)
