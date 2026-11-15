"""Console-side probe of the on-disk session store (S25, runtime/include/ps5lm/session_store.hpp).
Starts its own `flash_decode --serve --kv-store DIR` (stop ps5lm-server first) and checks the three cases the store
exists for, each against a real engine death (SIGKILL, what the OOM killer sends):
  A  a long prompt is prefilled and answered                       -> baseline time and tokens
  B  kill -9, restart, the same prompt                             -> resume=disk, same tokens, seconds not minutes
     then the conversation continues (APPEND more tokens)          -> only the new tokens are computed
  C  a second long prompt is killed half-way through its prefill   -> after the restart it resumes from the last
                                                                      periodic save, not from zero
  D  back to the first conversation                                -> resume=disk again (two conversations kept)
    python3 session_probe.py [N_TOKENS=30000] [APPEND=2000] [-- extra flash_decode args]
Default engine flags = PS5LM_ARGS of /etc/ps5lm/server.env (pass exact-routing flags after -- for a bit-exact compare).
"""
import glob, os, shutil, signal, subprocess, sys, time
from tokenizers import Tokenizer

args = sys.argv[1:]
extra = []
if "--" in args:
    i = args.index("--")
    args, extra = args[:i], args[i + 1:]
N = int(args[0]) if args else 30000
APP = int(args[1]) if len(args) > 1 else 2000
if not extra:
    for l in open("/etc/ps5lm/server.env"):
        if l.startswith("PS5LM_ARGS="):
            extra = l.strip().split("=", 1)[1].split()
STORE = os.environ.get("STORE", os.path.expanduser("~/ps5lm/sess-probe"))
if "--kv-store" not in extra:
    extra += ["--kv-store", STORE]
EVERY = int(os.environ.get("EVERY", "8192"))
if "--kv-store-every" not in extra:
    extra += ["--kv-store-every", str(EVERY)]
B = os.path.expanduser("~/ps5lm/build14")
M = os.path.expanduser("~/ps5lm/model")
HERE = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(M + "/tokenizer.json")
text = ""
for f in sorted(glob.glob("/usr/lib/python3.14/*.py")):
    text += open(f, errors="replace").read()
    if len(text) > 6 * (2 * N + APP):
        break
body = tok.encode(text).ids
head = tok.encode("<|im_start|>user\nRead this source code.\n\n").ids
head2 = tok.encode("<|im_start|>user\nSummarise the following module.\n\n").ids
tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
p1 = head + body[:N] + tail
p2 = head + body[:N] + body[N:N + APP] + tail
p3 = head2 + body[N + APP:2 * N + APP] + tail
cmd = ["sudo", B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto",
       "--predict", "1", "--snapshots", "2"] + extra
p = None


def start():
    global p
    subprocess.run(["bash", HERE + "/preflight.sh", "--fix"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.expanduser("~/ps5lm/bench/session_probe.err"), "a"),
                         text=True, bufsize=1)
    print(f"engine up in {time.time() - t0:.1f} s: {p.stdout.readline().strip()}", flush=True)


def kill9():
    subprocess.run("sudo kill -9 $(pgrep -x flash_decode)", shell=True)
    p.wait()
    time.sleep(2)
    print("-- engine killed with SIGKILL", flush=True)


def run(ids, label, snap, n=32, kill_at=None):
    print(f"== {label}: {len(ids)} tokens", flush=True)
    p.stdin.write(f"G n={n} stop=248046,248044 snap={snap} ids={','.join(map(str, ids))}\n")
    p.stdin.flush()
    t0 = time.time()
    out = []
    while True:
        l = p.stdout.readline()
        if not l:
            print("engine died", flush=True)
            sys.exit(1)
        f = l.split()
        if f[0] == "P" and kill_at and int(f[1]) >= kill_at:
            print(f"  at position {f[1]} after {time.time() - t0:.1f} s", flush=True)
            kill9()
            return None
        if f[0] == "T":
            out.append(int(f[1]))
        elif f[0] == "D":
            d = dict(x.split("=", 1) for x in f[1:] if "=" in x)
            print(f"  done in {time.time() - t0:.2f} s: resume {d.get('resume')}, reused {d.get('reused')} of {d.get('prompt')}, prefill_s {d.get('prefill_s')}, "
                  f"load_s {d.get('load_s')}, gen tps {d.get('tps')}", flush=True)
            p.stdin.write("I\n")  # answered only after the save that follows the reply
            p.stdin.flush()
            t1 = time.time()
            info = p.stdout.readline().strip()
            print(f"  saved in {time.time() - t1:.2f} s | {info}", flush=True)
            return out


if os.environ.get("KEEP") != "1":
    subprocess.run(["sudo", "rm", "-rf", STORE])
open(os.path.expanduser("~/ps5lm/bench/session_probe.err"), "w").close()
print("engine:", " ".join(cmd[2:]), flush=True)
start()
a = run(p1, "A  full prefill", len(head) + N)
kill9()
start()
b = run(p1, "B  the same prompt after kill -9 + restart", len(head) + N)
print(f"  tokens identical to A: {a == b}   A {a[:8]}  B {b[:8]}", flush=True)
run(p2, f"B2 the conversation continues (+{APP} tokens)", len(p2))
run(p3, "C  another conversation, killed half-way through its prefill", len(p3), kill_at=len(p3) * 6 // 10)
start()
run(p3, "C2 that prompt again after the restart", len(p3))
d = run(p1, "D  back to the first conversation", len(head) + N)
print(f"  tokens identical to A: {a == d}", flush=True)
p.stdin.write("Q\n")
p.stdin.flush()
p.wait()
subprocess.run("grep -a 'session store' ~/ps5lm/bench/session_probe.err | tail -30", shell=True)
subprocess.run(f"sudo du -sh {STORE}; ls {STORE}", shell=True)
