"""Console-side prefill probe (S24): prefill rate as a function of the position in the context.
Starts its own `flash_decode --serve` (stop ps5lm-server first: two engines cannot share the pool), feeds one long
prompt and timestamps the engine's progress lines, then appends more tokens to the same conversation (the
mid-session case: a tool result arriving at a long context) and times that too.
    python3 prefill_curve.py [N_TOKENS=120000] [APPEND=6000] [-- extra flash_decode args]
Default engine flags = the PS5LM_ARGS of /etc/ps5lm/server.env.
"""
import glob, os, subprocess, sys, time
from tokenizers import Tokenizer

args = sys.argv[1:]
extra = []
if "--" in args:
    i = args.index("--")
    args, extra = args[:i], args[i + 1:]
N = int(args[0]) if args else 120000
APP = int(args[1]) if len(args) > 1 else 6000
if not extra:
    for l in open("/etc/ps5lm/server.env"):
        if l.startswith("PS5LM_ARGS="):
            extra = l.strip().split("=", 1)[1].split()
B = os.path.expanduser("~/ps5lm/build14")
M = os.path.expanduser("~/ps5lm/model")
tok = Tokenizer.from_file(M + "/tokenizer.json")
text = ""
for f in sorted(glob.glob("/usr/lib/python3.14/*.py")):
    text += open(f, errors="replace").read()
    if len(text) > 6 * (N + APP):
        break
body = tok.encode(text).ids
head = tok.encode("<|im_start|>user\nRead this source code.\n\n").ids
tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").ids
p1 = head + body[:N] + tail
p2 = head + body[:N] + tail[:0] + body[N:N + APP] + tail  # the same prefix up to the end of the first body
cmd = ["sudo", B + "/flash_decode", "--serve", "--model", M, "--shaders", B + "/shaders", "--bits", "3", "--pool-gib", "auto",
       "--predict", "1", "--snapshots", "2"] + extra
print("engine:", " ".join(cmd[2:]), flush=True)
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
print(p.stdout.readline().strip(), flush=True)


def avail():
    return [int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0]


def run(ids, label, snap, n=8, step=8192):
    print(f"== {label}: {len(ids)} tokens", flush=True)
    p.stdin.write(f"G n={n} stop=248046,248044 snap={snap} ids={','.join(map(str, ids))}\n")
    p.stdin.flush()
    t0 = time.time()
    last_t, last_p, first = t0, None, None
    while True:
        l = p.stdout.readline()
        if not l:
            print("engine died", flush=True)
            sys.exit(1)
        f = l.split()
        if f[0] == "P":
            pos = int(f[1])
            now = time.time()
            if first is None:
                first = last_p = pos
                last_t = now
            if pos - last_p >= step:
                print(f"  pos {last_p:7d} -> {pos:7d}: {(pos - last_p) / (now - last_t):7.1f} tok/s   (t = {now - t0:6.1f} s, MemAvailable {avail()} MB)", flush=True)
                last_p, last_t = pos, now
        elif f[0] == "D":
            d = dict(x.split("=", 1) for x in f[1:] if "=" in x)
            print(f"  done in {time.time() - t0:.1f} s: reused {d.get('reused')}, prefill_s {d.get('prefill_s')}, gen tps {d.get('tps')}, hit {d.get('hit')}, "
                  f"reads/tok {d.get('reads_per_tok')}, gpu_ms {d.get('gpu_ms')}, io_ms {d.get('io_ms')}, MemAvailable {avail()} MB", flush=True)
            return


cut = len(head) + N  # snapshot at the end of the shared prefix, so the second request resumes there
run(p1, "full prefill", cut)
run(p2, f"append {APP} tokens at {N}", len(p2), n=64, step=2048)
p.stdin.write("Q\n")
p.stdin.flush()
p.wait()
