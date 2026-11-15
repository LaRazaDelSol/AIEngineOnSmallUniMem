"""Console-side end-to-end check of the session store through the HTTP server (S25).
A chat with a long document is taken two turns deep on the running ps5lm-server, the whole unit is then killed with
SIGKILL (front end and engine, as the OOM killer does), and after systemd restarts it the third turn is sent with the
full history, the way a chat client does. Pass = the third turn resumes from the drive (resume=disk) and reuses nearly
every prompt token.
    python3 session_e2e.py [DOC_TOKENS=12000]
"""
import glob, json, subprocess, sys, time, urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
text = ""
for f in sorted(glob.glob("/usr/lib/python3.14/*.py")):
    text += open(f, errors="replace").read()
    if len(text) > 4 * N:
        break
doc = text[: 4 * N]


def chat(msgs, n=120):
    body = {"model": "ps5", "messages": msgs, "max_tokens": n, "temperature": 0, "stream": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    dt = time.time() - t0
    u = r.get("usage", {})
    e = u.get("ps5lm", {})
    m = r["choices"][0]["message"]
    print(f"  {u.get('prompt_tokens')} prompt tokens, {u.get('completion_tokens')} generated in {dt:.1f} s | resume {e.get('resume')}, "
          f"reused {e.get('reused')}, prefill {e.get('prefill_s')} s, gen {e.get('tps')} tok/s", flush=True)
    return {"role": "assistant", "content": m.get("content") or ""}, e, u


def wait_up(limit=240):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            urllib.request.urlopen("http://127.0.0.1:8081/v1/models", timeout=2).read()
            return time.time() - t0
        except Exception:
            time.sleep(1)
    raise SystemExit("server did not come back")


wait_up()
msgs = [{"role": "system", "content": "You are a concise code reviewer."},
        {"role": "user", "content": "Here is a file.\n\n" + doc + "\n\nName the three most important functions in it."}]
print("turn 1 (document prefill)", flush=True)
a1, _, u1 = chat(msgs)
msgs += [a1, {"role": "user", "content": "Which of those would you refactor first, and why?"}]
print("turn 2 (continues in memory)", flush=True)
a2, _, _ = chat(msgs)
time.sleep(3)  # the save that follows the reply
print("kill -9 of the whole unit (front end + engine)", flush=True)
subprocess.run("sudo systemctl kill -s KILL ps5lm-server; sleep 2; sudo systemctl reset-failed ps5lm-server; sudo systemctl restart ps5lm-server", shell=True)
print(f"  server back after {wait_up():.0f} s", flush=True)
msgs += [a2, {"role": "user", "content": "Write the refactored version of that function."}]
print("turn 3 (after the restart, full history resent)", flush=True)
a3, e3, u3 = chat(msgs)
ok = e3.get("resume") == "disk" and int(e3.get("reused") or 0) > 0.9 * int(u3.get("prompt_tokens") or 1)
print("PASS" if ok else "FAIL", flush=True)
