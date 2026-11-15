"""Console-side served probe (S24): a few chat completions against the running ps5lm-server, with the wall time,
completion tokens / second, and the RSS of the front end and the engine after each (the front end's growth is what
eats the pool reserve and gets the unit OOM-killed).
    python3 serve_probe.py [rounds]
"""
import json, os, subprocess, sys, time, urllib.request

URL = "http://127.0.0.1:" + os.environ.get("PORT", "8081") + "/v1/chat/completions"  # PORT=8080 goes through the dashboard proxy
PROMPTS = [
    ("prose", "Write a detailed essay on the history of the Silk Road and its influence on culture and trade.", 300),
    ("code", "Implement a thread-safe LRU cache in C++20 with O(1) get/put, templated on key and value types. Include unit tests.", 300),
    ("qa", "Explain how a B-tree differs from a B+ tree and when each is the better choice.", 250),
    ("json", "Return a JSON array of 12 objects describing the planets and dwarf planets of the solar system with name, type, moons and radius_km.", 300),
]


def rss():
    out = subprocess.run(["ps", "-eo", "rss,comm"], capture_output=True, text=True).stdout.split("\n")
    r = {}
    for l in out[1:]:
        f = l.split()
        if len(f) == 2 and f[1] in ("python3", "flash_decode"):
            r[f[1]] = max(r.get(f[1], 0), int(f[0]) // 1024)
    return r


rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print("start", rss(), flush=True)
for rnd in range(rounds):
    for name, prompt, n in PROMPTS:
        body = {"model": "ps5", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": False}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        dt = time.time() - t0
        u = r.get("usage", {})
        ct = u.get("completion_tokens", 0)
        txt = (r["choices"][0]["message"].get("content") or "")[:70].replace("\n", " ")
        e = u.get("ps5lm", {})
        print(f"{name:6} {ct:4d} tok in {dt:6.2f} s = {ct / dt:5.1f} tok/s wall | engine gen {e.get('tps')} tok/s, prefill {e.get('prefill_s')} s "
              f"({e.get('resume')}, reused {e.get('reused')}/{u.get('prompt_tokens')}), hit {e.get('hit')}, reads/tok {e.get('reads_per_tok')}, gpu {e.get('gpu_ms')} io {e.get('io_ms')} ms, "
              f"spec acc {e.get('spec_acc')}, think {e.get('think_tokens')} | {rss()} | {txt!r}", flush=True)
