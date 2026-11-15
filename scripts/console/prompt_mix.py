#!/usr/bin/env python3
"""S35: how often does a served config make a given high-level choice? Sends the dashboard's exact request (low effort,
temperature 0.7, top-p 0.8, top-k 20) with seeds 1..N and classifies how the answer loads three.js.
    python3 prompt_mix.py TAG [N] [max_tokens] [port]
Writes ~/ps5lm/bench/mix_TAG.jsonl (one line per seed: class, reasoning/content heads)."""
import json, os, re, sys, time, urllib.request

TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 900
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 8081
PROMPT = os.environ.get("PROMPT", "Make a horse running using Three.js")


def classify(c):
    if re.search(r'type=["\']importmap', c): return "esm+importmap"
    if re.search(r'from\s+["\']three["\']', c): return "bare-three(no map)" if "importmap" not in c else "esm+importmap"
    if re.search(r'type=["\']module', c) and re.search(r'examples/jsm|/addons/', c): return "esm-addons-no-map"
    if re.search(r'type=["\']module', c): return "esm-core-only"
    if re.search(r'<script[^>]+src=["\'][^"\']*three(\.min)?\.js', c): return "global-script"
    return "undecided"


out = open(os.path.expanduser(f"~/ps5lm/bench/mix_{TAG}.jsonl"), "w")
counts = {}
for seed in range(1, N + 1):
    body = {"model": "ps5lm", "messages": [{"role": "user", "content": PROMPT}], "stream": False,
            "reasoning_effort": "low", "temperature": 0.7, "top_p": 0.8, "min_p": 0, "top_k": 20, "seed": seed,
            "max_tokens": MAX}
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(60):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600)); break
        except Exception as e:
            if attempt == 59: raise
            time.sleep(5)
    m = r["choices"][0]["message"]
    c, rs = m.get("content") or "", m.get("reasoning_content") or ""
    k = classify(c)
    counts[k] = counts.get(k, 0) + 1
    st = r.get("ps5lm", {})
    out.write(json.dumps({"seed": seed, "class": k, "tps": st.get("tps"), "reasoning": rs[:1500], "content": c[:2500]}) + "\n")
    out.flush()
    print(f"{TAG} seed {seed:2d}: {k:22s} {time.time() - t0:5.1f} s, {st.get('tps')} tok/s", flush=True)
print(f"{TAG} SUMMARY {json.dumps(counts)}", flush=True)
