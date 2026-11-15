"""Console-side served timing probe (S24): ONE streamed chat completion, repeated; prints the time to the first
generated text, the total, and what the engine reports, so wall time the engine does not account for shows up.
    python3 serve_gap.py [repeats=3] [max_tokens=100]
(Lives here, not in /tmp: the server's preflight clears tmpfs on every start.)
"""
import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:" + os.environ.get("PORT", "8081") + "/v1/chat/completions"
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
prompt = os.environ.get("PROMPT", "Describe three sorting algorithms and compare their complexity in a short table.")
for i in range(reps):
    body = {"model": "ps5", "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    first = None
    usage = {}
    ntok = 0
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        if not line.startswith(b"data: "):
            continue
        d = line[6:].strip()
        if d == b"[DONE]":
            break
        j = json.loads(d)
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                ntok += 1
                if first is None:
                    first = time.time() - t0
    tot = time.time() - t0
    e = usage.get("ps5lm") or {}
    acc = float(e.get("prefill_s") or 0) + (usage.get("completion_tokens", 0) / float(e.get("tps") or 1e9))
    print(f"run {i + 1}: first text {first if first is None else round(first, 2)} s, total {tot:.2f} s, {usage.get('completion_tokens')} tokens | engine: prefill {e.get('prefill_s')} s, "
          f"gen {e.get('tps')} tok/s, resume {e.get('resume')} reused {e.get('reused')}, reads/tok {e.get('reads_per_tok')}, spec acc {e.get('spec_acc')}, "
          f"think {e.get('think_tokens')}/{e.get('think_budget')} | accounted {acc:.2f} s, unaccounted {tot - acc:.2f} s", flush=True)
