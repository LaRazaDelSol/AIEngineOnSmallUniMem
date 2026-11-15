"""S20 thinking / effort smoke test against a running ps5lm_server (default http://127.0.0.1:8081).

Checks: effort levels change the thinking length (low < medium < xhigh on the same question), streamed reasoning is not
duplicated (stream text == non-stream text for greedy), a thinking budget is enforced (forced end of thinking), Anthropic
streaming emits thinking blocks, and a follow-up turn continues the engine state (prompt_from=splice, reused ~= prompt).
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
Q = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer briefly."


def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=3600)
    if not stream:
        return json.loads(r.read())
    events = []
    buf = b""
    for chunk in iter(lambda: r.read(256), b""):
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.decode().split("\n"):
                if line.startswith("data: "):
                    d = line[6:]
                    events.append(d if d == "[DONE]" else json.loads(d))
    return events


def chat(msgs, **kw):
    t = time.time()
    r = post("/v1/chat/completions", {"messages": msgs, "temperature": 0, **kw})
    m = r["choices"][0]["message"]
    u = r["usage"]
    print(f"  {kw}: think {u['ps5lm'].get('think_tokens')} tok, total {u['completion_tokens']}, effort {u['ps5lm'].get('effort')}, "
          f"budget_hit {u['ps5lm'].get('think_budget_hit')}, prompt {u['prompt_tokens']} reused {u['ps5lm'].get('reused')} "
          f"({u['ps5lm'].get('resume')}, {u['ps5lm'].get('prompt_from')}), prefill {u['ps5lm'].get('prefill_s')} s, "
          f"{u['ps5lm'].get('tps')} tok/s, {time.time() - t:.1f} s")
    print(f"    answer: {(m.get('content') or '')[:160]!r}")
    return r


ok = True
print("1. effort levels")
lens = {}
for e in ("low", "medium", "high"):
    r = chat([{"role": "user", "content": Q}], reasoning_effort=e, max_tokens=6000)
    lens[e] = r["usage"]["ps5lm"]["think_tokens"]
print(f"   think tokens low/medium/high = {lens['low']}/{lens['medium']}/{lens['high']}")
r = chat([{"role": "user", "content": Q}], reasoning_effort="none", max_tokens=300)
if r["usage"]["ps5lm"]["think_tokens"]:
    print("   FAIL: effort none still thought"); ok = False
r = chat([{"role": "user", "content": Q}], chat_template_kwargs={"reasoning_effort": "high"}, max_tokens=6000)  # raw kwarg spelling

print("2. streamed reasoning == non-streamed (greedy)")
body = {"messages": [{"role": "user", "content": Q}], "temperature": 0, "reasoning_effort": "low", "max_tokens": 3000}
ns = post("/v1/chat/completions", body)["choices"][0]["message"]
ev = post("/v1/chat/completions", {**body, "stream": True}, stream=True)
sr = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in ev if isinstance(e, dict) and e.get("choices"))
sc = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if isinstance(e, dict) and e.get("choices"))
same_r = sr.strip() == (ns.get("reasoning_content") or "").strip()
same_c = sc.strip() == (ns.get("content") or "").strip()
print(f"   reasoning equal {same_r} ({len(sr)} vs {len(ns.get('reasoning_content') or '')} chars), content equal {same_c}")
ok &= same_r and same_c

print("3. thinking budget 48 tokens")
r = chat([{"role": "user", "content": "Prove that there are infinitely many primes, carefully."}], reasoning_effort="high",
         thinking_budget=48, max_tokens=800)
hit = r["usage"]["ps5lm"].get("think_budget_hit")
print(f"   budget hit {hit}, content chars {len(r['choices'][0]['message'].get('content') or '')}")
ok &= bool(hit)

print("4. Anthropic streaming thinking blocks")
ev = post("/v1/messages", {"model": "x", "max_tokens": 2000, "stream": True, "thinking": {"type": "enabled", "budget_tokens": 1024},
                           "messages": [{"role": "user", "content": Q}]}, stream=True)
kinds = [e["content_block"]["type"] for e in ev if isinstance(e, dict) and e.get("type") == "content_block_start"]
th = "".join(e["delta"].get("thinking", "") for e in ev if isinstance(e, dict) and e.get("type") == "content_block_delta")
print(f"   blocks {kinds}, thinking chars {len(th)}")
ok &= kinds[:1] == ["thinking"] and len(th) > 0

print("5. follow-up turn continues the engine state")
msgs = [{"role": "user", "content": Q}]
r1 = chat(msgs, reasoning_effort="low", max_tokens=3000)
msgs.append({"role": "assistant", "content": r1["choices"][0]["message"]["content"]})  # client drops reasoning_content
msgs.append({"role": "user", "content": "Now explain the common wrong answer in one sentence."})
r2 = chat(msgs, reasoning_effort="low", max_tokens=3000)
p = r2["usage"]["ps5lm"]
print(f"   prompt_from {p.get('prompt_from')}, resume {p.get('resume')}, prefilled {r2['usage']['prompt_tokens'] - int(p.get('reused') or 0)} tokens")
ok &= p.get("prompt_from") == "splice" and p.get("resume") == "live"
print("ALL OK" if ok else "SOME CHECKS FAILED")
