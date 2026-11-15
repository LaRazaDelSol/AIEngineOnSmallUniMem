"""S20 long-prompt server check: a ~N-token document + question, streamed with ps5lm_progress; prints prefill rate,
progress events seen, answer head; then a follow-up turn (must continue live via the splice).
   python3 long_smoke.py FILE NTOK [URL]"""
import json
import sys
import time
import urllib.request

from tokenizers import Tokenizer

FILE, NTOK = sys.argv[1], int(sys.argv[2])
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8081"
tk = Tokenizer.from_file("/home/steam/ps5lm/model/tokenizer.json")
doc = tk.decode(tk.encode(open(FILE, encoding="utf-8", errors="replace").read(), add_special_tokens=False).ids[:NTOK])


def stream(msgs, **kw):
    body = {"messages": msgs, "stream": True, "temperature": 0, "ps5lm_progress": True, **kw}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=7200)
    buf, prog, content, reasoning, usage, first = b"", [], "", "", None, None
    for chunk in iter(lambda: r.read(512), b""):
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.decode().split("\n"):
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                j = json.loads(line[6:])
                if j.get("ps5lm_prefill"):
                    prog.append((round(time.time() - t0, 1), j["ps5lm_prefill"]["done"]))
                if j.get("usage"):
                    usage = j["usage"]
                d = (j.get("choices") or [{}])[0].get("delta", {})
                if (d.get("content") or d.get("reasoning_content")) and first is None:
                    first = time.time() - t0
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
    p = usage["ps5lm"]
    print(f"  prompt {usage['prompt_tokens']} tok, reused {p.get('reused')} ({p.get('resume')}, {p.get('prompt_from')}), prefill {p.get('prefill_s')} s "
          f"({p.get('prefill_tps')} tok/s), first token at {first and round(first, 1)} s, gen {usage['completion_tokens']} "
          f"(think {p.get('think_tokens')}) at {p.get('tps')} tok/s, {len(prog)} progress events {prog[:3]}...{prog[-2:]}")
    print(f"  answer: {content[:300]!r}")
    return content, reasoning


msgs = [{"role": "user", "content": "Here is a document:\n\n" + doc + "\n\nQuestion: in two sentences, what is this document about, "
                                    "and what tok/s numbers does it report?"}]
c, r = stream(msgs, reasoning_effort="low", max_tokens=1500)
msgs += [{"role": "assistant", "content": c}, {"role": "user", "content": "Which single change gave the biggest speedup? One sentence."}]
stream(msgs, reasoning_effort="low", max_tokens=800)
