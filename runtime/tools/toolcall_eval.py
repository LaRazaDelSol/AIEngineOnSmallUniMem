"""toolcall_eval — tool-calling quality gate for a pack (S14: Q2 vs Q4), through the real ps5lm server.

    python3 toolcall_eval.py [--url http://localhost:8081] [--out results.json] [--tag q2]

18 cases sharing ONE tool set (so the server's system/tools snapshot is reused and only the user turn is
prefilled). Greedy decoding, thinking off. Each case checks: call vs no-call, function name, and every
argument (exact / contains / numeric). Reports per-case pass/fail and the totals the pack-bpw decision needs:
parse rate (a well-formed <tool_call> when one is expected), name accuracy, full-argument accuracy.
"""
import argparse
import json
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "Search for files matching a glob or text pattern.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                     "max_results": {"type": "integer"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                    "required": ["command"]}}},
    {"type": "function", "function": {"name": "create_calendar_event", "description": "Create a calendar event.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 datetime"},
                                                     "duration_min": {"type": "integer"},
                                                     "attendees": {"type": "array", "items": {"type": "string"}}},
                    "required": ["title", "start", "duration_min"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate a math expression exactly.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]

# expect: None = must NOT call a tool (then `answer` lists substrings the reply must contain)
#         (name, {arg: check}) where check is ("eq", v) | ("has", [substrings]) | ("num", v) | ("list_has", [substrings])
CASES = [
    ("What's the weather in Berlin in fahrenheit?", ("get_weather", {"city": ("has", ["Berlin"]), "unit": ("eq", "f")})),
    ("Is it raining in Tokyo right now?", ("get_weather", {"city": ("has", ["Tokyo"])})),
    ("Find all Python files under src/ and show me at most 5.", ("search_files", {"pattern": ("has", ["py"]), "path": ("has", ["src"]), "max_results": ("num", 5)})),
    ("Run `ls -la /var/log` with a 10 second timeout.", ("run_shell", {"command": ("eq", "ls -la /var/log"), "timeout_s": ("num", 10)})),
    ("Create a calendar event titled 'Design review' on 2026-11-10 at 14:00 for 45 minutes with alice@example.com and bob@example.com.",
     ("create_calendar_event", {"title": ("has", ["Design review"]), "start": ("has", ["2026-11-10", "14:00"]), "duration_min": ("num", 45),
                                "attendees": ("list_has", ["alice@example.com", "bob@example.com"])})),
    ("Email carol@example.com with subject 'Build fixed' and tell her the CI pipeline is green again.",
     ("send_email", {"to": ("eq", "carol@example.com"), "subject": ("has", ["Build fixed"]), "body": ("has", ["green"])})),
    ("What is 17.5% of 2480? Use the calculator.", ("calculate", {"expression": ("has", ["2480"])})),
    ("What's the capital of Australia?", None, ["Canberra"]),
    ("Search for TODO comments in the docs folder.", ("search_files", {"pattern": ("has", ["TODO"]), "path": ("has", ["docs"])})),
    ("Count the lines in main.cpp using wc.", ("run_shell", {"command": ("has", ["wc", "main.cpp"])})),
    ("Tell me a one-sentence joke about computers.", None, []),
    ('Send an email to dev-team@example.org titled "Release v2.3" saying the release is tagged and the notes are in RELEASE.md.',
     ("send_email", {"to": ("eq", "dev-team@example.org"), "subject": ("has", ["Release v2.3"]), "body": ("has", ["RELEASE.md"])})),
    ("Compute (3**7 - 12) / 5 with the calculator.", ("calculate", {"expression": ("has", ["3", "7", "12", "5"])})),
    ("Weather in São Paulo, celsius please.", ("get_weather", {"city": ("has", ["Paulo"]), "unit": ("eq", "c")})),
    ("Make a 30-minute event called Standup at 2026-11-07T09:30.", ("create_calendar_event", {"title": ("has", ["Standup"]), "start": ("has", ["2026-11-07", "09:30"]), "duration_min": ("num", 30)})),
    ('Run this exact command: echo "hello, world" > out.txt', ("run_shell", {"command": ("has", ['echo "hello, world" > out.txt'])})),
    ("Find files named *.log in /tmp, limit 20 results.", ("search_files", {"pattern": ("has", ["*.log"]), "path": ("has", ["/tmp"]), "max_results": ("num", 20)})),
]
FOLLOWUP = {  # the tool result must be used, not called again
    "messages": [{"role": "user", "content": "What's the weather in Oslo?"},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                                                                       "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
                 {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"}],
    "answer": ["21"],
}


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def check_arg(v, chk):
    kind, want = chk
    if v is None:
        return False
    if kind == "eq":
        return str(v).strip() == want
    if kind == "has":
        return all(w.lower() in str(v).lower() for w in want)
    if kind == "num":
        try:
            return float(v) == float(want)
        except Exception:
            return False
    if kind == "list_has":
        items = v if isinstance(v, list) else [v]
        return all(any(w.lower() in str(i).lower() for i in items) for w in want)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--out", default="toolcall_results.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rows = []
    n_call = n_parse = n_name = n_args = n_nocall = n_nocall_ok = 0
    t0 = time.time()
    for case in CASES + [("<followup>", None, FOLLOWUP["answer"])]:
        prompt, expect = case[0], case[1]
        msgs = FOLLOWUP["messages"] if prompt == "<followup>" else [{"role": "user", "content": prompt}]
        r = post(a.url, {"messages": msgs, "tools": TOOLS, "max_tokens": 160, "temperature": 0})
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        row = {"prompt": prompt, "content": content, "calls": [c["function"] for c in calls], "ps5lm": r["usage"].get("ps5lm")}
        if expect is None:
            n_nocall += 1
            ok = not calls and all(w.lower() in content.lower() for w in case[2])
            n_nocall_ok += ok
            row["pass"] = ok
        else:
            n_call += 1
            name, args = expect
            parsed = len(calls) >= 1
            n_parse += parsed
            fn = calls[0]["function"] if parsed else {}
            try:
                got = json.loads(fn.get("arguments", "{}")) if parsed else {}
            except Exception:
                got = {}
            name_ok = parsed and fn.get("name") == name
            args_ok = name_ok and all(check_arg(got.get(k), c) for k, c in args.items())
            n_name += name_ok
            n_args += args_ok
            row.update({"pass": args_ok, "name_ok": name_ok, "bad_args": [k for k, c in args.items() if not check_arg(got.get(k), c)]})
        rows.append(row)
        print(f"{'PASS' if row['pass'] else 'FAIL'}  {prompt[:60]!r:64} calls={row['calls']} {('content=' + repr(content[:80])) if not calls else ''}"
              f"{(' bad=' + str(row.get('bad_args'))) if row.get('bad_args') else ''}", flush=True)
    summ = {"tag": a.tag, "call_cases": n_call, "parse_rate": n_parse / n_call, "name_acc": n_name / n_call, "full_arg_acc": n_args / n_call,
            "nocall_cases": n_nocall, "nocall_acc": n_nocall_ok / n_nocall,
            "overall": (n_args + n_nocall_ok) / (n_call + n_nocall), "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(summ))
    with open(a.out, "w") as f:
        json.dump({"summary": summ, "rows": rows}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
