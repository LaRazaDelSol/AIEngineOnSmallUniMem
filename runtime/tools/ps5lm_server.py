"""ps5lm_server — OpenAI + Anthropic compatible HTTP front end for `flash_decode --serve` (S14).

    sudo python3 ps5lm_server.py --model ~/ps5lm/model --bin ~/ps5lm/build14/flash_decode \
        --shaders ~/ps5lm/build14/shaders [--bits 2] [--pool-gib 8.5] [--port 8081] [--thinking]

The C++ engine owns the GPU, the expert pool and the recurrent/KV state; this process only does HTTP, the
chat template (the model's own chat_template.jinja), tokenisation and output parsing (<think>, qwen3-coder
<tool_call> XML). Requests are served one at a time (batch-1 engine); the engine keeps its state between
requests, so a follow-up turn only prefills its new tokens.

Endpoints: GET /health /v1/models /metrics; POST /v1/chat/completions, /v1/completions, /v1/messages.
"""
import argparse
import array
import ctypes
import bisect
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from tokenizers import Tokenizer

IM_END, EOT = 248046, 248044
STOP_IDS = [IM_END, EOT]
MODEL_NAME = "qwen3.8-flash-next-ps5"


# ----------------------------------------------------------------------------- engine process
class Engine:
    def __init__(self, a):
        cmd = ["sudo", a.bin, "--serve", "--model", a.model, "--shaders", a.shaders, "--bits", str(a.bits),
               "--pool-gib", str(a.pool_gib), "--predict", str(a.predict), "--max-ctx", str(a.max_ctx),
               "--snapshots", str(a.snapshots), "--residency", a.residency]
        if a.pack:  # S17: e.g. experts_exl3_268.pack (EXL3 2.68bpw, the fast Q4-class pack)
            cmd += ["--pack", a.pack]
        if a.spec:  # S18: MTP drafts + one-cb batched verify (exact for greedy and for sampling)
            cmd += ["--spec", str(a.spec), "--spec-gate", str(a.spec_gate)]
        if os.path.exists(a.residency):
            cmd += ["--warm", a.residency]
        if a.engine_args:  # S20: e.g. "--kv k8v4 --kv-base 16384" (quantized / elastic KV)
            cmd += a.engine_args.split()
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.p.stdout.readline().split()
        if not ready or ready[0] != "READY":
            raise RuntimeError(f"engine failed to start: {ready}")
        self.info = dict(kv.split("=") for kv in ready[1:])
        self.last_info = ""
        self.args = {"bits": a.bits, "pack": a.pack, "spec": a.spec, "spec_gate": a.spec_gate, "predict": a.predict, "max_ctx": a.max_ctx}
        self.lock = threading.Lock()
        self.last = {}
        self.totals = {"requests": 0, "prompt_tokens": 0, "reused_tokens": 0, "gen_tokens": 0, "gen_s": 0.0, "prefill_s": 0.0}

    def _send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def generate(self, ids, max_new, sp, snaps, cancelled, on_progress=None):
        """Yields token ids; the final stats dict is stored in self.last. Caller must hold self.lock.
        S20: `P <done> <total>` lines (prefill progress, one per batched chunk group) go to on_progress(done, total)."""
        f = lambda k, d: sp.get(k) if sp.get(k) is not None else d
        req = (f"G n={max_new} t={f('temperature', 0)} p={f('top_p', 1)} k={f('top_k', 0)} m={f('min_p', 0)} "
               f"r={f('repetition_penalty', 1)} pp={f('presence_penalty', 0)} fp={f('frequency_penalty', 0)} "
               f"s={f('seed', 0)} snap={','.join(str(s) for s in snaps if 0 < s <= len(ids))} "
               f"stop={','.join(map(str, STOP_IDS))} ids={','.join(map(str, ids))}")
        self._send(req)
        sent_cancel = False
        while True:
            parts = self.p.stdout.readline().split()
            if not parts:
                raise RuntimeError("engine exited")
            if parts[0] == "T":
                if cancelled() and not sent_cancel:
                    self._send("X")
                    sent_cancel = True
                yield int(parts[1])
            elif parts[0] == "P":
                if cancelled() and not sent_cancel:  # a client that left during a long prefill stops it
                    self._send("X")
                    sent_cancel = True
                if on_progress and len(parts) >= 3:
                    on_progress(int(parts[1]), int(parts[2]))
            elif parts[0] == "D":
                st = dict(kv.split("=", 1) for kv in parts[1:])
                self.last = st
                t = self.totals
                t["requests"] += 1
                t["prompt_tokens"] += int(st.get("prompt", 0))
                t["reused_tokens"] += int(st.get("reused", 0))
                t["gen_tokens"] += int(st.get("gen", 0))
                t["gen_s"] += float(st.get("gen_s", 0))
                t["prefill_s"] += float(st.get("prefill_s", 0))
                return

    def command(self, line):
        with self.lock:
            self._send(line)
            return self.p.stdout.readline().strip()

    def command_nowait(self, line):
        """S18: the engine's answer, or the previous one while a generation holds the engine."""
        if not self.lock.acquire(blocking=False):
            return self.last_info
        try:
            self._send(line)
            self.last_info = self.p.stdout.readline().strip()
            return self.last_info
        finally:
            self.lock.release()


# ----------------------------------------------------------------------------- template + parsing
def make_template(model_dir):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise ValueError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)  # HF semantics: no HTML escaping
    with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8") as fh:
        return env.from_string(fh.read())


TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def coerce(v, schema):
    """qwen3-coder parameters are raw text; convert by the tool's JSON schema type where known."""
    t = (schema or {}).get("type")
    if t == "string":
        return v
    try:
        return json.loads(v)
    except Exception:
        return v


def parse_output(text, tools, thinking):
    """-> (reasoning, content, [ {name, arguments(dict)} ])"""
    reasoning = ""
    if thinking:
        if "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
        else:
            return text.strip(), "", []
    calls = []
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or fn.get("input_schema") or {}).get("properties", {})
    for m in TOOL_RE.finditer(text):
        name = m.group(1).strip()
        args = {k.strip(): coerce(v, schemas.get(name, {}).get(k.strip())) for k, v in PARAM_RE.findall(m.group(2))}
        calls.append({"name": name, "arguments": args})
    if calls:
        text = text[: text.find("<tool_call>")]
    return reasoning, text.strip(), calls


def anthropic_to_messages(body):
    msgs = []
    if body.get("system"):
        s = body["system"]
        msgs.append({"role": "system", "content": s if isinstance(s, str) else "".join(b.get("text", "") for b in s)})
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            msgs.append({"role": m["role"], "content": c})
            continue
        text, calls, thought = [], [], []
        for b in c:
            ty = b.get("type")
            if ty == "text":
                text.append(b["text"])
            elif ty == "thinking":  # S20: a returned thinking block = the turn's reasoning_content
                thought.append(b.get("thinking", ""))
            elif ty == "tool_use":
                calls.append({"function": {"name": b["name"], "arguments": b.get("input", {})}})
            elif ty == "tool_result":
                rc = b.get("content", "")
                if isinstance(rc, list):
                    rc = "".join(x.get("text", "") for x in rc)
                msgs.append({"role": "tool", "content": rc})
        if text or calls or m["role"] == "assistant":
            mm = {"role": m["role"], "content": "".join(text)}
            if thought and m["role"] == "assistant":
                mm["reasoning_content"] = "".join(thought)
            if calls:
                mm["tool_calls"] = calls
            msgs.append(mm)
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema", {})}} for t in body.get("tools", [])]
    return msgs, tools


def openai_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            tc = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", c))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        fn["arguments"] = {}
                tc.append({"function": fn})
            m["tool_calls"] = tc
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


# ----------------------------------------------------------------------------- server
class State:
    engine: Engine
    tok: Tokenizer
    tmpl: jinja2.Template
    thinking_default: bool
    effort_default: str = None             # None = the template's own default (xhigh)
    budgets = {"low": 2048, "medium": 8192, "xhigh": 24576}  # S20 default thinking budgets (tokens) per effort
    default_max = 32768                     # S20: max_tokens when the request gives none (capped by the context left)
    turns = []                              # S20: recent finished turns (prompt text/ids + the raw generated ids)
    turns_lock = threading.Lock()
    turns_dir = None                        # S25: the turns on disk, next to the engine's session store (--kv-store)
    im_start = None                         # S33: id of <|im_start|>; enc: the incremental tokeniser (EncCache)
    enc = None


# S20 reasoning effort: the template accepts xhigh / medium / low; map every API's spelling onto it
EFFORTS = {"xhigh": "xhigh", "high": "xhigh", "max": "xhigh", "maximum": "xhigh", "medium": "medium", "default": "medium",
           "normal": "medium", "low": "low", "minimal": "low", "min": "low"}
EFFORT_OFF = {"none", "off", "disabled", "false", "no"}
# Qwen3's budget-forcing phrase: closes the thinking when the budget runs out, the answer follows
FORCE_END = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def thinking_options(body, anthropic):
    """-> (template kwargs with enable_thinking/reasoning_effort resolved, thinking budget in tokens or None)."""
    kw = dict(body.get("chat_template_kwargs") or {})
    eff = kw.pop("reasoning_effort", None)
    budget = body.get("thinking_budget", body.get("max_thinking_tokens"))
    r = body.get("reasoning")  # OpenRouter style: {"effort": ..., "max_tokens": ..., "enabled": ...}
    if isinstance(r, dict):
        eff = r.get("effort", eff)
        budget = r.get("max_tokens", budget)
        if r.get("enabled") is False:
            kw["enable_thinking"] = False
    if body.get("reasoning_effort") is not None:
        eff = body["reasoning_effort"]
    oc = body.get("output_config")  # Anthropic effort parameter
    if isinstance(oc, dict) and oc.get("effort") is not None:
        eff = oc["effort"]
    th = body.get("thinking")
    if isinstance(th, dict):
        ty = th.get("type")
        if ty == "disabled":
            kw["enable_thinking"] = False
        elif ty in ("enabled", "adaptive"):
            kw["enable_thinking"] = True
            b = th.get("budget_tokens")
            if b:
                budget = b
                if eff is None:
                    eff = "low" if b < 2048 else "medium" if b < 8192 else "xhigh"
    elif isinstance(th, bool):
        kw["enable_thinking"] = th
    if isinstance(eff, str):
        e = eff.strip().lower()
        if e in EFFORT_OFF:
            kw["enable_thinking"] = False
        elif e in EFFORTS:
            kw.setdefault("enable_thinking", True)
            kw["reasoning_effort"] = EFFORTS[e]
        else:
            raise ValueError(f"unknown reasoning effort {eff!r} (use low, medium, high/xhigh or none)")
    think = kw.get("enable_thinking", State.thinking_default)
    kw["enable_thinking"] = bool(think)
    if think:
        if "reasoning_effort" not in kw and State.effort_default:
            kw["reasoning_effort"] = State.effort_default
        if budget is None:
            budget = State.budgets.get(kw.get("reasoning_effort", "xhigh"))
    else:
        kw.pop("reasoning_effort", None)
    return kw, (int(budget) if think and budget else None)


IM_START = "<|im_start|>"

# S34: glibc gives each concurrently running request thread its own malloc arena and keeps every arena's high-water mark
# after the request: the same 145k-token conversation replayed with the dashboard polling grew this process 151 -> 181
# MiB and still climbing (208 MiB at the 2026-11-10 OOM kill; 134 at start). malloc_trim(0) after each request returns
# the free pages of all arenas (120-130 MiB flat in the same test). A few ms per request.
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def trim_heap():
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


class EncCache:
    """S33: incremental tokenisation. Encoding a 194k-token conversation costs 0.4 s of every request's TTFT, and an agent
    turn only appends to the previous prompt. The tokeniser splits added special tokens out before any pre-tokenising, so
    encode(A + B) == encode(A) + encode(B) whenever B starts with <|im_start|>: the ids of the longest shared prefix up to
    such a boundary are reused and only the rest is encoded. Each entry keeps the char offset and the token index of
    every <|im_start|>; an entry whose counts disagree (the literal never tokenised as the special token) is not kept."""

    def __init__(self, n=4):
        self.n, self.items, self.lock = n, [], threading.Lock()
        self.memo = {}

    def _bounds(self, text, ids):
        cs = [m.start() for m in re.finditer(re.escape(IM_START), text)]
        ts = [i for i, t in enumerate(ids) if t == State.im_start]
        return (cs, ts) if len(cs) == len(ts) else None

    @staticmethod
    def _common(a, b):
        lo, hi = 0, min(len(a), len(b))  # longest m with a[:m] == b[:m] (bisection on C-speed slice compares)
        while lo < hi:
            m = (lo + hi + 1) // 2
            if a[:m] == b[:m]:
                lo = m
            else:
                hi = m - 1
        return lo

    def encode(self, text):
        with self.lock:
            items = list(self.items)
        best = None
        for it in items:
            c = self._common(text, it[0])
            cs, ts = it[2]
            j = bisect.bisect_right(cs, c) - 1  # the last boundary inside the shared prefix: text[cs[j]:] starts with IM_START
            while j >= 0 and not text.startswith(IM_START, cs[j]):
                j -= 1
            if j >= 0 and (best is None or cs[j] > best[0]):
                best = (cs[j], it[1][: ts[j]])
        if best and best[0] > 0:
            ids = list(best[1]) + State.tok.encode(text[best[0]:], add_special_tokens=False).ids
        else:
            ids = State.tok.encode(text, add_special_tokens=False).ids
        bd = self._bounds(text, ids)
        if bd:
            with self.lock:
                self.items = [x for x in self.items if not text.startswith(x[0])][-(self.n - 1):] + [(text, array.array("i", ids), bd)]
        return ids

    def encode_memo(self, text):  # the system/tools block: identical across requests
        ids = self.memo.get(text)
        if ids is None:
            ids = State.tok.encode(text, add_special_tokens=False).ids
            if len(self.memo) > 16:
                self.memo.clear()
            self.memo[text] = ids
        return ids


def render(msgs, tools, kw):
    text = State.tmpl.render(messages=msgs, tools=tools or None, add_generation_prompt=True, **kw)
    ids, how = splice(msgs, tools, kw, text)  # S33: before any full encode (a spliced prompt never needed it)
    if ids is None:
        ids, how = State.enc.encode(text), "render"
    # Snapshot positions: the end of the system/tools block (shared across conversations) + the whole prompt.
    snaps = [len(ids)]
    cut = text.find("<|im_start|>user")
    if cut > 0:
        pre = State.enc.encode_memo(text[:cut])
        if len(pre) < len(ids) and ids[: len(pre)] == pre:
            snaps.insert(0, len(pre))
    return ids, snaps, text, how


def splice(msgs, tools, kw, text):
    """S20 follow-up turns: put back the engine's OWN tokens for the previous assistant turn (its raw reasoning + answer,
    exactly as generated), so the engine continues its live state (only the new user turn is prefilled) instead of
    re-prefilling a re-tokenised, reasoning-stripped copy. The template preserves past reasoning by default
    (preserve_thinking), so this is also the format the model expects."""
    with State.turns_lock:
        turns = list(State.turns)
    for c in reversed(turns):
        n = c["n"]
        if len(msgs) <= n + 1 or not text.startswith(c["text"]):
            continue
        m = msgs[n]
        if m.get("role") != "assistant" or (m.get("content") or "").strip() != c["content"]:
            continue
        if [((tc.get("function") or tc).get("name")) for tc in (m.get("tool_calls") or [])] != c["calls"]:
            continue
        try:
            upto = State.tmpl.render(messages=msgs[: n + 1], tools=tools or None, add_generation_prompt=False, **kw)
        except Exception:
            continue
        if not text.startswith(upto) or not upto.endswith("<|im_end|>\n"):
            continue
        rest = State.tok.encode(text[len(upto) - 1:], add_special_tokens=False).ids  # "\n<|im_start|>user..."
        return list(c["ids"]) + list(c["gen"]) + rest, "splice"
    return None, "render"


def remember_turn(text, ids, gen, n_msgs, content, calls, cut=False):
    # cut (S25): the turn ended at max_tokens. The template closes it with <|im_end|> when it comes back as history,
    # so that is what follows the engine's own tokens; without this the next turn re-tokenised the cut answer and
    # the engine fell back to the snapshot before the whole previous prompt.
    if gen and cut and gen[-1] not in STOP_IDS:
        gen = list(gen) + [IM_END]
    if not gen or gen[-1] != IM_END:
        return
    # ids as 4-byte arrays (a list of 120k Python ints is 4 MB; eight turns of a long conversation were 70 MB)
    turn = {"text": text, "ids": array.array("i", ids), "gen": array.array("i", gen), "n": n_msgs, "content": content.strip(),
            "calls": [c["name"] for c in calls]}
    with State.turns_lock:
        State.turns.append(turn)
        del State.turns[:-8]
    save_turn(turn)


# S25: the turns survive a restart. The engine keeps conversations on the SSD (--kv-store); they are found again only
# if the next request carries the same token ids, and for an assistant turn those are the ids the engine generated
# (raw reasoning included), which only this table knows. One JSON file per turn, the newest 8 kept.
def save_turn(turn):
    d = State.turns_dir
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        name = os.path.join(d, f"{time.time_ns()}.json")
        with open(name + ".tmp", "w") as f:
            json.dump({**turn, "ids": turn["ids"].tolist(), "gen": turn["gen"].tolist()}, f)
        os.replace(name + ".tmp", name)
        for old in sorted(x for x in os.listdir(d) if x.endswith(".json"))[:-8]:
            os.unlink(os.path.join(d, old))
    except OSError as e:
        print(f"turns: not saved ({e})", flush=True)


def load_turns():
    d = State.turns_dir
    if not d or not os.path.isdir(d):
        return
    for x in sorted(x for x in os.listdir(d) if x.endswith(".json"))[-8:]:
        try:
            t = json.load(open(os.path.join(d, x)))
            t["ids"], t["gen"] = array.array("i", t["ids"]), array.array("i", t["gen"])
            State.turns.append(t)
        except (OSError, ValueError, KeyError):
            pass
    print(f"turns: {len(State.turns)} restored from {d}", flush=True)


class Decoder:
    """Incremental detokeniser that never emits a partial UTF-8 character. S20: O(1) per token (prefix/read offsets,
    vLLM-style) instead of decoding the whole output every token (16k-token thinking made that quadratic)."""

    def __init__(self):
        self.ids, self.text, self.p, self.r = [], "", 0, 0

    def add(self, t):
        self.ids.append(t)
        prefix = State.tok.decode(self.ids[self.p:self.r], skip_special_tokens=False) if self.r > self.p else ""
        new = State.tok.decode(self.ids[self.p:], skip_special_tokens=False)
        if len(new) <= len(prefix) or new.endswith("�"):
            return ""
        d = new[len(prefix):]
        self.p, self.r = self.r, len(self.ids)
        self.text += d
        return d


def sampling(body):
    sp = {k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                                     "frequency_penalty", "seed")}
    return sp


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, data, event=None):
        try:
            if event:
                self.wfile.write(f"event: {event}\n".encode())
            self.wfile.write(f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def do_GET(self):
        if self.path in ("/", "/chat", "/index.html"):  # S17: browser chat page (same origin as the API)
            try:
                page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ps5lm_chat.html"), "rb").read()
            except OSError:
                return self._json(404, {"error": "ps5lm_chat.html missing next to the server"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/health":
            return self._json(200, {"status": "ok", **State.engine.info})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "ps5lm"}]})
        if self.path == "/metrics":
            t = dict(State.engine.totals)
            t["gen_tps"] = t["gen_tokens"] / t["gen_s"] if t["gen_s"] else 0
            return self._json(200, {"totals": t, "last": State.engine.last, "engine": State.engine.command_nowait("I"),
                                    "info": State.engine.info, "args": State.engine.args})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        try:
            if self.path == "/v1/chat/completions":
                return self.chat(body, anthropic=False)
            if self.path == "/v1/messages":
                return self.chat(body, anthropic=True)
            if self.path == "/v1/completions":
                return self.completion(body)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        finally:
            del body
            trim_heap()
        self._json(404, {"error": "not found"})

    # ---- core generation loop shared by every endpoint ----
    def run(self, ids, snaps, max_new, sp, stops, on_delta, think=False, budget=None, on_progress=None):
        """Streams text deltas to on_delta(str) -> bool (False = client gone). Returns (text, finish, stats, gen_ids).
        S20: think = the prompt opened a <think> block; budget = max thinking tokens: when it runs out the engine is
        stopped and FORCE_END is appended (the engine continues its live state, so the forced tokens are just a short
        batched prefill), then the answer is generated with the rest of max_new. gen_ids = every token the engine printed
        (+ the forced ones), for the follow-up splice."""
        dec = Decoder()
        text, finish = "", "length"
        gone, cancel = [False], [False]  # client left / stop the engine (stop string, budget, client)
        held = ""
        gen_ids, n_think, budget_hit = [], 0, False
        keep = max((len(s) - 1 for s in stops if s), default=0)
        tot = {"gen": 0, "gen_s": 0.0, "prefill_s": 0.0}
        first = None  # stats of the first engine request: where the prompt resumed (a forced end of thinking adds a second one)

        def push(d):
            nonlocal text, held, finish
            text += d
            hit = next((s for s in stops if s and s in text), None)
            if hit:
                text = text[: text.find(hit)]
                finish = "stop"
                cancel[0] = True  # cancels the engine at the next token
                return False
            emit = held + d  # hold back a possible partial stop string
            held = emit[-keep:] if keep else ""
            emit = emit[: len(emit) - len(held)] if keep else emit
            if emit and not on_delta(emit):
                gone[0] = cancel[0] = True
            return True

        with State.engine.lock:
            cur, cur_snaps, left, in_think = ids, snaps, max_new, think
            while True:
                done, forced, seg = False, False, []
                # Always drain the generator to the engine's "D" line (keeps the stdin/stdout protocol in sync).
                for t in State.engine.generate(cur, left, sp, cur_snaps, lambda: cancel[0], on_progress):
                    seg.append(t)
                    if done:
                        continue
                    if t in STOP_IDS:
                        finish = "stop"
                        done = True
                        continue
                    if in_think:
                        n_think += 1
                    d = dec.add(t)
                    if d and not push(d):
                        done = True
                        continue
                    if in_think and "</think>" in dec.text:
                        in_think = False
                    if in_think and budget and n_think >= budget and not cancel[0]:
                        forced = cancel[0] = True  # thinking budget spent: stop, then force the end of thinking
                        done = True
                st = dict(State.engine.last)
                first = first or st
                gen_ids += seg
                for k in tot:
                    tot[k] += float(st.get(k, 0) or 0)
                left -= len(seg)
                if not (forced and in_think and not gone[0] and left > 8):
                    break
                f_ids = State.tok.encode(FORCE_END, add_special_tokens=False).ids
                for t in f_ids:
                    d = dec.add(t)
                    if d:
                        push(d)
                in_think, cancel[0] = False, False
                cur = cur + seg + f_ids
                gen_ids += f_ids
                left -= len(f_ids)
                cur_snaps = []
                budget_hit = True
        if held and finish != "stop":
            on_delta(held)
        if st.get("reason") == "cancel" and finish != "stop":
            finish = "cancelled"
        for k in ("resume", "reused", "load_s"):
            if k in first:
                st[k] = first[k]
        st["gen"] = int(tot["gen"])
        st["gen_s"] = f"{tot['gen_s']:.3f}"
        st["prefill_s"] = f"{tot['prefill_s']:.3f}"
        st["think_tokens"] = n_think
        if budget_hit:
            st["think_budget_hit"] = 1
        return text, finish, st, gen_ids

    def chat(self, body, anthropic):
        if anthropic:
            msgs, tools = anthropic_to_messages(body)
            max_new = body.get("max_tokens")
        else:
            msgs, tools = openai_messages(body.get("messages", [])), body.get("tools")
            max_new = body.get("max_completion_tokens") or body.get("max_tokens")
        kw, budget = thinking_options(body, anthropic)
        think = kw["enable_thinking"]
        if body.get("tool_choice") == "none":
            tools = None
        ids, snaps, ptext, how = render(msgs, tools, kw)
        room = int(State.engine.info.get("max_ctx", 2048)) - len(ids) - 1
        if room < 1:
            raise ValueError(f"prompt is {len(ids)} tokens; the context is {State.engine.info.get('max_ctx')}")
        max_new = min(int(max_new or State.default_max), room)  # S20: thinking needs room (the old default was 1024)
        if budget:
            budget = min(budget, max(max_new - 256, 1))  # leave room for the answer after a forced end of thinking
        stops = body.get("stop") or body.get("stop_sequences") or []
        stops = [stops] if isinstance(stops, str) else stops
        stream = bool(body.get("stream"))
        rid = ("msg_" if anthropic else "chatcmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        sp = sampling(body)
        # Streaming split: reasoning (before </think>) / content / pending tool call. Every part advances a cursor, and a
        # partial tag at the end of the buffer is held back (S20: reasoning used to be re-sent from the start each delta).
        state = {"buf": "", "sent": 0, "in_tool": False, "think_done": not think, "lead": True}

        def safe_end(b, tag):
            for k in range(len(tag) - 1, 0, -1):
                if b.endswith(tag[:k]):
                    return len(b) - k
            return len(b)

        def visible(delta):
            """-> (content, reasoning) parts of the stream not yet forwarded."""
            state["buf"] += delta
            b = state["buf"]
            r = ""
            if not state["think_done"]:
                i = b.find("</think>", state["sent"])
                if i < 0:
                    e = safe_end(b, "</think>")
                    r = b[state["sent"]: e]
                    state["sent"] = max(state["sent"], e)
                    return "", r
                r = b[state["sent"]: i]
                state["sent"] = i + len("</think>")
                state["think_done"] = True
            if state["in_tool"]:
                return "", r
            if state["lead"]:  # the answer starts after "\n\n": drop the leading whitespace
                while state["sent"] < len(b) and b[state["sent"]] in " \n\r\t":
                    state["sent"] += 1
                if state["sent"] < len(b):
                    state["lead"] = False
            i = b.find("<tool_call>", state["sent"])
            if i >= 0:
                state["in_tool"] = True
                out = b[state["sent"]: i]
                state["sent"] = len(b)
                return out.rstrip(), r
            e = max(safe_end(b, "<tool_call>"), state["sent"])
            out = b[state["sent"]: e]
            state["sent"] = e
            return out, r

        ab = {"idx": -1, "kind": None}  # Anthropic streaming: the open content block

        def a_open(kind):
            if ab["kind"] == "thinking":
                self._sse({"type": "content_block_delta", "index": ab["idx"], "delta": {"type": "signature_delta", "signature": "ps5lm"}},
                          "content_block_delta")
            if ab["kind"]:
                self._sse({"type": "content_block_stop", "index": ab["idx"]}, "content_block_stop")
            ab["idx"] += 1
            ab["kind"] = kind
            if kind:
                blk = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" else {"type": "text", "text": ""}
                self._sse({"type": "content_block_start", "index": ab["idx"], "content_block": blk}, "content_block_start")

        if stream:
            self._sse_start()
            if anthropic:
                self._sse({"type": "message_start", "message": {"id": rid, "type": "message", "role": "assistant", "model": MODEL_NAME,
                                                                  "content": [], "stop_reason": None,
                                                                  "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
            else:
                self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                           "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

        progress = {"t": 0.0}

        def on_progress(done, total):  # S20: long prefills keep the stream alive and show progress
            if not stream or time.time() - progress["t"] < 0.5:
                return
            progress["t"] = time.time()
            if anthropic:
                self._sse({"type": "ping"}, "ping")
            elif body.get("ps5lm_progress"):
                self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": None}], "ps5lm_prefill": {"done": done, "total": total}})
            else:
                try:
                    self.wfile.write(f": prefill {done}/{total}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def on_delta(d):
            if not stream:
                return True
            c, r = visible(d)
            ok = True
            if anthropic:
                if r:
                    if ab["kind"] != "thinking":
                        a_open("thinking")
                    ok = self._sse({"type": "content_block_delta", "index": ab["idx"], "delta": {"type": "thinking_delta", "thinking": r}},
                                   "content_block_delta")
                if c:
                    if ab["kind"] != "text":
                        a_open("text")
                    ok = self._sse({"type": "content_block_delta", "index": ab["idx"], "delta": {"type": "text_delta", "text": c}},
                                   "content_block_delta")
            else:
                delta = {}
                if r:
                    delta["reasoning_content"] = r
                if c:
                    delta["content"] = c
                if delta:
                    ok = self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            return ok

        text, finish, st, gen_ids = self.run(ids, snaps, max_new, sp, stops, on_delta, think, budget, on_progress)
        if think and not text.startswith("<think>"):
            text = "<think>\n" + text
        reasoning, content, calls = parse_output(text, tools, think)
        if finish in ("stop", "length") and not stops:
            remember_turn(ptext, ids, gen_ids, len(msgs), content, calls, cut=finish == "length")
        ntok = int(st.get("gen", 0))
        usage = {"prompt_tokens": len(ids), "completion_tokens": ntok, "total_tokens": len(ids) + ntok,
                 "completion_tokens_details": {"reasoning_tokens": int(st.get("think_tokens", 0))},
                 "ps5lm": {k: st.get(k) for k in ("reused", "resume", "load_s", "prefill_s", "tps", "hit", "reads_per_tok", "gpu_ms", "io_ms",
                                                  "spec_cycles", "spec_acc", "think_tokens", "think_budget_hit", "prefill_tps")}}
        usage["ps5lm"].update({"prompt_from": how, "effort": kw.get("reasoning_effort", "xhigh") if think else "none",
                               "think_budget": budget})
        if anthropic:
            blocks = ([{"type": "thinking", "thinking": reasoning, "signature": "ps5lm"}] if reasoning else [])
            blocks += ([{"type": "text", "text": content}] if content else [])
            blocks += [{"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": c["name"], "input": c["arguments"]} for c in calls]
            stop_reason = "tool_use" if calls else ("end_turn" if finish == "stop" else "max_tokens")
            au = {"input_tokens": len(ids), "output_tokens": ntok}
            if stream:
                if ab["kind"] is None and not calls:
                    a_open("text")
                a_open(None)  # closes the open block
                base = ab["idx"]
                for i, b in enumerate(x for x in blocks if x["type"] == "tool_use"):
                    self._sse({"type": "content_block_start", "index": base + i, "content_block": {**b, "input": {}}}, "content_block_start")
                    self._sse({"type": "content_block_delta", "index": base + i,
                               "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}}, "content_block_delta")
                    self._sse({"type": "content_block_stop", "index": base + i}, "content_block_stop")
                self._sse({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": au}, "message_delta")
                self._sse({"type": "message_stop"}, "message_stop")
                return
            return self._json(200, {"id": rid, "type": "message", "role": "assistant", "model": MODEL_NAME, "content": blocks,
                                    "stop_reason": stop_reason, "usage": au, "ps5lm": usage["ps5lm"]})
        tool_calls = [{"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                       "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]
        fr = "tool_calls" if calls else finish
        if stream:
            if tool_calls:
                self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                           "choices": [{"index": 0, "delta": {"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]},
                                        "finish_reason": None}]})
            self._sse({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": fr}], "usage": usage})
            self._sse("[DONE]")
            return
        msg = {"role": "assistant", "content": content or (None if tool_calls else "")}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                         "choices": [{"index": 0, "message": msg, "finish_reason": fr}], "usage": usage})

    def completion(self, body):
        prompt = body.get("prompt", "")
        ids = State.tok.encode(prompt if isinstance(prompt, str) else prompt[0], add_special_tokens=False).ids
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        stream = bool(body.get("stream"))
        rid = "cmpl-" + uuid.uuid4().hex[:24]
        if stream:
            self._sse_start()
        on_delta = (lambda d: self._sse({"id": rid, "object": "text_completion", "model": MODEL_NAME,
                                         "choices": [{"index": 0, "text": d, "finish_reason": None}]})) if stream else (lambda d: True)
        text, finish, st, _ = self.run(ids, [len(ids)], body.get("max_tokens", 256), sampling(body), stops, on_delta)
        usage = {"prompt_tokens": len(ids), "completion_tokens": int(st.get("gen", 0))}
        if stream:
            self._sse({"id": rid, "object": "text_completion", "model": MODEL_NAME,
                       "choices": [{"index": 0, "text": "", "finish_reason": finish}], "usage": usage})
            self._sse("[DONE]")
            return
        self._json(200, {"id": rid, "object": "text_completion", "model": MODEL_NAME,
                         "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/ps5lm/model"))
    ap.add_argument("--bin", default=os.path.expanduser("~/ps5lm/build14/flash_decode"))
    ap.add_argument("--shaders", default=os.path.expanduser("~/ps5lm/build14/shaders"))
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--pack", default=None, help="expert pack file name in --model (default: the --bits pack)")
    ap.add_argument("--spec", type=int, default=0, help="speculative decoding depth (MTP drafts per verify; 1 = best measured)")
    ap.add_argument("--spec-gate", type=float, default=0.5, help="verify a draft only if the MTP head's top-1 probability >= this")
    ap.add_argument("--pool-gib", default="auto")
    ap.add_argument("--predict", type=int, default=1)
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--engine-args", default="", help="extra flash_decode arguments, e.g. '--kv k8v4 --kv-base 16384'")
    ap.add_argument("--snapshots", type=int, default=2)
    ap.add_argument("--residency", default=None, help="pool residency file (loaded at start, saved on quit)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking default (S20: on; --no-thinking = answers start immediately)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.set_defaults(thinking=True)  # S20: thinking on by default (context is no longer the constraint)
    ap.add_argument("--effort", default=None, help="default reasoning effort: low | medium | xhigh (default: the template's, xhigh)")
    ap.add_argument("--think-budgets", default="2048,8192,24576", help="default thinking budgets (tokens) for low,medium,xhigh; 0 = none")
    ap.add_argument("--default-max-tokens", type=int, default=32768, help="max_tokens when a request gives none (capped by the context)")
    a, extra = ap.parse_known_args()  # S20: unknown arguments go to the engine (PS5LM_ARGS=... --kv k8v4 --kv-base 16384)
    if extra:
        a.engine_args = (a.engine_args + " " + " ".join(extra)).strip()
    a.model = os.path.expanduser(a.model)
    a.residency = a.residency or os.path.join(a.model, f"residency_q{a.bits}.txt")
    State.tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
    State.im_start = State.tok.token_to_id(IM_START)
    State.enc = EncCache()
    State.tmpl = make_template(a.model)
    State.thinking_default = a.thinking
    if a.effort:
        State.effort_default = EFFORTS[a.effort.lower()]
    State.budgets = dict(zip(("low", "medium", "xhigh"), (int(x) or None for x in a.think_budgets.split(","))))
    State.default_max = a.default_max_tokens
    ea = a.engine_args.split()
    if "--kv-store" in ea[:-1]:  # S25
        State.turns_dir = os.path.join(ea[ea.index("--kv-store") + 1], "turns")
        load_turns()
    State.engine = Engine(a)
    print(f"ps5lm server on {a.host}:{a.port}  engine {State.engine.info}", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)

    def _term(signum, frame):  # SIGTERM (systemd stop, kill) -> clean shutdown; background jobs ignore SIGINT
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            State.engine._send("Q")
            State.engine.p.wait(timeout=30)
        except Exception:
            pass


if __name__ == "__main__":
    main()
