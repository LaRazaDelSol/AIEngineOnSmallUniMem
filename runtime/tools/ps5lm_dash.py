#!/usr/bin/env python3
"""
ps5lm_dash — live monitor + chat + stats proxy for the PS5 inference appliance (S18).

Dashboard for the PlayStation 5 engine: it sits in front of
ps5lm_server.py (flash_decode --serve) and forwards every /v1/* request (streaming included), recording
per-request timing, token counts and the engine's own per-request numbers (expert-pool hit rate, reads per
token, GPU / expert-wait ms per token, MTP draft acceptance), and samples what this console exposes: the APU die
temperature (k10temp: CPU and GPU share the die), the NVMe temperature, memory, CPU and the NVMe read rate
(= the expert stream). The PS5's amdgpu has no busy/power/clock telemetry, so those NVIDIA cards were replaced.
Serves the monitor + chat at /, raw JSON at /api/stats.

The only change it makes to requests: reasoning_effort values the Qwen3.8 chat template rejects (it accepts only
xhigh, medium, low) are mapped onto ones it accepts, and "none" turns thinking off. See normalize_effort().

Single file, Python standard library only. On the console (ps5lm-dash.service runs this):
  python3 ps5lm_dash.py --listen-host 0.0.0.0 --listen-port 8080 --upstream http://127.0.0.1:8081 \
      --model ~/ps5lm/model

Visual design and chat (tokens, components, monitor layout, sparklines, chat view, Markdown
renderer, sampling drawer, icons) adapted from
Strata — https://github.com/niko1221/strata — MIT License,
Copyright (c) 2026 Niko1221 and the Strata contributors.
Outfit font: Copyright 2021 The Outfit Project Authors, SIL Open Font License 1.1.
Full license texts are served at /licenses.
"""

import argparse
import gzip
import hashlib
import statistics
import http.client
import itertools
import json
import os
import re
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}
TRACKED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")

REQS = deque(maxlen=500)                  # finished requests, oldest first
INFLIGHT = {}                             # rid -> live request state
LOCK = threading.Lock()
IDS = itertools.count(1)
HIST_KEYS = ("soc_temp", "nvme_temp", "mem_used", "cpu", "disk_read", "tok_s")
HIST = {k: deque(maxlen=90) for k in HIST_KEYS}   # one sample every --sample seconds
HW = {}
STATIC = {}
UPSTREAM = {"ok": False, "model": None, "ctx": None}
ENGINE = {}                               # parsed from vLLM's /metrics
EFFORT_MAP = {"high": "xhigh", "max": "xhigh", "minimal": "low"}   # what the Qwen3.8 template accepts
LAST_VIEW = [0.0]                         # when a browser last polled; sampling slows down when nobody watches
ARGS = None


# ---------------------------------------------------------------- stream parsing

def merge_usage(target, obj):
    """Pull token usage out of OpenAI chat, Responses or Anthropic events."""
    if not isinstance(obj, dict):
        return
    candidates = [obj.get("usage")]
    if isinstance(obj.get("response"), dict):
        candidates.append(obj["response"].get("usage"))
    if isinstance(obj.get("message"), dict):
        candidates.append(obj["message"].get("usage"))
    for u in candidates:
        if not isinstance(u, dict):
            continue
        prompt = u.get("prompt_tokens", u.get("input_tokens"))
        completion = u.get("completion_tokens", u.get("output_tokens"))
        if prompt is not None:
            target["prompt"] = prompt
        if completion is not None:
            target["completion"] = completion
        details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            target["cached"] = details["cached_tokens"]
        if u.get("cache_read_input_tokens") is not None:
            target["cached"] = u["cache_read_input_tokens"]
        px = u.get("ps5lm") if isinstance(u.get("ps5lm"), dict) else (obj.get("ps5lm") if isinstance(obj.get("ps5lm"), dict) else None)
        if px:   # ps5lm: reused prompt tokens + the engine's per-request numbers
            if px.get("reused") is not None:
                target["cached"] = int(px["reused"])
            target["ps5lm"] = px
        cdetails = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
        if isinstance(cdetails, dict) and cdetails.get("reasoning_tokens") is not None:
            target["reasoning"] = cdetails["reasoning_tokens"]


def is_token_event(obj, raw):
    """True when an SSE event carries generated text, thinking or a tool call."""
    for c in obj.get("choices") or []:
        d = c.get("delta") or {}
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
            return True
        if c.get("text"):
            return True
    return b'"delta"' in raw and str(obj.get("type", "")).endswith("delta")


# ---------------------------------------------------------------- machine sampling

def _num(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def gpu_sample():
    """PS5: k10temp = the Oberon APU die (CPU + GPU), nvme = the expert SSD, amdgpu GTT / VRAM-carve use."""
    out = {}
    base = "/sys/class/hwmon"
    try:
        for h in os.listdir(base):
            name = _read(f"{base}/{h}/name")
            t = _num(_read(f"{base}/{h}/temp1_input") or "x")
            if t is None:
                continue
            if name == "k10temp":
                out["soc_temp"] = round(t / 1000, 1)
            elif name == "nvme":
                out["nvme_temp"] = round(t / 1000, 1)
    except OSError:
        pass
    d = "/sys/class/drm/card0/device"
    for k, f in (("gtt_used", "mem_info_gtt_used"), ("gtt_total", "mem_info_gtt_total"),
                 ("vram_used", "mem_info_vram_used"), ("vram_total", "mem_info_vram_total")):
        v = _num(_read(f"{d}/{f}") or "x")
        if v is not None:
            out[k] = v
    return out


def mem_sample():
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            mem[k] = int(v.strip().split()[0]) * 1024
    return {"mem_total": mem["MemTotal"], "mem_used": mem["MemTotal"] - mem["MemAvailable"],
            "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
            "hugetlb": mem.get("Hugetlb", 0)}   # the expert pool (2 MiB pages, pinned)


def read_cpu():
    with open("/proc/stat") as f:
        vals = list(map(int, f.readline().split()[1:]))
    return vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)


def read_disk():
    rd = wr = 0
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            name = p[2]
            whole_nvme = name.startswith("nvme") and "p" not in name[4:]
            whole_sd = name.startswith("sd") and not name[-1].isdigit()
            if whole_nvme or whole_sd:
                rd += int(p[5])
                wr += int(p[9])
    return rd * 512, wr * 512


def dir_size(path):
    total = 0
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat().st_size
        except OSError:
            pass
    return total


def _get_json(path, timeout=2):
    u = ARGS.upstream_parsed
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = json.loads(r.read() or b"{}")
    c.close()
    return r.status, body


def check_upstream():
    try:
        status, body = _get_json("/v1/models")
        data = body.get("data") or []
        UPSTREAM.update(ok=status == 200, model=data[0].get("id") if data else None)
        _, h = _get_json("/health")   # ps5lm: READY line of the engine (slots, pool_gib, max_ctx, ...)
        UPSTREAM.update(ctx=int(h["max_ctx"]) if h.get("max_ctx") else None, slots=h.get("slots"), pool_gib=h.get("pool_gib"),
                        resident_gib=h.get("resident_gib"))
    except Exception:
        UPSTREAM["ok"] = False


def check_metrics():
    """ps5lm /metrics: server totals, the engine's last request and its 'I' line (pool hit rate since start)."""
    try:
        _, m = _get_json("/metrics")
    except Exception:
        ENGINE.clear()
        return
    eng = dict(kv.split("=", 1) for kv in (m.get("engine") or "").split()[1:] if "=" in kv)
    args, tot, last = m.get("args") or {}, m.get("totals") or {}, m.get("last") or {}
    ENGINE.clear()
    ENGINE.update(pool_hit_pct=round(100 * float(eng["hit"]), 1) if eng.get("hit") else None,
                  slots=eng.get("slots"), reads=eng.get("reads"), engine_requests=eng.get("requests"),
                  spec=args.get("spec"), spec_gate=args.get("spec_gate"), pack=args.get("pack"), bits=args.get("bits"),
                  predict=args.get("predict"), gen_tps=round(tot["gen_tps"], 1) if tot.get("gen_tps") else None,
                  mtp_accept_pct=round(100 * float(last["spec_acc"]), 1) if last.get("spec_acc") else None)


def normalize_effort(req):
    """Map reasoning_effort onto what the Qwen3.8 template accepts. Returns True if changed."""
    effort = req.get("reasoning_effort")
    if not isinstance(effort, str):
        return False
    e = effort.lower()
    kwargs = req.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    if e == "none":
        req.pop("reasoning_effort")
        kwargs["enable_thinking"] = False
        req["chat_template_kwargs"] = kwargs
        return True
    if e in ("xhigh", "medium", "low"):
        if e == effort:
            return False
        req["reasoning_effort"] = e
        return True
    req["reasoning_effort"] = EFFORT_MAP.get(e, "xhigh")
    return True


def live_state():
    now = time.perf_counter()
    with LOCK:
        items = sorted((dict(v) for v in INFLIGHT.values()), key=lambda v: v["t0"])
    if not items:
        return {"state": "idle", "queued": 0}
    started = [v for v in items if v["first"] is not None]
    cur = started[0] if started else items[0]
    queued = len(items) - 1
    if cur["first"] is None:
        return {"state": "reading", "queued": queued, "elapsed": round(now - cur["t0"], 1)}
    gen_t = now - cur["first"]
    n = cur["events"]
    return {"state": "generating", "queued": queued, "generated": n, "max_tokens": cur["max_tokens"],
            "tok_s": round((n - 1) / gen_t, 1) if n > 1 and gen_t > 0 else None,
            "wait": round(cur["first"] - cur["t0"], 2), "elapsed": round(now - cur["t0"], 1)}


def sampler():
    last_cpu, last_disk, last_t, tick = read_cpu(), read_disk(), time.time(), 0
    check_upstream()
    while True:
        watched = time.time() - LAST_VIEW[0] < 30 or INFLIGHT
        time.sleep(ARGS.sample if watched else 10)
        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now
        hw = {}
        try:
            hw.update(gpu_sample())
        except Exception:
            pass
        try:
            hw.update(mem_sample())
        except Exception:
            pass
        try:
            c = read_cpu()
            dtot = c[1] - last_cpu[1]
            hw["cpu"] = round(100 * (1 - (c[0] - last_cpu[0]) / dtot), 1) if dtot > 0 else None
            last_cpu = c
        except Exception:
            pass
        try:
            d = read_disk()
            hw["disk_read_mb"] = round((d[0] - last_disk[0]) / dt / 1e6, 1)
            hw["disk_write_mb"] = round((d[1] - last_disk[1]) / dt / 1e6, 1)
            last_disk = d
        except Exception:
            pass
        try:
            hw["load"] = round(os.getloadavg()[0], 2)
        except Exception:
            pass
        if tick % 3 == 0:
            check_upstream()
        check_metrics()
        tick += 1
        live = live_state()
        with LOCK:
            HW.clear()
            HW.update(hw)
            for key, src in (("soc_temp", "soc_temp"), ("nvme_temp", "nvme_temp"),
                             ("mem_used", "mem_used"), ("cpu", "cpu"), ("disk_read", "disk_read_mb")):
                HIST[key].append(hw.get(src))
            HIST["tok_s"].append(live.get("tok_s") or 0)


def run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def load_static():
    s = {"host": socket.gethostname(), "cpu_count": os.cpu_count(),
         "listen": f"http://{ARGS.listen_host}:{ARGS.listen_port}", "mtp": ARGS.mtp,
         "started": datetime.now().isoformat(timespec="seconds")}
    cmdline = _read("/proc/cmdline") or ""
    cu = re.search(r"amdgpu\.ps5_cu_unlock=(\d)", cmdline)
    s["gpu_name"] = "AMD Oberon · GFX1013 (RDNA 2)"
    s["cus"] = 40 if cu and cu.group(1) == "3" else 36
    s["kernel"] = os.uname().release
    s["ai_mode"] = "ps5lm.ai=1" in cmdline
    cpu = re.search(r"model name\s*:\s*(.+)", _read("/proc/cpuinfo") or "")
    if cpu:
        s["cpu_name"] = cpu.group(1).strip()
    if ARGS.model:
        s["model_file"] = os.path.basename(os.path.normpath(ARGS.model))
        s["model_bytes"] = dir_size(ARGS.model)
    if ARGS.table:
        s["table_bytes"] = dir_size(ARGS.table)
    if ARGS.engine_dir:
        s["engine_commit"] = run(["git", "log", "-1", "--format=%h · %cd", "--date=short"], cwd=ARGS.engine_dir)
    STATIC.update(s)


# ---------------------------------------------------------------- request records

def record(rec):
    with LOCK:
        REQS.append(rec)
    try:
        with open(ARGS.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def load_history():
    try:
        with open(ARGS.log) as f:
            for line in f.readlines()[-REQS.maxlen:]:
                try:
                    REQS.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass


def ok(r):
    return r.get("status") == 200 and not r.get("aborted")


def stats():
    with LOCK:
        recent = list(REQS)
        hist = {k: list(v) for k, v in HIST.items()}
        hw = dict(HW)
    good = [r for r in recent if ok(r)]
    read = sum(max(0, (r.get("prompt") or 0) - (r.get("cached") or 0)) for r in good)
    out = sum(r.get("completion") or 0 for r in good)
    p_time = sum(r["ttft"] for r in good if r.get("ttft") and (r.get("prompt") or 0) - (r.get("cached") or 0) > 0)
    g_time = sum(r.get("gen_time") or 0 for r in good if r.get("completion"))
    last50 = good[-50:]
    last20 = good[-20:]

    def med(key):
        vals = [r[key] for r in last20 if r.get(key) is not None]
        return round(statistics.median(vals), 2) if vals else None

    prompt_all = sum(r.get("prompt") or 0 for r in good)
    reused_all = sum(r.get("cached") or 0 for r in good)
    return {
        "now": datetime.now().isoformat(timespec="seconds"),
        "live": live_state(),
        "upstream": dict(UPSTREAM),
        "engine": dict(ENGINE),
        "hw": hw,
        "history": hist,
        "static": STATIC,
        "per_request": {"gen_tps": [r.get("gen_tps") for r in last50], "prefill_tps": [r.get("prefill_tps") for r in last50],
                        "ttft": [r.get("ttft") for r in last50],
                        "hit_pct": [round(100 * r["hit"], 1) if r.get("hit") is not None else None for r in last50],
                        "accept_pct": [round(100 * r["spec_acc"], 1) if r.get("spec_acc") is not None else None for r in last50],
                        "reuse_pct": [round(100 * (r.get("cached") or 0) / r["prompt"]) if r.get("prompt") else 0 for r in last50]},
        "median_last20": {"gen_tps": med("gen_tps"), "ttft": med("ttft"), "prefill_tps": med("prefill_tps")},
        "totals": {"requests": len(recent), "read": read, "reused": reused_all,
                   "reused_pct": round(100 * reused_all / prompt_all) if prompt_all else None,
                   "output": out, "since": recent[0]["ts"] if recent else None,
                   "prefill_tps": round(read / p_time, 1) if p_time > 0 else None,
                   "decode_tps": round(out / g_time, 1) if g_time > 0 else None},
        "requests": list(reversed(recent[-50:])),
    }


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"     # close after each response; simplest correct streaming
    server_version = "ps5lm_dash"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache="no-store", etag=None, gz=None):
        data = body.encode() if isinstance(body, str) else body
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return
        zipped = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(data) > 1024
        if zipped:
            data = gz if gz is not None else gzip.compress(data, 5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        if etag:
            self.send_header("ETag", etag)
        if zipped:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/dashboard"):
            return self._send(200, PAGE_BYTES, "text/html; charset=utf-8", "no-cache", PAGE_ETAG, PAGE_GZ)
        if path == "/api/stats":
            LAST_VIEW[0] = time.time()
            return self._send(200, json.dumps(stats(), separators=(",", ":")), "application/json")
        if path == "/licenses":
            return self._send(200, LICENSES, "text/plain; charset=utf-8", "max-age=86400")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon", "max-age=86400")
        return self.proxy()

    def do_POST(self):
        return self.proxy()

    def do_OPTIONS(self):
        return self.proxy()

    def proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        stream, max_tokens = False, None
        if body and self.command == "POST":
            try:
                req = json.loads(body)
                if isinstance(req, dict):
                    stream = bool(req.get("stream"))
                    max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or req.get("max_output_tokens")
                    changed = self.path.startswith(("/v1/chat/completions", "/v1/responses", "/v1/messages")) \
                        and normalize_effort(req)
                    # ask for token usage in streams so stats work; harmless to clients
                    if stream and self.path.startswith("/v1/chat/completions") and "stream_options" not in req:
                        req["stream_options"] = {"include_usage": True}
                        changed = True
                    if changed:
                        body = json.dumps(req).encode()
            except Exception:
                pass

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Content-Length"] = str(len(body))
        u = ARGS.upstream_parsed
        tracked = self.command == "POST" and self.path.startswith(TRACKED_PATHS)
        rid = next(IDS)
        t0 = time.perf_counter()
        live = {"t0": t0, "first": None, "events": 0, "max_tokens": max_tokens}
        if tracked:
            with LOCK:
                INFLIGHT[rid] = live
        status, info, aborted = 502, {}, False
        try:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=ARGS.timeout)
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()

            is_sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
            buf, whole = b"", []
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    aborted = True        # client hung up; closing upstream stops generation
                    break
                if not tracked:
                    continue
                if not is_sse:
                    whole.append(chunk)
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    merge_usage(info, obj)
                    if is_token_event(obj, data):
                        live["events"] += 1
                        if live["first"] is None:
                            live["first"] = time.perf_counter()
            if tracked and whole:
                try:
                    merge_usage(info, json.loads(b"".join(whole)))
                except Exception:
                    pass
            conn.close()
        except Exception as e:
            if status == 502:
                try:
                    self._send(502, json.dumps({"error": f"dash: the PS5 engine server is unreachable: {e}"}), "application/json")
                except Exception:
                    pass
        finally:
            if tracked:
                with LOCK:
                    INFLIGHT.pop(rid, None)

        if not tracked:
            return
        t_end = time.perf_counter()
        first = live["first"]
        total = t_end - t0
        completion, prompt, cached = info.get("completion"), info.get("prompt"), info.get("cached") or 0
        ttft = (first - t0) if first else None
        gen_time = (t_end - first) if first else None
        record({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "path": self.path,
            "stream": stream,
            "status": status,
            "aborted": aborted,
            "prompt": prompt,
            "cached": cached,
            "completion": completion,
            "reasoning": info.get("reasoning"),
            "ttft": round(ttft, 2) if ttft is not None else None,
            "gen_time": round(gen_time, 2) if gen_time else None,
            "total": round(total, 2),
            "gen_tps": round((completion - 1) / gen_time, 1)
            if completion and completion > 1 and gen_time and gen_time > 0 else None,
            "e2e_tps": round(completion / total, 1) if completion and total > 0 else None,
            "prefill_tps": round((prompt - cached) / ttft, 1)
            if prompt and ttft and ttft > 0 and prompt - cached > 0 else None,
            # ps5lm engine numbers (expert pool hit rate, reads/token, GPU and expert-wait ms per fed token, MTP acceptance)
            **{k: _num(info.get("ps5lm", {}).get(src)) for k, src in (("hit", "hit"), ("reads_per_tok", "reads_per_tok"), ("gpu_ms", "gpu_ms"),
                                                                     ("io_ms", "io_ms"), ("spec_acc", "spec_acc"), ("engine_tps", "tps"))
               if info.get("ps5lm", {}).get(src) not in (None, "")},
        })


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3.8 · PS5</title>
<link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiI+CiAgPCEtLSBUaGUgdGFiIGljb246IHRoZSBsYXllciBzdGFjayB0aGUgYXBwJ3MgcGFuZWxzIHVzZSAoc2VydmUvd2ViL3Nwcml0ZS5zdmcsIGktbGF5ZXJzKSwKICAgICAgIGRyYXduIGJvbGQgZW5vdWdoIHRvIHN0YXkgcmVhZGFibGUgYXQgMTYgcHguIC0tPgogIDxwYXRoIGQ9Ik0xNiAyLjM4IDI4LjQyIDkuMDggMTYgMTUuNzggMy41OCA5LjA4WiIgZmlsbD0iIzEwYjk4MSIvPgogIDxwYXRoIGQ9Ik0zLjU4IDE0LjggMTYgMjEuNSAyOC40MiAxNC44IiBmaWxsPSJub25lIiBzdHJva2U9IiMxMGI5ODEiIHN0cm9rZS13aWR0aD0iMy42NyIKICAgICAgICBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIG9wYWNpdHk9IjAuOTIiLz4KICA8cGF0aCBkPSJNMy41OCAyMS4yOCAxNiAyNy45OCAyOC40MiAyMS4yOCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMTBiOTgxIiBzdHJva2Utd2lkdGg9IjMuNjciCiAgICAgICAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBvcGFjaXR5PSIwLjc4Ii8+Cjwvc3ZnPgo=">
<style>
/* Strata UI — tokens. Derived from the Gainsty design system.
   Light is default; dark via <html data-theme="dark">.
   Values marked (derived) are not in the Gainsty source and were added for the app. */

/* Outfit (SIL Open Font License 1.1, serve/web/fonts/OFL.txt): one variable font, weights 100-900 */
@font-face{font-family:"Outfit";src:url("data:font/woff2;base64,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") format("woff2-variations");font-weight:100 900;font-display:swap;unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}

:root{
  /* type */
  --st-font:"Outfit",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --st-font-mono:ui-monospace,"SF Mono","Cascadia Code",Consolas,"Liberation Mono",monospace; /* (derived) */
  --st-fs-xs:12px;  --st-fs-sm:13px; --st-fs-md:14px; --st-fs-base:16px;
  --st-fs-lg:18px;  --st-fs-xl:24px; --st-fs-stat:32px; --st-fs-display:40px;
  --st-fw-regular:400; --st-fw-medium:500; --st-fw-bold:700; --st-fw-black:900;
  --st-lh-tight:1.1; --st-lh-body:1.55;

  /* spacing (4pt) */
  --st-s-1:4px; --st-s-2:8px; --st-s-3:12px; --st-s-4:16px; --st-s-5:20px;
  --st-s-6:24px; --st-s-8:32px; --st-s-10:40px; --st-s-12:48px; --st-s-16:64px;

  /* radius */
  --st-r-sm:8px; --st-r-md:14px; --st-r-lg:20px; --st-r-xl:24px; --st-r-pill:9999px;

  /* motion (derived) */
  --st-ease:cubic-bezier(.2,.7,.2,1); --st-dur:160ms;

  /* color — light */
  --st-bg:#f9fafb;
  --st-surface:#ffffff;
  --st-surface-2:#f7f7f7;
  --st-line:#dee5ee;
  --st-line-soft:#eeeeee;
  --st-ink:#221e1f;
  --st-ink-soft:#303030;
  --st-ink-muted:#6f6f6f;        /* Gainsty #7a7a7a darkened slightly for 4.5:1 on white */
  --st-icon:#2F384C;             /* Anron default */

  --st-accent:#10b981;
  --st-accent-hover:#0c8a60;
  --st-accent-ink:#ffffff;
  --st-accent-tint:#ecfdf5;
  --st-accent-text:#0c8a60;      /* accent used as text on light */
  --st-focus:#405de6;

  --st-info:#0f97ff;     --st-info-tint:#e8f4ff;    --st-info-text:#0668c2;
  --st-warn:#e39b0b;     --st-warn-tint:#fff6e0;    --st-warn-text:#8a5a00;   /* (derived) */
  --st-danger:#e0283f;   --st-danger-tint:#fdecee;  --st-danger-text:#b01a2e; /* (derived) */

  --st-code-bg:#1b1f23;  --st-code-ink:#e6e9ec;  /* code blocks stay dark in both themes (derived) */

  --st-sh-card:0 18px 7px rgba(0,0,0,.01),0 10px 6px rgba(0,0,0,.02),0 4px 4px rgba(0,0,0,.03),0 1px 2px rgba(0,0,0,.04);
  --st-sh-bar:0 54px 21px rgba(0,0,0,.01),0 30px 18px rgba(0,0,0,.03),0 13px 13px rgba(0,0,0,.05),0 3px 7px rgba(0,0,0,.06);
  --st-sh-pop:0 10px 40px rgba(15,23,42,.08),0 1px 3px rgba(15,23,42,.06);
  color-scheme:light;
}

/* color — dark (all derived) */
[data-theme="dark"]{
  --st-bg:#0e1113;
  --st-surface:#161a1d;
  --st-surface-2:#1d2226;
  --st-line:#2a3036;
  --st-line-soft:#22272c;
  --st-ink:#eef1f3;
  --st-ink-soft:#c7cdd3;
  --st-ink-muted:#8e979f;
  --st-icon:#c7cdd3;

  --st-accent:#34d399;
  --st-accent-hover:#4ade80;
  --st-accent-ink:#052e20;
  --st-accent-tint:rgba(52,211,153,.12);
  --st-accent-text:#4ade80;
  --st-focus:#7b93ff;

  --st-info:#3aa9ff;     --st-info-tint:rgba(58,169,255,.12);  --st-info-text:#7cc4ff;
  --st-warn:#f5b53d;     --st-warn-tint:rgba(245,181,61,.12);  --st-warn-text:#f8c96a;
  --st-danger:#ff5a6e;   --st-danger-tint:rgba(255,90,110,.12);--st-danger-text:#ff8a98;

  --st-code-bg:#0a0c0e;  --st-code-ink:#e6e9ec;

  --st-sh-card:0 1px 2px rgba(0,0,0,.4);
  --st-sh-bar:0 12px 32px rgba(0,0,0,.45),0 2px 6px rgba(0,0,0,.35);
  --st-sh-pop:0 16px 48px rgba(0,0,0,.55),0 1px 3px rgba(0,0,0,.4);
  color-scheme:dark;
}

/* Strata UI — components. Requires tokens.css. */
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;background:var(--st-bg);color:var(--st-ink);font-family:var(--st-font);font-size:var(--st-fs-base);line-height:var(--st-lh-body);-webkit-font-smoothing:antialiased}
a{color:var(--st-accent-text);text-decoration:none} a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--st-focus);outline-offset:2px}

/* icon — <svg class="st-icon"><use href="icons/sprite.svg#i-chat"/></svg> */
.st-icon{width:20px;height:20px;flex:none;color:inherit}
.st-icon--sm{width:16px;height:16px} .st-icon--lg{width:24px;height:24px}

/* header */
.st-header{display:flex;align-items:center;gap:var(--st-s-4);height:60px;padding:0 var(--st-s-6);background:var(--st-surface);border-bottom:1px solid var(--st-line)}
.st-header__brand{font-weight:900;font-size:var(--st-fs-lg);letter-spacing:-.01em}
.st-header__spacer{flex:1}

/* status pill */
.st-pill{display:inline-flex;align-items:center;gap:8px;height:30px;padding:0 12px;border-radius:var(--st-r-pill);border:1px solid var(--st-line);background:var(--st-surface-2);font-size:var(--st-fs-sm);font-weight:500;color:var(--st-ink-soft);font-variant-numeric:tabular-nums}
.st-pill__dot{width:8px;height:8px;border-radius:50%;background:var(--st-ink-muted)}
.st-pill[data-state="generating"] .st-pill__dot{background:var(--st-accent);box-shadow:0 0 0 4px var(--st-accent-tint)}
.st-pill[data-state="reading"] .st-pill__dot{background:var(--st-info);box-shadow:0 0 0 4px var(--st-info-tint)}
.st-pill[data-state="queued"] .st-pill__dot{background:var(--st-warn)}

/* tabs */
.st-tabs{display:flex;gap:4px;padding:4px;background:var(--st-surface-2);border:1px solid var(--st-line);border-radius:var(--st-r-md)}
.st-tab{display:inline-flex;align-items:center;gap:8px;height:36px;padding:0 14px;border:0;border-radius:10px;background:transparent;color:var(--st-ink-muted);font:500 var(--st-fs-md)/1 var(--st-font);cursor:pointer}
.st-tab:hover{color:var(--st-ink)}
.st-tab[aria-selected="true"]{background:var(--st-surface);color:var(--st-ink);font-weight:700;box-shadow:var(--st-sh-card)}

/* card + metric */
.st-card{background:var(--st-surface);border:1px solid var(--st-line);border-radius:var(--st-r-lg);box-shadow:var(--st-sh-card);padding:var(--st-s-5)}
.st-metric{display:flex;flex-direction:column;gap:10px}
.st-metric__label{display:flex;align-items:center;gap:8px;font-size:var(--st-fs-sm);font-weight:500;color:var(--st-ink-muted)}
.st-metric__value{font-weight:900;font-size:var(--st-fs-stat);line-height:1;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.st-metric__unit{font-weight:500;font-size:var(--st-fs-md);color:var(--st-ink-muted);margin-left:4px}
.st-metric__spark{width:100%;height:32px;color:var(--st-accent)}

/* progress + gauge */
.st-progress{height:8px;border-radius:var(--st-r-pill);background:var(--st-surface-2);border:1px solid var(--st-line-soft);overflow:hidden}
.st-progress__bar{height:100%;border-radius:inherit;background:var(--st-accent);transition:width var(--st-dur) var(--st-ease)}
.st-progress[data-tone="info"] .st-progress__bar{background:var(--st-info)}
.st-progress[data-tone="warn"] .st-progress__bar{background:var(--st-warn)}
.st-progress[data-tone="danger"] .st-progress__bar{background:var(--st-danger)}
/* gauge: see README for SVG markup (stroke-dasharray on a 270° arc) */
.st-gauge{position:relative;width:120px;height:120px}
.st-gauge__track{stroke:var(--st-surface-2)} .st-gauge__fill{stroke:var(--st-accent)}

/* table */
.st-table{width:100%;border-collapse:collapse;font-size:var(--st-fs-md);font-variant-numeric:tabular-nums}
.st-table th{text-align:left;font-weight:500;font-size:var(--st-fs-xs);letter-spacing:.04em;text-transform:uppercase;color:var(--st-ink-muted);padding:10px 12px;border-bottom:1px solid var(--st-line)}
.st-table td{padding:12px;border-bottom:1px solid var(--st-line-soft);color:var(--st-ink-soft)}
.st-table tr:hover td{background:var(--st-surface-2)}
.st-table .num{text-align:right}

/* badges */
.st-badge{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 10px;border-radius:var(--st-r-pill);font-size:var(--st-fs-xs);font-weight:700;background:var(--st-surface-2);color:var(--st-ink-muted);border:1px solid var(--st-line)}
.st-badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.st-badge--reading{background:var(--st-info-tint);color:var(--st-info-text);border-color:transparent}
.st-badge--generating{background:var(--st-accent-tint);color:var(--st-accent-text);border-color:transparent}
.st-badge--queued{background:var(--st-warn-tint);color:var(--st-warn-text);border-color:transparent}
.st-badge--error{background:var(--st-danger-tint);color:var(--st-danger-text);border-color:transparent}

/* chat */
.st-msg{display:flex;flex-direction:column;gap:6px;max-width:720px}
.st-msg--user{align-self:flex-end;align-items:flex-end}
.st-bubble{padding:12px 16px;border-radius:var(--st-r-lg);font-size:var(--st-fs-base);line-height:1.55;color:var(--st-ink-soft)}
.st-msg--user .st-bubble{background:var(--st-accent);color:var(--st-accent-ink);border-bottom-right-radius:6px}
.st-msg--assistant .st-bubble{background:var(--st-surface);border:1px solid var(--st-line);border-bottom-left-radius:6px}
.st-msg__meta{font-size:var(--st-fs-xs);color:var(--st-ink-muted);display:flex;gap:8px;align-items:center}

/* collapsible — <details class="st-collapse"><summary>…</summary><div class="st-collapse__body">…</div></details> */
.st-collapse{border:1px solid var(--st-line);border-radius:var(--st-r-md);background:var(--st-surface-2)}
.st-collapse>summary{list-style:none;display:flex;align-items:center;gap:8px;padding:10px 14px;cursor:pointer;font-size:var(--st-fs-sm);font-weight:500;color:var(--st-ink-muted)}
.st-collapse>summary::-webkit-details-marker{display:none}
.st-collapse>summary .st-chev{margin-left:auto;transition:transform var(--st-dur) var(--st-ease)}
.st-collapse[open]>summary .st-chev{transform:rotate(180deg)}
.st-collapse__body{padding:0 14px 14px;font-size:var(--st-fs-sm);color:var(--st-ink-muted);line-height:1.6}

/* code */
.st-code{border-radius:var(--st-r-md);background:var(--st-code-bg);color:var(--st-code-ink);overflow:hidden}
.st-code__head{display:flex;align-items:center;justify-content:space-between;padding:8px 8px 8px 14px;font:500 var(--st-fs-xs)/1 var(--st-font);color:#9aa3ab;border-bottom:1px solid rgba(255,255,255,.06)}
.st-code pre{margin:0;padding:14px;overflow:auto;font:400 13px/1.6 var(--st-font-mono)}
.st-code .st-btn--icon{color:#9aa3ab} .st-code .st-btn--icon:hover{background:rgba(255,255,255,.08);color:#fff}

/* composer */
.st-composer{display:flex;flex-direction:column;gap:8px;padding:10px;background:var(--st-surface);border:1px solid var(--st-line);border-radius:var(--st-r-lg);box-shadow:var(--st-sh-bar)}
.st-composer:focus-within{border-color:var(--st-focus)}
.st-composer textarea{border:0;outline:0;resize:none;background:transparent;color:var(--st-ink);font:400 var(--st-fs-base)/1.5 var(--st-font);padding:6px 8px;min-height:48px}
.st-composer__bar{display:flex;align-items:center;gap:6px}
.st-composer__bar .spacer{flex:1}

/* drawer */
.st-drawer{position:fixed;top:0;right:0;bottom:0;width:360px;max-width:100vw;background:var(--st-surface);border-left:1px solid var(--st-line);box-shadow:var(--st-sh-pop);display:flex;flex-direction:column;transform:translateX(100%);transition:transform 220ms var(--st-ease)}
.st-drawer[data-open="true"]{transform:none}
.st-drawer__head{display:flex;align-items:center;justify-content:space-between;padding:18px 20px;border-bottom:1px solid var(--st-line)}
.st-drawer__body{padding:20px;display:flex;flex-direction:column;gap:22px;overflow:auto}

/* buttons */
.st-btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;height:42px;padding:0 18px;border-radius:var(--st-r-md);border:0;font:700 var(--st-fs-md)/1 var(--st-font);cursor:pointer;transition:background var(--st-dur),transform 120ms}
.st-btn:active{transform:translateY(1px)}
.st-btn--primary{background:var(--st-accent);color:var(--st-accent-ink);box-shadow:0 6px 16px rgba(16,185,129,.24),inset 0 1px 0 rgba(255,255,255,.18)}
.st-btn--primary:hover{background:var(--st-accent-hover)}
.st-btn--secondary{background:transparent;color:var(--st-ink);border:1.5px solid var(--st-line)}
.st-btn--secondary:hover{background:var(--st-surface-2)}
.st-btn--danger{background:var(--st-danger-tint);color:var(--st-danger-text)}
.st-btn--icon{width:36px;height:36px;padding:0;border-radius:10px;background:transparent;color:var(--st-ink-muted)}
.st-btn--icon:hover{background:var(--st-surface-2);color:var(--st-ink)}
.st-btn:disabled{opacity:.45;cursor:not-allowed}

/* form controls */
.st-field{display:flex;flex-direction:column;gap:8px}
.st-label{display:flex;justify-content:space-between;font-size:var(--st-fs-sm);font-weight:500;color:var(--st-ink-soft)}
.st-label output{font-variant-numeric:tabular-nums;color:var(--st-ink-muted)}
.st-input{height:44px;padding:0 14px;border-radius:var(--st-r-md);border:1.5px solid var(--st-line);background:var(--st-surface);color:var(--st-ink);font:400 var(--st-fs-md) var(--st-font)}
.st-input:focus{outline:0;border-color:var(--st-focus);box-shadow:0 0 0 3px color-mix(in srgb,var(--st-focus) 20%,transparent)}
.st-slider{width:100%;accent-color:var(--st-accent)}
.st-toggle{position:relative;width:40px;height:24px;border-radius:var(--st-r-pill);border:0;background:var(--st-line);cursor:pointer;transition:background var(--st-dur)}
.st-toggle::after{content:"";position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.2);transition:transform var(--st-dur) var(--st-ease)}
.st-toggle[aria-checked="true"]{background:var(--st-accent)}
.st-toggle[aria-checked="true"]::after{transform:translateX(16px)}

/* toast / alert */
.st-toast{display:flex;align-items:flex-start;gap:12px;padding:14px 16px;border-radius:var(--st-r-md);background:var(--st-surface);border:1px solid var(--st-line);box-shadow:var(--st-sh-pop);font-size:var(--st-fs-md);color:var(--st-ink-soft);max-width:420px}
.st-toast__title{font-weight:700;color:var(--st-ink)}
.st-toast--info .st-icon{color:var(--st-info)}
.st-toast--success .st-icon{color:var(--st-accent)}
.st-toast--warn .st-icon{color:var(--st-warn)}
.st-toast--error .st-icon{color:var(--st-danger)}
.st-toasts{position:fixed;right:20px;bottom:20px;display:flex;flex-direction:column;gap:10px;z-index:50}

/* layout — adapted from Strata serve/web/app.css (MIT) */
[hidden]{display:none!important}
html,body{height:100%}body{display:flex;flex-direction:column;overflow:hidden}
main{flex:1;min-height:0;display:flex}.view{flex:1;min-height:0;display:flex;flex-direction:column}.view--monitor,.view--about{overflow-y:auto}
.muted{color:var(--st-ink-muted)}.small{font-size:var(--st-fs-xs)}
.card-title{font-weight:var(--st-fw-bold);font-size:var(--st-fs-base);color:var(--st-ink)}
.st-header{position:relative;z-index:2;flex:none}
.st-header .st-tabs{margin-left:var(--st-s-4)}
.st-header__host{font-size:var(--st-fs-sm);font-weight:var(--st-fw-medium);color:var(--st-ink-muted);margin-left:-4px}
.st-pill[data-state="error"] .st-pill__dot{background:var(--st-danger);box-shadow:0 0 0 4px var(--st-danger-tint)}
.monitor,.about{max-width:1240px;width:100%;margin:0 auto;padding:var(--st-s-6);display:flex;flex-direction:column;gap:var(--st-s-5)}
.state-card{display:flex;flex-direction:column;gap:var(--st-s-3)}
.state-card__head{display:flex;align-items:center;gap:var(--st-s-4);flex-wrap:wrap}
.badges{display:flex;gap:var(--st-s-2);flex-wrap:wrap}
.badges .st-badge{opacity:.38;transition:opacity var(--st-dur)}
.badges .st-badge.on{opacity:1}
.state-card__row{display:flex;justify-content:space-between;gap:var(--st-s-4);font-size:var(--st-fs-sm);font-weight:var(--st-fw-medium);font-variant-numeric:tabular-nums}
.st-progress.busy .st-progress__bar{width:100%!important;animation:busy 1.4s ease-in-out infinite}
@keyframes busy{50%{opacity:.35}}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--st-s-4)}
.metric-card{padding:var(--st-s-4) var(--st-s-5)}
.st-metric__value small{font-size:var(--st-fs-md);font-weight:var(--st-fw-medium);color:var(--st-ink-muted);margin-left:4px}
.st-metric__sub{font-size:var(--st-fs-xs);color:var(--st-ink-muted);font-variant-numeric:tabular-nums;min-height:1em}
.st-metric__spark{display:block;overflow:visible}
.speed-values{display:flex;justify-content:space-between;gap:var(--st-s-3)}
.speed-values>div{display:flex;flex-direction:column;gap:10px;min-width:0}
.speed-values>div:last-child{text-align:right}
.speed-values .st-metric__value{font-size:clamp(20px,2.2vw,var(--st-fs-stat))}
.st-metric__spark .speed-prefill{color:color-mix(in srgb,var(--st-accent) 65%,black)}
.st-metric__spark[data-tone="warn"]{color:var(--st-warn)}
.st-metric__spark[data-tone="info"]{color:var(--st-info)}
.monitor-row{display:grid;grid-template-columns:minmax(260px,1fr) minmax(0,2.4fr);gap:var(--st-s-4)}
.ctx-card{display:flex;flex-direction:column;gap:var(--st-s-2)}
.ctx-gauge{position:relative;width:150px;height:150px;margin:var(--st-s-2) auto var(--st-s-3)}
.ctx-gauge .st-gauge{width:150px;height:150px}
.ctx-gauge .st-gauge__fill{transition:stroke-dasharray 400ms var(--st-ease)}
.ctx-gauge__label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:var(--st-fs-xs)}
.ctx-gauge__pct{font-weight:var(--st-fw-black);font-size:var(--st-fs-xl);line-height:1.1;font-variant-numeric:tabular-nums}
.bar-row{display:flex;justify-content:space-between;font-size:var(--st-fs-sm);font-weight:var(--st-fw-medium);margin-top:var(--st-s-2);font-variant-numeric:tabular-nums}
.req-card{display:flex;flex-direction:column;gap:var(--st-s-3);min-width:0}
.table-wrap{overflow:auto;max-height:420px}
#req-wrap .st-table th,#req-wrap .st-table td{white-space:nowrap;padding-inline:8px}
.req-head{display:flex;align-items:center;justify-content:space-between;gap:var(--st-s-2)}
.req-totals{margin:0;font-size:var(--st-fs-sm)}
.chart-card{display:flex;flex-direction:column;gap:var(--st-s-3)}
.chart{display:block;width:100%;height:120px;overflow:visible}
.legend{display:flex;gap:var(--st-s-4);flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.foot{margin:0;display:flex;justify-content:space-between;gap:var(--st-s-4);flex-wrap:wrap}
.req-totals:empty{display:none}
.about{max-width:860px}
.about .st-card{display:flex;flex-direction:column;gap:var(--st-s-3)}
.about p{margin:0}
.facts{display:grid;grid-template-columns:minmax(140px,max-content) 1fr;gap:10px var(--st-s-6);margin:0;font-size:var(--st-fs-md)}
.facts dt{color:var(--st-ink-muted)}
.facts dd{margin:0;color:var(--st-ink-soft);font-variant-numeric:tabular-nums;display:flex;align-items:center;gap:6px;overflow-wrap:anywhere}
.facts dd code{font:400 13px var(--st-font-mono)}
.facts .st-btn--icon{width:28px;height:28px}
.facts .st-btn--icon .st-icon{width:16px;height:16px}
@media (max-width:1000px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.monitor-row{grid-template-columns:1fr}}
@media (max-width:640px){.st-header{padding:0 var(--st-s-3);gap:var(--st-s-2)}.st-header .st-tabs{margin-left:0}.st-tab{padding:0 10px}
 .st-tab .st-icon,.st-header__host{display:none}#pill-text{display:none}.monitor,.about{padding:var(--st-s-4)}
 .facts{grid-template-columns:1fr;gap:2px}.facts dd{margin-bottom:var(--st-s-2)}}
/* chat, drawer, toasts — adapted from Strata serve/web/app.css (MIT) */
.chat-scroll{flex:1;min-height:0;overflow-y:auto}
.chat{max-width:860px;margin:0 auto;padding:var(--st-s-8) var(--st-s-6) var(--st-s-6);display:flex;flex-direction:column;gap:var(--st-s-6)}
.chat-empty{margin:12vh auto 0;text-align:center}
.chat-empty__title{font-weight:var(--st-fw-black);font-size:var(--st-fs-display);line-height:var(--st-lh-tight);letter-spacing:-.02em}
.chat-empty__sub{margin-top:var(--st-s-3);color:var(--st-ink-muted)}
.st-msg{width:100%}.st-msg--assistant{align-self:flex-start}.st-msg--assistant .st-bubble{width:100%}
.st-msg--user .st-bubble{white-space:pre-wrap;overflow-wrap:anywhere}
.st-bubble>:first-child{margin-top:0}.st-bubble>:last-child{margin-bottom:0}
.st-bubble p,.st-bubble ul,.st-bubble ol{margin:0 0 var(--st-s-3)}
.st-bubble ul,.st-bubble ol{padding-left:var(--st-s-6)}.st-bubble li+li{margin-top:var(--st-s-1)}
.st-bubble h3{font-size:var(--st-fs-lg);margin:var(--st-s-4) 0 var(--st-s-2);color:var(--st-ink)}
.st-bubble h4{font-size:var(--st-fs-base);margin:var(--st-s-3) 0 var(--st-s-2);color:var(--st-ink)}
.st-bubble code.inline{font:400 .9em var(--st-font-mono);background:var(--st-surface-2);border:1px solid var(--st-line-soft);border-radius:6px;padding:1px 5px}
.st-bubble .st-code{margin:var(--st-s-3) 0}
.st-bubble blockquote{margin:0 0 var(--st-s-3);padding-left:var(--st-s-3);border-left:3px solid var(--st-line);color:var(--st-ink-muted)}
.st-bubble hr{border:0;border-top:1px solid var(--st-line);margin:var(--st-s-4) 0}
.st-bubble table{border-collapse:collapse;margin:0 0 var(--st-s-3);font-size:var(--st-fs-md)}
.st-bubble th,.st-bubble td{border:1px solid var(--st-line);padding:6px 10px;text-align:left}
.msg-files{display:flex;flex-wrap:wrap;gap:var(--st-s-2);justify-content:flex-end}
.st-msg .st-collapse{width:100%}
.st-collapse__body.thinking{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow-y:auto}
.st-msg__meta .st-btn--icon{width:28px;height:28px}.st-msg__meta .st-btn--icon .st-icon{width:16px;height:16px}
.cursor::after{content:"";display:inline-block;width:8px;height:1.1em;margin-left:2px;vertical-align:-3px;border-radius:2px;background:var(--st-accent);animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
.msg-error{color:var(--st-danger-text);background:var(--st-danger-tint);border-radius:var(--st-r-md);padding:10px 14px;font-size:var(--st-fs-md)}
.composer-wrap{flex:none;padding:0 var(--st-s-6) var(--st-s-6)}
.st-composer{max-width:860px;margin:0 auto}.st-composer textarea{max-height:40vh}
.st-btn--send{width:42px;padding:0}
.composer-hint{font-size:var(--st-fs-xs);color:var(--st-ink-muted);margin-right:var(--st-s-2);font-variant-numeric:tabular-nums}
.attachments{display:flex;flex-wrap:wrap;gap:var(--st-s-2);padding:2px 4px 0}
.st-composer.dragging{outline:2px dashed var(--st-accent);outline-offset:3px}
.chip{display:inline-flex;align-items:center;gap:6px;height:30px;padding:0 6px 0 10px;border-radius:var(--st-r-pill);background:var(--st-surface-2);border:1px solid var(--st-line);font-size:var(--st-fs-xs);color:var(--st-ink-soft)}
.chip .st-btn--icon{width:22px;height:22px;border-radius:50%}.chip .st-btn--icon .st-icon{width:14px;height:14px}
.toggle-row{display:flex;align-items:center;justify-content:space-between;gap:var(--st-s-4);font-size:var(--st-fs-sm);font-weight:var(--st-fw-medium);color:var(--st-ink-soft)}
.toggle-row .st-toggle{flex:none}
.st-drawer{z-index:40}
#drawer-close .st-icon{transform:rotate(-90deg)}
.scrim{position:fixed;inset:0;background:rgba(15,23,42,.18);z-index:39}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:4px;background:var(--st-surface-2);border:1px solid var(--st-line);border-radius:var(--st-r-md)}
.seg button{height:32px;border:0;border-radius:10px;background:transparent;color:var(--st-ink-muted);font:500 var(--st-fs-sm)/1 var(--st-font);cursor:pointer}
.seg button[aria-checked="true"]{background:var(--st-surface);color:var(--st-ink);font-weight:var(--st-fw-bold);box-shadow:var(--st-sh-card)}
.drawer-actions{display:grid;grid-template-columns:1fr 1fr;gap:var(--st-s-3);margin-top:var(--st-s-2)}
.st-toast{animation:toast-in 180ms var(--st-ease)}
@keyframes toast-in{from{opacity:0;transform:translateY(6px)}}
@media (max-width:640px){.chat{padding:var(--st-s-5) var(--st-s-4)}.composer-wrap{padding:0 var(--st-s-3) var(--st-s-3)}}
</style>
<script>try{const v=localStorage.getItem("ps5dash.theme");document.documentElement.dataset.theme=(v&&JSON.parse(v))||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light")}catch(e){}</script>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" style="display:none"><symbol id="i-chat" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M21 11.5C21 16.194 16.971 20 12 20C10.8 20 9.65 19.78 8.6 19.37L4 20.5L5.2 16.8C3.83 15.34 3 13.5 3 11.5C3 6.806 7.029 3 12 3C16.971 3 21 6.806 21 11.5Z" fill="currentColor"></path><path d="M21 11.5C21 16.194 16.971 20 12 20C10.8 20 9.65 19.78 8.6 19.37L4 20.5L5.2 16.8C3.83 15.34 3 13.5 3 11.5C3 6.806 7.029 3 12 3C16.971 3 21 6.806 21 11.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8 11.5H8.01M12 11.5H12.01M16 11.5H16.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-activity" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M2 12C2 7.286 2 4.929 3.464 3.464C4.929 2 7.286 2 12 2C16.714 2 19.071 2 20.536 3.464C22 4.929 22 7.286 22 12C22 16.714 22 19.071 20.536 20.536C19.071 22 16.714 22 12 22C7.286 22 4.929 22 3.464 20.536C2 19.071 2 16.714 2 12Z" fill="currentColor"></path><path d="M2 12C2 7.286 2 4.929 3.464 3.464C4.929 2 7.286 2 12 2C16.714 2 19.071 2 20.536 3.464C22 4.929 22 7.286 22 12C22 16.714 22 19.071 20.536 20.536C19.071 22 16.714 22 12 22C7.286 22 4.929 22 3.464 20.536C2 19.071 2 16.714 2 12Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M6 12.5H8.5L10.5 8L13.5 16L15.5 11.5H18" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-info" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" fill="currentColor"></path><path d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 11V16.5M12 7.5H12.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-gauge" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M2 15C2 9.477 6.477 5 12 5C17.523 5 22 9.477 22 15C22 16.5 21.7 17.8 21.1 19H2.9C2.3 17.8 2 16.5 2 15Z" fill="currentColor"></path><path d="M2 15C2 9.477 6.477 5 12 5C17.523 5 22 9.477 22 15C22 16.5 21.7 17.8 21.1 19H2.9C2.3 17.8 2 16.5 2 15Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 15L16 10.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M13 15C13 15.552 12.552 16 12 16C11.448 16 11 15.552 11 15C11 14.448 11.448 14 12 14C12.552 14 13 14.448 13 15Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-gpu" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M2 7.5C2 6.12 3.12 5 4.5 5H19.5C20.88 5 22 6.12 22 7.5V15.5C22 16.88 20.88 18 19.5 18H4.5C3.12 18 2 16.88 2 15.5V7.5Z" fill="currentColor"></path><path d="M2 7.5C2 6.12 3.12 5 4.5 5H19.5C20.88 5 22 6.12 22 7.5V15.5C22 16.88 20.88 18 19.5 18H4.5C3.12 18 2 16.88 2 15.5V7.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12.5 11.5C12.5 13.433 10.933 15 9 15C7.067 15 5.5 13.433 5.5 11.5C5.5 9.567 7.067 8 9 8C10.933 8 12.5 9.567 12.5 11.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M16 9V14M19 9V14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M6 18V20.5M10 18V20.5M14 18V20.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-layers" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M12 3L22 8L12 13L2 8L12 3Z" fill="currentColor"></path><path d="M12 3L22 8L12 13L2 8L12 3Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M2 16L12 21L22 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-thermometer" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M14 14.76V5C14 3.895 13.105 3 12 3C10.895 3 10 3.895 10 5V14.76C8.8 15.45 8 16.73 8 18.2C8 20.4 9.8 22 12 22C14.2 22 16 20.4 16 18.2C16 16.73 15.2 15.45 14 14.76Z" fill="currentColor"></path><path d="M14 14.76V5C14 3.895 13.105 3 12 3C10.895 3 10 3.895 10 5V14.76C8.8 15.45 8 16.73 8 18.2C8 20.4 9.8 22 12 22C14.2 22 16 20.4 16 18.2C16 16.73 15.2 15.45 14 14.76Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 17.5V9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-bolt" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M13 2L4 14H11L10 22L20 9.5H13L13 2Z" fill="currentColor"></path><path d="M13 2L4 14H11L10 22L20 9.5H13L13 2Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-memory" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M2 8C2 6.895 2.895 6 4 6H20C21.105 6 22 6.895 22 8V15H2V8Z" fill="currentColor"></path><path d="M2 8C2 6.895 2.895 6 4 6H20C21.105 6 22 6.895 22 8V15H2V8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M6 9.5V11.5M10 9.5V11.5M14 9.5V11.5M18 9.5V11.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M4 15V18M8 15V18M12 15V18M16 15V18M20 15V18" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-cpu" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M6 9C6 7.343 7.343 6 9 6H15C16.657 6 18 7.343 18 9V15C18 16.657 16.657 18 15 18H9C7.343 18 6 16.657 6 15V9Z" fill="currentColor"></path><path d="M6 9C6 7.343 7.343 6 9 6H15C16.657 6 18 7.343 18 9V15C18 16.657 16.657 18 15 18H9C7.343 18 6 16.657 6 15V9Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M10 11C10 10.448 10.448 10 11 10H13C13.552 10 14 10.448 14 11V13C14 13.552 13.552 14 13 14H11C10.448 14 10 13.552 10 13V11Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9.5 2.5V6M14.5 2.5V6M9.5 18V21.5M14.5 18V21.5M2.5 9.5H6M2.5 14.5H6M18 9.5H21.5M18 14.5H21.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-disk" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M2 9C2 7.343 3.343 6 5 6H19C20.657 6 22 7.343 22 9V15C22 16.657 20.657 18 19 18H5C3.343 18 2 16.657 2 15V9Z" fill="currentColor"></path><path d="M2 9C2 7.343 3.343 6 5 6H19C20.657 6 22 7.343 22 9V15C22 16.657 20.657 18 19 18H5C3.343 18 2 16.657 2 15V9Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M6 12H12.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M17.5 12H17.51" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-copy" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M8 10C8 8.895 8.895 8 10 8H18C19.105 8 20 8.895 20 10V18C20 19.105 19.105 20 18 20H10C8.895 20 8 19.105 8 18V10Z" fill="currentColor"></path><path d="M8 10C8 8.895 8.895 8 10 8H18C19.105 8 20 8.895 20 10V18C20 19.105 19.105 20 18 20H10C8.895 20 8 19.105 8 18V10Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M16 8V6C16 4.895 15.105 4 14 4H6C4.895 4 4 4.895 4 6V14C4 15.105 4.895 16 6 16H8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-check" viewBox="0 0 24 24" fill="none"><path d="M4.5 12.5L9.5 17.5L19.5 6.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-sun" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M16 12C16 14.209 14.209 16 12 16C9.791 16 8 14.209 8 12C8 9.791 9.791 8 12 8C14.209 8 16 9.791 16 12Z" fill="currentColor"></path><path d="M16 12C16 14.209 14.209 16 12 16C9.791 16 8 14.209 8 12C8 9.791 9.791 8 12 8C14.209 8 16 9.791 16 12Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 2V4M12 20V22M4.93 4.93L6.34 6.34M17.66 17.66L19.07 19.07M2 12H4M20 12H22M4.93 19.07L6.34 17.66M17.66 6.34L19.07 4.93" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-moon" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M21 12.79A9 9 0 1 1 11.21 3A7 7 0 0 0 21 12.79Z" fill="currentColor"></path><path d="M21 12.79A9 9 0 1 1 11.21 3A7 7 0 0 0 21 12.79Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-attach" viewBox="0 0 24 24" fill="none"><path d="M21.44 11.05L12.25 20.24A6 6 0 0 1 3.76 11.75L12.33 3.18A4 4 0 1 1 18 8.84L9.41 17.41A2 2 0 0 1 6.58 14.58L15.07 6.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-new-chat" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M21 11.5C21 16.194 16.971 20 12 20C10.8 20 9.65 19.78 8.6 19.37L4 20.5L5.2 16.8C3.83 15.34 3 13.5 3 11.5C3 6.806 7.029 3 12 3C16.971 3 21 6.806 21 11.5Z" fill="currentColor"></path><path d="M21 11.5C21 16.194 16.971 20 12 20C10.8 20 9.65 19.78 8.6 19.37L4 20.5L5.2 16.8C3.83 15.34 3 13.5 3 11.5C3 6.806 7.029 3 12 3C16.971 3 21 6.806 21 11.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 8.5V14.5M9 11.5H15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-download" viewBox="0 0 24 24" fill="none"><path d="M4 15V17C4 18.657 5.343 20 7 20H17C18.657 20 20 18.657 20 17V15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 4V14M12 14L8 10M12 14L16 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-settings" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M12.22 2H11.78A2 2 0 0 0 9.78 4V4.18A2 2 0 0 1 8.78 5.91L8.35 6.16A2 2 0 0 1 6.35 6.16L6.2 6.08A2 2 0 0 0 3.47 6.81L3.25 7.19A2 2 0 0 0 3.98 9.92L4.13 10.02A2 2 0 0 1 5.13 11.74V12.25A2 2 0 0 1 4.13 13.99L3.98 14.08A2 2 0 0 0 3.25 16.81L3.47 17.19A2 2 0 0 0 6.2 17.92L6.35 17.84A2 2 0 0 1 8.35 17.84L8.78 18.09A2 2 0 0 1 9.78 19.82V20A2 2 0 0 0 11.78 22H12.22A2 2 0 0 0 14.22 20V19.82A2 2 0 0 1 15.22 18.09L15.65 17.84A2 2 0 0 1 17.65 17.84L17.8 17.92A2 2 0 0 0 20.53 17.19L20.75 16.8A2 2 0 0 0 20.02 14.07L19.87 13.99A2 2 0 0 1 18.87 12.25V11.75A2 2 0 0 1 19.87 10.01L20.02 9.92A2 2 0 0 0 20.75 7.19L20.53 6.81A2 2 0 0 0 17.8 6.08L17.65 6.16A2 2 0 0 1 15.65 6.16L15.22 5.91A2 2 0 0 1 14.22 4.18V4A2 2 0 0 0 12.22 2Z" fill="currentColor"></path><path d="M12.22 2H11.78A2 2 0 0 0 9.78 4V4.18A2 2 0 0 1 8.78 5.91L8.35 6.16A2 2 0 0 1 6.35 6.16L6.2 6.08A2 2 0 0 0 3.47 6.81L3.25 7.19A2 2 0 0 0 3.98 9.92L4.13 10.02A2 2 0 0 1 5.13 11.74V12.25A2 2 0 0 1 4.13 13.99L3.98 14.08A2 2 0 0 0 3.25 16.81L3.47 17.19A2 2 0 0 0 6.2 17.92L6.35 17.84A2 2 0 0 1 8.35 17.84L8.78 18.09A2 2 0 0 1 9.78 19.82V20A2 2 0 0 0 11.78 22H12.22A2 2 0 0 0 14.22 20V19.82A2 2 0 0 1 15.22 18.09L15.65 17.84A2 2 0 0 1 17.65 17.84L17.8 17.92A2 2 0 0 0 20.53 17.19L20.75 16.8A2 2 0 0 0 20.02 14.07L19.87 13.99A2 2 0 0 1 18.87 12.25V11.75A2 2 0 0 1 19.87 10.01L20.02 9.92A2 2 0 0 0 20.75 7.19L20.53 6.81A2 2 0 0 0 17.8 6.08L17.65 6.16A2 2 0 0 1 15.65 6.16L15.22 5.91A2 2 0 0 1 14.22 4.18V4A2 2 0 0 0 12.22 2Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-stop" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M5 8C5 6.343 6.343 5 8 5H16C17.657 5 19 6.343 19 8V16C19 17.657 17.657 19 16 19H8C6.343 19 5 17.657 5 16V8Z" fill="currentColor"></path><path d="M5 8C5 6.343 6.343 5 8 5H16C17.657 5 19 6.343 19 8V16C19 17.657 17.657 19 16 19H8C6.343 19 5 17.657 5 16V8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-send" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M21 3L2 10L10 13.5L13.5 21L21 3Z" fill="currentColor"></path><path d="M21 3L2 10L10 13.5L13.5 21L21 3Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M10 13.5L21 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-thinking" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M12 2.5C8.41 2.5 5.5 5.41 5.5 9C5.5 11.3 6.7 13.3 8.5 14.5V16.5C8.5 17.05 8.95 17.5 9.5 17.5H14.5C15.05 17.5 15.5 17.05 15.5 16.5V14.5C17.3 13.3 18.5 11.3 18.5 9C18.5 5.41 15.59 2.5 12 2.5Z" fill="currentColor"></path><path d="M12 2.5C8.41 2.5 5.5 5.41 5.5 9C5.5 11.3 6.7 13.3 8.5 14.5V16.5C8.5 17.05 8.95 17.5 9.5 17.5H14.5C15.05 17.5 15.5 17.05 15.5 16.5V14.5C17.3 13.3 18.5 11.3 18.5 9C18.5 5.41 15.59 2.5 12 2.5Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9.5 20.5H14.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-chevron" viewBox="0 0 24 24" fill="none"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-trash" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M5 6L6 19C6.08 20.1 7 21 8.1 21H15.9C17 21 17.92 20.1 18 19L19 6H5Z" fill="currentColor"></path><path d="M5 6L6 19C6.08 20.1 7 21 8.1 21H15.9C17 21 17.92 20.1 18 19L19 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M3 6H21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M9 6V4.5C9 3.672 9.672 3 10.5 3H13.5C14.328 3 15 3.672 15 4.5V6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M10 10.5V16.5M14 10.5V16.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-warning" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M10.24 3.99L2.4 17.5C1.62 18.83 2.58 20.5 4.12 20.5H19.88C21.42 20.5 22.38 18.83 21.6 17.5L13.76 3.99C12.98 2.66 11.02 2.66 10.24 3.99Z" fill="currentColor"></path><path d="M10.24 3.99L2.4 17.5C1.62 18.83 2.58 20.5 4.12 20.5H19.88C21.42 20.5 22.38 18.83 21.6 17.5L13.76 3.99C12.98 2.66 11.02 2.66 10.24 3.99Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M12 9V13M12 16.5H12.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol><symbol id="i-error" viewBox="0 0 24 24" fill="none"><path opacity="0.12" d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" fill="currentColor"></path><path d="M22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 6.47715 6.47715 2 12 2C17.5228 2 22 6.47715 22 12Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M15 9L9 15M9 9L15 15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></symbol></svg>
<header class="st-header">
  <span class="st-header__brand">Qwen3.8</span><span class="st-header__host" id="host">PS5</span>
  <nav class="st-tabs" role="tablist" aria-label="Views">
    <button class="st-tab" role="tab" data-tab="chat" aria-selected="true"><svg class="st-icon st-icon--sm"><use href="#i-chat"/></svg>Chat</button>
    <button class="st-tab" role="tab" data-tab="monitor" aria-selected="false"><svg class="st-icon st-icon--sm"><use href="#i-activity"/></svg>Monitor</button>
    <button class="st-tab" role="tab" data-tab="about" aria-selected="false"><svg class="st-icon st-icon--sm"><use href="#i-info"/></svg>About</button>
  </nav>
  <span class="st-header__spacer"></span>
  <span class="st-pill" id="pill" data-state="idle" role="status" aria-live="polite"><span class="st-pill__dot"></span><span id="pill-text">Connecting&hellip;</span></span>
  <button class="st-btn st-btn--icon" id="theme-btn" title="Light / dark" aria-label="Switch between light and dark"><svg class="st-icon"><use id="theme-icon" href="#i-moon"/></svg></button>
</header>

<main>
<section class="view view--chat" id="view-chat">
  <div class="chat-scroll" id="chat-scroll">
    <div class="chat" id="chat">
      <div class="chat-empty" id="chat-empty">
        <div class="chat-empty__title">Ask anything</div>
        <div class="chat-empty__sub">Qwen3.8-Flash-Next with 2.68-bit EXL3 experts streamed from the SSD, running on a PlayStation 5 (Zen 2 + RDNA 2, 16 GB). This chat is kept in your browser.</div>
      </div>
    </div>
  </div>
  <div class="composer-wrap">
    <form class="st-composer" id="composer">
      <div class="attachments" id="attachments" hidden></div>
      <textarea id="input" rows="2" placeholder="Ask anything&hellip;" aria-label="Message"></textarea>
      <div class="st-composer__bar">
        <button type="button" class="st-btn st-btn--icon" id="attach-btn" title="Attach a text file (or drop it here)" aria-label="Attach a file"><svg class="st-icon"><use href="#i-attach"/></svg></button>
        <input type="file" id="file" multiple hidden>
        <button type="button" class="st-btn st-btn--icon" id="new-btn" title="New chat" aria-label="New chat"><svg class="st-icon"><use href="#i-new-chat"/></svg></button>
        <button type="button" class="st-btn st-btn--icon" id="export-btn" title="Save this chat as Markdown" aria-label="Save this chat"><svg class="st-icon"><use href="#i-download"/></svg></button>
        <button type="button" class="st-btn st-btn--icon" id="sampling-btn" title="Sampling and thinking" aria-label="Sampling and thinking"><svg class="st-icon"><use href="#i-settings"/></svg></button>
        <span class="spacer"></span>
        <span class="composer-hint" id="composer-hint"></span>
        <button type="button" class="st-btn st-btn--secondary" id="stop-btn" hidden><svg class="st-icon st-icon--sm"><use href="#i-stop"/></svg>Stop</button>
        <button type="submit" class="st-btn st-btn--primary st-btn--send" id="send-btn" aria-label="Send"><svg class="st-icon"><use href="#i-send"/></svg></button>
      </div>
    </form>
  </div>
</section>

<section class="view view--monitor" id="view-monitor" hidden>
  <div class="monitor">
    <div class="st-card state-card">
      <div class="state-card__head">
        <span class="card-title">Model state</span>
        <span class="badges" id="state-badges">
          <span class="st-badge" data-s="idle">Idle</span>
          <span class="st-badge st-badge--reading" data-s="reading">Reading</span>
          <span class="st-badge st-badge--generating" data-s="generating">Generating</span>
          <span class="st-badge st-badge--queued" data-s="queued">Queued</span>
          <span class="st-badge st-badge--error" data-s="error">Offline</span>
        </span>
      </div>
      <div class="state-card__row"><span id="state-label">Connecting&hellip;</span><span class="muted" id="state-detail"></span></div>
      <div class="st-progress" id="state-progress"><div class="st-progress__bar" id="state-bar" style="width:0%"></div></div>
    </div>

    <div class="metrics" id="metrics"></div>

    <div class="st-card chart-card">
      <div class="req-head"><span class="card-title">Last 50 requests</span>
        <span class="legend small muted"><span><i style="background:var(--st-accent)"></i>Decode t/s · top <b id="ch-gen-max">–</b></span>
          <span><i style="background:var(--st-warn)"></i>Wait s · top <b id="ch-wait-max">–</b></span></span></div>
      <svg class="chart" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
        <g id="ch-wait" style="color:var(--st-warn)"><path class="area" fill="currentColor" opacity=".10"/><path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>
        <g id="ch-gen" style="color:var(--st-accent)"><path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>
      </svg>
      <p class="muted small" id="medians" style="margin:0"></p>
    </div>

    <div class="monitor-row">
      <div class="st-card ctx-card">
        <span class="card-title">Context fill</span>
        <div class="ctx-gauge">
          <svg class="st-gauge" viewBox="0 0 120 120" aria-hidden="true">
            <circle class="st-gauge__track" cx="60" cy="60" r="50" fill="none" stroke-width="10" stroke-linecap="round" stroke-dasharray="235.6 314.2" transform="rotate(135 60 60)"/>
            <circle class="st-gauge__fill" id="ctx-fill" cx="60" cy="60" r="50" fill="none" stroke-width="10" stroke-linecap="round" stroke-dasharray="0 314.2" transform="rotate(135 60 60)"/>
          </svg>
          <div class="ctx-gauge__label"><span class="ctx-gauge__pct" id="ctx-pct">0%</span><span class="muted" id="ctx-sub">&ndash;</span></div>
        </div>
        <div class="bar-row"><span>Unified memory</span><span class="muted" id="ram-text">&ndash;</span></div>
        <div class="st-progress" id="ram-progress"><div class="st-progress__bar" id="ram-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>Expert pool (since start)</span><span class="muted" id="kv-text">&ndash;</span></div>
        <div class="st-progress" data-tone="info"><div class="st-progress__bar" id="kv-bar" style="width:0%"></div></div>
        <div class="bar-row"><span>APU temperature</span><span class="muted" id="temp-text">&ndash;</span></div>
        <div class="st-progress" data-tone="warn"><div class="st-progress__bar" id="temp-bar" style="width:0%"></div></div>
      </div>
      <div class="st-card req-card">
        <div class="req-head"><span class="card-title">Recent requests</span><span class="muted small" id="req-count"></span></div>
        <div class="table-wrap" id="req-wrap">
          <table class="st-table">
            <thead><tr><th>Time</th><th>Status</th><th title="chat = OpenAI chat, messages = Anthropic, responses = OpenAI Responses, text = completions">API</th><th class="num">Prompt</th><th class="num">Reused</th><th class="num">Output</th>
              <th class="num" title="Time from sending the request to the first token: queue, prompt reading and network">Wait</th>
              <th class="num" title="New (not reused) prompt tokens divided by the wait">Prefill t/s</th>
              <th class="num" title="Output tokens per second after the first token, from the server's own token count">Decode t/s</th>
              <th class="num" title="Expert lookups served from the pinned pool (the rest are read from the SSD)">Hits</th>
              <th class="num" title="MTP draft tokens the batched verify accepted">Drafts</th>
              <th class="num">Duration</th></tr></thead>
            <tbody id="req-body"><tr><td colspan="12" class="muted">No requests yet</td></tr></tbody>
          </table>
        </div>
        <p class="muted req-totals" id="req-totals"></p>
      </div>
    </div>
    <p class="muted small foot"><span id="updated">Connecting&hellip;</span><span>History is kept on the console in dash.jsonl</span></p>
  </div>
</section>

<section class="view view--about" id="view-about" hidden>
  <div class="about">
    <div class="st-card"><span class="card-title">Model and engine</span><dl class="facts" id="facts-engine"></dl></div>
    <div class="st-card"><span class="card-title">This machine</span><dl class="facts" id="facts-hw"></dl></div>
    <div class="st-card"><span class="card-title">Connect your tools</span>
      <p class="muted">Any OpenAI- or Anthropic-compatible client works with these addresses. The API key can be anything; the server does not check it.</p>
      <dl class="facts" id="facts-api"></dl></div>
    <p class="muted small">Design adapted from <a href="https://github.com/niko1221/strata">Strata</a> (MIT). Outfit font under the SIL Open Font License. <a href="/licenses">Licenses</a></p>
  </div>
</section>
</main>

<aside class="st-drawer" id="drawer" data-open="false" aria-label="Sampling and thinking" aria-hidden="true">
  <div class="st-drawer__head"><span class="card-title">Sampling</span>
    <button class="st-btn st-btn--icon" id="drawer-close" aria-label="Close"><svg class="st-icon"><use href="#i-chevron"/></svg></button></div>
  <div class="st-drawer__body">
    <div class="st-field">
      <label class="st-label">Thinking <output id="o-thinking"></output></label>
      <div class="seg" id="s-thinking" role="radiogroup" aria-label="Thinking">
        <button type="button" role="radio" data-v="none">Off</button><button type="button" role="radio" data-v="low">Low</button><button type="button" role="radio" data-v="medium">Medium</button><button type="button" role="radio" data-v="xhigh">XHigh</button>
      </div>
      <span class="muted small">This console holds 2,048 tokens of context: thinking fills it fast</span>
    </div>
    <div class="st-field">
      <label class="st-label" for="s-temp">Temperature <output id="o-temp"></output></label>
      <input class="st-slider" type="range" id="s-temp" min="0" max="1.5" step="0.05">
      <span class="muted small">0 = always the most likely word (exact, repeatable)</span>
    </div>
    <div class="st-field">
      <label class="st-label" for="s-topp">Top-p <output id="o-topp"></output></label>
      <input class="st-slider" type="range" id="s-topp" min="0.05" max="1" step="0.05">
    </div>
    <div class="st-field">
      <label class="st-label" for="s-minp">Min-p <output id="o-minp"></output></label>
      <input class="st-slider" type="range" id="s-minp" min="0" max="0.3" step="0.01">
    </div>
    <div class="st-field">
      <label class="st-label" for="s-max">Max tokens <output>empty = until done</output></label>
      <input class="st-input" id="s-max" type="number" min="1" step="1" placeholder="Until done">
    </div>
    <div class="st-field">
      <label class="st-label" for="s-seed">Seed <output>empty = random</output></label>
      <input class="st-input" id="s-seed" type="number" min="1" step="1" placeholder="Random">
    </div>
    <div class="toggle-row"><span>Show thinking<br><span class="muted small">expanded while it streams</span></span>
      <button class="st-toggle" id="s-show" role="switch" aria-checked="true" aria-label="Show thinking"></button></div>
    <p class="muted small" style="margin:0">Defaults are Qwen's non-thinking ones: temperature 0.7, top-p 0.8, top-k 20. Temperature 0 is exact and the fastest here (the MTP draft head's guesses are accepted most often). These settings apply to this chat only.</p>
    <div class="drawer-actions">
      <button class="st-btn st-btn--secondary" id="s-reset" type="button">Reset</button>
      <button class="st-btn st-btn--primary" id="s-apply" type="button">Apply</button>
    </div>
  </div>
</aside>
<div class="scrim" id="scrim" hidden></div>
<div class="st-toasts" id="toasts" aria-live="polite"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const gib = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const store = {
  get(k, d) { try { const v = localStorage.getItem("ps5dash." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("ps5dash." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ---------------------------------------------------------------- theme + tabs
function applyTheme(t) { document.documentElement.dataset.theme = t; $("theme-icon").setAttribute("href", t === "dark" ? "#i-sun" : "#i-moon"); }
applyTheme(document.documentElement.dataset.theme || "dark");
$("theme-btn").onclick = () => { const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; applyTheme(t); store.set("theme", t); };
const TABS = ["chat", "monitor", "about"];
function showTab(name) {
  if (!TABS.includes(name)) name = "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const t of TABS) $(`view-${t}`).hidden = t !== name;
  if (location.hash.slice(1) !== name) history.replaceState(null, "", name === "chat" ? location.pathname : `#${name}`);
  if (name === "chat") $("input").focus();
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ---------------------------------------------------------------- the eight cards
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge"},
  {key: "hit", label: "Expert hits", icon: "gpu"},
  {key: "mem", label: "Unified memory", icon: "layers"},
  {key: "temp", label: "APU temp", icon: "thermometer", tone: "warn"},
  {key: "accept", label: "Draft accepted", icon: "bolt"},
  {key: "cache", label: "Prompt cache", icon: "memory", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu"},
  {key: "disk", label: "Expert stream", icon: "disk", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">–</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">–</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}
function setPill(state, text) { $("pill").dataset.state = state; $("pill-text").textContent = text; }
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" title="Copy" aria-label="Copy ${esc(k)}">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
document.addEventListener("click", (e) => { const b = e.target.closest("[data-copy]"); if (b) copyText(b.dataset.copy, b); });

// ---------------------------------------------------------------- render
function render(d) {
  const L = d.live || {}, hw = d.hw || {}, up = d.upstream || {}, st = d.static || {}, h = d.history || {};
  const reqs = d.requests || [], last = reqs.find((r) => r.status === 200 && !r.aborted);
  if (st.host) $("host").textContent = st.host.toUpperCase();
  if (up.model) modelId = up.model;

  // header pill
  if (!up.ok) setPill("error", "Model offline");
  else if (L.queued > 0) setPill("queued", `${L.queued} queued`);
  else if (L.state === "reading") setPill("reading", `Reading prompt · ${fmt(L.elapsed)}s`);
  else if (L.state === "generating") setPill("generating", `Generating · ${L.tok_s == null ? "…" : fmt(L.tok_s, 1) + " tok/s"}`);
  else setPill("idle", "Idle");

  // model state
  const on = !up.ok ? "error" : L.queued > 0 ? "queued" : L.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || (up.ok && b.dataset.s === L.state));
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0, busy = false;
  delete prog.dataset.tone;
  if (!up.ok) {
    label = "The engine is not answering"; detail = "starting (about 20 s), or stopped"; prog.dataset.tone = "danger"; busy = true;
  } else if (L.state === "reading") {
    label = "Reading prompt"; detail = `${fmt(L.elapsed, 1)} s`; prog.dataset.tone = "info"; busy = true;
  } else if (L.state === "generating") {
    label = "Generating";
    detail = `${fmt(L.generated)}${L.max_tokens ? ` / ${fmt(L.max_tokens)}` : ""} tokens · ${L.tok_s == null ? "…" : fmt(L.tok_s, 1)} tok/s · ${fmt(L.elapsed)} s`;
    if (L.max_tokens) pct = Math.min(100, (100 * L.generated) / L.max_tokens); else busy = true;
  } else if (last) {
    detail = `last: ${fmt(last.completion)} tokens${last.gen_tps ? ` at ${fmt(last.gen_tps, 1)} tok/s` : ""}` +
      `${last.gpu_ms != null ? ` · GPU ${fmt(last.gpu_ms, 1)} ms, expert wait ${fmt(last.io_ms, 1)} ms per token` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  prog.classList.toggle("busy", busy);
  $("state-bar").style.width = `${pct}%`;

  // cards
  const genNow = L.state === "generating";
  setMetric("speed", genNow ? (L.tok_s == null ? null : fmt(L.tok_s, 1)) : last && last.gen_tps != null ? fmt(last.gen_tps, 1) : null, "t/s",
            genNow ? "Decode now" : last ? "Decode last request" : "Decode");
  setMetric("prefill", last && last.prefill_tps != null ? fmt(last.prefill_tps) : null, "t/s", last ? "Prefill last request" : "Prefill");
  const pr = d.per_request || {}, eng = d.engine || {};
  spark("sp-speed", (pr.gen_tps || []).slice(-30));
  spark("sp-prefill", (pr.prefill_tps || []).slice(-30));
  setMetric("hit", last && last.hit != null ? fmt(100 * last.hit, 1) : null, "%",
            last && last.reads_per_tok != null ? `${fmt(last.reads_per_tok, 1)} SSD reads/token` : "of expert lookups in the pool");
  spark("sp-hit", (pr.hit_pct || []).slice(-30), 100);
  setMetric("mem", hw.mem_used == null ? null : gib(hw.mem_used), hw.mem_total ? `/ ${gib(hw.mem_total, 0)} GB` : "GB",
            hw.hugetlb ? `${gib(hw.hugetlb)} GB pinned expert pool` : "CPU and GPU share it");
  spark("sp-mem", h.mem_used, hw.mem_total);
  setMetric("temp", hw.soc_temp == null ? null : fmt(hw.soc_temp), "°C", hw.nvme_temp != null ? `CPU + GPU die · SSD ${fmt(hw.nvme_temp)} °C` : "CPU + GPU die");
  spark("sp-temp", h.soc_temp, 100);
  setMetric("accept", last && last.spec_acc != null ? fmt(100 * last.spec_acc) : null, "%",
            eng.spec ? `MTP drafts · gate ${fmt(eng.spec_gate, 2)}` : "speculative decoding off");
  spark("sp-accept", (pr.accept_pct || []).slice(-30), 100);
  setMetric("cache", last && last.prompt ? fmt((100 * (last.cached || 0)) / last.prompt) : null, "%",
            last && last.prompt ? `${kfmt(last.cached || 0)} of ${kfmt(last.prompt)} tokens reused` : "");
  spark("sp-cache", (pr.reuse_pct || []).slice(-30), 100);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", [st.cpu_count ? `${st.cpu_count} cores` : "", hw.load != null ? `load ${fmt(hw.load, 2)}` : ""].filter(Boolean).join(" · "));
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) setMetric("disk", null, "", "");
  else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1000, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              last && last.io_ms != null ? `expert wait ${fmt(last.io_ms, 1)} ms/token` : "NVMe read (experts)");
  }
  spark("sp-disk", h.disk_read);

  // context, memory, cache, temperature
  const ctx = up.ctx || 0;
  const used = last ? (last.prompt || 0) + (last.completion || 0) : 0;
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${kfmt(ctx)}` : "–";
  $("ram-text").textContent = hw.mem_total ? `${gib(hw.mem_used)} / ${gib(hw.mem_total, 0)} GB` : "–";
  const ramPct = hw.mem_total ? (100 * hw.mem_used) / hw.mem_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("kv-text").textContent = eng.pool_hit_pct == null ? "–" : `${fmt(eng.pool_hit_pct, 1)}% hits${up.slots ? ` · ${fmt(+up.slots)} slots, ${fmt(+up.pool_gib, 1)} GiB` : ""}`;
  $("kv-bar").style.width = eng.pool_hit_pct == null ? "0%" : `${Math.min(100, eng.pool_hit_pct)}%`;
  $("temp-text").textContent = hw.soc_temp == null ? "–" : `${fmt(hw.soc_temp)} °C`;
  $("temp-bar").style.width = hw.soc_temp == null ? "0%" : `${Math.min(100, hw.soc_temp)}%`;

  // last 50 requests chart + medians
  const top = (a) => Math.max(0, ...(a || []).filter((x) => x != null));
  spark("ch-gen", pr.gen_tps); spark("ch-wait", pr.ttft);
  $("ch-gen-max").textContent = pr.gen_tps && pr.gen_tps.length ? fmt(top(pr.gen_tps), 1) : "–";
  $("ch-wait-max").textContent = pr.ttft && pr.ttft.length ? fmt(top(pr.ttft), 2) : "–";
  const m = d.median_last20 || {};
  $("medians").textContent = m.gen_tps == null ? "No finished requests yet" :
    `Median of the last 20: decode ${fmt(m.gen_tps, 1)} tok/s · wait ${fmt(m.ttft, 2)} s · prefill ${fmt(m.prefill_tps)} tok/s`;
  $("updated").textContent = `Updated ${d.now.slice(11)}`;

  // recent requests
  const t = d.totals || {};
  $("req-count").textContent = t.requests ? `${fmt(t.requests)} kept` : "";
  $("req-body").innerHTML = reqs.length ? reqs.map((r) => {
    const s = r.status !== 200 ? ["error", `Error ${r.status}`] : r.aborted ? ["queued", "Stopped"] : ["generating", "Done"];
    return `<tr><td>${esc(r.ts.slice(11, 19))}</td><td><span class="st-badge st-badge--${s[0]}">${esc(s[1])}</span></td>
      <td class="muted">${esc(({"/v1/chat/completions": "chat", "/v1/completions": "text", "/v1/responses": "responses", "/v1/messages": "messages"})[(r.path || "").split("?")[0]] || r.path || "")}</td>
      <td class="num">${fmt(r.prompt)}</td><td class="num">${fmt(r.cached)}</td><td class="num">${fmt(r.completion)}</td>
      <td class="num">${r.ttft == null ? "–" : fmt(r.ttft, 2) + " s"}</td><td class="num">${fmt(r.prefill_tps)}</td>
      <td class="num">${fmt(r.gen_tps, 1)}</td><td class="num">${r.hit == null ? "–" : fmt(100 * r.hit, 1) + "%"}</td>
      <td class="num">${r.spec_acc == null ? "–" : fmt(100 * r.spec_acc) + "%"}</td><td class="num">${fmt(r.total, 1)} s</td></tr>`;
  }).join("") : `<tr><td colspan="12" class="muted">No requests yet</td></tr>`;
  if (t.requests && t.since) {
    const since = new Date(t.since).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
    $("req-totals").textContent = `Since ${since}: ${fmt(t.requests)} requests · ${fmt(t.read)} prompt tokens read` +
      `${t.prefill_tps ? ` at ${fmt(t.prefill_tps)} tok/s` : ""} (${fmt(t.reused)} reused${t.reused_pct != null ? `, ${t.reused_pct}% cache hits` : ""}) · ${fmt(t.output)} written` +
      `${t.decode_tps ? ` at ${fmt(t.decode_tps, 1)} tok/s` : ""}`;
  } else $("req-totals").textContent = "";

  // about
  const base = location.origin;
  facts($("facts-engine"), [
    ["Model", "Qwen3.8-Flash-Next · 48 layers, 512 experts + 1 shared per layer (top-10), hyper-connections, PLE n-gram table"],
    ["Experts", eng.pack ? `EXL3 ${eng.pack.includes("268") ? "2.68" : "3.18"} bpw (${eng.pack}), streamed from the NVMe into a pinned pool` : null],
    ["Dense weights", up.resident_gib ? `PS5Q4 4.5 bpw · ${fmt(+up.resident_gib, 2)} GiB resident (VRAM carve + GTT)` : null],
    ["Expert pool", up.slots ? `${fmt(+up.slots)} slots · ${fmt(+up.pool_gib, 2)} GiB on 2 MiB hugepages${eng.pool_hit_pct != null ? ` · ${fmt(eng.pool_hit_pct, 1)}% hits` : ""}` : null],
    ["Served as", up.model],
    ["Context", ctx ? `${fmt(ctx)} tokens` : null],
    ["Speculative decoding", eng.spec == null ? null : eng.spec ? `MTP head, ${eng.spec} draft per verify, confidence gate ${fmt(eng.spec_gate, 2)}${eng.mtp_accept_pct != null ? ` · last request ${fmt(eng.mtp_accept_pct, 1)}% accepted` : ""}` : "Off"],
    ["Engine", "flash_decode · Vulkan compute on RADV/ACO · one command buffer per token / verify batch"],
    ["Dashboard running since", st.started ? st.started.replace("T", " ") : null],
  ]);
  facts($("facts-hw"), [
    ["Host", st.host],
    ["GPU", st.gpu_name ? `${st.gpu_name} · ${st.cus} CUs${st.cus === 40 ? " (harvested CUs unlocked)" : ""}` : null],
    ["Memory", hw.mem_total ? `16 GB GDDR6 unified · ${gib(hw.mem_total, 1)} GB visible to Linux` : null],
    ["CPU", st.cpu_count ? `${st.cpu_name || "Zen 2"} · ${st.cpu_count} threads` : null],
    ["Kernel", st.kernel ? `${st.kernel}${st.ai_mode ? " · AI mode" : ""}` : null],
  ]);
  facts($("facts-api"), [
    ["OpenAI-compatible", `${base}/v1`, true],
    ["Anthropic-compatible", base, true],
    ["Model ID", up.model || "qwen3.8-flash-next-ps5", true],
    ["Stats (JSON)", `${base}/api/stats`, true],
  ]);
}

// ---------------------------------------------------------------- toasts + copy (from Strata)
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary"; b.style.height = "32px"; b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}
async function copyText(text, btn) {
  try {
    if (!navigator.clipboard || !window.isSecureContext) throw new Error("no async clipboard");
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // plain http (not a secure context): no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta);
    ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    if (use) { use.setAttribute("href", "#i-check"); setTimeout(() => use.setAttribute("href", "#i-copy"), 1500); }
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ---------------------------------------------------------------- Markdown (escaped first, then formatted; from Strata)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); const h = m[1].length <= 2 ? "h3" : "h4"; out.push(`<${h}>${inline(m[2])}</${h}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ---------------------------------------------------------------- Chat (from Strata, text only)
const DEFAULTS = {thinking: "none", temperature: 0.7, top_p: 0.8, min_p: 0, max: "", seed: "", show: true};
let settings = {...DEFAULTS, ...store.get("sampling", {})};
let messages = store.get("chat", []);
let attachments = [];                 // {name, text}
let busy = null;                      // {controller, msg}
let modelId = "qwen3.8-flash-next-ps5";

function saveChat() { store.set("chat", messages); }
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-files";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = markdown(m.text || "");
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function fileBlock(f) {           // a file's text in the message, fenced with more backticks than it contains itself
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) { return [m.text, ...(m.files || []).filter((f) => f.text != null).map(fileBlock)].filter((s) => s).join("\n\n"); }
function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") out.push({role: "user", content: userText(m)});
    else if (!(busy && busy.msg === m)) {               // the answer being asked for now is not history yet
      const msg = {role: "assistant", content: m.text || ""};
      if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
      out.push(msg);
    }
  }
  return out;
}
function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, files: attachments, time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: modelId, messages: apiMessages(), stream: true, stream_options: {include_usage: true}};
  if (settings.thinking === "none") body.chat_template_kwargs = {enable_thinking: false}; else body.reasoning_effort = settings.thinking;
  if (settings.temperature > 0) Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, min_p: +settings.min_p, top_k: 20});
  else body.temperature = 0;
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("/v1/chat/completions", {method: "POST", headers: {"Content-Type": "application/json"},
                                                    body: JSON.stringify(body), signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { const e = (await r.json()).error; msg = typeof e === "string" ? e : (e && e.message) || msg; } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the model server reported an error");
        if (j.usage) usage = j.usage;
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const thought = d.reasoning_content || d.reasoning;
        if (thought) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          m.reasoning += thought;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}`;
  } else if (m.stopped) m.meta = "Stopped";
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);
$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat(); renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${userText(m)}\n` :
    `## ${modelId}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `qwen-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// text files: the attach button or dropping them on the chat (from Strata; this model is text only)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql|cs|gd|gdshader|shader|hlsl|glsl|usf|ush)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data).`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon("attach", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...((e.dataTransfer || {}).types || [])].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}

// ---------------------------------------------------------------- the sampling drawer (from Strata)
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) loadDrawer();
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-minp").value = s.min_p;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  outputs();
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-minp").textContent = (+$("s-minp").value).toFixed(2);
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away (default)", low: "brief", medium: "balanced", xhigh: "thorough"}[sel.dataset.v] : "";
  for (const id of ["s-topp", "s-minp"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-minp"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "none", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              min_p: +$("s-minp").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  openDrawer(false);
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });
setBusy(false);
renderChat();

showTab(location.hash.slice(1));

// ---------------------------------------------------------------- polling
let timer = null, failures = 0;
function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }
async function tick() {
  try {
    const d = await (await fetch("/api/stats", {cache: "no-store"})).json();
    failures = 0;
    render(d);
    schedule(document.hidden ? 15000 : d.live && d.live.state !== "idle" ? 1000 : 2000);
  } catch (e) {
    if (++failures >= 3) { setPill("error", "Dashboard not reachable"); $("state-label").textContent = "This page cannot reach the dashboard"; }
    schedule(3000);
  }
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) tick(); });
tick();
</script>
</body>
</html>"""

LICENSES = r"""ps5lm_dash has its visual design adapted from Strata (https://github.com/niko1221/strata).

MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


======================================================================
Outfit font (embedded)
======================================================================

Copyright 2021 The Outfit Project Authors (https://github.com/Outfitio/Outfit-Fonts)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
https://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded, 
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
"""
PAGE_BYTES = PAGE.encode()
PAGE_GZ = gzip.compress(PAGE_BYTES, 9)
PAGE_ETAG = '"%s"' % hashlib.sha1(PAGE_BYTES).hexdigest()[:16]


# ---------------------------------------------------------------- main

def main():
    global ARGS
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Live monitor + chat + stats proxy for the PS5 engine (ps5lm_server)")
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=8080)
    p.add_argument("--upstream", default="http://127.0.0.1:8081", help="ps5lm_server base URL")
    p.add_argument("--log", default=os.path.expanduser("~/ps5lm/dash.jsonl"), help="Request history file")
    p.add_argument("--timeout", type=int, default=3600, help="Seconds to wait on the engine")
    p.add_argument("--sample", type=float, default=2.0, help="Seconds between machine samples")
    p.add_argument("--engine-dir", help="Serving recipe checkout, to show its version")
    p.add_argument("--model", help="Model checkpoint folder, to show its name and size")
    p.add_argument("--table", help="n-gram (PLE) table folder, to show its size")
    p.add_argument("--mtp", type=int, help="MTP speculative tokens, to show in About")
    ARGS = p.parse_args()
    ARGS.upstream_parsed = urlparse(ARGS.upstream)
    load_history()
    load_static()
    threading.Thread(target=sampler, daemon=True).start()
    srv = ThreadingHTTPServer((ARGS.listen_host, ARGS.listen_port), Handler)
    srv.daemon_threads = True
    print(f"ps5lm_dash: monitor + chat + API on http://{ARGS.listen_host}:{ARGS.listen_port}  ->  {ARGS.upstream}", flush=True)
    print(f"dash: request history in {ARGS.log}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
