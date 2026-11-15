#!/usr/bin/env python3
"""PS5 Gate 0/1 probe toolkit (drives the console over the network from the PC).

Run:  python -I tools/ps5probe/probe.py discover [--subnet 10.0.0.]
      python -I tools/ps5probe/probe.py native   --host 10.0.0.113 [--copytest] [--bwtest]
      python -I tools/ps5probe/probe.py linux    --host <ps5-linux-ip> [--user ps5]
      python -I tools/ps5probe/probe.py report   recon/ps5probe/linux-....json

`native`  talks to Pegasus DL (:6970) on the jailbroken PS5 OS: fingerprints the
          console, classifies mounts (internal SSD / USB / M.2 candidates), measures
          the HTTP pipe, and with --copytest measures on-disk copy throughput of an
          EXISTING file (falls to USB if no >=1 GiB file is on /data — see --bwtest).
          --bwtest self-creates an 8 GiB file on /data and same-mount-copies it:
          the internal-SSD read+write floor at 13.00 without Linux (supersedes the
          82.5 MB/s USB number). Upload takes ~6 min once; later runs reuse it.
`linux`   runs tools/ps5probe/gate0_1.sh on ps5-linux over ssh, evaluates
          every gate (G0/G1 + N6 dual-NVMe + N7 cache-replay) from the plan and the
          two frontier reports, and writes recon/gates/ps5-gate-<ts>.{json,md}.
          Add --n7 for the expert-cache replay (hit-rate-vs-capacity curve; writes a
          ~40 GiB store on first run, persisted for reuse).
`report`  re-evaluates a saved recon/ps5probe/linux-*.json without re-running the
          battery (physics flags below apply).

Writes are sandboxed: native mode only touches <mount>/ps5-probe (created and
deleted); linux mode only creates/deletes a scratch fio file. Stdlib only.
"""
import argparse, json, os, pathlib, re, socket, subprocess, sys, threading, time, urllib.parse, urllib.request, urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "recon" / "ps5probe"
REPORT_DIR = ROOT / "recon" / "gates"
GATE_SH = pathlib.Path(__file__).resolve().parent / "gate0_1.sh"
TS = time.strftime("%Y%m%d-%H%M%S")

# engine model defaults (Qwen3.8-Flash-Next: 63.4 GiB routed experts, top-10 of 512).
# gib_per_token 1.24 = W4 recompute from the HF config.json (480 expert reads @
# 2,764,800 B/token, UD-Q4_K_XL); the old k/experts*expert_gb arithmetic gave 1.40 (~13% high).
# expert_gb 63.4 = 512 x 48 x 2,764,800 B = 68.07 GB (S5 2026-10-21: the old 71.7 GiB was the
# stale 1.40-era figure and overstated the TTFT projection row ~13%; config-derived 3x2560x640).
DEF = dict(expert_gb=63.4, k=10, experts=512, gib_per_token=1.24)
# S10: verdict lines are computed from the ledger physics for BOTH packs, never canned strings.
# Mirrors of engine/ledger.py (this tool is standalone): EXL3 1.66 bpw G and the decimal-MB unit.
G_EXL3_1_66 = 0.443
MB_PER_GIB_S = 1073.741824
# fio spans below this are not store-shaped: on the DRAM-less internal drive a 1 GiB randread
# span reads at ~99% of seq. S11: the S10 chase numbers (3299/4858) were SLC-assisted reads of
# fresh files; NAND-true old-data rand 2.6M is 2290-2450, 1020k 1590-1890 MB/s.
STORE_SPAN_MB = 20480


def need_hit(mb_s, g, target=40.0):
    """Cache hit needed for `target` tok/s at fio bandwidth mb_s (decimal MB/s)."""
    return max(0.0, 1.0 - (mb_s / MB_PER_GIB_S) / (g * target))


def jget(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def call(method, base, path, body=None, raw=None, timeout=60, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None and raw is None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- discover
def cmd_discover(a):
    prefix = a.subnet
    ports = {6970: "Pegasus DL", 22: "ssh", 8000: "stager", 9021: "elfldr", 80: "http", 5353: "mdns"}
    found = {}
    lock = threading.Lock()

    def probe(ip):
        for p in ports:
            try:
                s = socket.create_connection((ip, p), timeout=0.4)
                s.close()
                with lock:
                    found.setdefault(ip, []).append((p, ports[p]))
                break  # one open port is enough to flag the host; second pass for detail
            except OSError:
                pass

    ths = [threading.Thread(target=probe, args=(f"{prefix}{i}",), daemon=True) for i in range(1, 255)]
    [t.start() for t in ths]
    for t in ths:
        t.join(timeout=6)
    if not found:
        log("no PS5 services found on %s.0/24 — is the console on and jailbroken?" % prefix)
        return 1
    for ip, ps in sorted(found.items()):
        log(f"{ip}: " + ", ".join(f"{p} ({n})" for p, n in ps))
    return 0


# ---------------------------------------------------------------- native
def classify(mounts):
    kind = {}
    for m in mounts:
        p, src = m.get("path", ""), str(m.get("source", ""))
        tot = m.get("totalBytes", 0) or 0
        if p == "/data" or "ssd0.user" in src:
            kind[p] = "internal-ssd"
        elif re.match(r"^/mnt/(ext|usb)", p) or "es0" in src:
            kind[p] = "usb"
        elif m.get("writable") and tot > 100e9:
            kind[p] = "M.2-candidate"
        else:
            kind[p] = "other"
    return kind


def cmd_native(a):
    base = f"http://{a.host}:6970"
    out = {"ts": TS, "mode": "native", "host": a.host}
    try:
        out["health"] = jget(base + "/api/health")
    except Exception as e:
        log(f"FAIL: Pegasus not reachable on {a.host}:6970 ({e}). Jailbreak + Pegasus must be running.")
        return 1
    log("Pegasus version: %s" % out["health"].get("version"))
    for key, path in [("settings", "/api/settings"), ("mounts", "/api/mounts"), ("locations", "/api/files/locations")]:
        try:
            out[key] = jget(base + path)
        except Exception as e:
            out[key] = {"error": str(e)}
    out["mount_kinds"] = classify(out.get("mounts", {}).get("mounts", []))
    log("mounts:")
    for m in out.get("mounts", {}).get("mounts", []):
        log(f"  {m['path']:12s} {out['mount_kinds'].get(m['path']):15s} {m.get('totalBytes', 0)/1e9:8.0f} GB  free {m.get('freeBytes', 0)/1e9:7.0f} GB  src={m.get('source')}")

    # HTTP pipe capacity (upper bound is the network, not the SSD — labelled as such)
    dir_ = "/data/ps5-probe"
    pre = max_op_id(base)  # W9-09: awaited, freshness-checked mkdir — an un-awaited
    call("POST", base, "/api/files/mkdir", {"parent": "/data", "name": "ps5-probe"})
    wait_op(base, "mkdir", dir_, 60, since=pre)  # mkdir can execute concurrent with later ops
    blob = os.urandom(32 << 20)
    t0 = time.monotonic()
    s, _ = call("PUT", base, f"/api/files/upload?directory={urllib.parse.quote(dir_)}&name=pipe.bin", raw=blob, timeout=300)
    up = len(blob) / 1e6 / (time.monotonic() - t0)
    t0 = time.monotonic()
    s, body = call("GET", base, f"/api/files/download?path={urllib.parse.quote(dir_ + '/pipe.bin')}", timeout=300)
    down = len(body) / 1e6 / (time.monotonic() - t0)
    out["pipe_mb_s"] = {"upload": round(up, 1), "download": round(down, 1),
                        "meaning": "network/HTTP ceiling, NOT nvme speed"}
    log(f"http pipe (network ceiling): upload {up:,.0f} MB/s, download {down:,.0f} MB/s")
    # awaited, freshness-checked delete (W2 #9: an un-awaited delete op can execute
    # CONCURRENTLY with a later mkdir/copy on /data — Pegasus quirk #3)
    pre = max_op_id(base)
    call("POST", base, "/api/files/delete", {"paths": [dir_ + "/pipe.bin"]})  # only OUR file:
    # deleting /data/ps5-probe whole would wipe a persisted N7 store (W9 #4)
    wait_op(base, "delete", dir_, 300, since=pre)

    # optional: on-disk copy throughput (real storage number, no network in the path)
    if a.copytest:
        out["copy_mb_s"] = {}
        big = None
        for m in out.get("mounts", {}).get("mounts", []):
            # only real storage (SSD / USB / M.2) — skips ShadowMount game-image mounts ("other"),
            # which would measure image-over-image and can fill a fixed-size image
            if out["mount_kinds"].get(m["path"], "other") == "other":
                continue
            room = (m.get("freeBytes") or 0) - (5 << 30)   # never let a copy take a mount past free-5GiB
            if m.get("writable") and room >= (1 << 30):
                cand = find_big_file(base, m["path"], 1 << 30, max_bytes=min(64 << 30, room))
                if cand:
                    big = (cand[0], cand[1], m["path"])
                    break
        if not big:
            log("copytest: no eligible file (>= 1 GiB, fits with 5 GiB headroom) on writable mounts; skipped")
        else:
            path, size, mnt = big
            sandbox = mnt + "/ps5-probe"   # empty dir on the SAME mount: keep-both can never rename onto
            pre = max_op_id(base)
            call("POST", base, "/api/files/mkdir", {"parent": mnt, "name": "ps5-probe"})  # the original, and
            wait_op(base, "mkdir", sandbox, 60, since=pre)                                  # cleanup deletes only the sandbox
            log(f"copytest source: {path} ({size/1e9:.1f} GB) -> {sandbox}/")
            pre = max_op_id(base)
            s, js = call("POST", base, "/api/files/copy", {"paths": [path], "destination": sandbox, "collision": "keep-both"})
            op = None
            if s == 200:
                el, op = wait_op(base, "copy", path, 1800, since=pre)
                if op and op.get("status") == "completed":
                    rate = size / 1e6 / max(0.01, el)
                    out["copy_mb_s"] = {"path": path, "mb_s": round(rate, 1), "note": "read+write of one file; read component >= half"}
                    log(f"on-disk copy: {rate:,.0f} MB/s effective (read+write) on {mnt}")
                else:
                    st = op.get("status") if op else "never-registered"
                    out["copy_mb_s"] = {"path": path, "error": f"copy op {st}: {(op or {}).get('error', '')}"}
                    log(f"copytest: copy op {st} ({(op or {}).get('error', 'no msg')}) — no rate measured")
            else:
                out["copy_mb_s"] = {"error": f"{s} {js}"}
            # cleanup LAST, after the copy op is terminal (concurrent-ops hazard), and verify it took
            for _ in range(2):
                pre = max_op_id(base)
                s2, _b = call("POST", base, "/api/files/delete", {"paths": [sandbox]})
                if s2 == 200:
                    wait_op(base, "delete", sandbox, 600, since=pre)
                try:
                    left = jget(base + "/api/files/list?path=" + urllib.parse.quote(sandbox), timeout=15).get("entries")
                except Exception:
                    left = []
                if not left:
                    break
                log("copytest: sandbox still present after delete; retrying")
    # optional: internal-SSD floor on /data, self-created (see run_bwtest)
    if getattr(a, "bwtest", False):
        out["bwtest"] = run_bwtest(base, out, a.bwtest_size)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    p = PROBE_DIR / f"native-{TS}.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"saved {p}")
    return 0


def find_big_file(base, path, min_bytes, max_bytes=1 << 40, depth=0, target=8 << 30):
    """Closest-to-8-GiB file in [min_bytes, max_bytes]: stable timing, bounded wear/footprint."""
    try:
        js = jget(base + f"/api/browse?path={urllib.parse.quote(path)}", timeout=20)
    except Exception:
        return None
    best, best_d = None, None
    def better(c):
        nonlocal best, best_d
        d = abs(c[1] - target)
        if best is None or d < best_d:
            best, best_d = c, d
    for e in js.get("entries", []):
        sz = e.get("sizeBytes", 0)
        if e.get("type") == "file" and min_bytes <= sz <= max_bytes:
            better((f"{path}/{e['name']}", sz))
        elif e.get("type") == "dir" and depth < 1 and e["name"].lower() not in ("ps5-probe", "system_tmp", "dev"):
            r = find_big_file(base, f"{path}/{e['name']}", min_bytes, max_bytes, depth + 1, target)
            if r:
                better(r)
    return best


def max_op_id(base):
    """Highest operationId currently registered. Snapshot BEFORE a POST, pass as
    wait_op(since=...) so a stale op from a previous run can never be matched
    during the 1-2 s registration lag (W2 finding #3)."""
    try:
        ops = jget(base + "/api/files/operations", timeout=15).get("operations", [])
    except Exception:
        return 0
    return max((o.get("operationId", 0) for o in ops), default=0)


def find_op(base, type_, src, since=0):
    """The newest op of `type_` whose `source` is exactly `src` AND whose operationId is
    > since (registration can lag the POST by ~2s; `since` kills stale-match races)."""
    try:
        ops = jget(base + "/api/files/operations", timeout=15).get("operations", [])
    except Exception:
        return None
    m = [o for o in ops if o.get("type") == type_ and o.get("source") == src
         and (not since or o.get("operationId", 0) > since)]  # since=0 = fresh ops table / server w/o ids
    return max(m, key=lambda o: o.get("operationId", 0)) if m else None


def wait_op(base, type_, src, timeout, since=0):
    """Wait for a FRESH op(type_, src) (operationId > since) to register, then for its
    terminal state. Returns (elapsed_s, op). Always snapshot max_op_id() BEFORE the POST.
    IMPORTANT: Pegasus runs queued ops CONCURRENTLY with running ones — never queue the
    sandbox delete until this returns; otherwise it deletes the destination out from
    under the copy."""
    t0 = time.monotonic()
    op = None
    while time.monotonic() - t0 < min(30, timeout):
        op = find_op(base, type_, src, since)
        if op:
            break
        time.sleep(1.0)
    if op is None:
        return time.monotonic() - t0, None
    while time.monotonic() - t0 < timeout:
        if op.get("status") not in ("queued", "running", "starting"):
            break
        time.sleep(1.0)
        op = find_op(base, type_, src, since) or op
    el = time.monotonic() - t0
    if op.get("status") == "completed" and op.get("updatedAt") and op.get("startedAt"):
        el = max(0.5, op["updatedAt"] - op["startedAt"])   # server-side transfer time, poll-lag free
    return el, op


def run_bwtest(base, out, size_gib):
    """Self-create a large file on /data, then same-mount copy it: the internal-SSD
    read+write floor at 13.00 native, no Linux needed (W2 finding #1: `copytest` can
    never satisfy P0 because it only copies EXISTING files, and no >=1 GiB file exists
    under /data). Upload once via the proven HTTP pipe (~6 min at 21 MB/s), reuse after.
    Every POST is followed by wait_op(since=<pre-POST max id>) -- single-op discipline."""
    res = {"size_gib": size_gib}
    if size_gib < 4:
        size_gib = 4   # below this the SLC/DRAM cache makes the number meaningless (W2 #14)
    room = 0
    for m in out.get("mounts", {}).get("mounts", []):
        if m.get("path") == "/data":
            room = m.get("freeBytes") or 0
    if room < (size_gib + 5) * (1 << 30):
        res["error"] = f"/data has {room/1e9:.0f} GB free; need {size_gib + 5} GiB headroom"
        log("bwtest: " + res["error"] + " -- skipped")
        return res
    store = "/data/ps5-probe"
    f = store + "/bwfill.bin"
    pre = max_op_id(base)
    call("POST", base, "/api/files/mkdir", {"parent": "/data", "name": "ps5-probe"})
    wait_op(base, "mkdir", store, 60, since=pre)

    have = 0
    try:
        for e in jget(base + f"/api/browse?path={urllib.parse.quote(store)}", timeout=20).get("entries", []):
            if e.get("type") == "file" and e.get("name") == "bwfill.bin":
                have = e.get("sizeBytes", 0)
    except Exception:
        pass
    want = size_gib * (1 << 30)
    size = want
    if have >= want:
        size = have
        res["store"] = "reused (%.1f GiB on /data)" % (have / 2**30)
        log(f"bwtest: reusing existing {have/2**30:.1f} GiB store")
    elif have > 0:
        # W9 #4b: a stale store of the wrong size must be PURGED, not overwritten in
        # place (upload collision behaviour onto an existing file is untested)
        log(f"bwtest: purging stale {have/2**30:.1f} GiB store (asked {size_gib} GiB) ...")
        pre = max_op_id(base)
        call("POST", base, "/api/files/delete", {"paths": [f]})
        wait_op(base, "delete", f, 300, since=pre)
        have = 0
    if 0 < have < want:  # unreachable after purge; defensive
        have = 0
    if have == 0:
        log(f"bwtest: streaming {size_gib} GiB of urandom to /data (~{size_gib*512/21/60:.0f} min at the measured pipe rate) ...")
        t0 = time.monotonic()
        s, _b = call("PUT", base, f"/api/files/upload?directory={urllib.parse.quote(store)}&name=bwfill.bin",
                     raw=_RandFile(want), timeout=3600, headers={"Content-Length": str(want)})
        el = time.monotonic() - t0
        res["upload_mb_s"] = round(want / 1e6 / max(0.01, el), 1)
        if s != 200:
            res["error"] = f"upload HTTP {s}: {_b[:200]!r}"
            log("bwtest: " + res["error"])
            _bw_cleanup(base, purge=True)
            return res
        got = 0
        try:  # verify the upload landed whole (a truncated file would silently skew the copy rate)
            for e in jget(base + f"/api/browse?path={urllib.parse.quote(store)}", timeout=20).get("entries", []):
                if e.get("type") == "file" and e.get("name") == "bwfill.bin":
                    got = e.get("sizeBytes", 0)
        except Exception:
            pass
        if got != want:
            res["error"] = f"upload verify: {got} bytes != {want} (re-run bwtest)"
            log("bwtest: " + res["error"])
            _bw_cleanup(base, purge=True)  # a truncated store must NOT persist
            return res
        log(f"bwtest: uploaded in {el/60:.1f} min ({res['upload_mb_s']} MB/s)")

    sandbox = store + "/dst"
    pre = max_op_id(base)
    call("POST", base, "/api/files/mkdir", {"parent": store, "name": "dst"})
    wait_op(base, "mkdir", sandbox, 60, since=pre)
    pre = max_op_id(base)
    s, js = call("POST", base, "/api/files/copy", {"paths": [f], "destination": sandbox, "collision": "keep-both"})
    if s != 200:
        res["error"] = f"copy POST {s}: {js[:200]!r}"
        log("bwtest: " + res["error"])
    else:
        el, op = wait_op(base, "copy", f, 1800, since=pre)
        if op and op.get("status") == "completed":
            rate = size / 1e6 / max(0.5, el)  # bytes ACTUALLY copied (reused store may be larger than want)
            res["bwtest_mb_s"] = round(rate, 1)
            res["meaning"] = ("same-device read+write floor on /data; true seq-read >= rate. "
                              "Supersedes the 82.5 MB/s USB number. Maps to P0: copy >= 150 MB/s "
                              "=> read floor >= 150 MB/s.")
            log(f"bwtest: internal-SSD r+w floor {rate:,.0f} MB/s on /data")
        else:
            res["error"] = f"copy op {op.get('status') if op else 'never-registered'}"
            log("bwtest: " + res["error"])
    # cleanup, single-op, verify it took
    _bw_cleanup(base)
    try:
        free_after = next((m.get("freeBytes", 0) for m in out.get("mounts", {}).get("mounts", [])
                           if m.get("path") == "/data"), 0)
        res["free_after_bytes"] = free_after
    except Exception:
        pass
    return res


def _bw_cleanup(base, purge=False):
    """Delete the bwtest SANDBOX (dst) with freshness-checked awaited deletes + list
    verification, retrying once each (W2 design guard rails). The bwfill.bin store —
    and any N7 store sharing /data/ps5-probe — PERSISTS across runs unless purge=True
    (W9 #4: wiping the parent dir silently destroyed the persisted stores, and an
    80 GiB rebuild at 21 MB/s is 64 minutes of your life back)."""
    targets = ("/data/ps5-probe/bwfill.bin", "/data/ps5-probe/dst") if purge else ("/data/ps5-probe/dst",)
    for tgt in targets:
        for _ in range(2):
            pre = max_op_id(base)
            s2, _b = call("POST", base, "/api/files/delete", {"paths": [tgt]})
            if s2 == 200:
                wait_op(base, "delete", tgt, 600, since=pre)
            try:
                left = jget(base + "/api/files/list?path=" + urllib.parse.quote(tgt), timeout=15).get("entries")
            except Exception:
                left = []
            if not left:
                break
            log(f"bwtest cleanup: {tgt} still present after delete; retrying")


class _RandFile:
    """File-like source of `total` bytes of urandom, generated in 4 MiB chunks so an
    8 GiB upload never needs 8 GiB of RAM. urllib streams objects with .read()."""
    def __init__(self, total):
        self.left = total

    def read(self, n=1 << 20):
        n = min(n, self.left)
        out = bytearray()
        while len(out) < n:
            out += os.urandom(min(1 << 22, n - len(out)))
        self.left -= len(out)
        return bytes(out)


# ---------------------------------------------------------------- linux
def cmd_linux(a):
    script = GATE_SH.read_text().replace("\r\n", "\n")
    target = f"{a.user}@{a.host}" if a.user else a.host
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
           "-p", str(a.port), target, "bash -s"]
    import shlex
    env_pfx = f"FIO_SECONDS={a.fio_seconds} FIO_SIZE_MB={a.fio_size_mb}"
    # W9-05: the script reads these as env vars — every flag must reach it over ssh,
    # or it silently runs defaults after promising the user's mount / cap sweep
    if getattr(a, "g1_target", ""):
        env_pfx += " G1_TARGET=" + shlex.quote(a.g1_target)
    if getattr(a, "settle_s", 0):
        env_pfx += f" FIO_SETTLE_S={a.settle_s}"
    timeout = 6 * a.fio_seconds + 400 + max(0, a.fio_size_mb // 512) + getattr(a, "settle_s", 0)
    # S11: the full battery may create the 100G rand scratch (+ its settle) on first run
    if not getattr(a, "n7_only", False):
        timeout += 400
    if getattr(a, "n7", False):
        env_pfx += f" N7=1 N7_SIZE_GB={a.n7_size} N7_KEEP={getattr(a, 'n7_keep', 1)}"
        if getattr(a, "n7_cap_mb", 0):
            env_pfx += f" N7_CAP_MB={a.n7_cap_mb}"
        # S11: fixed-token mode + real-pack-geometry store + N7-only fast sweeps
        if getattr(a, "n7_tokens", 0):
            env_pfx += f" N7_TOKENS={a.n7_tokens}"
        if getattr(a, "n7_slot_bytes", 0):
            env_pfx += f" N7_SLOT_BYTES={a.n7_slot_bytes} N7_SPL={getattr(a, 'n7_spl', 512)}"
        if getattr(a, "n7_only", False):
            env_pfx += " N7_ONLY=1"
        timeout += 500
    log(f"running gate battery on {target} (fio ~{4*a.fio_seconds+60}s + ~{a.fio_size_mb//3072}s scratch create"
        f"{', n7 store+replay +~5min first run' if getattr(a, 'n7', False) else ''}) ...")
    try:
        # W10 fix: send BYTES. text=True on Windows translates the piped \n to
        # \r\n (os.linesep) on the stdin pipe and remote bash chokes on CRLF.
        r = subprocess.run(cmd, input=(env_pfx + "\n" + script).encode("utf-8"),
                           capture_output=True, timeout=timeout)
        r.stdout = r.stdout.decode("utf-8", "replace")
        r.stderr = r.stderr.decode("utf-8", "replace")
    except FileNotFoundError:
        log("ssh client not found on PATH")
        return 1
    except subprocess.TimeoutExpired:
        log("ssh timed out — is pLinux booted and the IP right? (reboot of the loader is often the fix)")
        return 1
    if r.returncode != 0:
        log(f"ssh exit {r.returncode}: {r.stderr.strip()[:400]}")
        log("if this was a password prompt, set up a key first (see research/design-history/PS5-PROBE-RUNBOOK.md step 4)")
        return 1
    js = None
    m = re.search(r"JSON_START\n(.*?)\nJSON_END", r.stdout, re.S)
    if m:
        try:
            js = json.loads(m.group(1))
        except Exception:
            pass
    if js is None:
        log("no JSON in remote output (no python3 on target?); saving raw")
        js = {"raw": r.stdout[-4000:]}
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_DIR / f"linux-{TS}.json").write_text(json.dumps(js, indent=1))
    evaluate(js, a)
    return 0


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_dd(s):
    if not s:
        return None
    m = re.search(r"([\d.]+)\s?(kB|MB|GB)/s", s, re.I)
    if not m:
        return None
    v = float(m.group(1))
    return v * {"kb": 1e-3, "mb": 1, "gb": 1e3}[m.group(2).lower()]


def seq_bw(js):
    v = num(js.get("fio_seq1m_mb_s"))
    return v if v else parse_dd(js.get("dd_seq_direct"))


def evaluate(js, a):
    rows = []

    def row(gate, name, ok, detail):
        rows.append((gate, name, ok, detail))

    did = str(js.get("device_id", ""))
    row("G0", "booted / reachable", True, f"{js.get('os', '?')} | cmdline: {str(js.get('kernel_cmdline', ''))[:90]}")
    row("G0", "amdgpu bound to GPU", bool(did), f"pci: {str(js.get('pci_vga', ''))[:120]}")
    vk = str(js.get("vk_summary", ""))
    ident = "GFX1013" if "1013" in vk else ("GFX1030/GFX103" if "GFX103" in vk else "?")
    row("G0", "GPU identity (A4: gfx1013 vs Oberon)", ident != "?", f"vulkaninfo says {ident}: {vk[:140]}")
    ext = js.get("vk_ext_host")
    row("G0", "VK_EXT_external_memory_host", ext == 1 if isinstance(ext, int) else False, f"matches={ext}")
    pl, pp = num(js.get("ttm_pages_limit")), num(js.get("ttm_page_pool_size"))
    ttm_ok = pl == 4194304 and pp == 4194304
    ttm_state = "OK" if ttm_ok else ("SILENT-CLOBBERED to 12GiB (tmpfiles; see runbook)" if pl == 3145728 or pp == 3145728 else f"too low/unknown ({pl}/{pp})")
    # gttsize param is root-only (0600) on the 7.x kernel -> blank for steam; GTT total from sysfs is world-readable
    gtt = num(js.get("gtt_total"))
    gtt_note = f"gtt_total={gtt / 2**30:.2f} GiB" if gtt else f"gttsize={js.get('amdgpu_gttsize') or 'unreadable (root-only)'}"
    vram = num(js.get("vram_total"))
    vram_note = f"vram carve-out={vram / 2**20:.0f} MiB" if vram else f"vram_total={js.get('vram_total')}"
    row("G0", "ttm pages_limit+page_pool_size = 4194304", ttm_ok, ttm_state + f" | {gtt_note} | {vram_note}")
    m2 = js.get("block_devs", "")
    devs = [d for d in str(m2).split("|") if d]
    # S5 fix: count NVMe ONLY — a >=100 GB USB boot SSD enumerates as sd? and must never
    # fake a second NVMe (the old check false-PASSed "M.2 present" on big boot drives).
    big = [d for d in devs if ":" in d and d.split(":", 1)[0].startswith("nvme")
           and (num(d.split(":")[1]) or 0) * 512 > 100e9]  # "name:sectors:model|", W9-08: guard colon-less entry
    # Linux track: the internal Gen4 M.2 is the store and the only NVMe (the Sony SSD never
    # enumerates) -> one device is the PASS condition; a second one is what N6 striping needs.
    row("G0", "NVMe store device present", len(big) >= 1,
        f"{len(big)} NVMe >100 GB ({'N6 stripe possible' if len(big) >= 2 else 'single device: N6 stripe n/a'}) | block devs: {m2[:200]}")

    seq, rand = seq_bw(js), num(js.get("fio_rand3m_mb_s"))
    dev = js.get("g1_device") or js.get("scratch_mount") or "?"
    dev_note = f" [device {js.get('g1_device')} {js.get('g1_device_model')} via {js.get('g1_target', 'auto')} mount {js.get('scratch_mount')}]"
    if seq:
        verdict = (f"= {seq / MB_PER_GIB_S:.3f} GiB/s; 40 tok/s needs h>={need_hit(seq, G_EXL3_1_66):.1%} @G={G_EXL3_1_66} (EXL3 1.66), "
                   f"h>={need_hit(seq, a.gib_per_token):.1%} @G={a.gib_per_token} (Q4) — seq is the ceiling, only a sequentialized miss stream gets it")
        row("G1", "sequential read 1M qd32 O_DIRECT", seq >= 3000, f"{seq:.0f} MB/s {verdict}{dev_note}")
    else:
        row("G1", "sequential read measurement", False, "no fio and dd fallback empty (install fio: apt install fio)")
    span = num(js.get("fio_size_mb"))
    span_ok = bool(span) and span >= STORE_SPAN_MB
    # S11: the rand row runs on its OWN 100G SLC-honest scratch — its span is that
    # scratch's size, not the seq/4k scratch (fio_size_mb). Missing kv = pre-S11 run.
    rspan = num(js.get("g1_rand_scratch_mb")) or span
    rspan_ok = bool(rspan) and rspan >= STORE_SPAN_MB
    rspan_note = f" [rand scratch {rspan:.0f} MiB, SLC-honest]" if num(js.get("g1_rand_scratch_mb")) else ""
    if rand and seq:
        ratio = rand / seq
        fix = "suspect LBA-alignment: re-run m2_init/m2_install from updated ps5-linux-tools" if ratio < 0.6 else "healthy"
        if rspan_ok:
            row("G1", f"engine-shaped randread 3MiB qd8 over {rspan:.0f} MiB span", rand >= 3500,
                f"{rand:.0f} MB/s ({ratio:.0%} of seq) — {fix}; 40 tok/s needs h>={need_hit(rand, G_EXL3_1_66):.1%} @G={G_EXL3_1_66}{rspan_note}{dev_note}")
        else:
            # not a pass/fail: a small span reads like seq on this drive and must never be quoted as engine-shaped
            row("G1", f"randread 3MiB qd8 over {rspan or '?'} MiB span (SPAN-LIMITED)", None,
                f"{rand:.0f} MB/s ({ratio:.0%} of seq) — NOT engine-shaped: span < {STORE_SPAN_MB} MiB; "
                f"quote the NAND-true 100G number (S11 old-data: 2290-2450 MB/s 2.6M, 1590-1890 MB/s 1020k; "
                f"S10's 3299/2795 were SLC-assisted){rspan_note}{dev_note}")
    if num(js.get("fio_rand4k_mb_s")):
        row("G1", "randread 4k qd64 (metadata reality)", num(js["fio_rand4k_mb_s"]) >= 1500, f"{num(js['fio_rand4k_mb_s']):.0f} MB/s")
    buf = num(js.get("buffered_reread_mb_s"))
    if buf:
        row("G1", "buffered page-cache re-read (tail window)", buf >= 10000, f"{buf:.0f} MB/s — flash-moe 'trust the OS' regime")
    if js.get("scaling_governor") or js.get("ps5_control"):
        row("G0", "clock/boost state recorded (W2 #15)", True,
            f"governor={js.get('scaling_governor')} boost={js.get('cpu_boost')} ps5_control={str(js.get('ps5_control'))[:60]} hwmon sampled={'yes' if js.get('hwmon') else 'no'}")

    # N6: concurrent dual-device stripe ceiling (frontier report gate)
    agg = num(js.get("n6_aggregate_mb_s"))
    if agg:
        ok = agg >= 9000
        row("N6", "concurrent dual-NVMe reads (stripe ceiling)", ok,
            f"{agg:.0f} MB/s aggregate = {js.get('n6_a_mb_s')} + {js.get('n6_b_mb_s')} on {js.get('n6_devices')} — "
            + ("STRIPE GO: 11-13 GB/s band alive, 40-60 reachable if N5 hit holds" if ok else
               "stripe underdelivers (<9 GB/s); 40-60 depends on hit-rate + MTP alone"))
    else:
        row("N6", "concurrent dual-NVMe reads", False,
            f"{js.get('n6_devices', 'not measured')} — single device: no stripe, the store ceiling is the G1 seq number")

    # N7: page-cache replay (hit-rate-vs-capacity curve point). Per the S2 audit this is
    # a MEASUREMENT, not a pass/fail gate: >=80% sits at/above the published static
    # ceiling (0.79, crow-nest) so PASS/FAIL framing was misleading (FLOW kill list).
    h = num(js.get("n7_hit_pct"))
    if h is not None:
        cap_pct = None
        store_mb = (num(js.get("n7_store_gb")) or 0) * 1024
        cap_mb = num(js.get("n7_cache_mb"))
        reached = num(js.get("n7_cgroup_current"))
        if store_mb and cap_mb:
            cap_pct = 100 * cap_mb / store_mb
        hf = h / 100
        bands = []
        for bw_name, bw_mb in (("seq", seq), ("rand", rand if rspan_ok else None)):
            if bw_mb and hf < 1:
                t = {g: bw_mb / MB_PER_GIB_S / (g * (1 - hf)) for g in (a.gib_per_token, G_EXL3_1_66)}
                bands.append(f"{bw_name} {bw_mb:.0f} MB/s -> Q4 {t[a.gib_per_token]:.1f} / EXL3-1.66 {t[G_EXL3_1_66]:.1f} tok/s "
                             f"(40 needs h>={need_hit(bw_mb, G_EXL3_1_66):.1%})")
        band = "; ".join(bands) or "no bandwidth measured"
        chunk = num(js.get("n7_chunk_bytes"))
        if chunk:
            band += f" [store slots {chunk / 2**20:.2f} MiB = Q4-size; smaller-slot packs hold more experts per GiB -> this hit under-credits them]"
        row("N7", f"expert-cache replay: hit {h:.1f}% (cap {cap_pct:.0f}% of store, {js.get('n7_slots_per_layer')} slots/layer)" if cap_pct else f"expert-cache replay: hit {h:.1f}%",
            None,
            (f"pool reached {reached / 2**30:.2f} GiB of cap {cap_mb / 1024:.2f} GiB (hit tracks reached, not cap), " if reached and cap_mb else "")
            + f"{js.get('n7_tokens')} tokens, SSD {js.get('n7_ssd_mb_s')} MB/s, "
            f"MemAvailable {js.get('n7_mem_available_mb', 'n/a')} MiB, "
            f"zipf s={js.get('n7_zipf_s')} [{js.get('n7_mode')}] — {band}"
            " [synthetic Zipf stand-in, no router; replace with engine traces when the engine exists]")

    # throughput projection — from THIS console's numbers (frontier physics)
    bpt = a.gib_per_token  # GiB streamed per token (1.24 = W4 recompute from config.json)
    proj = {}
    # S10 unit fix: fio MB/s are decimal; ÷1073.741824 -> GiB/s (the old ÷1000
    # labeled it GiB/s and inflated every projected tok/s ~7.4% — user-caught
    # 2026-10-27; mirror of ledger.MB_PER_GIB_S, kept inline: this tool is standalone).
    bw_hi = max((agg or 0), (seq or 0)) / 1073.741824
    bw_lo = (seq or 0) / 1073.741824   # GiB/s available (stripe if N6 passed, else single)
    hs = (h / 100) if h is not None else None
    if bw_hi:
        proj["cold(h=0)"] = round(bw_lo / bpt, 2)
        if hs is not None:
            proj["streamed(h=%.2f measured)" % hs] = round(bw_hi / (bpt * max(1 - hs, 0.02)), 1)
        else:
            proj["streamed(h=0.77 expected)"] = round(bw_hi / (bpt * 0.23), 1)
            proj["streamed(h=0.88 expected)"] = round(bw_hi / (bpt * 0.12), 1)
        # N4 has ZERO instrumentation (W2 #25): the multiplier is an UNMEASURED assumption
        stream_best = max((v for k, v in proj.items() if k.startswith("streamed")), default=0)
        proj["with MTP x1.44 (N4 UNMEASURED)"] = round(stream_best * 1.44, 1)
        if hs is not None:
            proj["EXL3-1.66 streamed(h=%.2f, G=%.3f)" % (hs, G_EXL3_1_66)] = round(bw_hi / (G_EXL3_1_66 * max(1 - hs, 0.02)), 1)
        proj["TTFT_once_per_request_s"] = round(a.expert_gb / max(bw_lo, 0.1), 1)
    row("G1", f"projected tok/s @{bpt}GiB-per-token/{a.k}of{a.experts}", None, json.dumps(proj))

    w = {"ts": TS, "host": getattr(a, "host", None), "js": js, "rows": rows, "projection": proj}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"ps5-gate-{TS}.json").write_text(json.dumps(w, indent=1))
    md = [f"# PS5 gate report — {TS}\n\nHost: {getattr(a, 'host', None)} (physics: G={getattr(a, 'gib_per_token', 1.24)} GiB/token Q4, {G_EXL3_1_66} EXL3 1.66; bandwidth MB/s decimal)\n"]
    md.append("| gate | check | result | detail |\n|---|---|---|---|")
    for g, n, ok, d in rows:
        md.append(f"| {g} | {n} | {'PASS' if ok is True else 'MEASURED' if ok is None else 'CHECK'} | {d} |")
    md.append(f"\nRaw: `recon/ps5probe/linux-{TS}.json`  ·  thresholds from research/reports/'PS5 MoE streaming engine plan - adversarial review.md'")
    (REPORT_DIR / f"ps5-gate-{TS}.md").write_text("\n".join(md))
    log("\n".join(md))
    log(f"\nwrote recon/gates/ps5-gate-{TS}.md (+ .json)")


# ---------------------------------------------------------------- report
def cmd_report(a):
    p = pathlib.Path(a.json)
    if not p.exists():
        log(f"no such file: {p}")
        return 1
    js = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    js = js.get("js", js) if isinstance(js, dict) else js   # accept the reports/ wrapper too
    if "rows" in js and "js" in js:
        js = js["js"]
    log(f"re-evaluating {p} ({len(js)} keys)")
    evaluate(js, a)
    return 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="probe.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover"); d.add_argument("--subnet", default="10.0.0."); d.set_defaults(fn=cmd_discover)
    n = sub.add_parser("native"); n.add_argument("--host", default="10.0.0.113"); n.add_argument("--copytest", action="store_true")
    n.add_argument("--bwtest", action="store_true", help="self-create an 8 GiB file on /data and same-mount-copy it: internal-SSD r+w floor, no Linux needed (upload ~6 min first run)")
    n.add_argument("--bwtest-size", type=int, default=8, help="bwtest scratch size in GiB (floor 4)")
    n.set_defaults(fn=cmd_native)
    l = sub.add_parser("linux"); l.add_argument("--host", required=True); l.add_argument("--user", default="ps5")
    l.add_argument("--port", type=int, default=22); l.add_argument("--fio-seconds", type=int, default=15)
    l.add_argument("--fio-size-mb", type=int, default=102400, help="G1 scratch file, GiB-honest by default (>=3x any SLC cache, W2 #14); shrink only for tiny mounts")
    l.add_argument("--n7", action="store_true", help="run the expert-cache replay (writes a ~40G store first run, +~2 min; store PERSISTS for reuse, N7_KEEP=0 to delete)")
    l.add_argument("--n7-size", type=int, default=40, help="N7 synthetic expert store size in GiB (72 = 512 slots/layer, engine-accurate, needs ~87G free — W9-13)")
    l.add_argument("--n7-cap-mb", type=int, default=0, help="N7 hot-cache cap in MB (0 = min(35%% RAM, 3072)); sweep 1024/2048/3072/6144 for the hit-vs-capacity curve")
    l.add_argument("--n7-keep", type=int, default=1, help="1 = persist the N7 store for reuse; 0 = delete after run")
    l.add_argument("--n7-tokens", type=int, default=0, help="S11: fixed MEASURED-token count per replay (0 = legacy time-boxed); identical draw counts make hit%% cap-comparable")
    l.add_argument("--n7-slot-bytes", type=int, default=0, help="S11: real pack geometry — exact slot bytes, e.g. 1019904 = 1.66 bpw (store is 48 x spl x slot, geometry-named)")
    l.add_argument("--n7-spl", type=int, default=512, help="S11: slots per layer for the geometry store (512 = the real model)")
    l.add_argument("--n7-only", action="store_true", help="S11: skip the G1/N6 fio battery — N7 replay only (fast x3 sweeps); G0 kvs still emit")
    l.add_argument("--g1-target", default="", help="mount/path for the G1 scratch (e.g. the internal or M.2 ext4 mount); default: auto = largest-free local mount. findmnt -t ext4 on the console to name it")
    l.add_argument("--settle-s", type=int, default=0, help="cache-drain between write and read phases (script default 45)")
    for f in ("expert_gb", "gib_per_token"):
        l.add_argument("--" + f, type=float, default=DEF[f])
    l.add_argument("--k", type=int, default=DEF["k"]); l.add_argument("--experts", type=int, default=DEF["experts"])
    l.set_defaults(fn=cmd_linux)
    r = sub.add_parser("report"); r.add_argument("json", help="recon/ps5probe/linux-*.json or recon/gates/ps5-gate-*.json")
    for f in ("expert_gb", "gib_per_token"):
        r.add_argument("--" + f, type=float, default=DEF[f])
    r.add_argument("--k", type=int, default=DEF["k"]); r.add_argument("--experts", type=int, default=DEF["experts"])
    r.add_argument("--host", default=None)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
