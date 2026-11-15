"""S17: build an EXL3 expert pack for ANY turboderp/Qwen3.8-Flash-Next-exl3 branch by HTTP range requests, straight
into the 4 KiB-slot pack (no shard staging: disk need = the pack itself).

Same pack format as tools/s15/pack_exl3.py (byte-preserving: each routed expert is ONE contiguous 12-tensor span
  down.{suh,svh,mul1,trellis} gate.{...} up.{...}
copied verbatim into slot base[l] + e * slot[l]). Writes <prefix>.pack, <prefix>.txt ("layer K2 slot base span",
K2 = 2K: 5 = K2.5, 6 = K3, 7 = K3.5) and <prefix>.json. Resumable (<prefix>.done lists finished fetch ranges).

Integrity: TLS in transit + exact Content-Range/length per request + per-expert structural checks (template offsets,
suh/svh finite fp16 in a sane range, mul1 constant) + a re-fetch of a random 1% of the experts compared byte for byte.

  python3 fetch_pack_exl3.py BRANCH OUT_DIR PREFIX [--workers 8] [--chunk-mb 48]
  e.g. python3 fetch_pack_exl3.py 2.68bpw_h6_ng8 ~/ps5lm/model experts_exl3_268
"""
import json, os, random, re, struct, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BRANCH, OUT, PREFIX = sys.argv[1], sys.argv[2], sys.argv[3]
WORKERS = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
CHUNK = (int(sys.argv[sys.argv.index("--chunk-mb") + 1]) if "--chunk-mb" in sys.argv else 48) << 20
REPO = "https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/" + BRANCH + "/"
L, E, PAGE = 48, 512, 4096
RX = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mul1)$")
ORDER = [(p, t) for p in ("down", "gate", "up") for t in ("suh", "svh", "mul1", "trellis")]
UA = {"User-Agent": "ps5lm-research"}


def get(url, a=None, b=None, tries=6):
    for i in range(tries):
        try:
            h = dict(UA)
            if a is not None:
                h["Range"] = f"bytes={a}-{b - 1}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                d = r.read()
            if a is not None:
                cr = r.headers.get("Content-Range", "")
                assert len(d) == b - a and cr.startswith(f"bytes {a}-{b - 1}/"), (len(d), b - a, cr)
            return d
        except Exception as ex:  # noqa: BLE001 - network: retry with backoff
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} {url[-40:]} [{a},{b}): {ex}", flush=True)
            time.sleep(2 + 3 * i)


idx = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
shards = sorted({s for k, s in idx.items() if RX.match(k)})
heads = {}
for s in shards:
    n = struct.unpack("<Q", get(REPO + s, 0, 8))[0]
    heads[s] = (8 + n, json.loads(get(REPO + s, 8, 8 + n)))
print(f"{BRANCH}: {len(shards)} shards hold routed experts", flush=True)

spans, tmpl_of = {}, {}
for s, (base, h) in heads.items():
    per = {}
    for k, v in h.items():
        m = RX.match(k) if k != "__metadata__" else None
        if m:
            per.setdefault((int(m.group(1)), int(m.group(2))), []).append(
                (base + v["data_offsets"][0], base + v["data_offsets"][1], m.group(3), m.group(4), v["shape"], v["dtype"]))
    for key, ts in per.items():
        ts.sort()
        assert len(ts) == 12, key
        st, en = ts[0][0], ts[-1][1]
        assert sum(b - a for a, b, *_ in ts) == en - st, f"{key}: span has foreign bytes"
        assert [(p, t) for _, _, p, t, _, _ in ts] == ORDER, f"{key}: unexpected tensor order"
        tmpl = [(p, t, a - st, b - a, shape, dt) for a, b, p, t, shape, dt in ts]
        assert tmpl_of.setdefault(en - st, tmpl) == tmpl, f"{key}: layout differs"
        spans[key] = (s, st, en - st)
assert len(spans) == L * E, f"found {len(spans)} experts"

span_len = []
for l in range(L):
    lens = {spans[(l, e)][2] for e in range(E)}
    assert len(lens) == 1, f"layer {l}: mixed expert sizes {lens}"
    span_len.append(lens.pop())
slot = [((n + PAGE - 1) // PAGE) * PAGE for n in span_len]
base, acc = [], 0
for l in range(L):
    base.append(acc)
    acc += slot[l] * E
total = acc
K_of = {n: next(x for x in tmpl_of[n] if x[0] == "gate" and x[1] == "trellis")[4][2] / 16 for n in tmpl_of}
K2 = [int(round(2 * K_of[n])) for n in span_len]
print(f"pack {total / 1e9:.2f} GB; K per layer: " + ",".join(f"{k / 2:g}" for k in K2), flush=True)

# fetch plan: per shard, runs of experts that are adjacent in the shard, cut into ~CHUNK ranges
jobs = []
for s in shards:
    ex = sorted((st, n, key) for key, (sh, st, n) in spans.items() if sh == s)
    cur, cur_a, cur_b = [], None, None
    for st, n, key in ex:
        if cur and (st != cur_b or cur_b + n - cur_a > CHUNK):
            jobs.append((s, cur_a, cur_b, cur))
            cur = []
        if not cur:
            cur_a = st
        cur.append((key, st, n))
        cur_b = st + n
    if cur:
        jobs.append((s, cur_a, cur_b, cur))
print(f"{len(jobs)} range requests, {sum(b - a for _, a, b, _ in jobs) / 1e9:.2f} GB", flush=True)

pack_path = os.path.join(OUT, PREFIX + ".pack")
fd = os.open(pack_path, os.O_RDWR | os.O_CREAT, 0o644)
if os.fstat(fd).st_size < total:
    os.ftruncate(fd, total)  # sparse until written
done_path = os.path.join(OUT, PREFIX + ".done")
done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
lock = threading.Lock()
stats = {"bytes": 0, "jobs": 0}
t0 = time.time()


def check_expert(key, buf):
    """structural sanity of one expert span (catches misaligned/zeroed/garbled ranges)"""
    for p, t, off, nb, shape, dt in tmpl_of[len(buf)]:
        if t in ("suh", "svh"):
            v = np.frombuffer(buf, dtype=np.float16, count=nb // 2, offset=off).astype(np.float32)
            assert np.isfinite(v).all() and 0 < np.abs(v).max() < 1e4 and np.abs(v).min() > 0, f"{key} {p}.{t} bad scales"
        elif t == "mul1":
            assert struct.unpack_from("<I", buf, off)[0] == 0x83DCD12D, f"{key} {p}.mul1"
        else:
            assert any(buf[off:off + 4096]), f"{key} {p}.trellis all zero"


def run(job):
    s, a, b, items = job
    tag = f"{s}:{a}"
    if tag in done:
        return
    d = get(REPO + s, a, b)
    for key, st, n in items:
        buf = d[st - a:st - a + n]
        check_expert(key, buf)
        l, e = key
        os.pwrite(fd, buf + b"\0" * (slot[l] - n), base[l] + e * slot[l])
    with lock:
        with open(done_path, "a") as f:
            f.write(tag + "\n")
        stats["bytes"] += b - a
        stats["jobs"] += 1
        if stats["jobs"] % 20 == 0:
            dt = time.time() - t0
            print(f"  {stats['jobs']}/{len(jobs)} jobs, {stats['bytes'] / 1e9:.1f} GB, {stats['bytes'] / dt / 1e6:.0f} MB/s", flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(run, jobs))
os.fsync(fd)

# spot re-fetch: 1% of the experts, byte compare against what the pack holds
rng = random.Random(17)
sample = rng.sample(sorted(spans), max(64, L * E // 100))
for key in sample:
    s, st, n = spans[key]
    l, e = key
    assert get(REPO + s, st, st + n) == os.pread(fd, n, base[l] + e * slot[l]), f"re-fetch mismatch {key}"
print(f"re-fetch check: {len(sample)} experts identical", flush=True)
os.close(fd)

tj = {str(n): [{"proj": p, "kind": t, "off": o, "bytes": nb, "shape": sh, "dtype": dt} for p, t, o, nb, sh, dt in tmpl_of[n]] for n in sorted(tmpl_of)}
meta = {"format": "ps5lm-exl3-pack-v1", "source": REPO, "layers": L, "experts": E, "span_bytes": span_len, "slot_bytes": slot,
        "base": base, "total_bytes": total, "K": [k / 2 for k in K2], "layouts": tj}
json.dump(meta, open(os.path.join(OUT, PREFIX + ".json"), "w"), indent=1)
with open(os.path.join(OUT, PREFIX + ".txt"), "w") as f:
    f.write("# layer K2 slot_bytes base span_bytes  (" + BRANCH + ")\n")
    for l in range(L):
        f.write(f"{l} {K2[l]} {slot[l]} {base[l]} {span_len[l]}\n")
print(f"done in {time.time() - t0:.0f} s", flush=True)
