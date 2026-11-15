"""S15b: byte-preserving repack of turboderp's EXL3 Qwen3.8-Flash-Next shards into the engine's expert pack.

Each routed expert in the shards is ONE contiguous span of 12 tensors (verified for all 24,576 experts of 3.18bpw):
  down.{suh,svh,mul1,trellis} gate.{suh,svh,mul1,trellis} up.{suh,svh,mul1,trellis}
The span is copied verbatim to a 4 KiB-aligned slot (O_DIRECT reads, 4-byte aligned GPU loads). Slot size is per layer
(K=3 -> 455 pages, K=3.5 -> 530 pages). Expert (l, e) lives at base[l] + e * slot[l]; the per-layer table is in
experts_exl3.json next to the pack. Everything that is NOT a routed expert (MTP head, EXL3 dense tensors, norms...)
is first copied to exl3_nonexpert.safetensors (valid safetensors), so the shards can be deleted after the repack.

  python3 pack_exl3.py SRC_DIR OUT_DIR [--delete-shards]
"""
import json, os, re, struct, sys

SRC, OUT = sys.argv[1], sys.argv[2]
DELETE = "--delete-shards" in sys.argv
L, E, PAGE = 48, 512, 4096
RX = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mul1)$")
ORDER = [(p, t) for p in ("down", "gate", "up") for t in ("suh", "svh", "mul1", "trellis")]


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


shards = sorted(f for f in os.listdir(SRC) if re.match(r"model-\d+-of-\d+\.safetensors$", f))
heads = {s: header(os.path.join(SRC, s)) for s in shards}

# ---- 1. per-expert spans + per-layer geometry, with a strict layout check
spans = {}           # (l, e) -> (shard, start, length)
rel_template = {}    # length -> [(proj, kind, rel_off, nbytes, shape)]
for s, (base, h) in heads.items():
    per = {}
    for k, v in h.items():
        m = RX.match(k) if k != "__metadata__" else None
        if m:
            per.setdefault((int(m.group(1)), int(m.group(2))), []).append((base + v["data_offsets"][0], base + v["data_offsets"][1], m.group(3), m.group(4), v["shape"], v["dtype"]))
    for key, ts in per.items():
        ts.sort()
        assert len(ts) == 12, key
        st, en = ts[0][0], ts[-1][1]
        assert sum(b - a for a, b, *_ in ts) == en - st, f"{key}: span has foreign bytes"
        assert [(p, t) for _, _, p, t, _, _ in ts] == ORDER, f"{key}: unexpected tensor order"
        tmpl = [(p, t, a - st, b - a, shape, dt) for a, b, p, t, shape, dt in ts]
        if en - st in rel_template:
            assert rel_template[en - st] == tmpl, f"{key}: layout differs from template"
        else:
            rel_template[en - st] = tmpl
        assert key not in spans, f"{key}: duplicate"
        spans[key] = (s, st, en - st)
assert len(spans) == L * E, f"found {len(spans)} experts"

layer_len = []
for l in range(L):
    lens = {spans[(l, e)][2] for e in range(E)}
    assert len(lens) == 1, f"layer {l}: mixed expert sizes {lens}"
    layer_len.append(lens.pop())
slot = [((n + PAGE - 1) // PAGE) * PAGE for n in layer_len]
base, acc = [], 0
for l in range(L):
    base.append(acc)
    acc += slot[l] * E
total = acc
tmpl_json = {str(n): [{"proj": p, "kind": t, "off": o, "bytes": nb, "shape": sh, "dtype": dt} for p, t, o, nb, sh, dt in rel_template[n]]
             for n in sorted(rel_template)}
K_of = {n: next(x for x in rel_template[n] if x[0] == "gate" and x[1] == "trellis")[4][2] / 16 for n in rel_template}
meta = {"format": "ps5lm-exl3-pack-v1", "source": os.path.abspath(SRC), "layers": L, "experts": E,
        "span_bytes": layer_len, "slot_bytes": slot, "base": base, "total_bytes": total,
        "K": [K_of[n] for n in layer_len], "layouts": tmpl_json}
print(f"pack: {total / 1e9:.2f} GB, K per layer: " + ",".join(f"{k:g}" for k in meta["K"]), flush=True)

# ---- 2. non-expert tensors -> exl3_nonexpert.safetensors (before anything is deleted)
ne_path = os.path.join(OUT, "exl3_nonexpert.safetensors")
if not os.path.exists(ne_path):
    items = []
    for s, (b0, h) in heads.items():
        for k, v in h.items():
            if k != "__metadata__" and not RX.match(k):
                items.append((k, s, b0 + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"]))
    items.sort()
    hdr, off = {}, 0
    for k, s, a, n, dt, sh in items:
        hdr[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = ne_path + ".part"
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)) + hb)
        for k, s, a, n, dt, sh in items:
            with open(os.path.join(SRC, s), "rb") as fi:
                fi.seek(a)
                left = n
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    fo.write(chunk)
                    left -= len(chunk)
    os.replace(tmp, ne_path)
    print(f"non-expert tensors: {len(items)} -> {off / 1e9:.2f} GB", flush=True)

# ---- 3. experts -> pack, shard by shard (shard deleted after its experts are written and re-read back)
pack_path = os.path.join(OUT, "experts_exl3.pack")
fd = os.open(pack_path, os.O_RDWR | os.O_CREAT, 0o644)
if os.fstat(fd).st_size < total:
    os.ftruncate(fd, total)  # sparse until written
done_path = os.path.join(OUT, "experts_exl3.done")
done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
for s in shards:
    if s in done:
        continue
    keys = sorted(k for k, v in spans.items() if v[0] == s)
    with open(os.path.join(SRC, s), "rb") as fi:
        for (l, e) in keys:
            _, st, n = spans[(l, e)]
            fi.seek(st)
            buf = fi.read(n)
            assert len(buf) == n
            dst = base[l] + e * slot[l]
            os.pwrite(fd, buf + b"\0" * (slot[l] - n), dst)
    os.fsync(fd)
    # verify: every expert of this shard reads back identical (page cache dropped by the reader's O_DIRECT-less path is
    # fine here: fsync above makes the bytes durable; compare content)
    with open(os.path.join(SRC, s), "rb") as fi:
        for (l, e) in keys:
            _, st, n = spans[(l, e)]
            fi.seek(st)
            assert os.pread(fd, n, base[l] + e * slot[l]) == fi.read(n), f"verify failed {l} {e}"
    with open(done_path, "a") as fdn:
        fdn.write(s + "\n")
    print(f"{s}: {len(keys)} experts written + verified", flush=True)
    if DELETE:
        os.remove(os.path.join(SRC, s))
        print(f"  deleted {s}", flush=True)
os.close(fd)
json.dump(meta, open(os.path.join(OUT, "experts_exl3.json"), "w"), indent=1)
print("done", flush=True)
