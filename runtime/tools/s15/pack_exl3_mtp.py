"""S16: append the MTP head's 512 routed experts (EXL3, K=3) to experts_exl3.pack as layer 48.

The MTP experts were kept in exl3_nonexpert.safetensors by pack_exl3.py (sorted by NAME there, so the 12 tensors of an
expert are NOT in slot order). Each expert is re-assembled into the slot layout of the trunk's K=3 layers (the
"layouts" template in experts_exl3.json for span 1,862,412: down/gate/up x suh/svh/mul1/trellis at fixed offsets),
written at base[48] = old total, and read back + compared tensor by tensor. experts_exl3.txt gets a "48 ..." line and
experts_exl3.json gets base/slot/span/K for layer 48 (layers stays 48: the trunk geometry is unchanged).

  python3 pack_exl3_mtp.py MODEL_DIR [PREFIX] [--nonexpert PATH]
  PREFIX defaults to experts_exl3 (the 3.18 pack from pack_exl3.py). For a pack from fetch_pack_exl3.py, e.g. the
  2.68 pack, pass its prefix (experts_exl3_268) and --nonexpert pointing at the 3.18 branch's
  exl3_nonexpert.safetensors: the MTP layer is K=3 in every branch and is appended the same way.
"""
import json, os, struct, sys

args = [a for a in sys.argv[1:]]
ne = None
if "--nonexpert" in args:
    i = args.index("--nonexpert"); ne = args[i + 1]; del args[i:i + 2]
D = args[0]
PREFIX = args[1] if len(args) > 1 else "experts_exl3"
E, PAGE, ML = 512, 4096, 48
meta = json.load(open(os.path.join(D, PREFIX + ".json")))
ne = ne or os.path.join(D, "exl3_nonexpert.safetensors")
with open(ne, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
data0 = 8 + n
P = "mtp.layers.0.mlp.experts.{}.{}_proj.{}"

span = 1862412
tmpl = meta["layouts"][str(span)]
slot = ((span + PAGE - 1) // PAGE) * PAGE
k3 = [i for i, s in enumerate(meta["span_bytes"][:ML]) if s == span]
assert k3 and meta["slot_bytes"][k3[0]] == slot, "K=3 slot size mismatch"
base = meta["total_bytes"]
if len(meta["base"]) > ML:
    assert meta["base"][ML] == base - slot * E or meta["base"][ML] == base, "unexpected json state"
    base = meta["base"][ML]

pack = os.path.join(D, PREFIX + ".pack")
fd = os.open(pack, os.O_RDWR)
size0 = os.fstat(fd).st_size
assert size0 in (base, base + slot * E), f"pack size {size0} is neither the trunk total nor trunk+MTP"
with open(ne, "rb") as fi:
    def tensor(e, proj, kind, nbytes, shape, dtype):
        v = hdr[P.format(e, proj, kind)]
        a, b = v["data_offsets"]
        assert b - a == nbytes and v["shape"] == shape and v["dtype"] == dtype, (e, proj, kind, v, nbytes, shape, dtype)
        fi.seek(data0 + a)
        return fi.read(nbytes)

    for e in range(E):
        buf = bytearray(slot)
        for t in tmpl:
            buf[t["off"]:t["off"] + t["bytes"]] = tensor(e, t["proj"], t["kind"], t["bytes"], t["shape"], t["dtype"])
        os.pwrite(fd, bytes(buf), base + e * slot)
    os.fsync(fd)
    for e in range(E):  # read back, tensor by tensor
        got = os.pread(fd, slot, base + e * slot)
        for t in tmpl:
            assert got[t["off"]:t["off"] + t["bytes"]] == tensor(e, t["proj"], t["kind"], t["bytes"], t["shape"], t["dtype"]), f"verify {e} {t['proj']}.{t['kind']}"
        assert got[span:] == b"\0" * (slot - span)
os.close(fd)

meta["base"] = meta["base"][:ML] + [base]
meta["slot_bytes"] = meta["slot_bytes"][:ML] + [slot]
meta["span_bytes"] = meta["span_bytes"][:ML] + [span]
meta["K"] = meta["K"][:ML] + [3.0]
meta["total_bytes_with_mtp"] = base + slot * E
json.dump(meta, open(os.path.join(D, PREFIX + ".json.tmp"), "w"), indent=1)
os.replace(os.path.join(D, PREFIX + ".json.tmp"), os.path.join(D, PREFIX + ".json"))

txt = os.path.join(D, PREFIX + ".txt")
lines = [l for l in open(txt).read().splitlines() if not l.startswith(f"{ML} ")]
lines.append(f"{ML} 6 {slot} {base} {span}")
open(txt + ".tmp", "w").write("\n".join(lines) + "\n")
os.replace(txt + ".tmp", txt)
print(f"MTP experts: {E} x {slot} B at {base} (pack {base + slot * E} B), written + verified", flush=True)
