"""Range-fetch a sample of BF16 experts (gate_up + down) of Qwen3.8-Flash-Next from HF for the S15 quantizer study.
Hot / median / cold experts by the real-trace frequency, at layers 2, 10, 30, 45. Writes <out>/L{l}_E{e}.npz-like raw files."""
import json, os, struct, sys, urllib.request
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from policy_sweep import load  # noqa

BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
idx = json.load(open(sys.argv[2]))["weight_map"]
LAYERS = [2, 10, 30, 45]
freq = {l: Counter() for l in LAYERS}
for i in (1, 2, 3):
    toks, _, _ = load(__import__("os").path.join(__import__("os").environ.get("PS5LM_TRACES", "traces"), "real2_%d.txt") % i)
    for tk in toks:
        for l in LAYERS:
            freq[l].update(tk[l])


def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "ps5lm-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = r.read()
    assert len(d) == b - a, (len(d), b - a)
    return d


hdr_cache = {}


def header(shard):
    if shard not in hdr_cache:
        url = BASE + shard
        n = struct.unpack("<Q", get(url, 0, 8))[0]
        hdr_cache[shard] = (8 + n, json.loads(get(url, 8, 8 + n)))
    return hdr_cache[shard]


plan = {}
for l in LAYERS:
    ranked = [e for e, _ in freq[l].most_common()]
    n = len(ranked)
    pick = ranked[:6] + ranked[n // 2 - 3:n // 2 + 3] + ranked[-6:]
    plan[l] = pick
    print(l, "used", n, "pick", pick, [freq[l][e] for e in pick], flush=True)
json.dump({str(l): {"pick": plan[l], "freq": {str(e): freq[l][e] for e in plan[l]}} for l in LAYERS}, open(os.path.join(OUT, "plan.json"), "w"))

for l in LAYERS:
    for kind, rows, cols in (("gate_up_proj", 1280, 2560), ("down_proj", 2560, 640)):
        name = f"model.language_model.layers.{l}.mlp.experts.{kind}"
        shard = idx[name]
        d0, h = header(shard)
        info = h[name]
        assert info["shape"] == [512, rows, cols], info
        per = rows * cols * 2
        for e in plan[l]:
            fn = os.path.join(OUT, f"L{l}_E{e}_{kind}.bf16")
            if os.path.exists(fn) and os.path.getsize(fn) == per:
                continue
            a = d0 + info["data_offsets"][0] + e * per
            open(fn, "wb").write(get(BASE + shard, a, a + per))
        print("done", name, flush=True)
print("ALL DONE")
