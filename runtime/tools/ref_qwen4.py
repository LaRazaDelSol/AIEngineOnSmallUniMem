"""Golden reference for Qwen3.8-Flash-Next text decoding (CPU torch), built from the VERBATIM HF module
source (transformers modeling_qwen4_exp.py, extracted by AST with no-op shims for hub/kernel decorators).

Weights come from the SAME converted model the PS5 engine uses:
  dense/<name>.bin (raw BF16 + manifest.json), routed experts dequantised on demand from the PS5Q pack,
  n-gram rows gathered from ngram_bf16.bin (memmap). Layers are built on the meta device and streamed one at a
  time, so the 51B-param n-gram table and the 512 x 48 experts are never materialised.

    python3 ref_qwen4.py --model ~/ps5lm/model --hf modeling_qwen4_exp.py --ids 248044,9707,11 --out golden.npz
        [--pack experts_q4.pack --bits 4] [--layers 48] [--dense-q4 DIR]

Output npz: ids, hidden[L+1, T, 4*2560] (stream state after embedding and after each layer), route_ids[L, T, 10],
route_w[L, T, 10], logits_top (T, 20) ids+values, argmax (T).
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import os
import sys
import time
import types
import typing

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NEEDED = {
    "Qwen4ExpTextRotaryEmbedding", "Qwen4ExpTextRMSNorm", "Qwen4ExpTextRMSNormGated", "apply_mask_to_padding_states",
    "causal_conv1d_update", "causal_conv1d_fn", "l2norm", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
    "Qwen4ExpTextGatedDeltaNet", "rotate_half", "apply_rotary_pos_emb", "Qwen4ExpTextQSAIndexer", "repeat_kv",
    "eager_attention_forward", "Qwen4ExpTextAttention", "Qwen4ExpTextMLP", "Qwen4ExpTextExperts", "Qwen4ExpTextTopKRouter",
    "Qwen4ExpTextSparseMoeBlock", "Qwen4ExpTextGatedResidual", "_splitmix64", "_build_layer_multipliers", "_is_prime",
    "_find_nth_prime_after", "Qwen4ExpTextNGramEmbedding", "Qwen4ExpTextPLELayer", "Qwen4ExpTextDecoderLayer",
}


class _Sub:
    def __getitem__(self, _):
        return object


def _deco(*_a, **_k):
    return lambda f: f


def load_hf(path: str) -> dict:
    ns = dict(
        torch=torch, nn=nn, F=F, math=math, itertools=itertools, Callable=typing.Callable,
        ACT2FN={"silu": nn.SiLU(), "sigmoid": nn.Sigmoid()}, Cache=object, DynamicCache=object, Unpack=_Sub(),
        TransformersKwargs=dict, GradientCheckpointingLayer=nn.Module, use_kernel_forward_from_hub=_deco,
        use_kernel_func_from_hub_with_fallback=_deco, use_kernelized_func=_deco, use_experts_implementation=lambda c: c,
        force_accelerate_hooks=_deco, dynamic_rope_update=lambda f: f, ROPE_INIT_FUNCTIONS={},
        ALL_ATTENTION_FUNCTIONS=types.SimpleNamespace(get_interface=lambda impl, default: default),
        Qwen4ExpTextConfig=object, is_torchdynamo_exporting=lambda: False, torch_compilable_check=lambda *a, **k: None,
    )
    tree = ast.parse(open(path, encoding="utf-8").read())
    got = set()
    for node in tree.body:
        keep = isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NEEDED
        keep |= isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.startswith("_") for t in node.targets)
        if keep:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
            if hasattr(node, "name"):
                got.add(node.name)
    missing = NEEDED - got
    if missing:
        raise SystemExit(f"HF source lacks {missing}")
    return ns


def make_config(model_dir: str) -> types.SimpleNamespace:
    tc = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    c = types.SimpleNamespace(**tc)
    c.layer_types = ["indexed_attention" if t == "full_attention" else t for t in c.layer_types]
    c.norm_topk_prob = getattr(c, "norm_topk_prob", True)
    c.seed = getattr(c, "seed", 1234)
    c._attn_implementation = "eager"
    c.ple_embed_dim = c.ple_embed_dim or c.hidden_size
    return c


class Weights:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.man = json.load(open(os.path.join(model_dir, "manifest.json")))

    def get(self, name: str) -> torch.Tensor:
        info = self.man[name]
        raw = np.fromfile(os.path.join(self.dir, "dense", name + ".bin"), dtype=np.uint8)
        if info["dtype"] == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif info["dtype"] == "F32":
            arr = raw.view(np.float32)
        elif info["dtype"] == "I64":
            arr = raw.view(np.int64)
        elif info["dtype"] == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        else:
            raise SystemExit(f"dtype {info['dtype']} for {name}")
        return torch.from_numpy(arr.reshape(info["shape"]).copy())


def ps5q_dequant(rows: np.ndarray, K: int, bits: int) -> np.ndarray:
    """rows: uint8 [R, row_bytes] in PS5Q layout -> float32 [R, K]."""
    nb = K // 32
    sb = (nb * 2 + 15) // 16 * 16
    qb = 16 if bits == 4 else 8
    sc = rows[:, : nb * 2].copy().view(np.float16).astype(np.float32)  # [R, nb]
    q = rows[:, sb: sb + nb * qb].reshape(-1, nb, qb)
    if bits == 4:
        v = np.concatenate([q & 0xF, q >> 4], axis=2).astype(np.float32) - 8.0
    else:
        v = np.concatenate([(q >> (2 * s)) & 3 for s in range(4)], axis=2).astype(np.float32) - 2.0
    return (v * sc[:, :, None]).reshape(-1, K)


class PackExperts:
    """Indexable like the HF fused parameter: obj[e] -> dequantised [2*FFN, H] (gate_up) or [H, FFN] (down)."""

    def __init__(self, pack: np.memmap, layer: int, kind: str, bits: int, H: int = 2560, FF: int = 640):
        self.pack, self.layer, self.kind, self.bits, self.H, self.F = pack, layer, kind, bits, H, FF
        rb = lambda K: (((K // 32) * 2 + 15) // 16 * 16 + (K // 32) * (16 if bits == 4 else 8) + 15) // 16 * 16
        self.rgu, self.rd = rb(H), rb(FF)
        self.slot = 2 * FF * self.rgu + H * self.rd

    def __getitem__(self, e) -> torch.Tensor:
        base = (self.layer * 512 + int(e)) * self.slot
        if self.kind == "gate_up":
            rows = np.asarray(self.pack[base: base + 2 * self.F * self.rgu]).reshape(2 * self.F, self.rgu)
            return torch.from_numpy(ps5q_dequant(rows, self.H, self.bits))
        off = base + 2 * self.F * self.rgu
        rows = np.asarray(self.pack[off: off + self.H * self.rd]).reshape(self.H, self.rd)
        return torch.from_numpy(ps5q_dequant(rows, self.F, self.bits))


class NGramRows(nn.Module):
    def __init__(self, path: str, width: int = 160):
        super().__init__()
        self.tab = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, width)
        self.weight = types.SimpleNamespace(device=torch.device("cpu"))  # HF forward reads .weight.device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).cpu().numpy()
        u = self.tab[flat].astype(np.uint32) << 16
        return torch.from_numpy(u.view(np.float32).copy()).reshape(*ids.shape, -1)


def q4_rows(md: str, key: str, K: int, r0: int, r1: int) -> torch.Tensor:
    rb = (((K // 32) * 2 + 15) // 16 * 16 + (K // 32) * 16 + 15) // 16 * 16
    raw = np.memmap(os.path.join(md, "dense_q4", key + ".q4"), dtype=np.uint8, mode="r")
    rows = np.asarray(raw[r0 * rb: r1 * rb]).reshape(r1 - r0, rb)
    return torch.from_numpy(ps5q_dequant(rows, K, 4))


def apply_dense_q4(layer, li: int, md: str):
    """Overwrite every Linear the engine quantises with the engine's own PS5Q4 blocks (exact equivalence mode)."""
    p = f"model.language_model.layers.{li}."
    def setw(lin, t):
        with torch.no_grad():
            lin.weight.copy_(t.reshape(lin.weight.shape))
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        m = getattr(layer, hc)
        setw(m.input_mix_weight_down, q4_rows(md, p + hc + ".down", 10240, 0, 320))
        setw(m.input_mix_weight_up, q4_rows(md, p + hc + ".up", 320, 0, 10240))
        setw(m.block_inject_weight, q4_rows(md, p + hc + ".inj", 10240, 0, 4))
    if layer.layer_type == "linear_attention":
        g, k = layer.linear_attn, p + "linear_attn.inproj"
        setw(g.in_proj_qkv, q4_rows(md, k, 2560, 0, 10240))
        setw(g.in_proj_z, q4_rows(md, k, 2560, 10240, 16384))
        setw(g.in_proj_b, q4_rows(md, k, 2560, 16384, 16432))
        setw(g.in_proj_a, q4_rows(md, k, 2560, 16432, 16480))
        setw(g.out_proj, q4_rows(md, p + "linear_attn.out", 6144, 0, 2560))
    else:
        a, k = layer.self_attn, p + "self_attn.qkv"
        setw(a.q_proj, q4_rows(md, k, 2560, 0, 12288))
        setw(a.k_proj, q4_rows(md, k, 2560, 12288, 12800))
        setw(a.v_proj, q4_rows(md, k, 2560, 12800, 13312))
        setw(a.o_proj, q4_rows(md, p + "self_attn.o", 6144, 0, 2560))
    setw(layer.mlp.shared_expert_gate, q4_rows(md, p + "sgate", 2560, 0, 1))
    slot = PackExperts(np.memmap(os.path.join(md, "dense_q4", p + "mlp.shared_slot.q4"), dtype=np.uint8, mode="r"), 0, "gate_up", 4)
    gu = slot[0]
    slot.kind = "down"
    setw(layer.mlp.shared_expert.gate_proj, gu[:640])
    setw(layer.mlp.shared_expert.up_proj, gu[640:])
    setw(layer.mlp.shared_expert.down_proj, slot[0])
    if layer.ple is not None:
        setw(layer.ple.key_proj, q4_rows(md, p + "ple.key", 2560, 0, 10240))
        setw(layer.ple.value_proj, q4_rows(md, p + "ple.value", 2560, 0, 2560))


def load_module(mod: nn.Module, prefix: str, W: Weights, skip=()):
    mod.to_empty(device="cpu")
    for name, p in itertools.chain(mod.named_parameters(), mod.named_buffers()):
        full = prefix + name
        if any(s in name for s in skip):
            continue
        if full not in W.man:
            raise SystemExit(f"missing weight {full}")
        t = W.get(full)
        if tuple(t.shape) != tuple(p.shape):
            raise SystemExit(f"shape {full}: {tuple(t.shape)} vs {tuple(p.shape)}")
        with torch.no_grad():
            p.copy_(t.to(p.dtype))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", default="experts_q4.pack")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--dense-q4", action="store_true", help="use the engine PS5Q4 dense weights (exact-equivalence check)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    md = os.path.expanduser(a.model)
    ns = load_hf(a.hf)
    cfg = make_config(md)
    W = Weights(md)
    pack = np.memmap(os.path.join(md, a.pack), dtype=np.uint8, mode="r")
    ids = torch.tensor([[int(x) for x in a.ids.split(",")]], dtype=torch.long)
    T = ids.shape[1]
    t0 = time.time()

    emb = np.memmap(os.path.join(md, "dense", "model.language_model.embed_tokens.weight.bin"), dtype=np.uint16, mode="r").reshape(cfg.vocab_size, -1)
    x = torch.from_numpy((emb[ids[0].numpy()].astype(np.uint32) << 16).view(np.float32).copy())[None]  # [1,T,H]
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    rot = ns["Qwen4ExpTextRotaryEmbedding"](cfg)
    cos, sin = rot(x, pos)
    mask = torch.full((1, 1, T, T), torch.finfo(torch.float32).min).triu(1)
    h = x.repeat(1, 1, cfg.hc_count)
    hidden = [h[0].numpy().copy()]
    route_ids = np.zeros((a.layers, T, cfg.num_experts_per_tok), np.int32)
    route_w = np.zeros((a.layers, T, cfg.num_experts_per_tok), np.float32)
    for li in range(a.layers):
        with torch.device("meta"):
            layer = ns["Qwen4ExpTextDecoderLayer"](cfg, li)
        ex = layer.mlp.experts
        del ex._parameters["gate_up_proj"], ex._parameters["down_proj"]
        ex.gate_up_proj = PackExperts(pack, li, "gate_up", a.bits)
        ex.down_proj = PackExperts(pack, li, "down", a.bits)
        skip = ()
        if layer.ple is not None:
            del layer.ple.ple_embedding._modules["ngram_embedding"]
            skip = ("ngram_embedding",)
        load_module(layer, f"model.language_model.layers.{li}.", W, skip)
        if a.dense_q4:
            apply_dense_q4(layer, li, md)
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = NGramRows(os.path.join(md, "ngram_bf16.bin"))
        cap = {}
        layer.mlp.gate.register_forward_hook(lambda m, i, o: cap.update(scores=o[1], idx=o[2]))
        h = layer(h, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None, past_key_values=None, ple_input_ids=ids)
        route_ids[li] = cap["idx"].numpy()
        route_w[li] = cap["scores"].float().numpy()
        hidden.append(h[0].numpy().copy())
        print(f"layer {li:2d} done  |h| {h.norm().item():.3f}  {time.time() - t0:.0f}s", flush=True)
        del layer

    out = {"ids": ids[0].numpy(), "hidden": np.stack(hidden), "route_ids": route_ids, "route_w": route_w}
    if a.layers == cfg.num_hidden_layers:
        with torch.device("meta"):
            mixer = ns["Qwen4ExpTextGatedResidual"](cfg, use_combine=False)
        load_module(mixer, "model.language_model.hyper_connection_mixer.", W)
        y = mixer(h)[0]  # [T, H]
        head = np.memmap(os.path.join(md, "dense", "lm_head.weight.bin"), dtype=np.uint16, mode="r").reshape(cfg.vocab_size, -1)
        logits = np.zeros((T, cfg.vocab_size), np.float32)
        for s in range(0, cfg.vocab_size, 16384):
            blk = (head[s: s + 16384].astype(np.uint32) << 16).view(np.float32)
            logits[:, s: s + 16384] = y.numpy() @ blk.T
        top = np.argsort(-logits, axis=1)[:, :20]
        out.update(logits_top_ids=top, logits_top_vals=np.take_along_axis(logits, top, 1), argmax=top[:, 0], final=y.numpy())
        print("argmax:", top[:, 0].tolist())
    np.savez(a.out, **out)
    print(f"saved {a.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
