"""Pack manifest: the streaming layout of expert bytes + per-token traffic (G).

The engine's contribution per the composed-engine feasibility register
(F29): the *pack tool* — turn a model descriptor (layers, experts, top-k,
bits-per-weight) into a verified streaming layout manifest, and recompute G
from it. Layout is slot-contiguous per layer so the runtime can issue
direct-memory staging reads per (layer, expert) slot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

# DERIVED (S3 re-derivation): at the Q4 anchor, G=1.24 GiB/token over
# 48 MoE layers x (top-10 + 1 shared) = 528 expert reads/token ->
# ~2.406 MiB per expert slot. Back out expert params from a 4.5 effective bpw.
EXPERT_SLOT_MIB_AT_Q4 = 1.24 * 1024 / (48 * 11)  # 2.4029 MiB
EFFECTIVE_Q4_BPW = 4.5  # ASSUMED anchor for parameter back-out
EXPERT_PARAMS = round(EXPERT_SLOT_MIB_AT_Q4 * 1024 * 1024 * 8 / EFFECTIVE_Q4_BPW)


@dataclass
class ModelConfig:
    """Model descriptor. Numbers follow Qwen3.8-Flash-Next as audited (STATE.md)."""

    name: str = "Qwen3.8-Flash-Next"
    moe_layers: int = 48          # MEASURED (config.json, S3)
    n_experts: int = 512          # MEASURED
    top_k: int = 10               # MEASURED (top-10 + 1 shared)
    shared_experts: int = 1       # MEASURED
    expert_params: int = EXPERT_PARAMS  # DERIVED from the Q4 anchor
    provenance: dict[str, str] = field(default_factory=lambda: {"source": "STATE.md S3/S6"})


def expert_slot_bytes(cfg: ModelConfig, bpw: float) -> int:
    """Bytes of one expert slot at the given bits-per-weight."""
    if bpw <= 0:
        raise ValueError("bpw must be positive")
    return int(math.ceil(cfg.expert_params * bpw / 8))


def g_per_token_gib(cfg: ModelConfig, bpw: float) -> float:
    """G: expert bytes moved per token, GiB. Shared experts ride along."""
    reads = cfg.top_k + cfg.shared_experts
    slot = expert_slot_bytes(cfg, bpw)
    return reads * cfg.moe_layers * slot / 1024**3


def build_manifest(cfg: ModelConfig, bpw: float, hot_experts: dict[int, list[int]] | None = None) -> dict[str, Any]:
    """Build the streaming-layout manifest.

    hot_experts: optional {layer: [expert ids]} to pin at the *front* of each
    layer's arena so hot slots share cache lines/erase blocks with neighbours
    (arena seeding, F29). Default: identity order.
    """
    slot = expert_slot_bytes(cfg, bpw)
    reads = cfg.top_k + cfg.shared_experts
    layers = []
    for layer in range(cfg.moe_layers):
        order = list(range(cfg.n_experts))
        if hot_experts:
            pinned = hot_experts.get(layer, [])
            order = pinned + [e for e in order if e not in pinned]
        offset = layer * cfg.n_experts * slot
        slots = {expert: offset + i * slot for i, expert in enumerate(order)}
        layers.append({"layer": layer, "slot_bytes": slot, "slots": slots})
    manifest = {
        "schema": "ps5lm-pack/1",
        "model": cfg.name,
        "bpw": bpw,
        "moe_layers": cfg.moe_layers,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "shared_experts": cfg.shared_experts,
        "expert_slot_bytes": slot,
        "pack_bytes": cfg.moe_layers * cfg.n_experts * slot,
        "g_gib_per_token": g_per_token_gib(cfg, bpw),
        "provenance": {
            "g_derivation": f"({reads} reads/layer) x {cfg.moe_layers} layers x {slot} B",
            "expert_params": "DERIVED from Q4 anchor (G=1.24 @ 48x11 reads), EFFECTIVE_Q4_BPW=4.5 ASSUMED",
            **cfg.provenance,
        },
        "layers": layers,
    }
    problems = verify_manifest(manifest)
    if problems:
        raise ValueError(f"internal error: built manifest failed verification: {problems}")
    return manifest


def verify_manifest(manifest: dict[str, Any]) -> list[str]:
    """Consistency checks. Returns a list of problems (empty = verified)."""
    p: list[str] = []
    if manifest.get("schema") != "ps5lm-pack/1":
        p.append(f"unknown schema {manifest.get('schema')!r}")
    layers = manifest.get("layers") or []
    if len(layers) != manifest.get("moe_layers"):
        p.append("layer count mismatch")
    slot = manifest.get("expert_slot_bytes", 0)
    if slot <= 0:
        p.append("non-positive slot size")
    seen_offsets: set[int] = set()
    for lay in layers:
        slots = lay.get("slots") or {}
        if len(slots) != manifest.get("n_experts"):
            p.append(f"layer {lay.get('layer')}: expert count mismatch")
        for expert, off in slots.items():
            if not 0 <= expert < manifest.get("n_experts", 0):
                p.append(f"layer {lay.get('layer')}: expert id {expert} out of range")
            if off in seen_offsets:
                p.append(f"duplicate offset {off}")
            seen_offsets.add(off)
            if slot and off % slot != 0:
                p.append(f"offset {off} not slot-aligned")
    expected_pack = manifest.get("moe_layers", 0) * manifest.get("n_experts", 0) * slot
    if manifest.get("pack_bytes") != expected_pack:
        p.append("pack_bytes mismatch")
    # G recompute must match the stored value (the ledger is the source of truth).
    reads = manifest.get("top_k", 0) + manifest.get("shared_experts", 0)
    g = reads * manifest.get("moe_layers", 0) * slot / 1024**3
    if abs(g - manifest.get("g_gib_per_token", -1.0)) > 1e-6:
        p.append("g_gib_per_token does not match layout")
    return p


def save(manifest: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    # JSON object keys are strings; slots must be int expert ids.
    for lay in m.get("layers", []):
        if "slots" in lay:
            lay["slots"] = {int(k): v for k, v in lay["slots"].items()}
    return m
