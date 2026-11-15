"""Routing-trace generation for the R2 acceptance gate.

A trace is what the gate replays: per token, per MoE layer, the set of expert
ids the router would have read. Profiles model the traffic shapes the project
cares about (audited in S5: "MTP OFF used the wrong prior for repetitive agent
traffic" — so `agent` has heavy repetition by default).

Pure synthetic until console-side dumps exist (R3/R4 keystone: real routing
dumps from probe.py). Every trace records its profile + seed so runs are
reproducible.
"""

from __future__ import annotations

import json
import random
from typing import Any


def generate(
    tokens: int,
    moe_layers: int,
    n_experts: int,
    top_k: int,
    profile: str = "agent",
    seed: int = 7,
) -> dict[str, Any]:
    """Generate a routing trace.

    Profiles:
      agent  - repetitive tool-calling workload: a small working set per burst,
               Zipf(1.5) popularity, burst switches every ~64 tokens.
      chat   - conversational: milder repetition, Zipf(1.1).
      uniform- adversarial worst case: uniform popularity, no bursts.
    """
    if profile not in {"agent", "chat", "uniform"}:
        raise ValueError(f"unknown profile {profile!r}")
    rng = random.Random(seed)
    alpha = {"agent": 1.5, "chat": 1.1, "uniform": 0.0}[profile]
    burst_len = {"agent": 64, "chat": 256, "uniform": 10**9}[profile]

    # Global popularity weights per layer (Zipf-ish over expert ranks).
    layer_weights: list[list[float]] = []
    for _ in range(moe_layers):
        if alpha <= 0:
            w = [1.0] * n_experts
        else:
            w = [1.0 / (i + 1) ** alpha for i in range(n_experts)]
        layer_weights.append(w)

    tokens_out: list[list[list[int]]] = []  # [token][layer] -> expert ids
    # Per-layer burst working sets: reuse recent experts within a burst.
    burst_sets: list[list[int]] = [
        rng.sample(range(n_experts), min(48, n_experts)) for _ in range(moe_layers)
    ]
    for t in range(tokens):
        if t % burst_len == 0:  # burst switch: refresh working sets
            burst_sets = [
                rng.sample(range(n_experts), min(48, n_experts)) for _ in range(moe_layers)
            ]
        token_layers = []
        for layer in range(moe_layers):
            w = layer_weights[layer]
            picks: list[int] = []
            # uniform profile: all picks from the global distribution (no
            # burst prior) - the documented adversarial worst case.
            n_from_burst = 0 if profile == "uniform" else max(1, int(top_k * 0.7))
            if n_from_burst:
                bs = burst_sets[layer]
                picks.extend(rng.sample(bs, min(n_from_burst, len(bs))))
            while len(picks) < top_k:
                e = rng.choices(range(n_experts), weights=w, k=1)[0]
                if e not in picks:
                    picks.append(e)
            token_layers.append(sorted(picks))
        tokens_out.append(token_layers)

    return {
        "schema": "ps5lm-trace/1",
        "profile": profile,
        "seed": seed,
        "tokens": tokens,
        "moe_layers": moe_layers,
        "n_experts": n_experts,
        "top_k": top_k,
        "provenance": {"kind": "synthetic", "note": "replace with probe.py routing dumps (R3/R4)"},
        "routing": tokens_out,
    }


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    if trace.get("schema") != "ps5lm-trace/1":
        raise ValueError(f"unknown trace schema {trace.get('schema')!r}")
    return trace
