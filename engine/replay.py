"""R2 acceptance gate: replay a routing trace against a pack manifest.

Simulates the runtime's slot cache -- ONE LRU pool shared by all layers, keyed
by (layer, expert), capacity = cache budget / slot size; mirrors
runtime/include/ps5lm/slot_cache.hpp Policy::LRU read-for-read (gate S1) and reports the achieved hit rate, the implied
tok/s at a measured store bandwidth, and a PASS/FAIL verdict against the R2
gate. This is the acceptance harness for any pack (REAP trim, EXL3, layered).

The gate is defined in research/design-history/RESEARCH-TRACK.md: a pass OR a fail is a public
datapoint; what is not allowed is an unpriced claim.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from . import ledger
from .ledger import tok_s


class SlotCache:
    """Per-layer LRU over expert slots with a fixed capacity in slots."""

    def __init__(self, capacity_slots: int) -> None:
        if capacity_slots <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity_slots
        self._slots: OrderedDict[int, None] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def touch(self, expert: int) -> None:
        if expert in self._slots:
            self.hits += 1
            self._slots.move_to_end(expert)
        else:
            self.misses += 1
            if len(self._slots) >= self.capacity:
                self._slots.popitem(last=False)
            self._slots[expert] = None


def replay(
    manifest: dict[str, Any],
    trace: dict[str, Any],
    cache_gib: float,
    bw_gib_s: float,
    tok_s_target: float,
    gate_hit: float = 0.8,
) -> dict[str, Any]:
    """Run the gate. Returns a report dict; `verdict` is PASS/FAIL/UNREACHABLE.

    - PASS:  hit >= gate_hit AND projected tok/s >= tok_s_target.
    - FAIL:  ran to completion, missed at least one bar.
    - UNREACHABLE: even at hit->1 the bandwidth cannot serve tok_s_target
      (the physics bars it; no cache policy can fix it).
    """
    for key in ("expert_slot_bytes", "moe_layers", "layers"):
        if key not in manifest:
            raise ValueError(f"manifest missing {key!r}")
    if trace.get("schema") != "ps5lm-trace/1":
        raise ValueError("trace must be ps5lm-trace/1")

    slot = manifest["expert_slot_bytes"]
    capacity = int(cache_gib * 1024**3 // slot) if slot else 0
    if capacity <= 0:
        raise ValueError("cache budget smaller than one expert slot")

    # 2026-11-03 fix: the budget is ONE pool shared by all layers. The previous
    # version built one LRU per layer, each with the WHOLE budget -- i.e. 48x the
    # memory -- so at >=1.2 GiB every layer held all 512 experts and only
    # compulsory misses remained (the PoC's "97.5% @ 8 GiB" was that artifact).
    cache = SlotCache(capacity)
    for token_layers in trace["routing"]:
        for layer, experts in enumerate(token_layers):
            for expert in experts:
                cache.touch((layer, expert))

    hits, misses = cache.hits, cache.misses
    total = hits + misses
    hit_rate = hits / total if total else 0.0
    g = manifest["g_gib_per_token"]

    required = 1.0 - bw_gib_s / (g * tok_s_target)
    if required >= 1.0 or required > ledger.REQUIRED_HIT_CEIL:
        verdict = "UNREACHABLE"
        projected = 0.0
    else:
        projected = tok_s(bw_gib_s, g, hit_rate)
        verdict = "PASS" if (hit_rate >= gate_hit and projected >= tok_s_target) else "FAIL"

    return {
        "schema": "ps5lm-gate-report/1",
        "verdict": verdict,
        "hit_rate": round(hit_rate, 4),
        "gate_hit": gate_hit,
        "projected_tok_s": round(projected, 2),
        "tok_s_target": tok_s_target,
        "cache_gib": cache_gib,
        "cache_slots_total": capacity,
        "g_gib_per_token": round(g, 3),
        "bw_gib_s": bw_gib_s,
        "expert_reads_total": total,
        "pack_bytes": manifest["pack_bytes"],
        "pack_gib": round(manifest["pack_bytes"] / 1024**3, 2),
        "trace": {k: trace[k] for k in ("profile", "seed", "tokens")},
    }


def report_markdown(report: dict[str, Any]) -> str:
    """Render the report the way probe sessions land in reports/."""
    lines = [
        "# R2 gate report",
        "",
        f"**Verdict: {report['verdict']}** "
        f"(hit {report['hit_rate']:.1%} vs gate {report['gate_hit']:.0%}; "
        f"projected {report['projected_tok_s']} tok/s vs target {report['tok_s_target']})",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| pack size | {report['pack_gib']} GiB |",
        f"| G (expert bytes/token) | {report['g_gib_per_token']} GiB |",
        f"| store bandwidth (input) | {report['bw_gib_s']} GiB/s |",
        f"| cache budget | {report['cache_gib']} GiB = {report['cache_slots_total']} slots (one pool, all layers) |",
        f"| trace | {report['trace']['profile']} seed={report['trace']['seed']} tokens={report['trace']['tokens']} |",
        f"| expert reads replayed | {report['expert_reads_total']} |",
        "",
        "*Synthetic trace - a public datapoint, not a console measurement. "
        "Replace with probe.py routing dumps per RESEARCH-TRACK R3/R4.*",
    ]
    return "\n".join(lines) + "\n"


def save(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
