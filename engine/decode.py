"""Decoder requirement: tok/s -> experts/s and G-params/s that ANY runtime must sustain.

The S9 reprice made the 40-band conditional on "a decoder for the EXL3 format
(now the critical path)". This module prices that decoder as a first-class
gate output (the plan's Risk-2 fallback: "the gate quantifies the required
decode rate (expert reads/s) as a first-class output").

The load-bearing insight (2026-10-27 spike): **the decode requirement is
independent of the cache hit rate.** The N7 slot cache saves expert *bytes*
from the SSD, not dequant *compute* — every token still dequantizes all 480
routed expert reads no matter how hot the cache is. So the requirement is a
closed form in tok/s alone, and it can be adjudicated against measured
anchors with zero console time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import ModelConfig

# --- Provenance-tagged constants --------------------------------------------

# MEASURED (S5 re-derivation from the real HF config.json): one expert is
# 3 x 2560 x 640 = 4,915,200 params (gate/up/down projections). This is the
# EXACT audited count; manifest.EXPERT_PARAMS is a back-out from the rounded
# G anchor and is ~9% low -- do not mix them.
EXPERT_PARAMS_EXACT = 3 * 2560 * 640  # 4,915,200

# DERIVED: routed reads per token = top_k x moe_layers. The shared expert
# (1/layer) rides every token too, but it is always-hot: a runtime can keep
# all 48 shared experts PRE-DEQUANTIZED in GTT for 48 * 4,915,200 * 2 B =
# 472 MB (F29 budget has room), so the *streaming* decoder only owes the
# 480 routed reads. Both numbers are exposed; verdicts use ROUTED.
def reads_per_token(cfg: ModelConfig, include_shared: bool = False) -> int:
    reads = cfg.top_k * cfg.moe_layers
    if include_shared:
        reads += cfg.shared_experts * cfg.moe_layers
    return reads


# MEASURED-ADJACENT ANCHOR [D-PS5, F28 register 2026-10-23]: PS5LM-Plus
# (llama.cpp + ggml Vulkan, native title, PS5 GPU) runs dense Qwen3.8-27B at
# 20.3 tok/s (conservative end of their 20.3-20.7). Every token dequantizes
# the full 27B on the fly inside the GEMMs, so the GPU demonstrably sustains
# 27.0e9 x 20.3 = 548.1 G params/s of fused dequant+GEMM on THIS console.
# It is a FLOOR on the PS5 GPU's decode path, measured on the exact silicon
# we target (GFX1013 / Cyan Skillfish).
ANCHOR_PS5_GPU_DENSE_27B_TOK_S = 20.3
ANCHOR_PS5_GPU_DENSE_27B_PARAMS = 27.0e9
ANCHOR_PS5_GPU_GPARAMS_S = ANCHOR_PS5_GPU_DENSE_27B_PARAMS * ANCHOR_PS5_GPU_DENSE_27B_TOK_S / 1e9  # 548.1

# Second anchor, same source [F28]: ProsperoAI (AGC kernels) ~30 tok/s @ 9B
# -> 270 G params/s. Kept for consistency, not used in verdicts.
ANCHOR_PROSPERO_9B_GPARAMS_S = 9.0e9 * 30.0 / 1e9

# Verdict margins: GO needs the anchor to cover the requirement with this
# much headroom (fused dequant shares the GPU with attention, KV, routing).
GO_HEADROOM_MIN = 2.0
MARGINAL_HEADROOM = 1.0


@dataclass(frozen=True)
class DecodeRequirement:
    """The decoder workload at a given decode throughput."""

    tok_s: float
    experts_per_s: float          # routed reads dequantized per second
    gparams_s: float              # billions of params dequantized per second
    gemm_gflop_s: float           # 2 FLOP/param fused into the same pass

    @property
    def fp16_materialize_gb_s(self) -> float:
        """Write bandwidth IF a runtime materialized experts in fp16 first.

        ggml-style runtimes fuse dequant into the GEMM (no materialization),
        so this is a bound for materialize-then-compute designs only.
        """
        return self.experts_per_s * EXPERT_PARAMS_EXACT * 2 / 1e9

    @property
    def headroom_vs_ps5_gpu(self) -> float:
        """Measured PS5-GPU floor divided by the requirement. >1 = feasible."""
        return ANCHOR_PS5_GPU_GPARAMS_S / self.gparams_s

    @property
    def verdict(self) -> str:
        h = self.headroom_vs_ps5_gpu
        if h >= GO_HEADROOM_MIN:
            return "GO"
        if h >= MARGINAL_HEADROOM:
            return "MARGINAL"
        return "DEAD"


def requirement(
    tok_s: float,
    cfg: ModelConfig | None = None,
    include_shared: bool = False,
) -> DecodeRequirement:
    """Decode workload required to sustain tok/s, hit-rate-INDEPENDENT."""
    if tok_s <= 0:
        raise ValueError("tok/s must be positive")
    cfg = cfg or ModelConfig()
    eps = tok_s * reads_per_token(cfg, include_shared)
    gps = eps * EXPERT_PARAMS_EXACT / 1e9
    return DecodeRequirement(
        tok_s=tok_s,
        experts_per_s=eps,
        gparams_s=gps,
        gemm_gflop_s=2 * eps * EXPERT_PARAMS_EXACT / 1e9,
    )


def spike_verdict(tok_s: float, cfg: ModelConfig | None = None) -> DecodeRequirement:
    """The S9-queue item 4 verdict: is the 40-band's decoder requirement
    covered by the measured PS5-GPU anchor?"""
    return requirement(tok_s, cfg)


def headline_check() -> dict[str, bool]:
    """Regression invariants for the spike (2026-10-27).

    - 40 tok/s needs 19,200 routed experts/s = 94.4 G params/s.
    - Headroom vs the measured PS5-GPU anchor is >= 5x.
    - The async-deep EXL3 1.66 row (43.7 tok/s) also clears GO.
    """
    r40 = requirement(40.0)
    r437 = requirement(43.7)
    return {
        "experts_per_s_at_40": abs(r40.experts_per_s - 19200.0) < 1e-6,
        "gparams_s_at_40_is_94_4": abs(r40.gparams_s - 94.37) < 0.1,
        "headroom_at_40_ge_5x": r40.headroom_vs_ps5_gpu >= 5.0,
        "async_deep_43_7_still_go": r437.verdict == "GO",
    }
