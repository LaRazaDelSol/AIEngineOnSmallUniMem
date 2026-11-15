"""Physics ledger: tok/s = SSD_BW / (G * (1 - hit)).

Re-derived from the model's real config.json on 2026-10-20 (the old 1.40 GiB
figure was ~13% high). This module is the single source of the formula so the
pack tool, the gate and the reports can never drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Provenance-tagged constants -------------------------------------------

# MEASURED (config.json recompute, 2026-10-20, S3): Q4 full-pack expert traffic.
G_Q4_GIB = 1.24

# MEASURED (S6, EXL3 1.66-2.18 bpw quality-charted packs of the exact model):
# 2-bit-class per-token expert traffic range.
G_EXL3_LOW_GIB = 0.44
G_EXL3_HIGH_GIB = 0.58

# MEASURED (S6 register 2026-10-23-exl3-flash-next-compression.md): exact
# per-token expert traffic of the three quality-charted EXL3 packs.
G_EXL3_1_66_GIB = 0.443
G_EXL3_2_05_GIB = 0.558
G_EXL3_2_18_GIB = 0.581

# DERIVED: for the 40 tok/s headline to hold at h=0.896 (G=1.24), the store must
# deliver 5.158 GiB/s. Used as the ledger's reference single-NVMe bandwidth.
BW_REFERENCE_GIB_S = (1.0 - 0.896) * G_Q4_GIB * 40.0  # = 5.1584

# DERIVED (S3-S5 audit): hit rates beyond this bar need a 128 GB-class box, not
# a 16 GB console — treat as practically unreachable. 40 tok/s <=> 0.896 and
# 60 <=> 0.931 sit *below* it (dead as a plan, but not physically impossible).
REQUIRED_HIT_CEIL = 0.99

# DERIVED (S10 unit correction, 2026-10-27): fio reports DECIMAL MB/s
# (bw_bytes/1e6); 1 GiB/s = 1073.741824 MB/s. The S8/S9 reports and the
# reprice CLI divided by 1000 and labeled the result GiB/s — a ~7.4% tok/s
# inflation (user-caught verdict 2026-10-27). Fixed at both consumers
# (engine reprice, probe.py projection); every GiB/s in the ledger comes
# through this constant now.
MB_PER_GIB_S = 1073.741824


def mb_s_to_gib_s(mb_s: float) -> float:
    """fio MB/s (decimal) -> GiB/s. Do NOT divide by 1000 (S10 unit fix)."""
    if mb_s < 0:
        raise ValueError("bandwidth must be non-negative")
    return mb_s / MB_PER_GIB_S


@dataclass(frozen=True)
class Band:
    """One row of the honest tok/s band."""

    label: str
    g_gib: float
    hit: float
    bw_gib_s: float

    @property
    def tok_s(self) -> float:
        return tok_s(self.bw_gib_s, self.g_gib, self.hit)


def tok_s(bw_gib_s: float, g_gib: float, hit: float) -> float:
    """Decode throughput under the streaming model. 0 <= hit < 1."""
    if not 0.0 <= hit < 1.0:
        raise ValueError(f"hit rate must be in [0, 1), got {hit}")
    if g_gib <= 0 or bw_gib_s <= 0:
        raise ValueError("bandwidth and G must be positive")
    return bw_gib_s / (g_gib * (1.0 - hit))


def required_hit(bw_gib_s: float, g_gib: float, tok_s_target: float) -> float:
    """Minimum cache hit rate needed to reach tok_s_target."""
    if tok_s_target <= 0:
        raise ValueError("tok/s target must be positive")
    h = 1.0 - bw_gib_s / (g_gib * tok_s_target)
    if h >= 1.0 or h > REQUIRED_HIT_CEIL:
        raise ValueError(
            f"practically unreachable: needs hit={h:.3f} > {REQUIRED_HIT_CEIL} "
            "(raise bandwidth, cut G, or lower the target)"
        )
    if h < 0.0:
        return 0.0
    return h


def band_table(g_gib: float, bw_gib_s: float) -> list[Band]:
    """The honest band grid: hit rates 0.0..0.95 on a fixed step."""
    rows = []
    for hit in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        rows.append(Band(label=f"h={hit:.2f}", g_gib=g_gib, hit=hit, bw_gib_s=bw_gib_s))
    return rows


def reprice_rows(
    g_values: list[tuple[str, float]],
    hit_points: list[tuple[str, float]],
    bw_gib_s: float,
) -> list[tuple[str, float, dict[str, float]]]:
    """One row per (label, hit) with tok/s decoded for every G.

    The Alpha deliverable: measured [D-PS5] constants in, honest tok/s out.
    """
    return [
        (label, hit, {name: tok_s(bw_gib_s, g, hit) for name, g in g_values})
        for label, hit in hit_points
    ]


def reprice_table(
    g_values: list[tuple[str, float]],
    hit_points: list[tuple[str, float]],
    bw_gib_s: float,
) -> str:
    """Markdown table of reprice_rows, in the repo's report style."""
    names = [n for n, _ in g_values]
    lines = [
        "| pool (N7 cap) | hit | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for label, hit, cells in reprice_rows(g_values, hit_points, bw_gib_s):
        lines.append(f"| {label} | {hit:.1%} | " + " | ".join(f"{v:.1f}" for v in cells.values()) + " |")
    return "\n".join(lines)


def headline_check() -> dict[str, bool]:
    """Regression invariants from the audit (S3 re-derivation).

    - 40 tok/s at G=1.24 requires hit=0.896 at the reference bandwidth.
    - 60 tok/s at G=1.24 requires hit=0.931.
    - The 40-60 plan is dead: both need >90% hit at single-NVMe bandwidth.
    """
    h40 = required_hit(BW_REFERENCE_GIB_S, G_Q4_GIB, 40.0)
    h60 = required_hit(BW_REFERENCE_GIB_S, G_Q4_GIB, 60.0)
    return {
        "h40_is_0_896": abs(h40 - 0.896) < 5e-3,
        "h60_is_0_931": abs(h60 - 0.931) < 5e-3,
        "headline_dead": h40 > 0.85 and h60 > 0.9,
    }
