"""Generate the PoC evidence: band chart (SVG, source-only) + gate report.
Run from repo root:  python tools/gen_poc_evidence.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import ledger

OUT = Path(__file__).resolve().parents[1] / "docs" / "evidence"
OUT.mkdir(exist_ok=True)

# --- 1. Band chart: tok/s vs hit rate for the three pack classes -------------
W, H, ML, MB = 720, 440, 60, 50
MAXX, MAXY = 0.95, 90.0
def px(h): return ML + int(h / MAXX * (W - ML - 30))
def py(t): return H - MB - int(t / MAXY * (H - MB - 40))

series = [
    ("Q4 full pack (G=1.24)", "#d95f02", ledger.G_Q4_GIB),
    ("EXL3 2.18 bpw (G=0.58)", "#7570b3", 0.58),
    ("EXL3 1.66 bpw (G=0.44)", "#1b9e77", 0.44),
]
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="monospace" font-size="12">',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{ML}" y="24" font-weight="bold">PS5LM streaming ledger: tok/s vs cache hit rate (store 5.158 GiB/s)</text>',
]
for t in range(0, int(MAXY) + 1, 15):
    parts.append(f'<line x1="{ML}" y1="{py(t)}" x2="{W-30}" y2="{py(t)}" stroke="#eee"/>')
    parts.append(f'<text x="{ML-38}" y="{py(t)+4}">{t:>3}</text>')
for h in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
    parts.append(f'<text x="{px(h)-10}" y="{H-MB+18}">{h:.1f}</text>')
parts.append(f'<line x1="{ML}" y1="{py(0)}" x2="{W-30}" y2="{py(0)}" stroke="#999"/>')
parts.append(f'<line x1="{ML}" y1="40" x2="{ML}" y2="{py(0)}" stroke="#999"/>')
parts.append(f'<text x="{W//2-40}" y="{H-12}">cache hit rate</text>')
parts.append(f'<text x="14" y="{H//2}" transform="rotate(-90 14 {H//2})">tok/s</text>')
# dead-zone line: the 40-60 headline needs h=0.896-0.931
parts.append(f'<line x1="{px(0.896)}" y1="40" x2="{px(0.896)}" y2="{py(0)}" stroke="#cc0000" stroke-dasharray="4 3"/>')
parts.append(f'<text x="{px(0.896)-118}" y="52" fill="#cc0000">40 tok/s needs h=0.896</text>')
for label, color, g in series:
    pts = " ".join(f"{px(h)},{py(min(ledger.tok_s(ledger.BW_REFERENCE_GIB_S, g, h), MAXY))}"
                   for h in [i / 100 for i in range(0, 96, 2)])
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
ly = 74
for label, color, g in series:
    parts.append(f'<line x1="{W-250}" y1="{ly}" x2="{W-230}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
    parts.append(f'<text x="{W-224}" y="{ly+4}">{label}</text>')
    ly += 18
parts.append("</svg>")
(OUT / "poc-band-chart.svg").write_text("\n".join(parts), encoding="utf-8")
print("wrote docs/evidence/poc-band-chart.svg")

# --- 2. Gate report from a real CLI run (2.05 bpw pack, 2000-token trace) -----
def run(*args):
    r = subprocess.run([sys.executable, "-m", "engine", *args], capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stderr
    return r.stdout

lines = ["# PoC evidence — engine end-to-end run (2026-10-26)", "",
         "All commands: `python -m engine …` from a fresh clone; suite: `python -m unittest discover -s engine/tests` (16 tests).", "",
         "## 1. Ledger invariants (S3 re-derivation enforced by tests)", "", "```", run("ledger").strip(), "```", ""]
run("pack", "build", "--bpw", "2.05", "--out", str(OUT / "_tmp_pack.json"))
lines += ["## 2. Pack manifest (EXL3-class, 2.05 bpw)", "", "```",
          "$ python -m engine pack build --bpw 2.05 --out pack.json",
          f"pack: {json.load(open(OUT / '_tmp_pack.json'))['pack_bytes'] / 1024**3:.1f} GiB, "
          f"G={json.load(open(OUT / '_tmp_pack.json'))['g_gib_per_token']:.3f} GiB/token", "```", ""]
run("trace", "gen", "--tokens", "2000", "--profile", "agent", "--out", str(OUT / "_tmp_trace.json"))
out = run("gate", "replay", "--pack", str(OUT / "_tmp_pack.json"), "--trace", str(OUT / "_tmp_trace.json"),
          "--cache-gib", "8", "--target", "20", "--report", str(OUT / "poc-gate-report.md"))
(OUT / "_tmp_trace.json").unlink()
lines += ["## 3. R2 acceptance gate (agent trace, 8 GiB cache = Linux-track budget, target 20 tok/s)", "", "```", out.strip(), "```"]
run("trace", "gen", "--tokens", "2000", "--profile", "uniform", "--seed", "11",
    "--out", str(OUT / "_tmp_trace2.json"))
out2 = run("gate", "replay", "--pack", str(OUT / "_tmp_pack.json"), "--trace", str(OUT / "_tmp_trace2.json"),
           "--cache-gib", "0.25", "--target", "20", "--report", str(OUT / "poc-gate-uniform.md"))
(OUT / "_tmp_trace2.json").unlink()
(OUT / "_tmp_pack.json").unlink()
lines += ["## 4. Same pack, worst-case traffic AND a tight cache (0.25 GiB = PS5-OS-title-class) - the gate discriminates", "",
          "```", out2.strip(), "```"]
(OUT / "poc-gate-run.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote docs/evidence/poc-gate-run.md")
