"""Convert a ps5lm-trace/1 JSON (engine/trace.py, or a real routing dump) to the
flat text format the C++ runtime reads (runtime/include/ps5lm/trace_txt.hpp).

    py runtime/tools/trace_to_txt.py trace.json trace.txt

Format (ps5lm-trace-txt/1):
    line 1:  ps5lm-trace-txt/1 <tokens> <layers> <experts>
    then tokens*layers lines, token-major:  <k> <e1> ... <ek>
"""
from __future__ import annotations

import json
import sys


def convert(src: str, dst: str) -> None:
    with open(src, encoding="utf-8") as f:
        t = json.load(f)
    if t.get("schema") != "ps5lm-trace/1":
        raise SystemExit(f"{src}: not a ps5lm-trace/1 file")
    routing = t["routing"]
    layers = t["moe_layers"]
    with open(dst, "w", encoding="ascii", newline="\n") as out:
        out.write(f"ps5lm-trace-txt/1 {len(routing)} {layers} {t['n_experts']}\n")
        for ti, tok in enumerate(routing):
            if len(tok) != layers:
                raise SystemExit(f"token {ti}: {len(tok)} layers, expected {layers}")
            for ids in tok:
                out.write(f"{len(ids)} {' '.join(map(str, ids))}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2])
