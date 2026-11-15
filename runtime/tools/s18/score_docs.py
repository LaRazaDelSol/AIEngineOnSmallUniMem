"""S18 quality instrument, step 1: tokenize fixed HUMAN-written texts (never model output, which would favour the
model that generated it) into the ids file `flash_decode --score` and ref_logprobs.py both read.
    python3 score_docs.py --tokenizer ~/ps5lm/model/tokenizer.json --out docs.ids [--tokens 512]
"""
import argparse, glob, os, sys
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", default=os.path.expanduser("~/ps5lm/model/tokenizer.json"))
ap.add_argument("--out", default="docs.ids")
ap.add_argument("--tokens", type=int, default=512)
a = ap.parse_args()
py = sorted(glob.glob("/usr/lib/python3*/"))[-1]
SOURCES = [  # (label, path, start offset in characters)
    ("gpl3-legal-prose", "/usr/share/licenses/common/GPL3/license.txt", 2000),
    ("heapq-essay", py + "heapq.py", 0),
    ("textwrap-code", py + "textwrap.py", 0),
    ("json-docstring", py + "json/__init__.py", 0),
    ("argparse-code", py + "argparse.py", 4000),
    ("statistics-docs", py + "statistics.py", 0),
    ("cpp-slot-cache", os.path.expanduser("~/ps5lm/src/runtime/include/ps5lm/slot_cache.hpp"), 0),
    ("email-prose", py + "email/__init__.py", 0),
]
tok = Tokenizer.from_file(a.tokenizer)
with open(a.out, "w") as f, open(a.out + ".labels", "w") as lab:
    for name, path, off in SOURCES:
        if not os.path.exists(path):
            print("skip (missing):", path, file=sys.stderr)
            continue
        text = open(path, encoding="utf-8", errors="replace").read()[off:off + 12000]
        ids = tok.encode(text).ids[: a.tokens]
        f.write(",".join(map(str, ids)) + "\n")
        lab.write(f"{name}\t{path}\t{len(ids)}\n")
        print(name, len(ids), file=sys.stderr)
