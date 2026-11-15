#!/usr/bin/env python3
"""Decode a session-store conversation to text: python3 sess_text.py DIR [model_dir]
meta: "PS5LMSS1" u32 fp_len fp u64 last_use u32 n_tok u32 n_ck u32 ck[n_ck] u32 toks[n_tok]"""
import struct, sys, os
from tokenizers import Tokenizer

d = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/ps5lm/model")
b = open(os.path.join(d, "meta"), "rb").read()
assert b[:8] == b"PS5LMSS1", b[:8]
o = 8
fl, = struct.unpack_from("<I", b, o); o += 4
fp = b[o:o + fl].decode(errors="replace"); o += fl
o += 8
n, nck = struct.unpack_from("<II", b, o); o += 8
ck = struct.unpack_from(f"<{nck}I", b, o); o += 4 * nck
toks = struct.unpack_from(f"<{n}I", b, o)
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
sys.stdout.write(f"### {d}: {n} tokens, checkpoints {list(ck)}\n")
sys.stdout.write(tok.decode(list(toks), skip_special_tokens=False))
sys.stdout.write("\n")
