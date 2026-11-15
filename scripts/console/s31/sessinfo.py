"""List stored conversations: tokens, resume points, tag (format fingerprint). python3 sessinfo.py STORE_DIR"""
import os, struct, sys
d = sys.argv[1]
for s in sorted(os.listdir(d)):
    m = os.path.join(d, s, "meta")
    if not os.path.isfile(m): continue
    b = open(m, "rb").read()
    fl = struct.unpack_from("<I", b, 8)[0]
    tag = b[12:12 + fl].decode(errors="replace")
    q = 12 + fl + 8
    nt, nc = struct.unpack_from("<II", b, q)
    ck = struct.unpack_from(f"<{nc}I", b, q + 8)
    print(f"{s}: {nt} tokens, {nc} resume points (last {ck[-1] if ck else 0}), tag {tag[:60]}")
