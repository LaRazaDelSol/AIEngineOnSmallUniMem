"""Text in -> flash_decode on the PS5 GPU -> text out (greedy).

    sudo python3 run_text.py --model ~/ps5lm/model --bin ~/ps5lm/build/flash_decode --shaders ~/ps5lm/build/shaders \
        --prompt "The capital of France is" [--gen 32] [--pool-gib 8] [--chat]
--chat wraps the prompt in the model's chat template (thinking disabled) via tokenizer_config/chat_template.jinja
when available; otherwise raw completion.
"""
import argparse
import os
import subprocess
import sys
import time

from tokenizers import Tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--shaders", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--pool-gib", type=float, default=8)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra flash_decode args after --")
    a = ap.parse_args()
    md = os.path.expanduser(a.model)
    tok = Tokenizer.from_file(os.path.join(md, "tokenizer.json"))
    text = a.prompt
    if a.chat:
        text = f"<|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(text).ids
    print(f"prompt ({len(ids)} tokens): {text!r}", flush=True)
    t0 = time.time()
    p = subprocess.Popen([a.bin, "--model", md, "--shaders", a.shaders, "--ids", ",".join(map(str, ids)),
                          "--gen", str(a.gen), "--pool-gib", str(a.pool_gib)] + a.extra, stdout=subprocess.PIPE, text=True)
    out_ids: list[int] = []
    buf = ""
    sys.stdout.write("output: ")
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        if ch in ",\n":
            if buf:
                out_ids.append(int(buf))
                buf = ""
                sys.stdout.write("\r" + "output: " + tok.decode(out_ids))
                sys.stdout.flush()
        else:
            buf += ch
    p.wait()
    print(f"\n[{len(out_ids)} tokens in {time.time() - t0:.1f}s wall incl. load] rc={p.returncode}")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
