"""Prompt -> comma-separated token ids in the chat format the engine's reference runs use (thinking off).

    python3 tok.py "prompt text"          # model dir: $PS5LM_MODEL or ~/ps5lm/model
"""
import os
import sys

from tokenizers import Tokenizer

model = os.environ.get("PS5LM_MODEL", os.path.expanduser("~/ps5lm/model"))
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
p = sys.argv[1]
text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
print(",".join(map(str, tok.encode(text).ids)))
