"""Render a chat through the model's template (as ps5lm_server does) and print the ids, comma-separated.
   python3 render_ids.py EFFORT|none "question"   (decode with --decode "ids")"""
import json
import sys

import jinja2
from tokenizers import Tokenizer

M = "/home/steam/ps5lm/model"
tk = Tokenizer.from_file(M + "/tokenizer.json")
if sys.argv[1] == "--decode":
    print(tk.decode([int(x) for x in sys.argv[2].replace("\n", ",").split(",") if x.strip()], skip_special_tokens=False))
    sys.exit()
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
t = env.from_string(open(M + "/chat_template.jinja", encoding="utf-8").read())
kw = {"enable_thinking": sys.argv[1] != "none"}
if sys.argv[1] not in ("none", "default"):
    kw["reasoning_effort"] = sys.argv[1]
text = t.render(messages=[{"role": "user", "content": sys.argv[2]}], add_generation_prompt=True, **kw)
print(",".join(map(str, tk.encode(text, add_special_tokens=False).ids)))
