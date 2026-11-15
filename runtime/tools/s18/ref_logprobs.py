"""S18 quality instrument, step 2 (runs next to the reference vLLM server): teacher-forced top-20
next-token log-probabilities of every position of every document (vLLM `prompt_logprobs`).
    python3 ref_logprobs.py --url http://127.0.0.1:8001 --ids docs.ids --out ref.txt
ref.txt lines: "doc pos id1,...,id20 lp1,...,lp20 lp_actual"   (pos = the token whose NEXT token is predicted)
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8001")
ap.add_argument("--ids", default="docs.ids")
ap.add_argument("--out", default="ref.txt")
ap.add_argument("--model", default=None)
a = ap.parse_args()
model = a.model or json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
docs = [list(map(int, l.split(","))) for l in open(a.ids) if l.strip()]
with open(a.out, "w") as f:
    for d, ids in enumerate(docs):
        body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20}
        req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
        # pl[i] = logprobs of token i given tokens < i (pl[0] is None) -> position i-1 predicts token i
        for i in range(1, len(ids)):
            ent = pl[i] or {}
            items = sorted(((int(k), v["logprob"], v.get("rank", 99)) for k, v in ent.items()), key=lambda x: x[2])
            top = [x for x in items if x[2] <= 20][:20]
            act = ent.get(str(ids[i]), {}).get("logprob")
            f.write(f"{d} {i - 1} {','.join(str(x[0]) for x in top)} {','.join(f'{x[1]:.6f}' for x in top)} {act if act is not None else 'nan'}\n")
        print("doc", d, "positions", len(ids) - 1, flush=True)
