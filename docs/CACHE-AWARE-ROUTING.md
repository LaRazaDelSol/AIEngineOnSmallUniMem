# Cache-aware routing (S24, 2026-11-09)

Decode on the console was bound by the one SSD: every token needs the experts its routers pick, about one in nine
is not in memory, and the drive was already saturated (S23). A second drive is not available. This document is the
lever that removes the wall without one: let the router prefer experts that are already in memory when the choice
is a near-tie.

It has two parts with different guarantees:

| flag | what it changes | output |
|---|---|---|
| `--route-mtp B` | the MTP **draft** layer only | **unchanged** (the verify pass is exact; md5 `18179c0e1f51` PASS) |
| `--route-bias B` | the 48 trunk layers | **changes** (lossy; quality measured below) |

## 1. Mechanism

The model routes each token, per layer, to the 10 experts with the highest router logits out of 512 and weights
them by a softmax over those 10 (`norm_topk_prob`). In this engine the host is already in that loop: the GPU's
top-k kernel writes ids and weights to a mailbox, the host binds slots and sends the slot table and the weights
back. So the whole feature is host code plus one changed constant:

1. `topk.comp` is dispatched with `k = 16` instead of 10 (`kr(l)` in `flash_engine.hpp`): the mailbox row already
   had room for 16 ids and 16 weights.
2. `route_select()` scores each candidate `log(w) + B` if its expert is **in the pool or already being read**, `log(w)` otherwise,
   keeps the 10 best in rank order and renormalises the weights over the kept set. `B` is in logit units: an
   expert outside the true top 10 replaces a missing one only if its logit is within `B` of it.
3. Everything downstream (binding, flags A/B, kernels) is unchanged. With both flags at 0 the dispatch constant
   is 10 and no code path differs, so the exact build stays bit-identical.

Details that matter:

- **In flight counts as resident** (default). A prefetch that was already issued is sunk cost, so it earns the
  bonus. The alternative (`--route-ready`: only landed experts earn it) was measured and is not better: 56.8 tok/s
  at 5.0% displaced weight versus 55.7-57.8 at 2.4-3.0%.
- **Batches.** In a verify batch a later token counts the experts earlier tokens of the batch picked in this layer
  as resident (their reads are issued anyway).
- **Prefill.** Without `--route-pf` the prompt is routed exactly (the true top 10; only the weights are
  renormalised on the host). With it, the token-group path and the wide layer-major path are biased too; inside a
  chunk an expert an earlier token requested counts as resident.
- **The prefetch predictor stays on.** It is what keeps the pool following the true routing. Turning it down is
  the fastest way to lose quality (section 2.2).
- `--route-keep R` never displaces the top `R` ranks; `--route-cand C` limits the candidates (10..16);
  `--route-fetch N` reads displaced experts in the background while fewer than `N` reads are in flight (measured:
  slower, no fidelity gain, off).

## 2. Measurements (console, 2026-11-09, EXL3-2.68 pack, serving flags, cold pool, one run each)

### 2.1 Speed

`scripts/console/route_matrix.sh`; tok/s of the generation phase, expert reads per token in brackets.

| prompt | exact (S23 flags) | `--route-mtp 1` (exact) | + `--route-bias 0.5` | + `--spec 3 --pld 4` |
|---|---|---|---|---|
| reference C++ prompt, 300 tokens | 45.50 (47.1) | 45.93 (46.0) | **57.63** (31.8) | 58.02 (38.2) |
| prose essay, 256 tokens | 41.67 (57.1) | 42.47 (55.7) | **52.81** (39.4) | 51.25 (42.1) |
| copy-heavy code edit, 500 tokens after 784 | 40.44 (69.8) | 40.78 (69.1) | **57.67** (41.9) | 59.70 (51.5) |

The copy-heavy edit produced the same text with and without the bias (md5 `5f515f20c1e7` in all four columns).
Served through the HTTP front end (`scripts/console/serve_probe.py`, thinking on, 250-300 tokens, engine-side
generation rate): prose 50-52, code 53-55, Q&A 46-50, JSON 51-53 tok/s (S23 served prose: 34.7). Wall rate of a
whole short request including prefill and the reasoning tokens: 38-41 tok/s.

Where the time went on the reference prompt, per verify cycle: gate B (GPU waiting for expert reads) 12.4 ms to
4.6 ms, demand misses 35.8 to 3.6, SSD busy 66% to 57%. GPU work is unchanged (24.9 ms), so the GPU is the bound
again and the S23 conclusion "nothing helps until bytes per second go up" no longer holds with the bias on.

Sweeps on the reference prompt (tok/s, displaced weight share):

| bias | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| prefetch top-8 (default) | 55.2, 1.9% | 57.8, 2.4% | 55.8, 3.0% | 56.1, 3.2% |

| bias 1, prefetch top-N | 0 | 2 | 4 | 8 |
|---|---|---|---|---|
| tok/s, displaced | 55.5, 8.8% | 58.5, 7.7% | 60.8, 6.3% | 55.8, 3.0% |

With bias 1 and prefetch top-4 the drive is idle enough for deeper drafts to pay: `--spec 3` 64.6 tok/s on the
reference prompt, `--spec 3 --pld 4` 73.9 tok/s on the copy-heavy edit (exact: 40.4). That setting costs
measurable quality (next table), so it is not the serving default.

### 2.2 Quality

`scripts/console/score.sh` (the S18 instrument): 7 documents, 3,532 teacher-forced positions, scored against the
unmodified W4A16 reference model's top-20 (`top-1`, `KL20`, `ppl`), and `runtime/tools/s24/score_pair.py` against
the same engine with exact routing (`agree`, `KL vs exact`, `dNLL`; the quantisation noise cancels there).
Scoring is token-serial and resets per document, so the pool is colder than in decoding and about twice as much
weight is displaced as in section 2.1: these rows are the harsher case.

| routing | top-1 | KL20 | ppl | displaced weight | agree with exact | KL vs exact | dNLL (nats) | reads/token |
|---|---|---|---|---|---|---|---|---|
| exact | 0.9323 | 0.1532 | 1.630 | 0 | 1 | 0 | 0 | 92.6 |
| bias 0.25 | 0.9318 | 0.1585 | 1.641 | 4.1% | 0.9725 | 0.016 | +0.006 | 67.2 |
| **bias 0.5** | **0.9335** | **0.1542** | **1.628** | 5.4% | 0.9672 | 0.021 | -0.001 | 61.6 |
| bias 1 | 0.9312 | 0.1506 | 1.627 | 6.0% | 0.9683 | 0.020 | -0.002 | 58.9 |
| bias 2 | 0.9340 | 0.1559 | 1.632 | 6.3% | 0.9652 | 0.025 | +0.001 | 58.5 |
| bias 8 | 0.9326 | 0.1554 | 1.636 | 6.3% | 0.9643 | 0.024 | +0.004 | 58.2 |
| bias 1, prefetch top-4 | 0.9270 | 0.1635 | 1.648 | 9.2% | 0.9564 | 0.034 | +0.011 | 29.1 |
| bias 8, prefetch off | 0.9173 | 0.2086 | 1.690 | 15.3% | 0.9369 | 0.084 | +0.036 | 15.8 |

Long documents (2 x 6,000 tokens, `--max-ctx 8192 --kv q8`): exact 0.9642 / 0.1191 / 1.192; bias 1 with prefetch
top-4 0.9641 / 0.1232 / 1.199.

Reading:

- With the default prefetch, a bias anywhere from 0.25 to 8 is inside the noise of this instrument against the
  reference (top-1 within 0.002, perplexity within 0.7%). Against exact routing the output distribution moves by
  0.02 nats of KL, about one eighth of what the 2.68 bpw quantisation itself moves it (0.153), and the argmax
  changes on 3% of positions.
- Quality follows the **displaced weight share**, not the bias: up to about 6% nothing is measurable, 9% costs 1%
  of perplexity, 15% costs 3.7%. The predictor's prefetch is what keeps it low, which is why the serving setting
  keeps the prefetch at top-8 and takes the smaller speed gain.
- Limits of this evidence: one instrument, 3,532 + 12,000 positions, no task benchmark (the pack's GPQA-Diamond
  number was measured by its publisher with exact routing). Exact routing is deterministic; biased routing depends
  on which reads have been issued, so two runs of the same prompt can differ.

Prefill (`--route-pf`), long documents with the first 3,000 tokens of each through the prefill path and the next
3,000 scored (5,998 positions, `--max-ctx 8192 --kv q8 --score-prefill 3000`):

| routing | top-1 | KL20 | ppl | reads/token |
|---|---|---|---|---|
| exact | 0.9592 | 0.1364 | 1.232 | 111.8 |
| bias 0.5, prompt exact | 0.9587 | 0.1369 | 1.232 | 76.8 |
| bias 0.5 + `--route-pf` | 0.9588 | 0.1365 | 1.232 | 69.9 |

Prefill speed with `--route-pf`: 784-token prompt on a cold pool 133.5 to 176.7 tok/s; served short prompts
(75-130 tokens, warm pool) 1.6-2.0 s to 1.2-1.6 s before the first token.

The 3.18 bpw pack (`--pack experts_exl3.pack`, the Q4-class one) with the same flags:

| pack, routing | tok/s (reference prompt) | top-1 | KL20 | ppl |
|---|---|---|---|---|
| 2.68, exact | 45.5 | 0.9323 | 0.1532 | 1.630 |
| 2.68, bias 0.5 (served) | 57.6 | 0.9335 | 0.1542 | 1.628 |
| 3.18, exact | 33.4 | 0.9352 | 0.1347 | 1.594 |
| 3.18, bias 0.5 | 47.1 | 0.9363 | 0.1409 | 1.611 |

So 3.18 with the bias is both faster and closer to the reference than 2.68 with exact routing. It is the setting
to serve if quality matters more than the last 10 tok/s (remove `--pack` in `server.env`).

### 2.3 What did not work

| variant | result |
|---|---|
| `--route-ready` (only landed experts earn the bonus) | 56.8 tok/s, 5.0% displaced: no faster, more displaced |
| `--route-fetch 8` (background reads of displaced experts) | 53.1 tok/s, 4.4% displaced: the reads compete with the kept misses |
| `--route-fetch 8 --route-keep 2` | 49.6 tok/s |
| `--route-mtp 3 --mtp-lossy` (drafts never wait) | 44.6 tok/s, acceptance 0.848: worse than `--route-mtp 1` |
| prefetch off | fastest drive relief, clear quality loss (table above) |

## 3. Use

```
# exact output, about +1%: the draft layer only
--route-mtp 1
# serving setting since S24 (lossy within the noise of the instrument): /etc/ps5lm/server.env
--route-mtp 1 --route-bias 0.5 --route-pf
# back to exact routing: remove --route-bias (and --route-pf)
```

Check a setting before serving it: `bash scripts/console/score.sh TAG --route-bias B ...` (2 minutes), then
`python3 runtime/tools/s24/score_pair.py sc_exact.txt sc_TAG.txt`.

## 4. S35 audit of the served setting (2026-11-13, `scripts/console/route_audit.sh`)

Served pack (3.18), `--kv k4v4`, current build, one run each. Quality = S18 instrument; speed = smoke prompts with the
served decode flags (`--spec 3 --pld 4`), generation phase.

| routing | top-1 | KL20 | ppl | displaced (scoring) | ref tok/s | prose tok/s |
|---|---|---|---|---|---|---|
| exact (`--route-mtp 1`) | 0.9357 | 0.1403 | 1.600 | 0 | 33.2 | 28.9 |
| **served**: bias 1, prefetch 4, `--route-pf`, tier | 0.9304 | 0.1521 | 1.622 | 9.9% | 53.3 | 49.9 |
| served without the tier | 0.9301 | 0.1557 | 1.625 | 10.1% | - | - |
| served + `--route-keep 3` | 0.9309 | 0.1500 | 1.622 | 8.9% | 54.0 | 48.3 |
| bias 0.5, prefetch 4 | 0.9306 | 0.1477 | 1.615 | 8.1% | 49.5 | 45.2 |
| bias 1, prefetch 6 | 0.9306 | 0.1514 | 1.615 | 8.2% | 52.3 | 49.3 |

- The served setting costs +8% KL20 and +1.4% ppl against exact routing (paired vs exact: dNLL +0.014 +- 0.006 nats,
  argmax changes on 4% of positions). Still about level with 2.68 + exact routing (0.1532 / 1.630).
- The loss is concentrated on uncertain positions: exact top-1 p >= 0.9 (83% of tokens) is untouched (0.2% flips);
  p < 0.6 (8%) flips 28-54% and KL doubles vs bias 0.5. Worst document: the hard C++ one, ppl 7.44 -> 7.97.
- The tier does not add damage. Gentler settings recover at most ~0.004 KL / 0.007 ppl, inside single-run noise.
- Displaced experts get no LFU use (`find()` only); their resident substitutes do, so the pool learns the biased
  routing. With a 36-token half-life this is a fast equilibrium, not a drift: 6k-token documents displace 9.0%,
  the same as short ones. Untested lever: count a displaced expert as a fractional (shadow) LFU use.

### 4.1 Shadow uses for displaced experts (`--route-shadow F`, `--route-shadow-fetch T`): measured, no gain

A displaced true top-K expert earns an LFU use of F x (its weight share x K) without a read; with T it is also read in
the background once its decayed count reaches T. `scripts/console/shadow_audit.sh`, served flags + the variant:

| variant | top-1 | KL20 | ppl | displaced (scoring) | KL vs exact (paired) | ref / prose tok/s | displaced (decode) |
|---|---|---|---|---|---|---|---|
| served (this run) | 0.9289 | 0.1586 | 1.636 | 9.9% | 0.041 | 56.0 / 50.2 | 7.6% |
| served (route_audit run) | 0.9304 | 0.1521 | 1.622 | 9.9% | 0.037 | 53.3 / 49.9 | 7.6% |
| shadow 0.5 | 0.9315 | 0.1530 | 1.622 | 9.8% | 0.040 | 54.6 / 50.1 | 7.6% |
| shadow 1 | 0.9312 | 0.1524 | 1.626 | 9.8% | 0.041 | 56.6 / 51.0 | 7.5% |
| shadow 1, fetch at 2 | 0.9318 | 0.1528 | 1.630 | 9.0% | 0.043 | 55.8 / 50.3 | 6.8% |
| shadow 1, fetch at 1 | 0.9287 | 0.1546 | 1.631 | 8.1% | 0.039 | 52.2 / 44.3 | 5.9-6.3% |

The two served runs differ by 0.004 KL vs exact / 0.006 KL20: every variant is inside that. Shadow uses alone do not
change what is displaced (an LFU score only matters when the expert is admitted, and displaced experts rarely are);
shadow fetch lowers displacement by 1-2 points but the quality instrument cannot see it, and T=1 costs 7-12% speed.
Cause: the pool holds ~23% of the 24,576 trunk experts; displacement comes from capacity, not from eviction order.
Flags stay in the binary, off by default (exact smoke md5 18179c0e1f51 PASS with them off).
