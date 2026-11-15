# How small can a position get? (research 2026-11-10, built and served S32 2026-11-11)

## S32 result (served)

`--kv k4v4` + fp16 pooled keys: **15,808 -> 7,696 B per position** (128k context: 2.07 -> 1.01 GB of pool; the SSD
store and its load time halve too). K and V are both rotated by the 256-point Walsh-Hadamard (the query gets the same
rotation in `attn_prep`, so q.k is unchanged) and stored as 16-level uniform midrise codes with step 0.3352 x the
row's RMS: the MSE-optimal uniform step for Gaussian data, which rotated rows are (TurboQuant's argument, without its
non-uniform codebook, so the decode is one FMA per element as before). Row 132 B (32 words + fp32 step).

| KV (S18 instrument, 3,532 positions vs W4A16) | B/position | top-1 | KL20 | ppl |
|---|---|---|---|---|
| q8 (served until S32) | 15,808 | 0.9346 | 0.1532 | 1.633 |
| k8v4 (S20, absmax int4 V) | 12,480 | 0.9309 | 0.1590 | 1.634 |
| **k4v4 (S32)** | **7,696** | 0.9332 | 0.1548 | 1.628 |

Smoke (default flags) PASS `18179c0e1f51`. Long-context quality not measured here; the vLLM nvfp4 KV run on this
model (4-bit, block scales) reported needle 70/70 from 65k to 235k and ppl +0.4-0.6% (vLLM issue 54426).
Pooled keys fp16: SGLang stores them BF16; in-batch pooled keys are rounded to the same fp16 bits, so a block scores
identically whether it was pooled in the batch or read from the cache.

Question: DeepSeek-V4.1-Flash keeps 890 B per token of context; we keep 15,808. How far down can we
go with the model we have (no retraining), and do DSpine or other drafters help?

## 1. Where our 15,808 B go (S30b, measured from `flash_engine.hpp`)

13 layers own a cache (12 attention layers + the MTP layer), 2 KV heads of 256 dims each.

| part | per layer | x13 | share |
|---|---|---|---|
| K, int8 + scales (272 B per head) | 544 | 7,072 | 45% |
| V, int8 + scales (272 B per head) | 544 | 7,072 | 45% |
| pooled indexer keys, fp32, 128 dims per 4-token block | 128 | 1,664 | 10% |
| raw indexer keys | ring since S30b | 0 | - |

At 128k that is 2.07 GB taken out of the expert pool (about 11% of decode speed at 3% per 0.5 GiB); at 262k, 4.1 GB.
The pooled keys are also read in full for every token (24 MB per layer at 190k), so their size is an indexer-time
cost as well as a memory one.

## 2. Where DeepSeek's 890 B come from (arXiv 2609.19969, read in full HTML)

- **Most of it is architecture, learned in pretraining.** Only 4 of 40 layers own a main KV ("Full"); 4 "Reindex"
  layers reuse the KV and rescore it; 30 "Reuse" layers reuse the KV and the top-k indices too. K and V are one 512-d
  MLA latent, not separate K and V. The encoder half merges 2 tokens per entry (m = 2).
- **The rest is FP4 (E2M1, one E4M3 scale per 16 channels), applied after RoPE, with QAT in post-training.** The
  indexer keys are FP4 with QAT too. The 128-token sliding-window cache stays FP8 and is not counted in the 890.
- No ablations, no training-free results. My reconstruction (3 encoder Full layers x 288 B / 2 + 1 decoder Full x
  288 B + FP4 indexer keys) lands at about 900 B, so the claim is consistent; the paper gives no breakdown.
- Persistent tier: the SSD keeps only the global KV (1/8 of V4's, 72 h retention); the window cache is not
  persisted and is replayed from the last 128 tokens on a hit.

**What carries over to us without training:** FP4-class storage and keeping indexer keys small. **What does not:**
cross-layer reuse, the shared K=V latent and token merging. Each needs the model to have been trained that way. The
closest training-free relative (cross-layer SVD of K/V, xKV-style) is research-grade here: our K is RMS-normed per
head and rotated after projection, so it is not a linear function of a shared latent. Not recommended first.

## 3. Training-free ladder for this model (per position, all 13 layers)

| step | B/position | 128k pool | risk | state |
|---|---|---|---|---|
| today (S30b) | 15,808 | 2.07 GB | - | served |
| A. pooled keys fp32 -> fp16 | 14,976 | 1.96 GB | near-ties only | **served S32** |
| A'. pooled keys int8, one scale per block | 14,573 | 1.91 GB | selection may change at near-ties | design |
| B. + V int4 rotated (k8v4) | 11,648 | 1.53 GB | KL +0.002-0.005 at 6k (R85); unmeasured past 6k | kernels exist (`*_kq4*`), `--kv q4` |
| C. + K int4 rotated (built as k4v4, 132 B rows) | 7,696 | 1.01 GB | measured: KL +0.0016 vs q8 | **served S32** |
| D. tiering on top of C: only pooled keys + the newest 4,096 positions + a hot set stay in the pool | ~2,200 effective at 128k, ~1,100 at 262k | ~0.29 GB | miss stalls and SSD contention with expert reads; depends on churn (unmeasured) | design |

Step D is the lever that matches DeepSeek's number where it matters (pool bytes):

- QSA reads at most 2,048 positions per query per layer (512 blocks). Everything else in the cache sits idle that token.
- The session store (S25) already writes every row to the SSD. Pages the pool evicts are clean: nothing to write back.
- A block of 4 positions of one layer is 4 x 576 B = 2.3 KB at step C, about one 4 KiB read.
- Budget check: if 5% of the 512 selected blocks change per token per layer, that is 330 reads (1.4 MB) per token,
  70 MB/s at 50 tok/s, about 1% of the drive. At 25% churn, 7 MB per token: that competes with the expert reads and D
  only pays above about 64k tokens.
- To stay exact, a token whose selection misses a page has to wait for it. Prefetching from the previous token's
  selection (QSA picks overlap heavily from one token to the next) hides most of that. Whether "most" is 90% or 99%
  decides whether D works.

## 4. DSpine (asked)

Correction: the 613-978 tok/s in Tencent's table is ONE GPU (DSpine supports `tp_size == 1` only), summed over
8 concurrent requests (`--max-running-requests 8`; "the 8 sessions"). GPU model not stated. "k11" is in no source.
Single stream: ~105 tok/s on one RTX 3090 (community llama.cpp port, Qwen3.8-27B dense).


DSpine (Tencent, arXiv 2609.36173) is a speculative-decoding drafter that drafts a 16-token block from five tapped
hidden states of the target model. It does not shrink the context. It adds its own windowed KV, and the GGUF card
says verification keeps one recurrent-state copy per drafted token (about 150 MB each there; ours is 118 MB, so a
16-token block would cost about 1.8 GB of pool). The only Flash-Next build
(`ProCreations/Qwen3.8-Flash-Next-DSpine-preview`) says on its own card that it is slower than the native MTP
(tau 1.82-1.87 on CPU, estimated 0.6-0.7x the MTP's throughput). Our MTP is at 0.80 draft acceptance at 20k (S30b).
Verdict: no. "k11" does not appear in any DSpine repo or card. Their names use `-b<N>` (block size).

## 5. Selection churn, measured (S32, `PS5LM_SELSTAT=1`, `scripts/console/selstat.py`)

The 122,880-token agent conversation s000010, prefilled, then 512 tokens decoded greedily with exact routing and no
drafts. For every block the indexer selects (512.8 per layer per token), how long ago it was last selected:

| window (newest 4,096) | previous token | <= 4 | <= 16 | <= 64 | <= 256 | older | never |
|---|---|---|---|---|---|---|---|
| 24.9% | 39.6% | 13.3% | 9.9% | 5.5% | 2.6% | 0.5% | 3.8% |

All 12 layers look alike (never 3.2-4.4%, window 15-36%). A hot set that keeps what was selected in the last 256 tokens
misses 4.3% of selections: about 22 blocks per layer per token, 262 per token, 0.55 MB of reads (9 MB/s at 17 tok/s).
The drive can do that without trouble. Tiering is viable on the access pattern.

**Why it is not built (S32):** the elastic KV takes its pages from retired expert slots and gives slots back only
when the whole context shrinks back to `kv_base` (`ek_shrink` -> `ek_return_all`); pages freed in between only go to
`ek_free_` for later KV growth. Compacting a 123k context after its prefill would therefore stop further slot loss but
return no slot to the expert pool. The decode would not get faster. The build needs, on top of the slot-table attention
(`attn_split`: rows below the cut through a per-layer block -> slot table, missing blocks skipped and fetched from the
session store's `kv` file after the token):
1. page defragmentation: copy the live pages (window + hot slots) into the pages of the slots retired first, rebind,
   and unretire the slots at the `ek_rb_` boundary;
2. re-expansion before any prefill (the prefill's queries select all over the context): re-retire, read rows
   `[0, cut)` back from the store;
3. ordering with the store (cut <= committed rows), snapshots (`kv_hist_`), the verify path's T columns, the MTP layer.
Payoff: about 0.8 GB of pool back at 123k (~5% decode), ~1.8 GB at 262k (~11%), nothing below ~16k tokens.

## 6. Next measurements, in order

1. **Selection churn** (decides D): instrument `idx_select` to count the selected blocks per layer that were not
   selected for the previous token, on a stored 120k and 194k conversation (`--ctx-load`, CLI, exact routing). Report
   the mean and p99 per layer, and the hit rate of "previous token's selection + newest 4,096" as a prefetch.
2. **A** (pooled keys fp16): an A/B with tokens identical on the 194k conversation. It also cuts indexer reads in half.
3. **B/C quality past 6k**: the 16k/32k documents against the reference server (OPEN-ITEMS 2.5 item 4), for
   q8, k8v4 and k4v4.

Sources: [DeepSeek-V4.1-Flash, arXiv 2609.19969](https://arxiv.org/abs/2609.19969);
[DSpine 27B card](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16);
[DSpine GGUF card](https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF);
[Flash-Next DSpine preview](https://huggingface.co/ProCreations/Qwen3.8-Flash-Next-DSpine-preview);
R85 in the development ledger.
