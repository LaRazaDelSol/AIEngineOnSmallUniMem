# S33: where long context still costs, and what was removed (2026-11-11)

Measured on a stored 194,474-token agent conversation (s000009's tokens prefilled once under S32 k4v4 into
`~/ps5lm/s33/store`, 676 s), served flags unless noted. Scripts: `~/ps5lm/s33/{lc_probe.py,s33_ab.py,pf8k.py,anat.sh}`.

## Before S33: the long-context penalty, by part

| | 4k context | 197k context | penalty |
|---|---|---|---|
| decode, served flags (verify cycle) | 54.5 ms | 65.7 ms | +20%: GPU +5.9 ms, expert I/O +6 ms (demand misses 55 -> 67 per cycle: the KV holds 1.5 GB of the pool) |
| decode, exact routing (verify cycle) | 113 ms | 125 ms | +10% |
| prefill, 8,192-token chunk | 345 tok/s | 233 tok/s | -32% |
| server TTFT per request (Python) | ~0 | 0.40 s | the whole conversation re-tokenised every request |

The served dashboard's 14-17 tok/s at 140-190k (vs 41-45 short) was q8 KV (S30b): 3.9 GB of pool at 194k. With k4v4
(S32) the decode gap is the +20% above.

GPU profile of one attention layer at 197k, T = 2 verify (`PS5LM_VPROF=1`): indexer scores 158 us (11 at 4k), top-512
selection 115 us (8 at 4k), sparse attention 77 us (same at both). x 12 layers + 3 MTP drafts per cycle.

## Changes (all exact: tokens identical to build14 in every phase, smoke PASS 18179c0e1f51)

1. **Radix selection** (`idx_select.comp`, `-DRADIX`): the 512th largest score bits by three histogram digits
   (bits 30..20, 19..9, 8..0) instead of a 31-step bisection; then one coalesced pass collects the blocks above /
   at the threshold into LDS and each entry's output slot is its rank by block index (the ordered compaction's output);
   more than 1,024 ties at the threshold fall back to the ordered compaction. 115 -> 64 us per layer at 197k.
   `PS5LM_SEL_BISECT=1` keeps the bisection.
2. **Decode scores one thread per block** (`idx_score_d.comp`): 158 -> 144 us. The keys live in elastic KV pages
   (hugetlb, 64 KiB sparse pages); ~87 GB/s is what this access gets, LDS staging did not do better. `PS5LM_SCORE_OLD=1`.
3. **Tiled wide-prefill scores** (`idx_score_w.comp` + `idx_select_wps`): idx_select_w scored all pooled keys inside
   each query's workgroup (12 MB per token per layer at 194k); now each key is read once per 32 queries.
   8,192-token chunk at 197k **233 -> 294 tok/s**; 2,048-token append 10.6 -> 8.8 s. 64 queries per pass measured the
   same (the indexer is no longer the limit). `PS5LM_WIDX_OLD=1`.
4. **Incremental tokenisation** (`ps5lm_server.py` EncCache): the ids of the longest prefix shared with a recent
   prompt, cut at an `<|im_start|>` (added special tokens are split before pre-tokenising, so encoding is additive
   there), plus the prompt splice before any full encode. Agent append at 194k: 0.38 -> 0.03 s. Verified identical to
   a full encode on appends, a mid-conversation prune, random edits and a literal `<|im_start|>` in user text.
5. `flash_decode --ids @FILE`.

After: decode GPU per cycle at 197k is +3.4 ms over 4k (was +5.9); prefill penalty -15% (was -32%).

## What is left

- **Expert I/O from the KV's pool share** (~6 ms per verify cycle at 197k, the larger half of the decode penalty,
  and most of the remaining prefill gap). The lever is QSA tiering (KV-FOOTPRINT-RESEARCH.md §5); S33 measured its
  costs: an exact version stalls once per attention layer on the misses (~22 random 4 KB reads, ~100 us at qd 22),
  and every prefill must re-expand (1.5 GB at 5.8 GB/s with the S33 direct KV load = 0.26 s). Slot return now exists
  (feat/s33-efficiency `ek_reclaim`); still needed: page defragmentation, the block -> slot attention path and
  the store fetch. Estimated net at 197k: +6-8% decode, ~0 below 64k.
- **Small appends**: a 26-token append at 197k takes 1.5 s, almost all expert misses for tokens unlike the recent
  context (hit 0.70); not a context-length cost.
- Decode scores (1.7 ms per cycle at 197k) are bound by the pages' bandwidth; int8 pooled keys would halve it but
  can change the selection at near-ties.
