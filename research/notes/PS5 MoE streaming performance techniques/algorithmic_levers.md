# Algorithmic / model-side levers for streamed-MoE decode on a bandwidth-constrained console (PS5, 16 GB, SSD streaming)

> **Status (2026-10-21):** ACTIVE evidence base — numbers are claims; re-derive against the W4 recompute before citing (the parent report had ledger arithmetic errors, W1/FLOW kill list).

Target: Qwen3.8-Flash-Next (125B main + 51B N-gram table, 6B active, 512 routed experts + 1 shared/layer, top-10, 48 MoE layers = 36 GDN + 12 QSA attention layers, Gated Residual, MTP head; released 2026-08-26) ([LMSYS/SGLang blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/), [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next)). BF16 ≈ 354 GB ([Kaitchup GGUF benchmark](https://kaitchup.substack.com/p/qwen38-flash-next-gguf-benchmark)).

Baseline arithmetic used throughout (my derivation, consistent with cited measurements): routed expert ≈ (≈118B routed params)/(512×48) ≈ 4.8M params ≈ **2.7–3.0 MB at Q4-class**. Routed-expert bytes/token ≈ 48×10×2.8 MB ≈ **1.3–1.5 GB/token** at Q4 pure-streamed → ~4 tok/s at ~5.5 GB/s effective. Cross-check: community reports 20 tok/s at Q2 (≈0.7 GB/token) on a desktop with 32 GB RAM ([llama.cpp discussion #27149, Dampfinchen comment](https://github.com/ggml-org/llama.cpp/discussions/27149)) — consistent.

## Q1. MTP for streaming MoE

### Takeaway
MTP is the strongest single lever and the math works even for SSD-streamed experts, provided acceptance length AL ≥ ~2: unique-expert I/O for verifying k tokens grows sub-linearly in k (measured: verifying 4 tokens ≈ 2.5× one token's expert loads, not 4×, from temporal routing correlation). GPU-resident results on this exact model are strong (accept length 3.3 incl. bonus token; "30→80 tok/s" claims); flash-moe's "MTP = break-even" on Apple is explained by a draft implementation where I/O scaled ~linearly per token — accepting k tokens per verify with measured expert overlap changes the math to ~30–35% fewer expert-bytes/token. llama.cpp MTP is merged for some models (May 2026) but not yet for Flash-Next.

### Cited Findings
- Qwen3.8-Flash-Next ships an official **MTP module** supporting speculative decoding; native context 262,144 ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)).
- SGLang day-0: with MTP + "IndexShare" (QSA top-k selection frozen from the draft-extend pass and reused across all draft steps; accept length unchanged), NVFP4 checkpoint decodes **540 tok/s at batch 1** (TP4 B200) at **accept length 3.3 including the bonus token** → ≈2.3 accepted draft tokens + 1 verified/bonus per target forward. The MTP draft model is **one layer** (it disables the N-gram/PLE table; the target keeps PLE through prefill/decode/verification) ([LMSYS/SGLang blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)).
- Verifying K+1 tokens in an MoE activates only **~2.5× top-k unique experts**, not (K+1)×, because adjacent tokens share experts (38% overlap at distance 1, decaying to ~30% at distance 4; 2–4× above the independence baseline; stable across 13 Spec-Bench categories and 7 languages) ([Cohere engineering blog](https://www.cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding); underlying temporal-correlation documentation: [arXiv 2505.16056](https://arxiv.org/abs/2505.16056)).
- Measured target-side cost ratio on a real production MoE (k=8/N=128): T_t(4)/T_t(1) = **1.25× empirical** (1.46× by Amdahl model with routed-expert fraction f=0.30); observed SD speedup 1.95× at BS=1 with AL=2.73; draft cost ≈15% of one target decode ([Cohere](https://www.cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)). MoESD predicts the non-monotonic MoE speedup-vs-batch curve (sweet spot at moderate batch where verification is nearly free) ([MoESD](https://arxiv.org/pdf/2505.19645), cited and validated by Cohere).
- flash-moe (Qwen3.5-397B-A17B streamed from SSD on M3 Max, 4.36 tok/s): tested **MTP speculative decoding → break-even**, because "MoE I/O scales per-token (unlike dense)"; also "speculative early routing" −38%, temporal expert prediction −18% (25% hit), MLP routing predictor 31% accuracy, LZ4 expert compression −13% ([flash-moe README](https://github.com/danveloper/flash-moe)).
- RTX GPUs: "30 Tokens Per Second to 80: The MTP Feature That Doubles Qwen3.8-Flash-Next on RTX GPUs, and Why llama.cpp Still Won't Merge It" ([Medium](https://medium.com/@inamdaraditya/30-tokens-per-second-to-80-the-mtp-feature-that-doubles-qwen3-8-flash-next-19d85dcdaa74), fetch blocked/403 — title/claim only). A Reddit thread running Flash-Next at 65 tok/s on 12 GB VRAM describes "predicting multiple tokens at once and they are all accepted," "up to 50% more tokens" ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wp7zyb/qwen38flashnext_on_12gb_vram_65_tokens_per_second/)); MTP+MTPLX is credited for 60+ tok/s on M5 Max 128 GB ([YouTube](https://www.youtube.com/watch?v=jFqrXIjttE8)); Jetson AGX Thor: 46.7 tok/s ([NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-at-46-7-tok-s-on-jetson-agx-thor/383269)). The coordinator-reported "30–60%" megathread claim was not directly verified (megathread fetch failed) but is consistent with this set.
- llama.cpp MTP status: MTP support **merged 16 May 2026** for Qwen3.5-class models ("build from the MTP PR branch — has been merged 16th May 2026"; Qwen3.6-27B 22→42 tok/s with two extra flags) ([ModelScope unsloth MTP GGUF page](https://www.modelscope.cn/models/unsloth/Qwen3.5-9B-MTP-GGUF), [NVIDIA forum thread](https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298)); Qwen3-Next MTP PR **#25589** (yomaytk) ([Reddit mirror](https://www.reddit.com/r/LocalLLaMA/comments/1veca9y/model_mtp_support_for_qwen3next_by_yomaytk_pull/)); for Flash-Next: "llama.cpp support merged… **MTP support hasn't been merged yet**" ([Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1w03zdo/llamacpp_support_for_qwen38flashnext_has_been/)), and unsloth's Q4 GGUF "doesn't support MTP parameters using llama.cpp" ([unsloth GGUF discussion #21](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/21)); feature request issue [#27741](https://github.com/ggml-org/llama.cpp/issues/27741).
- SP-MoE (arXiv 2510.10302): SD-aware expert offloading — during the draft phase, feed the draft model's per-layer attention outputs into the *target's* gate networks to prefetch the verification's critical experts; expert loading is 69.4% of SD-MoE decode latency; neighboring draft tokens activate overlapping expert sets; top-1 expert prediction accuracy ~88–89% via cross-model gating; **1.07×–3.5× TPOT speedup** over SOTA offloading baselines; warns naive over-prefetch beyond ~3–5 layers causes thrashing ([arXiv](https://arxiv.org/html/2510.10302v2)).

### Inferences
- **Derivation — why MTP beats break-even on streamed MoE.** Let AL = accepted tokens/forward (3.3 for Flash-Next). Streaming cost/forward ≈ unique_experts(AL) × expert_bytes × 48. From Cohere's measured temporal overlap (≈0.3–0.38 adjacent-token expert overlap, which is correlation-driven and largely N-independent), unique_experts(k)/k at k=4 ≈ 2.5/4·(k ratio) → verified-k tokens cost ≈ 1 + 0.62(k−1) × one-token expert bytes. Speedup ≈ AL / [1 + 0.62(AL−1) + ε_draft]. For AL=3.3 with a cheap draft: 3.3/2.42 ≈ **1.36× (+36%)**; AL=2 → +23%; break-even at AL≈1.6. This lands in the community's claimed 30–60% band and explains flash-moe's null result: if their draft path re-streamed experts per draft step with ~1× scaling and modest accept length, net ≈ 1.0.
- flash-moe's per-token I/O framing understates the win: the sub-linearity is *empirical routing correlation*, which their own numbers (71% page-cache hit) prove exists on-streaming hardware too. The fix on a streaming engine is to **share the SSD/page-cache working set between draft steps and verification** (single expert store, content-addressed by (layer, expert id)), so a verification of 3.3 tokens touches ~2.4 tokens' worth of bytes.
- The one-layer MTP draft head has its own small MoE layer → its expert loads add ≈1/48 ≈ 2% per step — negligible; PLE is disabled in the draft head (per SGLang) so it adds no n-gram IO.
- On PS5: target verification is SSD-bound but the draft forward is GDDR-bound (fast); the SSD pipe can be kept busy during draft steps — exactly SP-MoE's "drafting stage provides idle I/O budget," with the SSD substituted for PCIe.

### Gaps
- Did not verify the megathread's exact 30–60% wording (Reddit fetch blocked); rely on the corroborating GPU threads above.
- No published end-to-end **MTP-over-SSD-streaming** measurement for Qwen3.8-Flash-Next on any device (flash-moe's is the closest, negative result, on a different model).
- Whether the official MTP head weights are included in unsloth Q4 GGUFs (Q4 reportedly lacks MTP params for llama.cpp — likely needs BF16 sidecar tensors; find/pack a MTP-complete GGUF).

### Maturity & PS5 adaptation
Maturity: high on GPU (SGLang/vLLM/MTPLX), low in llama.cpp for this model. PS5: implement MTP natively in the Vulkan streaming engine: (a) extract MTP head weights from the HF repo into the GGUF (keep head weights at high precision, they are one layer); (b) freeze QSA top-k across draft steps (IndexShare — trivially portable to QSA decode kernels); (c) during the ~2–3 draft forwards (GDDR-only, ~ms each), run SSD reads for the union of predicted experts for the verify pass on the async IO queue — predict with the target's own router at draft-extend (draft tokens' hidden states are close enough; SP-MoE shows ~88% top-1 with a *cross-model* gate, better same-model); (d) on a miss, stall is bounded by one expert (≤3 MB, <0.6 ms at 5.5 GB/s). Expected: bytes/token −25…−35%, tokens/s +30…+50% on repetitive/code text.

## Q2. Speculative decoding with a RAM-resident dense draft

### Takeaway
Same math as MTP but with the draft in GDDR (negligible SSD cost): because MoE verification IO grows sub-linearly with verified-token count, even low-cost accept lengths beat direct decode — a drafter reaching AL≥2 needs only ~62% acceptance on the first draft token to gain >20%. Published SD+offload systems (SP-MoE) confirm 1.07–3.5× gains; the accepted-token IO amortization is the key mechanism, and temporal routing correlation (~30% overlap) is what makes verification cheaper than the (K+1)× worst case.

### Cited Findings
- SP-MoE: first SD-aware MoE-offloading framework; 1.07×–3.5× TPOT speedup; draft/verify expert overlap and cutoff-layer prefetch policy; draft stage = idle I/O window ([arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)).
- Reddit experiment combining SSD offload + speculative decoding for exactly this motivation ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1sjbx2c/offloading_to_ssd_speculative_decoding_via/)).
- Cohere: verification of 4 tokens = 20.36 unique experts vs 29.1 uniform baseline; T_t(4)/T_t(1)=0.80 at BS=1 (i.e., verifying 4 tokens costs 1.25× one); SD gives 1.95–2.18× on MoE at BS=1 ([Cohere blog](https://www.cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)).
- Systems overview of six SSD expert-serving runtimes notes standard SD (draft model + single-pass verify) as the universal MoE-SSD multiplier ([originshq survey](https://originshq.com/blog/moe-ssd-expert-serving-runtimes/)).
- A dense 0.8B-class draft would live fully in GDDR; its per-step cost on PS5 GDDR (≈448 GB/s) is bandwidth-trivial (≈1–3 ms/step) — inference below.

### Inferences
- **Derivation — acceptance-rate threshold at 5.5 GB/s.** Direct decode ≈ 1.35 GB/token ≈ 245 ms/token (Q4). With k draft tokens: cost/step ≈ 245 ms × [1 + 0.62·(accepted_prefix+1 − 1)] + draft_time×k. For geometric-ish acceptance p per token: AL ≈ 1/(1−p) capped. Break-even vs direct: AL > [1 + 0.62(AL−1)] → AL > 1.63 → **p ≥ ~0.39 per draft token** — i.e., a drafter that's right 40% of the time per token already helps, far weaker a requirement than dense-model SD intuition (p≥0.5–0.6), because rejected tokens' expert loads partly overlap accepted tokens' (Cohere's correlation data). At p=0.6 (AL≈2.5): 2.5/[1+0.93]=1.30×; at p=0.7 (AL≈3.3): 1.36×.
- A self-drafted MTP head (Q1) beats an external dense draft on *quality* (same model, no drift) but an external draft in RAM is simpler to ship and can run longer chains; on PS5 both are GDDR-cheap, so prefer MTP-when-present, else dense draft with cross-model top-k overlap (SP-MoE's cross-model gating trick transfers directly: reuse the *draft's* hidden state through the *target's* router to prefetch during drafting).
- Caveat: PS5 CPU/GPU must run the game or other work; draft tokens that are wrong still cost a verify pass, and unlike the GPU case the verify pass is IO-dominated, so keep k≤3.

### Gaps
- No published measurement of "small dense draft (Qwen3.x-0.8B) + Flash-Next from SSD" — the threshold derivation above is mine, built on cited overlap/cost-ratio measurements.
- No vLLM/SGLang public result combining expert-SSD-offload + spec-decode on console-class hardware.

### Maturity & PS5 adaptation
Maturity: papers exist (SP-MoE), no shipped console implementation. PS5: keep Qwen3.8-Flash-Next-0.8B (or MTPLX-style self-draft) pinned in ~1 GB of GDDR; run it on the CPU/GPU during SSD transfers; use draft-step hidden states to drive target-router prefetch; cap chain length at 2–3; drop to 0 accepted → same as direct decode (worst case bounded).

## Q3. Expert caching / prediction / pinning

### Takeaway
Temporal + popularity locality of routing is real and large on streaming hardware (flash-moe: 71% page-cache hit with zero custom cache; Cohere: 38% adjacent-token expert overlap; TinyGiant: calibrated pinning 42-56% hit vs 6% random). But learned/temporal *predictors* have repeatedly failed in practice (flash-moe -18%; MLP predictor 31% acc), while the cheap statistical versions — heavy-hitter pinning, expert-contiguous layout, double-buffered pipelining — are what actually work. Group-level routing to cut k (top-10 -> top-8) directly cuts bytes/token ~20% and is the highest-certainty IO lever after MTP.

### Cited Findings
- flash-moe: OS page cache alone hits **~71%** expert hit rate over the streamed 209 GB model; every custom cache (Metal LRU, malloc, LZ4-compressed) was **slower**; temporal expert prediction -18% (25% hit -> wasted SSD bandwidth); MLP routing predictor 31% accuracy; expert file clustering 0% (NVMe ignores scatter at 7 MB) ([flash-moe README](https://github.com/danveloper/flash-moe)).
- TinyGiant / llama.cpp discussion #27149: expert-aware streaming cuts per-layer I/O **11.4x** (345 MB -> 30 MB for Qwen3-30B-A3B), 4.7 tok/s @ ~2 GB RAM on M1; **expert-contiguous re-layout cuts page-fault amplification 36x** (interleaved GGUF: 1 expert = 6,912 pages); pipelined C benchmark: 2.40/3.81/4.10/4.45 tok/s at 50/75/88/100% cache-hit (compute ceiling 4.4 — at 88% hit, SSD reads fully hidden); **text-calibrated heavy-hitter pinning: 42% hit from 10 tokens of calibration, 56% with LRU, vs 6% for random profiles**; Q4 experts (~2.2 MB) pipelined = compute-bound even at 0% hit ([discussion](https://github.com/ggml-org/llama.cpp/discussions/27149), [tinygiant repo](https://github.com/jerryjokesalot/tinygiant)).
- Router skew: the most popular expert in a layer is up to **10x the uniform baseline** (k=8/N=128); empirical active-expert counts run 0.71-0.87x the uniform prediction at batch 4-32 ([Cohere](https://www.cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)). Expert activations follow a power-law/heavy-hitter distribution (MoE-ECS, [arXiv 2312.17238](https://arxiv.org/html/2312.17238v1)) — the paper's exact "top-X% of experts serve Y% of tokens" figure was not captured in fetched text (see Gaps).
- Predictive prefetch done right (GPU+CPU-RAM tier): MoE-Infinity activation-aware offloading ([paper](https://www.semanticscholar.org/paper/MoE-Infinity%3A-Activation-Aware-Expert-Offloading-Xue-Fu/5e612ce3f2fdde6906aba85f1e7ab5113add1212)); OD-MoE claims 99.94% prediction accuracy with ~75% of GPU-cached decode speed (ResearchGate page only — weak source, treat as unverified) ([ResearchGate](https://www.researchgate.net/publication/400888358_Two-Stage_Expert_Offloading_for_Domain-Aware_MoE_Inference)); SP-MoE gating-based cross-token prediction ~88% top-1 ([arXiv](https://arxiv.org/html/2510.10302v2)). FlashMoE (Jan 2026) proposes ML-based cache management for SSD expert offload specifically to cut SSD I/O ([arXiv 2601.17063](https://arxiv.org/html/2601.17063v1), abstract-level only).
- A Guanaco-variant llama.cpp hack pins initial experts from an imatrix and "gets faster as you use it... most conversations stick to a small set of experts" ([llama.cpp discussion #27149 comment](https://github.com/ggml-org/llama.cpp/discussions/27149)).
- llama.cpp already has streaming-cache plumbing converging with this: PR #25294 `--moe-stream-cache` (device slot cache + CPU id-remap), PR #27861 LRU cache, issue #20757 two-tier GPU+RAM expert cache with pluggable eviction ([cross-referenced in discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)).
- Strata engine runs Flash-Next Q2 at 200 tok/s prefill / **20 tok/s decode** on 32 GB RAM + 6 GB VRAM, model >> RAM; its gain attributed largely to coalescing each (layer,expert) into one contiguous read ([HN thread](https://news.ycombinator.com/item?id=47604354); [discussion comment](https://github.com/ggml-org/llama.cpp/discussions/27149)).

### Inferences
- Effective hit-rate model for this project: decode tok/s ~= 1/(hit*t_gpu + (1-hit)*t_ssd + t_other). TinyGiant's curve (0->88% hit: 2.4->4.1 tok/s) shows each hit point is worth a lot below ~85% and nothing above. PS5 GDDR is ~80x faster than SSD, so the same shape applies with a sharper knee; target >=80% hit via: (a) expert-contiguous packer (36x fewer faults, matches Strata's finding), (b) heavy-hitter pinning of top-N experts sized to the ~5-6 GB expert budget, (c) LRU for the rest.
- Co-activation "super-experts" (packing co-routed experts contiguously): no published end-to-end result found; adjacent evidence is positive (co-activation overlap 38%/step -> grouping k-tuples could cut random IOs from 480/token toward ~150/token if pairs survive). The cheaper 80% version is per-(layer,expert) contiguity, which every successful system converged on independently (tinygiant relayout, Strata, flash-moe repack_experts.py).
- Learned prefetch across tokens is not worth it on SSD (two negative results above); spend the budget on calibration-based pinning instead — TinyGiant shows 10 tokens of real text calibration ~= oracle top-8.
- Group routing to cut k: no public top-8 variant of Flash-Next found; but 512-expert/top-10 makes k-reduction unusually headroom-rich: k->8 = -20% bytes/token; fallback is router-calibrated renormalized averaging over the surviving 8; do it by measuring per-layer gate-mass of the bottom-2 picks on a calibration corpus and only cutting where that mass < epsilon.

### Gaps
- Exact Zipf exponent / "top-10% of experts serve X% of tokens" number for Qwen3.8-Flash-Next specifically — not published anywhere found; MoE-ECS's precise fraction was in paper body text I could not fetch fully.
- No published temporal expert-reuse trace for Flash-Next conversation traffic (flash-moe's 71% is Qwen3.5-397B).
- Whether llama.cpp PR #25294/#27861 function on the Vulkan backend + SSD tier (not stated).

### Maturity & PS5 adaptation
Maturity: hot-set pinning + contiguous layout = production-proven in 3 independent engines (flash-moe, tinygiant, Strata); learned prediction = failed twice. PS5: do all of it PC-side: the expert-store packer emits expert-contiguous, co-activation-grouped files sorted by calibrated heavy-hitter rank; reserve ~4 GB GDDR for pinned top-N experts (~1,400 experts ~= top-6% by calibrated frequency; with 10x skew these should serve a large share of tokens); LRU-stream the rest double-buffered; re-tune pinning on-device from the user's own router traces after first use.

## Q4. Expert pruning / dedup / merging for this family

### Takeaway
50% expert-count pruning of this exact model exists and shipped (ISTA-DASLab GSQ-RCO coder variant: 58 GB file, "50% fewer experts"); one-shot compression literature says at 50% compression *pruning beats merging* on generative quality (REAP), and MMLU-style aggregates hide tool-call/agentic degradation. Crucially for streaming: **pruning does not cut bytes/token** (same k, fewer-but-unchanged experts) — its streaming value is smaller footprint -> higher cache/page hit rate and denser packing; the bytes/token lever is cutting k, not N.

### Cited Findings
- "Qwen3.8-Flash-Next Shrunk to 58GB — 50% Fewer Experts" — GSQ-RCO Coder variant from ISTA-DASLab ([YouTube](https://www.youtube.com/watch?v=gtYdl3m5KMQ); first-party HF repo/evals not fetched — gap).
- REAP ("REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression"): at 50% compression, expert *pruning* consistently beats *merging* (SVD-style pair merges) on generative benchmarks; shipped as pruned gpt-oss variants ([arXiv 2510.13999](https://arxiv.org/html/2510.13999v3); [HF example model](https://huggingface.co/AmanPriyanshu/gpt-oss-10.8b-specialized-all-pruned-moe-only-15-experts)).
- Unified one-shot expert-pruning formulation with physical expert removal + router shrink ([GitHub](https://github.com/ZongfangLiu/unified-expert-pruning)); STEP: prune with fewer, signal-rich tokens ([OpenReview](https://openreview.net/forum?id=4iupzej9nT)).
- "moe-stream's PrunedHub" models as named were **not found** (see Gaps); likely a conflation with the REAP/PrunedLM-style gpt-oss series above.
- Live expert-pruning practice in vLLM: dead vs dominant experts observable at runtime ([NVIDIA forum](https://forums.developer.nvidia.com/t/why-you-should-reap-it-yourself-live-moe-expert-pruning-in-vllm/364098)).
- Community claim that 50% expert prune "tripled speed" (coordinator-reported): consistent with caching physics (halving N ~doubles page-cache hit fraction) but no first-party Flash-Next measurement found.

### Inferences
- Streaming effect of pruning N -> N/2 at fixed k: bytes/token unchanged (~1.3 GB at Q4); the tok/s change comes entirely from hit rate and RSS. On a 16 GB PS5 with ~5-6 GB expert budget, halving the pool moves the operating point up the TinyGiant hit-rate curve (e.g., 75% -> 85%+ ~= +15-25% tok/s) — real, but not 3x; 3x claims likely come from RAM-cached setups or Q2-vs-Q4+prune comparisons.
- Merging halves N without touching k and preserves output at high N, but REAP shows it loses to pruning at 50% on generation tasks; for streaming both are equal-footprint wins, so prefer prune + light distill.
- Quality floor for 50% pruning at top-10-of-512 is gentler per-removal than top-2-of-8 (each removal is a smaller share of routing mass), but Flash-Next's agentic/coding specialization means MMLU is the wrong metric — watch tool-call JSON validity (cf. Q5/Q8 2-bit JSON breakage).

### Gaps
- No published MMLU/tool-use delta for the GSQ-RCO 50%-pruned Flash-Next (demo video only); need ISTA-DASLab's HF eval cards.
- No prune-vs-merge head-to-head on Qwen3-Next-80B or Flash-Next (REAP evidence is gpt-oss).
- No distillation-after-pruning recipe published for Next-class models.

### Maturity & PS5 adaptation
Maturity: pruning pipelines exist for this model; merging discouraged by the evidence. PS5: ship packer variants — "pruned-pack" (256 experts, k=10: same bytes/token, 2x footprint cut, +hits) and "k=8-calibrated pack" (-20% bytes/token). Validate tool-calling locally before shipping.

## Q5. Mixed-precision expert tiering & cheaper bytes

### Takeaway
Frequency-aware mixed precision is an established ICML/ICCV result (MxMoE, MoPEQ, GEMQ): allocate bits by expert sensitivity and activation frequency. In streaming this maps to: hot pinned experts at Q5/Q6, bulk routed experts at Q4-class, cold tail at low-Q3/IQ2-territory — but the community floor is harsh (2-bit breaks JSON/tool calling in this model family), so realistic averages are ~3.5-4 bpw (-20-25% bytes/token), not IQ1. Entropy-coding quants has SqueezeLLM precedent but no GGUF implementation found; codebook methods (QuIP#/QTIP) add matvec cost RDNA handles worse than straight Q4 dequant.

### Cited Findings
- MxMoE (ICML 2025): linear-block-level mixed precision driven by per-block quantization sensitivity **and expert activation frequency**; automated mixed-precision assignment beats uniform precision at equal total bits ([arXiv 2505.05799](https://arxiv.org/abs/2505.05799), [code](https://github.com/cat538/MxMoE), [ICML page](https://icml.cc/virtual/2025/poster/43995)).
- MoPEQ: per-expert bit-width assignment ([ICCV-W 2025](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/html/Chitty-Venkata_MoPEQ_Mixture_of_Mixed_Precision_Quantized_Experts_ICCVW_2025_paper.html)); GEMQ: global expert-level mixed precision via linear programming ([PDF](https://openreview.net/attachment?id=5Q4ts0pr9T&name=originally_submitted_PDF)); automated fine-grained MoE quantization ([ACL 2025](https://aclanthology.org/2025.findings-acl.1386.pdf)).
- flash-moe 2-bit experts: **5.74 tok/s vs 4.36 at 4-bit, but breaks JSON/tool calling** (`\name\` instead of `"name"`); 4-bit is the production config; LZ4 expert compression: **-13%** (decompress overhead exceeds IO savings) ([flash-moe README](https://github.com/danveloper/flash-moe)).
- Kaitchup ran a full Q4->Q1 GGUF accuracy sweep on Flash-Next (BF16 ~= 354 GB) ([benchmark](https://kaitchup.substack.com/p/qwen38-flash-next-gguf-benchmark)); unsloth: model needs >=75 GB RAM normally; its 1-bit quant is "larger than usual" due to the large embedding tables ([unsloth docs](https://unsloth.ai/docs/models/qwen3.8-next)); SGLang shipped an NVFP4 checkpoint day-0 ([LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)).
- TinyGiant: with Q4 experts pipelined, streaming is **compute-bound even at 0% cache hit** — at Q4 the SSD stops being the bottleneck; below-Q4 trades quality for IO headroom you may not need ([discussion](https://github.com/ggml-org/llama.cpp/discussions/27149)).
- SqueezeLLM precedent for entropy-coded codebook inference (custom codebooks + efficient variable-length encodings) ([arXiv 2306.07629](https://arxiv.org/abs/2306.07629)).
- SSD expert streaming energy analysis supports decode-time streaming as viable ([arXiv 2508.06978](https://www.alphaxiv.org/abs/2508.06978)).

### Inferences
- Tiering policy from evidence: Q5/Q6 for top ~5-10% heavy-hitters + all shared/attention weights; bulk routed experts Q4-class; only the cold tail (<0.1% frequency) at IQ2/IQ3 — hot-path quality is where tool-calling lives. Average ~3.5-4 bpw => ~1.0-1.1 GB/token vs 1.35 at uniform Q4 (-20-25%).
- ANS/entropy-coded streams: plausible 10-20% "free bandwidth", but flash-moe's LZ4 result (-13%) bounds the risk: decode must be free. On PS5 this is actually attractive *if* the custom SSD decompression block can be driven with a custom table — otherwise GPU-side ANS decode competes with the ALU budget that RDNA needs for dequant-matvec. No GGUF+ANS measurement found (gap).
- QuIP#/QTIP on RDNA2 (GCN/RDNA3-class, no bf16 tensor cores): codebook gather + Hadamard matvec likely slower than integer Q4 dequant path; no RDNA comparison published for these (gap); recommend QTIP-equivalent only if forced to ~2 bpw ([QTIP](https://arxiv.org/abs/2406.11235)).

### Gaps
- No published frequency-based mixed-precision result *on a Qwen3-Next/Flash-Next model* (evidence is Mixtral/DeepSeek/gpt-oss era).
- No GGUF+ANS measurement; no RDNA codebook-dequant benchmark for QuIP#/QTIP found.

### Maturity & PS5 adaptation
Maturity: paper-solid (MxMoE/MoPEQ/GEMQ), packaging-side unexplored. PS5: PC-side packer computes per-expert routing frequency (calibration corpus) + sensitivity proxy, emits tiered store; Vulkan dequant branches per tier (same layout, per-tensor bpw); optionally, OFML/ANS on the custom decompression block for the cold tier.

## Q6. N-gram embeddings, GDN, QSA: bandwidth profiles of Flash-Next's non-expert parts

### Takeaway
All three extras are *good news* for a bandwidth-constrained box: the 51B-param N-gram (PLE) table is huge but touched at only 16 rows/token (host-resident + prefetch works at zero throughput cost — proven in SGLang); GDN layers (36/48) use constant-size state, shrinking KV traffic dramatically; QSA bounds full-attention KV reads to ~2K tokens/layer and shrinks the index cache 80%, with the caveat that beyond a few thousand tokens the *indexer* dominates QSA cost (fixed by IndexShare-style reuse). Net: unlike dense/standard-MoE models, Flash-Next's per-token non-expert bandwidth stays ~constant with context length.

### Cited Findings
- PLE (N-gram embedding) sits at decoder block 2 (layer id 2); 51.2B params ~= 95.4 GiB BF16; per token: 16 rows x 160 values (eight 2-gram hash heads on (t-1,t), eight 3-gram heads on (t-2,t-1,t)) concatenated to [2560]; SGLang keeps the table in **pinned host memory**, gathers 16 rows/token with a Triton UVA kernel on a side stream, overlapping with block 1: target weights -23.46 GiB/GPU, KV capacity +78.5%, throughput -0.07% (geomean), outputs bit-exact ([LMSYS/SGLang blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)).
- Per-request PLE state: last 2 token IDs + a short-conv history [10240, 9] — trivial.
- Architecture: 48 layers = 36 GDN (constant-size state) + 12 QSA full-attention; only 12/48 layers store growing KV ([LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/), [Qwen blog](https://qwen.ai/blog?id=qwen3.8-flash-next)).
- QSA (c4): indexer scores query vs compressed keys (4 raw keys averaged per index key, 4x128-d heads); keeps best **512 blocks -> 2048 logical positions** (+<=3 current tokens) for sparse GQA over **original uncompressed K/V**; adds 1 BF16 compressed index key per 4 tokens (index-cache overhead -80%); prefill+decode kernels exist (TRTLLM-Gen / packed FA / paged scorer) ([LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)).
- Beyond a few thousand tokens, **the indexer (not the attention it feeds) sets QSA layer cost**, and speculative decoding multiplies indexer calls by draft steps — hence IndexShare (freeze the top-k selection across draft steps; accept length unchanged; draft indexer work per iteration: N -> 1) ([LMSYS](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/)).
- GDN recurrence is cheap BLAS on CPU: flash-moe ran GatedDeltaNet with cblas_sscal/ssgemv/sger, 64% faster than scalar (same op family, 64-head 128x128 state) ([flash-moe README](https://github.com/danveloper/flash-moe)).

### Inferences
- PS5 GDDR/SSD budget per token (my derivation): PLE = 16 rows x 320 B = ~5 KB random reads (16 seeks — fine from SSD with NCQ, or Q8-compressed 48 GB table in page cache; Q4 = 26 GB too big to matter); GDN state r/w ~1 MB/layer x 36 = ~70 MB/token GDDR-only (never SSD); QSA KV = 12 layers x ~2051 positions x ~2 KB ~= 50 MB/token from resident KV; indexer keys ~L/4 x 512 B. **All of these are an order of magnitude below the 1.3 GB of expert traffic** — experts remain the only lever that matters for SSD streaming.
- Long-context RSS on PS5: KV grows only in 12 layers: at 32K ctx ~= 12 x 32768 x 2048 B ~= 800 MB + indexer keys ~50 MB — an order of magnitude less than a dense 48-layer model's 3.2 GB; QSA makes long-context decode bandwidth ~flat. No published KV-streaming-to-SSD need for this model.
- If the 48 GB Q4 PLE table is too fat for the user's install-size budget, quantize PLE rows to Q8 (51 GB) or even Q4 (26 GB) — 16 scattered rows/token means PLE quantization costs nothing in bandwidth (it's seek-bound); quality impact unstudied (gap).

### Gaps
- No published per-token bandwidth breakdown for Flash-Next decode (all numbers above are my derivations from LMSYS architecture facts).
- No published study of KV-state streaming needs for GDN+QSA hybrids.

### Maturity & PS5 adaptation
Maturity: SGLang-proven host-offload pattern; nowhere ported to GGUF. PS5: keep PLE table in a raw file + 256 MB page-cache ring (LRU on hash heads — natural recency), prefetch 16 rows async on the same IO queue as experts, quantized Q8; keep GDN states + 12-layer QSA KV fully resident (they're small); implement IndexShare-style indexer freeze for the MTP draft loop.

## Q7. Long-context / throughput interplay: batch-of-1 vs tiny-batch

### Takeaway
For a single user on a console, batch-of-1 is optimal for latency, and the published batching win for MoE offload ("shared-expert batching") is about *serving*: batch-aware opportunistic routing (OEA, ICML 2026) cuts MoE decode latency 39% on Qwen3-30B at batch 16 by having tokens piggyback already-loaded experts; Cohere's data shows at batch >= 4 nearly all popular experts are loaded anyway, so extra streams are nearly free IO-wise. For PS5 single-user use: batching 2 requests ~halves the *marginal* expert IO only for overlapping routes — worth doing only if latency budget allows (OEA-style rerouting is retraining-free and portable).

### Cited Findings
- OEA / "Opportunistic Expert Activation: Batch-Aware Expert Routing" (ICML 2026, Oncescu et al.): dynamically re-route tokens to experts already loaded for other tokens in the batch; **-39% MoE-layer decode latency on Qwen3-30B, -15% on Qwen3-235B at batch 16** with standard-error-adjusted accuracy preserved; retraining-free ([ICML page](https://icml.cc/virtual/2026/poster/62958), [OpenReview](https://openreview.net/forum?id=bSLIDztFwx)).
- Cohere: expected unique experts at batch 4 = 25.4 empirical vs 29.1 uniform (of 128) — saturation means additional batch members cost little marginal expert IO; sparser models (lower k/N) stay bandwidth-bound to higher batch, pushing the SD sweet spot right ([Cohere](https://www.cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding); MoESD, [arXiv 2505.19645](https://arxiv.org/pdf/2505.19645)).
- Prior offloading systems increase request batch to accumulate compute time to hide expert loading ([SP-MoE related work](https://arxiv.org/html/2510.10302v2)).
- CommitMoE (AAAI 2026) treats inactive-expert offload to RAM/NVMe with fallback-free serving ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415)).

### Inferences
- Two concurrent Flash-Next requests at batch 2: unique routed experts ~= 10 + 10x(1-overlap) ~= 16/layer (using Cohere's 38% step-1 overlap, transferable per their cross-domain validation) => 2 tokens in ~1.6 tokens' IO = 0.8 IO/token: 25% better than two sequential decodes *if* the user tolerates ~1.6x per-token latency for both. For a single human user this is a wash; for background agent tasks (indexing, summarizing while the user chats) it's a genuine 1.2-1.6x aggregate throughput win.
- OEA-style piggybacking on a streaming engine = "when the LRU already holds/prefetches expert X for token A, bias token B's router pick toward X among near-ties" — directly implementable in a streaming llama.cpp engine since the residency map already exists; expected effect on PS5 at batch 2-4: fewer SSD reads per token, at some quality cost that grows with rerouting aggressiveness (OEA kept accuracy at bs16 for Qwen3-30B with care).
- Long context itself barely changes expert IO (routes are context-dependent but the k-per-token is fixed); QSA/GDN keep attention-side flat (see Q6), so PS5 throughput is context-robust — a notable selling point.

### Gaps
- No measured Flash-Next route-overlap-vs-context-length data.
- No published "two local requests over one streamed expert store" measurement (all batching data is serving-stack batch).

### Maturity & PS5 adaptation
Maturity: OEA is published, not yet in any local engine. PS5: implement a "quiet batch" mode — background tasks (RAG indexing, session summary) co-scheduled with interactive decode, with the router nudged toward resident experts; expose a "power mode" toggle rather than doing it by default.

## Q8. Quality floors: minimum usable quantization for Flash-Next class

### Takeaway
The empirical floor for *agentic* use is ~3-4 bit: 2-bit experts on this architecture class break JSON/tool-call syntax (documented in flash-moe; echoed by the community IQ3 thread), while MMLU-style scores stay deceptively OK; Q4-class is the "production" config per every serious implementation; Q2 (IQ2_XS-class) is usable for chat at much higher speed (20 tok/s reports via Strata) but not for tools. The honest bytes/token floor is therefore ~0.7-0.9 GB/token (Q3-class) for tools, ~1.35 GB (Q4) for maximum reliability — below that you are buying speed with broken tool calls.

### Cited Findings
- flash-moe: 2-bit experts produce `\name\` instead of `"name"` in JSON — "not suitable for tool use"; 4-bit is production ([README](https://github.com/danveloper/flash-moe)).
- Community IQ3-family quality question for Flash-Next specifically ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wuark9/how_smart_is_the_iq3_family_of_qwen_38_flash_next/)); full Q4->Q1 accuracy sweep published by Kaitchup (numbers behind paywall/fetch limit — gap) ([benchmark](https://kaitchup.substack.com/p/qwen38-flash-next-gguf-benchmark)).
- Q2-class Flash-Next is fast on constrained rigs: 20 tok/s decode on 32 GB RAM + 6 GB VRAM via Strata ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/27149)); unsloth notes even the 1-bit quant is unusually large due to embedding-table overhead ([docs](https://unsloth.ai/docs/models/qwen3.8-next)).
- Mixed-precision literature (MxMoE/MoPEQ, Q5) implies uniform-low-bit is the wrong way to go below Q4 anyway — sensitivity varies per block/expert ([arXiv 2505.05799](https://arxiv.org/abs/2505.05799)).
- Non-expert weights must stay high precision: every serious streaming engine keeps shared weights/attention/router at BF16/FP32-ish (5.5 GB non-expert at native precision in flash-moe; "keep in RAM" shared 0.43 GB + router 0.05 GB in tinygiant layout) ([flash-moe](https://github.com/danveloper/flash-moe), [tinygiant](https://github.com/jerryjokesalot/tinygiant)).

### Inferences
- The 4-bit tool-call cliff is a *grammar* problem (logit perturbation breaks exact-syntax tokens), so it will bite harder on the cold experts' weights when they're low-bit only at the tail of a long generation; tiering (Q5 hot / low-Q3 cold, Q5) is the principled escape: most tokens' tool syntax decisions are made via hot-path experts.
- Minimum honest config for a "does real work" PS5 build: Q4-class experts + BF16-ish shared/router/MTP-head => ~60 GB install, 1.3-1.5 GB/token => ~4 tok/s direct, ~5.5-6 with MTP+pinning (stacked with Q1/Q3 gains). A "chat speed" build at Q3-class (~0.9-1.0 GB/token) => ~6 tok/s direct, trading reliability on long tool chains (IQ2-class: fast but tool-calling unreliable per flash-moe/IQ3 threads).

### Gaps
- Exact Kaitchup per-quant accuracy numbers for Flash-Next (table not fetchable — paywall/rendering); a follow-up fetch of the page HTML or the unsloth GGUF discussion threads would close this.
- No published tool-call-benchmark-vs-bpw curve for Flash-Next (only anecdotal JSON-breakage reports).

### Maturity & PS5 adaptation
Maturity: well-mapped by community, not by first-party evals. PS5: ship two packs by default (Q4 "work" pack, Q3 "chat" pack), pin router/shared/MTP at BF16, keep an "agentic guard" that warns when using Q3 packs for tool-heavy sessions.

---

## Cross-cutting notes for the report writer

- Stacked-effect estimate for PS5 (my synthesis): direct Q4 baseline ~1.35 GB/token ~= 4 tok/s @5.5 GB/s. Add: pinning+contiguous layout to ~80% hit (Q3, +60-90% by TinyGiant curve shape), MTP with shared expert working set (Q1, +30-40%), tiered quantization (Q5, +20-25%), k=8 or pruned pack (Q4, +15-20%) — components interact multiplicatively through the hit-rate curve and additively through bytes/token; realistic combined: **8-15 tok/s on Q4-class with tools intact**. These are estimates, not measurements; no end-to-end console measurement exists.
- Convergence signal: flash-moe, tinygiant, Strata, Guanaco-hack, and llama.cpp PRs #25294/#27861 all independently arrived at (contiguous expert layout + frequency-based pinning + pipelined streaming + trust-the-page-cache); this project should adopt that consensus pattern plus MTP, which none of the streaming engines has yet made work over SSD.
- PS5-specific caveats flagged for the hardware researcher: unified GDDR shared by GPU+SSD DMA (flash-moe found SSD DMA and GPU compute cannot be profitably overlapped on Apple unified memory — serial pipeline optimal; PS5's GDDR may behave differently, must be measured), hardware decompression block reuse for ANS/OFML streams, and ~5.5 GB/s effective SSD read for game-engine workloads.
