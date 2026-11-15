# Residency moonshots: shrinking / better-utilizing the resident expert set of Qwen3.8-Flash-Next on 13–16 GB

> **Status (2026-10-21):** ACTIVE evidence base (W4/W8). Constants per the W4 recompute (G = 1.24 GiB/token); REAP-288 quant correction S5-F23 (Q4_K_M does not reduce G).

Research notes, 2026-10-17. Target: 125B-A6B MoE (48 layers, 512 experts, 10 routed + 1 shared, GDN+QSA hybrid attention, 51B PLE n-gram table, 4B MTP head; ~180B total params, 329.7 GiB BF16, 360 GB checkpoint download). For scale reference: Q4_0 GGUF = 93.7 GiB total with 65.4 GiB routed-expert data ([llama.cpp #29887 benchmark setup](https://github.com/ggml-org/llama.cpp/pull/29887)); Unsloth UD-IQ3_XXS ≈ 78 GB ([fork report via #29887](https://github.com/ggml-org/llama.cpp/pull/29887)); qtum IQ1_M-layered = 76.4 GiB ([qtum repo](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF)); the task's 71.7 GiB expert figure is consistent with a Q3/Q4-class expert store.

---

## 1. Extreme quantization frontier for this model class (late 2026)

### Takeaway
For Flash-Next itself, the published community position is: everything down to ~IQ2/Q2 works on single-turn benchmarks but agentic quality collapses below Q3-class; the shipped sweet spot is ~3-bit (GSQ-RCO IQ3_S recommended by kaitchup; UD-IQ3_XXS/IQ3_M by others). True 1–2-bit for the *routed experts* has only been shipped in "layered" builds where the PLE n-gram table pays for the expert bits, and those show 2.5×+ perplexity damage. No BitNet-b1.58-at-MoE-scale result exists at 100B+ as far as I could find.

### Cited findings
- kaitchup ran a dedicated "Qwen3.8 Flash Next GGUF Benchmark: Q4 to Q1, 12 compressed GGUFs, 42.7M tokens" (Sep 30, 2026, paywalled) — the per-quant table is not accessible; exact numbers are a gap — [The Kaitchup archive](https://kaitchup.substack.com/archive)
- kaitchup's agentic-coding follow-up on Flash-Next GGUFs (Oct 11, 2026): "all the GGUFs I tested performed very well on non-agentic tasks, even down to 'Q1'; agentic tasks are a different story. Even higher-end Q2 variants, like UD-Q2_K_XL, show significant degradation. UD-Q4_K_XL performs on par with the original model, as does GSQ-RCO IQ3_S, which is far smaller and thus my recommendation" (community agrees on X per the same post) — [The Kaitchup](https://kaitchup.substack.com/p/agentic-coding-with-qwen38-flash)
- Sibling-model anchor (dense Qwen3.8-27B, same family): below ~9–9.5 GB "you're giving up too much"; UD-Q2_K_XL (9.5 GB) keeps >95% of BF16; UD-IQ1_M (6.7 GB) "loses more than half of BF16 accuracy on some coding evaluations", worse than a 4-bit Qwen3-4B on LiveCodeBench, "not far from random" on GPQA Diamond — [Qwen3.8 27B Q4→Q1 benchmark](https://kaitchup.substack.com/p/qwen38-27b-27b-gguf-benchmark-q4-to-q1); [follow-up](https://kaitchup.substack.com/p/qwen38-needs-preserved-thinking-q1)
- Shipped sub-2-bit-class Flash-Next quant (qtum, "layered"): IQ2_XS-layered = 83.5 GiB @ 3.98 bpw, wikitext-2 PPL 2.78 (BF16 = 1.72); IQ1_M-layered = 76.4 GiB @ 3.65 bpw, PPL 4.41. Trick: the 51B PLE n-gram table tolerates Q4_0 ("measured: Q8_0 → Q4_0 costs nothing"), and the freed space is spent on expert gate/up projections, improving PPL ~14% vs the uniform tier at smaller size — [qtum/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF)
- Bits-vs-experts ablation at constant disk (same model): Q2@REAP-K320 vs Q3@REAP-K256 (57.3 GiB both): HumanEval-164 95.1% vs 89.6%; rare-name probe 9/10 vs 7/10; fabrication 25% vs 42% — "at equal disk, 64 extra experts per layer beat 1 extra bit per weight on every knowledge metric". But pruned builds fabricate more than the unpruned UD-IQ1_M (0% on their 12-question probe) — "experts store knowledge, bits store precision" — [REAP-320 GGUF commit](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF/commit/ba0dc6a0ec349c61a8a9923d879371eefaa1c48a)
- Strata "Coder" build = Q2-class with **half the experts removed** (REAP-256-style): 91% of full model's SWE-bench Verified per its authors; README warns it's "weaker outside code, including CJK"; HN commenters call Q2 "a poor trade" and note long-horizon coding degrades — [Strata via developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)
- MxMoE (ICML 2025): MoE-aware mixed precision beats uniform at the average bit rate — 2.4 lower WikiText-2 PPL than GPTQ at 2.25-bit; W5A5 within ~full-precision PPL while 29.4% faster than W8A8 — [arXiv:2505.05799](https://arxiv.org/abs/2505.05799)
- BitNet b1.58: the largest native-ternary open model is 2B (BitNet b1.58 2B4T); ternary MoE GGUFs found are ~1GB toys. **No published 1.58-bit ternary port or "routing preserved at b1.58" paper for any 100B+ MoE** (incl. Qwen3-Next-80B) surfaced — [BitNet b1.58 2B4T](https://arxiv.org/html/2504.12285v2); [UlukaDev/bitnet-ternary-moe-gguf](https://huggingface.co/UlukaDev/bitnet-ternary-moe-gguf)

### Inferences
- Honest quality at 1–2 bit for Flash-Next: usable on single-turn evals, degraded to "avoid" on agentic trajectories. IQ1_M-layered's PPL 4.41 vs BF16 1.72 is consistent with the 27B sibling's >50%-accuracy-loss behavior. The frontier for *agentic* Flash-Next is ~2.5–3.5 bpw on the expert store (IQ3_S-class or Q2_K_XL-class with strong imatrix/GSQ), not 1–2 bit.
- Sub-2-bit only becomes viable when spent asymmetrically: cheap PLE + medium experts (qtum layered), or more experts @ fewer bits (REAP). A "Q1 experts / Q6 attention+PLE" hybrid has not been shipped for this model that I found; the 27B sibling's Q1 disaster suggests the *dense/attention* side can carry Q6+ but experts at IQ1 are still lossy on their own.

### PS5 adaptation
On a 13–16 GB APU, the expert store's on-disk format determines the streaming math: at Q2-class (~57–60 GiB routed data) the PS5's NVMe must feed ~3.6 GiB/token at 6B active — plausible only at ~5–8 GB/s effective (55–60 GB/s console SSD would allow more, but decode stalls at ~40 tok/s × 3.6 GiB ≈ 145 GB/s — so *residency*, not streaming, must carry ≥90% of reads regardless of quant). Dropping experts from Q3→Q2 halves the streamed-miss cost per token, which is the single biggest lever for the miss path; IQ1-class saves little more (REAP showed expert-count beats expert-bits) while destroying agentic quality. Realistic PS5 target: experts at Q2_XS/IQ2_K_XL-class (~3.0–3.3 bpw, ~50–55 GiB), PLE table at Q4_0/Q8_0 mmap'd, dense/attention Q6 — total file ~60–65 GiB, per-token expert read ~3.0–3.3 GiB.

### Gaps
- Exact per-quant accuracy table from kaitchup's Sep-30 Flash-Next article (paywalled; URL not found).
- No published sub-2-bit experiment that keeps *routed experts* at IQ1/TQ1 while holding attention/shared-experts at Q6+ for this model class (the 27B evidence is dense-only).
- No BitNet/ternary-at-MoE-scale or "1-bit DeepSeek/Qwen3-Next" results found.

---

## 2. Mixed-precision by expert frequency + sensitivity (tiered quantization)

### Takeaway
The academic method exists and is validated (MxMoE: allocate bits per linear block by sensitivity × activation frequency; big wins at 2.25–5 bit averages). In the Flash-Next ecosystem, the shipped analogues are (a) component-tiered quantization (qtum layered: cheap PLE, richer experts) and (b) expert-count tiering (REAP/Strata Coder: delete cold experts, keep survivors at higher bits). Nobody has shipped a published *frequency-tiered per-expert-bit* Flash-Next GGUF; Strata's engine effectively does the equivalent at runtime (hot experts on GPU).

### Cited findings
- MxMoE: bitwidth allocated at **linear-block level** (gate/up/down separately), ILP solver minimizing L^r·T^(1−r) with r=0.75; assigns lower precision to *frequently activated* experts (they become compute-bound); roofline: W4A16 > W8A8 when AI<83, W2A16 > W4A4 when AI<42; 1.6–2.7× (memory-bound) and 3–3.4× (compute-bound) throughput vs full precision; +25% throughput and better PPL than uniform W4A16 on Qwen1.5-MoE — [arXiv:2505.05799](https://arxiv.org/abs/2505.05799); [PMLR v267](https://proceedings.mlr.press/v267/duanmu25a.html); [code](https://github.com/cat538/mxmoe)
- MoPEQ: named in the task but **no search hit** for a paper by this name — gap (see below).
- Shipped tiered packs for Flash-Next:
  - qtum "layered" quants: PLE n-gram at Q4_0 (free), space moved into expert gate/up projections; ~14% PPL improvement at smaller size vs uniform tier — [qtum](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF)
  - Unsloth dynamic (UD-Q*_XL): "embeddings and output kept at Q8_0 for higher quality at a Q4 footprint" — the standard component-tiering — [AtomicChat README](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/raw/a986af61d99830ae1b12a6e85de17e30b4769ce4/README.md)
  - ISTA-DASLab GSQ-RCO Flash-Next build exists (the "Coder" 58 GiB / 50%-fewer-experts variant and mmproj; for the 27B sibling: GSQ-RCO IQ3_XXS 10.1 GB, IQ2_S 9.3 GB, recommended by kaitchup at par with BF16 on agentic evals) — [ISTA-DASLab repo via kaitchup](https://kaitchup.substack.com/p/qwen38-27b-gguf-benchmark-q4-to-q1); [YouTube coverage](https://www.youtube.com/watch?v=gtYdl3m5KMQ); [kaitchup agentic update](https://kaitchup.substack.com/p/agentic-coding-with-qwen38-flash)
  - Strata's engine keeps "the most-used experts on the GPU" with an offline `expert-profile.bin` — frequency-tiered *residency*, not bits — [Strata via developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026); [NVFP4-Strata repo usage](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)
- REAP K320 (same expert set at Q2 vs Q3) shows expert-count/bits trading is quantified in the community, at 95.1% vs 96.3% HumanEval — [commit](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF/commit/ba0dc6a0ec349c61a8a9923d879371eefaa1c48a)

### Inferences
- No one has published "average bpw at equal quality" for a frequency-tiered 125B MoE in bits-per-expert form; the closest measured artifacts are qtum's layered PPL table (component-tiered) and kaitchup's finding that GSQ-RCO IQ3_S ≈ BF16 agentic at ~3 bpw.
- A frequency-tiered expert GGUF (hot experts Q6, cold Q2–IQ2) is publishable adjacent art (MxMoE) but unclaimed for Flash-Next — a differentiation opportunity for this project's PS5 pack.

### PS5 adaptation
PS5 wants exactly this: pin the ~2–3K hottest experts (by coding-agent trace) at Q6_K in the 13–16 GB pool and stream the tail at IQ2_XS. At ~8.4 GB/expert-GB-budget… concretely: 71.7 GiB/24,576 slots ≈ 3.0 MB/expert at Q4-ish; a 14 GB pool = ~4,700 slots. Frequency-tiered: 4,700 slots at Q6 (~4.6 MB each, half-count) vs 4,700 at Q4. If the hot set is Q6 and the streamed tail is IQ2-class, effective per-token bytes for a 90%-hit workload ≈ 0.9×(6/4)×miss-adjusted… the honest framing: tiering buys ~1.15–1.3× effective pool capacity at measured-equal quality (MxMoE's equal-accuracy allocation logic), i.e., 90% hit at what LRU-Q4 would only achieve at ~25–30% capacity.

### Gaps
- No "MoPEQ" paper found under that name (searched; possibly a misremembered acronym — closest are MxMoE and HOBBIT's mixed-precision expert offloading, [arXiv:2411.01433](https://arxiv.org/html/2411.01433v2)).
- No published average-bpw-at-equal-quality table for any *frequency*-tiered Flash-Next GGUF.

---

## 3. Hot-set concentration engineering (making routing more cache-friendly)

### Takeaway
Two independent lines converge: (1) Flash-Next-class routing on decode has enough temporal locality that LRU caches achieve 72–94.6% hits at 10–29% capacity (see Q5), while Qwen3-30B-A3B-class routing is near-uniform and LRU thrashes (25.1% at 75% capacity) — **frequency-pinning beats LRU 3× there**; (2) a 2025–2026 literature (ELDR, GRACE, local-routing-consistency metrics) explicitly engineers routers/placement to reduce the number of distinct experts activated. Prefetch quality doesn't help single-stream decode (WiSP: PCIe-bound, not prediction-bound), so the lever is *which experts are resident*, i.e., static frequency pins + warm LRU.

### Cited findings
- BlobPager (Qwen3-30B-A3B, 128 experts/layer): frequency-pinned 96/128 = 77.9% held-out hit vs LRU 25.1%; "budget share, not locality, decides capacity: routing is near-uniform (top-16 experts = 13.7% of draws vs 12.5% uniform)"; decode concentrates more than prefill — live hit rates prefill 77.68% / decode 82.76%, mean 1.38 misses of 8 draws per layer — [llama.cpp discussion #29234](https://github.com/ggml-org/llama.cpp/discussions/29234)
- ELDR: Expert-Locality-Aware Decode Routing for PD-disaggregated MoE serving — a decode-time routing algorithm that reduces the number of activated experts — [arXiv:2607.00466](https://arxiv.org/abs/2607.00466)
- "Not All Models Suit Expert Offloading: On Local Routing Consistency of MoE" proposes Segment Routing Best-Performance metrics; expert offloading gains depend on the model's intrinsic routing locality — [arXiv:2505.16056](https://tldr.takara.ai/p/2505.16056)
- GRACE-MoE: grouping + replication with locality-aware routing, affinity-aware offloading — [Semantic Scholar](https://www.semanticscholar.org/paper/GRACE-MoE%3A-Grouping-and-Replication-with-Routing-Han-Pan/1fbaabccef99f0ea1a26b21bdd2c53741ea22103); ICDCS'25 locality-aware expert placement: "expert layers are not uniformly accessed, but exhibit a stable locality" — [paper](https://iqua.ece.toronto.edu/papers/chenghao-icdcs25.pdf)
- WiSP: predictive prefetch "does not help in single-stream decode: the bottleneck is PCIe bandwidth, not prediction quality" — allocation (expert pool vs KV split) is the real lever; routing-aware paging gives 2.0× decode over static offload at 24 GiB — [arXiv:2606.21868](https://arxiv.org/html/2606.21868v2)
- FlashMoE (arXiv Jan 2026): ML-based cache admission improves hit rate up to +51% over LRU/LFU, 2.6× speedup — [arXiv:2601.17063](https://arxiv.org/html/2601.17063v1)
- OD-MoE: on-demand expert loading with 99.94% activation-prediction accuracy at ~75% of fully-cached decode speed — [arXiv:2512.03927](https://arxiv.org/html/2512.03927v1)
- Task-type split: BlobPager's traces show decode > prefill concentration (Qwen3-30B); no published coding-agent-vs-chat reuse study for Flash-Next surfaced — gap. Strata ships `expert-profile.bin` (offline profiling of expert usage) to seed its cache; NVFP4 pack runs `--expert-profile data\expert-profile.bin --expert-cache auto` — [HF repo](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)
- Expert-reuse statistics from the mainline cache (closest to per-task): 72–89% LRU hit rates at 10–29% pool capacity on Flash-Next (Q5 table) imply the hot-set curve is steep: ~16% of slots → ~77% of draws (−cmoe), vs 12.5% budget-share expected from top-10-of-512 — [PR #29887 via developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)

### Inferences
- Flash-Next's router is much more skewed than Qwen3-30B-A3B's (measured: 77% hit at 16% capacity vs budget-share 18.75% for the 30B model at frequency pins) — likely from its expert-bias/aux-loss design; this is the single best piece of news for PS5: the base router already concentrates, so even naive LRU gets ~72–77% at 10–16% pool capacity.
- Reusing BlobPager's LRU→frequency-pin swap idea (+52.8 pts of hit rate on the 30B model) is a drop-in for a paging sidecar; combined with a coding-agent-specific offline pinset (Strata-style `expert-profile`), a PS5 pool could realistically convert its 18–22% capacity into 85–90%+ served-from-RAM.
- Router retraining for locality (ELDR-style decode-affinity) has never been applied to a shipped 125B GGUF; a router *calibration* (temperature/similarity-clamp at runtime) is the PS5-feasible version.

### PS5 adaptation
Maximize reuse per pool byte with three stacked tricks, all published pieces: (1) offline frequency pins from a mixed coding-agent corpus (Strata/BlobPager E3 pattern) covering ~60–70% of draws; (2) warm LRU admission for the remainder (LRU already ~72–77% at 10–16% on this model); (3) per-layer budget skew — give pools to the layers whose expert entropy is lowest (BlobPager E1-style trace tells you which). At 14 GB ≈ 20% of a 71.7 GiB expert store, the measured curve says ~82–88% from RAM as-is; frequency-pinning (3× LRU on the uniform-routing 30B; ~1.2× expected here since routing is already skewed) plausibly reaches the 90% target without any router surgery.

### Gaps
- No published coding-agent-vs-chat-vs-completion expert-reuse percentages for a 512-expert model; only BlobPager's prefill-vs-decode split and the cache PR's aggregate rates.
- No "locality-aware router" or "expert affinity" fine-tune has been applied to Flash-Next-class open weights; ELDR is serving-side (PD-disaggregated) and unfixed to any GGUF.

---

## 4. Compression of residency itself (precision of the pool, dedup, sharing)

### Takeaway
Measured facts: the PLE n-gram table compresses to Q4_0 essentially free (this is the biggest single "resident budget" win found for this model); GGUFs are already expert-major so paging needs no repack; expert *redundancy* literature exists (CAMERA micro-expert deltas, Unified-MoE holistic compression) but no one published an exact/near-duplicate expert fraction for Flash-Next. The pinned-pool-vs-streamed precision split (hot Q6 / cold IQ2) is implied by MxMoE logic but unshipped for this model.

### Cited findings
- PLE n-gram table: Q8_0 → Q4_0 "costs nothing" (measured); the 51B table (≈25 GiB at Q8, ≈13 GiB at Q4) is a prime target; Strata runs it as `ple-fp8.gguf`, and the laptop-CPU engine runs "125B main model plus 51B PLE" on RAM paths — [qtum](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF); [NVFP4-Strata usage](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata); [QwenLM discussion #2336](https://github.com/QwenLM/Qwen/discussions/2336)
- GGUF layout: "the GGUF is already expert-major (3 contiguous runs per expert on Q4_K_M) → no repack, no new format needed for demand paging" — [BlobPager E2](https://github.com/ggml-org/llama.cpp/discussions/29234)
- CAMERA (AAAI): MoE redundancy modeled via shared base weight + per-expert low-rank deltas (micro-expert joint compression) — [arXiv:2508.02322](https://arxiv.org/html/2508.02322v4); Unified-MoE (TMLR): holistic compression study (pruning+quantization+collapsing) of MoE — [repo](https://github.com/DaizeDong/Unified-MoE-Compression)
- REAP expert pruning as residency compression: K=320/512 (−37.5% expert slots) at Q3 keeps HumanEval 96.3% but 25% fabrication rate on a 12-question probe (unpruned IQ1_M: 0%); K=256 duo: 89.6% / 42%; Strata Coder (K≈256, 57.7 GiB): 91% of full SWE-bench Verified — [REAP commit](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF/commit/ba0dc6a0ec349c61a8a9923d879371eefaa1c48a); [Strata README via developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)
- KV side of the residency split: WiSP's MV-WSA finds the optimal VRAM split between expert pool and KV by marginal latency benefit/byte, with a KV-admission floor; live resizing gains up to 1.19× over a fixed offline split — [arXiv:2606.21868](https://arxiv.org/html/2606.21868v2)
- MTP head (4B) excluded at GGUF conversion by qtum; Strata keeps MTP as a separate component with `--spec 6 --spec-min-p 0.7` (speculative, ~1.6–1.8× acceptance gain per its README) — [qtum](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF); [Strata/developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)

### Inferences
- The pool-vs-stream precision split is unexplored in products: pinning hot experts at Q6_K while streaming the tail at Q2_XS lowers miss-path bandwidth AND keeps hot-path quality — a direct MxMoE transplant (its allocator already balances exactly this sensitivity×frequency objective).
- No evidence of *exact*-duplicate experts across layers in Flash-Next; "cross-layer expert sharing" papers exist in the redundancy-compression literature (Unified-MoE collapsing) but no shipped layer-agnostic Flash-Next variant was found.

### PS5 adaptation
Concrete free wins for a 13–16 GB budget: PLE table at Q4_0 (~12–13 GiB saved vs Q8 — even if it stays mmap-on-SSD, the touched-page footprint shrinks ~2×), MTP excluded or tiny, KV at q8_0/i8 (Gerganov's own preset uses q8_0 KV; Strata's `--kv k8v4` cuts KV 23% and is faster). That can plausibly move 1.5–2.5 GB into the expert pool, which the Q5 curve prices at roughly +5–10 points of hit rate — worth 8–15 tok/s near the 40 tok/s cliff. Frequency-tiered pool precision (Q6 hot / IQ2 cold) is the natural next step and nobody has shipped it for this model.

### Gaps
- No measured exact/near-duplicate expert fraction for Flash-Next or any 512-expert model.
- No published experiment on RAM-pool precision tiers (Q8-pinned vs Q4-streamed) end to end.

---

## 5. Measured hit rates vs pool capacity on 512-expert / 100B+ streaming

### Takeaway
The best curve available is on *exactly this model*, from llama.cpp PR #29887 (merged Oct 7, 2026): with all experts in host RAM (Q4_0, 65.4 GiB experts), LRU cache hit rate = 72–77% at 10–16% capacity, 89% at ~29%, 94.6% at ~24.5% on the R9700; the community threshold for the cache to pay off is ≥75% hit rate. Adjacent systems (BlobPager 77.9%@75% capacity with frequency pins; flash-moe 71%@17% via OS page cache; Strata traces) bracket the same shape.

### Cited findings
- **PR #29887** (`--moe-cache-mib`, LRU GPU cache over host-RAM experts; model = Qwen3.8-Flash-Next Q4_0, 93.7 GiB total / 65.4 GiB experts, EPYC 7742, PCIe 4.0; batch ≤32 tokens for cache path):
  | GPU | Config | Experts in VRAM | t/s | Hit |
  |---|---|---|---|---|
  | RTX 4090 | `--fit` | 16.5 GB (25%) | 25.0 | — |
  | RTX 4090 | `--fit` + 6,440 MiB cache | 10.4+6.4 (26%) | 39.4 | 72% |
  | RTX 4090 | `-cmoe` + 10,700 MiB (16.4%) | 10.7 | 40.7 | 77% |
  | RTX 5090 | `-cmoe` + 18,600 MiB (28.4%) | 18.6 | 67.8 | 89% |
  — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887) via [developersdigest write-up](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
  - Radeon AI PRO R9700 32 GB (ROCm/HIP, `-cmoe` simulating a small card): 11.0 t/s no-cache → 14.5 at 8,000 MiB (**85% hit**) → 27.6 at 16,000 MiB (**94.6% hit**); prompt speed didn't drop on that box.
  - RTX 5060 Ti **16 GB** + 62 GB RAM, whole UD-IQ3_XXS (78 GB): 14.95→19.26 t/s (1.29×) with 8,000 MiB cache (~10–12% of expert bytes), prefill ~89 t/s unchanged.
  - 2× RTX 4090: gen 34.78→64.39 t/s (`-cmoe` + 15 GB), prefill 440.7→321.5 (0.73×); middle `--fit`+6.4 GB keeps 91% prefill at 1.74× gen.
  - Post-merge counterpoint (GLM 5.3 Flash, 2 GPUs): 9,991 MiB cache / 99.6 GiB host experts (10%) → **43.5% hit at ≤8-token ubatches**, gen halved 33→16 t/s — [PR #30112 issue comment](https://github.com/ggml-org/llama.cpp/pull/30112#issuecomment-6067849080)
  - Sizing rules: author's rule ≈10% of expert size as *starting point*; testers converge on ≥75% hit as break-even; Gerganov's fastest 5090 preset: `-cmoe 1`, `--moe-cache-mib 16384`, 128K ctx, **q8_0 KV**, batch 4096.
- flash-moe (danveloper, 397B / 209 GB @4-bit on 48 GB M-series Mac, ~17% capacity): OS page cache yields **71% hit**, streaming the rest from NVMe at 17.5 GB/s, 4.4 tok/s — [repo](https://github.com/danveloper/flash-moe); [writeup](https://www.arunbaby.com/ml-system-design/0069-flash-moe-macbook-397b-consumer-hardware/); [author post](https://chenguangliang.com/en/posts/blog109_flash-moe-local-inference/)
- BlobPager (Qwen3-30B-A3B, 75% capacity pool): frequency pins 77.9% vs LRU 25.1%; miss path 145 µs/expert draw; hybrid executor 1.62×; **dead valley** at 32–40/48 CPU layers (15.7–17.3 t/s) vs `--cpu-moe 12` at 50.3 t/s — [discussion #29234](https://github.com/ggml-org/llama.cpp/discussions/29234)
- Expert-aware SSD streaming thread: ~42% hit rate achieved vs 6% for the Mac-style approach in that configuration; sequential/pipelined table around 50% hit → 2.20 t/s — [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- Mac mini 64 GB SSD streaming: "GPU still waits 27% of decode on experts" even with as-many-hot-as-possible — [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1wy5fma/moe_ssd_streaming_on_a_64_gb_mac_mini_gpu_still/)
- Apple expert-residency PR #23238 (Qwen3-30B-A3B Q6_K, M3 Pro 36 GB): slot pools regress gen 40.4→20.3 even with zero eviction (per-layer CPU-op sync); N_SLOTS=64 saves 11.3 GiB at 17.7 t/s — [PR](https://github.com/ggml-org/llama.cpp/pull/23238); and issue #20757's LFU Metal pool hit "~97–99% steady-state, matching offline sim" but negative memory win from mmap accounting; two-tier PoC: 47% hit at 100 tokens → 67% at 500+ — [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- Lidenburg (RFC #24528 thread): ~60% faster than `--n-cpu-moe` at equal VRAM on 32+16 GB — [discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
- Strata (engine-level expert cache + spec decode): 94 t/s Q2_0 (12 GB GPU + 64 GB RAM, RTX 5070); RX 9070 XT 16 GB: Q2_0 60, IQ2_XS 52, Coder 44 t/s; UD-Q4_K_XL (SSD-reading) only 7–8.5 t/s; 124 t/s reported on 4090+128 GB DDR5 — [developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)
- REAP-320 on 32 GB card: fully-resident 98.5 t/s; LRU-128 expert cache 47.6 t/s; `--n-cpu-moe 3` 86.9 t/s — [commit](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF/commit/ba0dc6a0ec349c61a8a9923d879371eefaa1c48a)
- 16 GB GPU + 32 GB RAM report: 17–26 t/s, ~76 GB model split across VRAM/RAM/NVMe, ~620 t/s prefill at 16K — [Reddit](https://www.reddit.com/r/LocalLLM/comments/1wucof9/qwen38flashnext_125b_at_1726_toks_on_a_16_gb_gpu/)

### Inferences
- Hit-rate-vs-capacity for Flash-Next-class (512 experts, top-10): ~10% → 72–77%; ~16% → 77%; ~24–25% → 85–94.6%; ~29% → 89–94.6%. The curve is steep then flat; the 90%-hit target sits at ~24–29% capacity with **plain LRU**. A PS5 pool of 13–16 GB against a 71.7 GiB store = 18–22% → expect ~82–88% from LRU alone; frequency pinning and warm-up (BlobPager precedent) are what bridge the last 5–10 points to 90%+.
- The GLM 5.3 Flash 43.5%@10% datapoint warns the curve is model-dependent; Flash-Next's own traces (77% at 16%) confirm it's on the good side, but don't extrapolate from 10%→72% to any pool size.
- Decode-only benefit: the cache path is batch≤32, prefill streams separately — for PS5 that's fine (prefill can tolerate SSD bandwidth, decode cannot).

### PS5 adaptation
Map the curve onto PS5: 14 GB usable ≈ 20% of 71.7 GiB → 82–88% from RAM with stock LRU, i.e., 12–18% of the ~3.0–3.6 GiB/token expert reads hitting NVMe → at 40 tok/s that's 15–25 GB/s of sustained SSD reads needed just for misses — at or beyond the console's ~5.5 GB/s. Conclusion from the curve: *pure LRU at PS5 capacity is ~1–2 points of hit short of the 90% target and misses are brutally expensive*; the design must be frequency-pinned-first (static hot set from agent traces), LRU-second, and prefetch must be *bandwidth-planned* (WiSP: prediction doesn't help; bytes do). 40–60 tok/s at 90% RAM-served needs the pool to behave like ~25% capacity, i.e., tiered precision (Q4) + dedup + KV/PLE savings adding 1.5–2.5 effective GB.

### Gaps
- No published per-task-type (coding agent vs chat) hit-rate traces for Flash-Next; the R9700's 94.6% is the best measured point at 16 GB-class cache.
- No Strata/tinygiant trace dumps with hit-vs-capacity tables found (Strata's expert-profile is shipped but its E-curve isn't published).

---

## 6. KV/state side: freeing the non-expert resident budget

### Takeaway
Published levers: KV to K8/V4 (measured −23% KV memory and faster, Strata `--kv k8v4`; llama.cpp preset uses q8_0 KV), PLE n-gram table to fp8/Q4_0 (measured free), and VRAM/context sizing tables that trade context for expert residency (Strata: 16 GB → 64K ctx). No one published a full component-level resident-budget breakdown (dense bytes / GDN state / KV) for this model.

### Cited findings
- Strata flags: `--kv k8v4` = "INT8 keys + rotated Q4_0 values: 23% less KV memory, and faster"; `--resident-experts` low-RAM mode "copy into RAM" for the expert store; `--gpus`/`--layer-split auto`; `--expert-profile data\expert-profile.bin --expert-cache auto` — [xhinker/Strata](https://xhinker.medium.com/latest-strata-enable-rtx-3090-run-qwen3-8-flash-next-6ece698a1ca3)
- Strata NVFP4-GPTQ usage on Flash-Next: `--kv int8`, `--ple-gguf ple-fp8.gguf`, `--mtp ... --spec 6 --spec-min-p 0.7`; context-vs-VRAM table: 32 GB→262K, 24 GB→131K, **16 GB→65,536**, 12 GB→32,768 — [HF repo](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)
- Gerganov's llama.cpp 5090 preset: `cmoe=1, moe-cache-mib=16384, ctx 131072, q8_0 KV, batch/ubatch 4096` — [PR #29887 via developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
- Architecture facts (what the non-expert budget contains): 48 layers, GDN linear-attention interleaved with QSA micro-block sparse attention; 51B PLE n-gram (20M bigram/trigram) embeddings "offload well on memory-constrained accelerators"; QSA "cuts long-context latency" — [AtomicChat README](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF/raw/a986af61d99830ae1b12a6e85de17e30b4769ce4/README.md); [Unsloth docs](https://unsloth.ai/docs/models/qwen3.8-next)
- PLE table compression is free at Q4_0 (measured) — [qtum](https://huggingface.co/qtum/Qwen3.8-Flash-Next-GGUF)
- WiSP: expert-pool vs KV-cache split is one allocation problem; marginal-value allocation with KV floor is the only tested policy near-optimal on both prefill and decode; online resizing up to 1.19× — [arXiv:2606.21868](https://arxiv.org/html/2606.21868v2)

### Inferences
- GDN+QSA makes Flash-Next's KV/state side unusually cheap (linear-attention state + block-sparse), so the 1–3 GB state budget is mostly PLE table and dense weights, not KV — the biggest reclaimable chunk is PLE table precision (fp8/Q4_0 already proven) and the 51B table's residency policy (mmap + selective populate beats pinning).
- At 32–64K context the measured configs (Strata 16 GB → 64K) imply state+KV fits ≤1–2 GB with int8 KV — consistent with the task's 1–3 GB estimate; K8V4-class quantization is shipped and faster, not just cheaper.

### PS5 adaptation
The Strata 16 GB row of the context table is the closest published PS5-class datapoint: it says "16 GB of memory class → run at 64K context with int8 KV". For this project: mmap the PLE table Q4_0 (Q8_0→Q4_0 measured lossless), keep GDN recurrent state fp32→bf16 (no published measurement found for this specific switch — treat as unverified), q8_0 KV at minimum (Gerganov preset) or k8v4 (−23%, measured, Strata), and cap context at 32K for the 40–60 tok/s tier. Reclaim target: 1.5–2.5 GB into the pool = +5–10 hit-rate points = the difference between the 30s and 40s of tok/s.

### Gaps
- No published GDN-state precision experiment (bf16 vs fp32) for Flash-Next or GDN models generally surfaced in these searches.
- No published PS5-class resident-budget breakdown (Strata's expert map internals not public beyond CLI flags; ehepp/npanj configs not found at all).

---

## 7. Published "Flash-Next on 16 GB device" attempts

### Takeaway
Several 16 GB-class attempts exist and are published — a 16 GB GPU + 32 GB RAM PC (17–26 tok/s, model split VRAM/RAM/NVMe), a 16 GB RX 9070 XT + 47 GB RAM via Strata (44–60 tok/s, experts in RAM), and a 16 GB RTX 5060 Ti + 62 GB RAM with `--moe-cache-mib 8000` (19.26 t/s). But nothing on 16 GB *unified* (no discrete VRAM): no PS5, Steam Deck, Switch 2, ROG Ally, or M1-16GB attempt surfaced. Unsloth claims a 75 GB RAM no-VRAM path and there's a native-C laptop-CPU engine with an "automatic 8 GB path"; nobody has published the unified-16 GB configuration.

### Cited findings
- 16 GB GPU + 32 GB DDR5 + NVMe, ~76 GB model, experts across VRAM/RAM/NVMe: **17–26 t/s**, ~620 t/s prefill @16K — [Reddit r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1wucof9/qwen38flashnext_125b_at_1726_toks_on_a_16_gb_gpu/); companion video ["16GB GPU: Fast, But Is It Smart?"](https://www.youtube.com/watch?v=B77LasklmZU)
- RX 9070 XT (16 GB VRAM, 47 GB RAM), Strata: Q2_0 60 / IQ2_XS 52 / Coder 44 t/s — [developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)
- RTX 5060 Ti 16 GB + 62 GB RAM, UD-IQ3_XXS 78 GB, `--cpu-moe --moe-cache-mib 8000`: 14.95→19.26 t/s, prefill ~89 t/s — [fork commit via PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887#ref-commit-6b99580)
- Strata requirements: 12 GB+ VRAM **and** 32 GB+ RAM (VRAM holds hot experts, RAM the full expert set; first load 35–55 GB into RAM freezes the PC 1–3 min); README recommends 32 GB RAM for Coder, 48 GB IQ2_XS/Q2_0, 64 GB sweet spot — [developersdigest](https://www.developersdigest.tech/blog/qwen-3-8-flash-next-strata-run-locally-2026)
- Unsloth docs: "can run locally on devices with 75 GB RAM/unified memory with no GPU VRAM required" (their llama.cpp RAM path); native-C laptop-CPU engine (no GPU/Python) running 125B + 51B PLE with an "automatic 8 GB path" — [Unsloth docs](https://unsloth.ai/docs/models/qwen3.8-next); [QwenLM discussion #2336](https://github.com/QwenLM/Qwen/discussions/2336) / [unsloth discussion #10396](https://github.com/unslothai/unsloth/discussions/10396)
- Pure-CPU 180B ("no GPU required") runs exist but are "actually usable"-grade speed — [YouTube](https://www.youtube.com/watch?v=KGkKpAGeoXM)

### Inferences
- The published envelope: 16 GB *of GPU* works (44–60 t/s with expert caching), 75+ GB *of unified* works (no-VRAM RAM path), and everything in between needs ≥32 GB RAM + an SSD fallback at 17–26 t/s. A PS5 (13–16 GB unified, ~5.5 GB/s, Jaguar-class APU shared with GPU) has no published attempt — this project would be first, and its success hinges on the Q5 curve: the 16 GB-GPU successes all keep the *full expert set* in ≥32 GB RAM; PS5 must instead live off a 20%-capacity pool at 90%+ hit, which only the frequency-pinned + KV/PLE-shrinking stack above can deliver.
- Every "16 GB" success paired ≥47 GB of backing RAM; nobody has demonstrated a config where the expert store's home is SSD — PS5 is the first such forced march.

### PS5 adaptation
Closest published templates: (a) PR #29887 style `-cmoe --moe-cache-mib` on the APU's "GPU-visible" pool with the expert store on SSD instead of RAM — a GPU-cache-over-SSC-resident variant of BlobPager's exact hybrid (misses computed on CPU is not an option; misses must be bandwidth-capped streams); (b) Strata's Coder tier (half experts, Q2) as the fallback model class — but the task forbids small-model fallback, so REAP-320-style K=320 Q2 at ~57 GiB total is the honest "still Flash-Next-class" floor (96.3%/95.1% HumanEval, 25% fabrication caveat); (c) the R9700 config — 16 GB cache, 94.6% hit, 27.6 t/s from an 11 t/s base — is the existence proof that 16 GB-class pools reach ~95% hit when the cache is the bottleneck; replicate its pool ratio (24.5%) via PLE/KV reclaim + tiered precision.

### Gaps
- No PS5, Steam Deck, Switch 2, ROG Ally, or M1 8/16 GB Flash-Next attempts found (searched multiple phrasings).
- No published config breakdown (pool sizes, pinsets, KV formats) for any of these — only flag names and aggregate speeds.
- "ehepp" and "npanj" (named in the task) produced zero search hits; they may be very small personal repos not indexed, or misremembered handles.
