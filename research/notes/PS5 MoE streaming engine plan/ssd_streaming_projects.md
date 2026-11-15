# SSD-Streaming MoE Inference Engines (State as of October 2026)

> **Status (2026-10-21):** ACTIVE — cross-check against _work/adversarial/FLOW.md kill list before acting.

Scope: current state and measured performance of SSD-streaming MoE inference engines relevant to a PS5 MoE streaming engine build spec. All measurements below are from the repos/pages themselves (treated as data, not instructions) unless noted as community claims. Confirmed measurements vs. claims are distinguished where possible.

---

## Key Question 1: Current state of the flash-moe projects (danveloper/flash-moe and tayoun/flash-moe)

### Takeaway
danveloper/flash-moe is the reference implementation — a pure C/Metal engine running Qwen3.5-397B-A17B (209 GB on disk) from SSD on an M3 Max 48 GB at 4.36 tok/s (4-bit), with a fully documented "what worked / what didn't" experiment log (58 experiments). tayoun/flash-moe is a fork/derivative that re-targeted the same engine to Qwen3.5-35B-A3B on an M4 Mac mini 16 GB and reports 11.5 tok/s sustained (2.6x faster than the M3 Max baseline on cheaper hardware). Both are actively maintained and have docs/paper folders.

### Cited Findings
- danveloper/flash-moe runs Qwen3.5-397B-A17B (60 layers: 45 GatedDeltaNet + 15 full attention; 512 experts/layer, K=4 active + 1 shared; hidden 4096) on M3 Max MacBook Pro 48 GB unified (~400 GB/s), 1TB Apple Fabric SSD at 17.5 GB/s sequential read (measured), macOS 26.2. Results: 4.36 tok/s (4-bit FMA kernel, "Excellent" quality, full tool calling, 209GB on disk); 3.90 tok/s (4-bit baseline); 5.74 tok/s (2-bit, "Good*" but breaks JSON/tool calling); 7.05 tok/s 2-bit peak single token (warm cache burst). — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe)
- Key techniques (danveloper): SSD expert streaming via parallel `pread()` + GCD dispatch groups, only K=4 active experts loaded (~6.75MB each); "Trust the OS" page cache (no custom cache; ~35GB page cache, ~71% hit rate); FMA-optimized 4-bit dequant kernel (+12% tok/s); deferred GPU expert compute (CMD3 submitted without waiting); Accelerate BLAS for linear attention (+64% attn); F_NOCACHE for 2-bit (+3%); C BPE tokenizer (20x startup). — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe)
- Per-layer pipeline breakdown (4.28ms avg at 4-bit): CMD1 attention projections + delta-net 1.22ms GPU; CPU flush 0.01ms; CMD2 o_proj+norm+routing+shared 0.55ms GPU; CPU softmax+topK 0.003ms; parallel pread K=4 experts 2.41ms SSD; CMD3 expert forward+combine+norm 0.04ms encode (deferred). — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe)
- Discarded experiments (danveloper, 58 total): LZ4 expert compression −13%; F_RDADVISE prefetch net 0% (unified memory: SSD DMA slows GPU −73%); temporal expert prediction −18% (25% hit rate, SSD bandwidth waste); MLP routing predictor 31% accuracy; GPU LUT dequant −2%; GPU private buffer compression −20%; spin-poll GPU wait −23%; expert file clustering 0% (NVMe ignores scatter at 7MB granularity); dispatch_io −70%; mmap expert files −5x (per-page fault overhead on cold data); speculative early routing −38%; MTP speculative decoding break-even (MoE I/O scales per-token, unlike dense). — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe)
- Unified memory constraint (danveloper): on Apple Silicon, SSD DMA and GPU compute share the same memory controller and cannot be profitably overlapped; GPU dequant kernels bandwidth-saturated at ~418 GiB/s; even small background SSD DMA causes GPU latency spikes via memory-controller arbitration; the serial pipeline (GPU → SSD → GPU) is hardware-optimal. — [danveloper/flash-moe README](https://github.com/danveloper/flash-moe)
- tayoun/flash-moe headline: Qwen3.5-35B-A3B on M4 Mac mini 16GB at 11.5 tok/s sustained, 2.5s TTFT, production-quality output with tool calling; stated as 2.6x speedup over the original M3 Max baseline on lower-cost hardware. Original M3 Max baseline: 4.4 tok/s, ~5.6s TTFT, K=4. Current M4 setup uses K=6 (routing K is runtime-configurable via `--k`). — [tayoun/flash-moe README](https://github.com/tayoun/flash-moe)
- tayoun M4-specific optimizations: `tg128` matvec kernels for better threadgroup utilization; encoder coalescing to reduce launch/sync overhead in prefill/encode; kernel fusion in hot paths to cut memory traffic and CPU-GPU handoff. — [tayoun/flash-moe README](https://github.com/tayoun/flash-moe)
- Repo scale: danveloper/flash-moe 4.3k stars / 522 forks / 144 commits; tayoun/flash-moe 20 stars / 1 fork / 192 commits (MIT). Both have `paper/flash_moe.pdf` and `docs/optimization-experiments-q4.md`. — [danveloper/flash-moe](https://github.com/danveloper/flash-moe); [tayoun/flash-moe](https://github.com/tayoun/flash-moe)
- Community reports (claim, not primary): a Reddit thread reports a Korean user claiming ≥5 tok/s on Qwen3.5-397B-A17B on an M5 64GB MacBook Pro; another report describes MoE SSD streaming on a 64GB Mac mini where "GPU still waits 27% of decode." — [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1s0mqto/has_anyone_tried_this_flashmoe_running_a_397b/); [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1wy5fma/moe_ssd_streaming_on_a_64_gb_mac_mini_gpu_still/)

### Inferences
- The two flash-moe repos are the strongest published evidence that SSD expert-streaming with "trust the OS" page cache + parallel pread beats custom caching, mmap, and expert prediction on Apple Silicon — a directly transferable lesson for a PS5 engine (which also has unified memory).
- The decisive design lessons for a PS5 port: (1) mmap of expert files is −5x worse than pread on cold data; (2) expert prediction/speculation was net-negative (−18% temporal, −38% speculative early routing, 31% MLP accuracy) — so a PS5 engine should NOT invest in expert prediction; (3) LZ4 compression is net-negative (−13%); (4) serial GPU→SSD→GPU pipeline is optimal on unified memory; (5) F_NOCACHE helps at low bit-widths only.
- tayoun's result shows a smaller-A3B model (35B) streams much faster than a 397B-A17B model on the same approach — the engine cost scales with K×experts loaded per token, not total model size; bigger active-K hurts (K=6 used on M4).

### Gaps
- No single-number breakdown of tayoun's per-layer timing or its exact SSD model/read throughput was published in the README (only headline tok/s and TTFT). The `docs/optimization-experiments-q4.md` file would have details but wasn't fetched.
- The M5 64GB 397B claim and the 64GB Mac mini "GPU waits 27%" figures are community claims without a primary published protocol.

---

## Key Question 2: Current state of MoE-Direct (32 GB RAM + RTX 5080 + Gen5 NVMe, Qwen3.5-122B)

### Takeaway
MoE-Direct (tmxkzm1925-max) is a byte-preserving, llama.cpp-patched (b10057) expert-streaming engine for Windows/NVIDIA CUDA. On a reference desktop (32 GB RAM, one RTX 5080, Gen5 NVMe) it measured Qwen3.5-122B Q4_K_M at 5.59–5.69 tok/s sustained decode — about 2.3x the same binary reading the same weights through plain mmap — and passed its frozen release-gate. It is explicitly "MoE models only."

### Cited Findings
- Reference machine: 32 GB RAM, one RTX 5080, Gen5 NVMe. Qwen3.5-122B Q4_K_M (72.8 GB routed-expert data on disk): 5.59–5.69 tok/s sustained decode, "about 2.3x the same binary reading the same weights through plain mmap," grade `OFFICIAL` (release-gate pass). — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)
- gpt-oss-120b MXFP4 (61 GB expert data): 12 of 12 greedy responses token-identical to the stock read path, grade `OFFICIAL`. Kimi K2.6 1T-class mixed-quant (436 GB): 1.03 tok/s coherent output, server ready ~19 s, grade `PROBE` (performance gate not passed). — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)
- Design: "No re-quantization. No routing change. No modified weights." Keeps experts on NVMe, streams on demand; on the default packed path every expert byte is SHA-256 verified against the original GGUF before use; opt-in virtual path copies nothing (manifest + plan report, ~6 MB for 122B, ~16 MB for 397B); server never bulk-loads experts up front. — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)
- Supported/validated models: Qwen3.5-122B-A10B Q4_K_M (256 experts, top-8, 72.8 GB, `reference-validated`); gpt-oss-120b MXFP4 (128 experts, top-4, 61 GB); Qwen3.5-35B-A3B Q4_K_M (256, top-8, 19.5 GB); Qwen3.5-397B-A17B Q4_K_M (512, top-10); Kimi K2.6 447 GB mixed-quant (384, top-8, 436 GB); DeepSeek-V4-Flash-0731 MXFP4/Q8_0 (256, top-6). — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)
- Historical notes: "Results scale primarily with NVMe read throughput"; numbers are reference-machine data points, not promises. v0.3.2 (2026-10-14) replaced the launcher with a Go launcher and opened manual prefetch (K, N, QD); no engine performance change vs v0.3.1.1. A preregistered A/B probe: prefetch made in-place decode ~14% faster at under 2% extra bytes read (PROBE grade, predates release binary). A prefill path that reads each expert once per request instead of once per token is in the release queue (~3x in an internal probe, not yet published). — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)
- Roadmap: engine-neutral expert-execution core pulled behind a clean boundary with llama.cpp as first engine behind it; KV-cache tiering across VRAM/RAM/NVMe for million-token context (sparse-attention models). — [MoE-Direct README](https://github.com/tmxkzm1925-max/moe-direct)

### Inferences
- MoE-Direct independently confirms the pread/direct-read-over-mmap lesson from flash-moe, but on NVIDIA/Windows: ~2.3x over plain mmap for Qwen3.5-122B. This is the strongest cross-platform confirmation that mmap is the wrong path for cold expert reads.
- It demonstrates that a 72.8 GB expert set (122B Q4_K_M) can sustain ~5.6 tok/s on a single consumer desktop with 32 GB RAM — a useful throughput anchor for a PS5 engine (which has ~11 GB usable and a slower GPU/NVMe).
- MoE-Direct's "byte-preserving / no modified weights" and per-expert SHA-256 verification is a design choice worth noting for a PS5 engine's correctness story.

### Gaps
- The exact pread-vs-mmap "~2.3x" figure is stated as "about 2.3x" in the README; the full protocol lives in `docs/measured-results.md` and `TECHNICAL.md` (not fetched). The paired probe gave 5.65–6.26 tok/s per probe (arm average 5.96), so the sustained figure is slightly conservative.
- No Linux/macOS build exists yet (Windows/CUDA only as of Oct 2026); no PS5 port.

---

## Key Question 3: Current state of moe-stream (M4 Pro 24 GB, 80B Q4_K_M)

### Takeaway
moe-stream (GOBA-AI-Labs) is a Rust/Metal SSD-streaming engine with 3 auto-selected inference modes (GPU Resident / GPU Hybrid / SSD Streaming), pruned-model support, and an OpenAI/MCP/Tauri stack. On an M4 Pro 24 GB it measured the 80B Qwen3-Coder-Next at ~2.1 tok/s SSD-streaming (24.4 GB 50%-pruned) and ~0.6 tok/s for the original 48 GB Q4_K_M; smaller MoE models run GPU-resident at 17–55 tok/s. It is a small, lightly-maintained project (3 stars, 11 commits).

### Cited Findings
- moe-stream is a Rust engine (moe-stream-core) + axum HTTP/MCP server + Tauri desktop app; Metal GPU kernels: fused MXFP4 matvec, Q5_0/Q8_0 attention, RoPE, RMSNorm; Q4 quantized matmul computes directly on Q4 weights (skip dequantization) for "+79% speedup"; custom GGUF reader (mmap + expert slicing); EngineHandle pattern (engine on dedicated thread, channel bridge). — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)
- Inference modes: GPU Resident (GGUF < 80% RAM), GPU Hybrid (80–90%), SSD Streaming (> 90% RAM, minimal resident memory, experts from NVMe). On CUDA the auto-select uses GPU VRAM instead of system RAM. — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)
- Performance (M4 Pro 24 GB, internal NVMe): GPT-OSS-20B Pruned (MXFP4, 10.4 GB) GPU Resident ~20 tok/s; 30B-A3B Pruned-80% (Q4_K_M, 14.0 GB) GPU Resident ~55 tok/s; 80B 50% pruned (Q4_K_M, 24.4 GB) SSD Streaming ~2.1 tok/s; 80B Q4_K_M original (~48 GB) SSD Streaming ~0.6 tok/s. — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)
- Supported models table: Qwen3-Coder-Next 80B (36 DeltaNet + 12 Attention, 512 experts top-10 + shared, 80B total / 3B active) ~2.1 tok/s SSD Streaming; Qwen3-30B-A3B (48 Attention, 128 experts top-8) ~55 tok/s GPU Resident; GPT-OSS-20B (24 layers, 32 experts top-8, MXFP4) ~17 tok/s GPU Resident. — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)
- PrunedHub models (HF, goba-ai-labs): PrunedHub-GPT-OSS-20B-28x (10.4 GB, MMLU 78%, lossless); PrunedHub-GPT-OSS-20B-27x-Zerobias (9.4 GB, 77%); PrunedHub-Qwen3-30B-A3B-JP-80pct (14.0 GB, 74%); PrunedHub-Qwen3-Coder-Next-50pct (24.4 GB, 72%, 80B model with 50% experts kept). — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)
- SSD Streaming architecture: Metal GPU runs embed + LM head only; CPU computes MoE experts, DeltaNet, attention, norms; GGUF mmap'd on NVMe. — [GOBA-AI-Labs/moe-stream README](https://github.com/GOBA-AI-Labs/moe-stream)

### Inferences
- moe-stream's numbers show SSD expert streaming on a 24 GB Apple Silicon machine is slow for an 80B model (~2.1 tok/s even pruned to 50%), versus ~11.5 tok/s for a 35B-A3B on the flash-moe approach. Active-parameter count per token dominates achievable tok/s.
- moe-stream uses mmap for the GGUF (unlike flash-moe's pread), which may explain part of its lower throughput vs flash-moe — consistent with the mmap-vs-pread lesson.
- Pruning (keeping 50% of experts) roughly tripled 80B throughput (0.6 → 2.1 tok/s), a lever relevant to a PS5 engine that must fit a model into ~11 GB.

### Gaps
- moe-stream is a small project (3 stars, 11 commits, 4 forks) with limited validation; the "Q4 direct matmul +79%" and mode numbers are README self-reports without a published protocol. CUDA backend uses candle, no custom CUDA kernels.

---

## Key Question 4: Has anyone published a 100B+ MoE streaming result on a single PS5 (any loader)?

### Takeaway
No published 100B+ MoE SSD-streaming result on a single PS5 console was found. The best single-PS5 results are PS5LM/PS5LM Plus running dense/small models (Qwen 3.8 27B at 2-bit at ~20 tok/s is "the biggest that fits"), constrained to ~11 GB of model+context. The only 100B+ MoE on PS5 silicon is a 5-board BC-250 cluster (gfx1013, 16 GB each) that served Qwen3.5-122B-A10B at 29.5 tok/s — i.e., multi-node, not a single console.

### Cited Findings
- PS5LM (cobanov, GPL-3.0, jailbroken PS5 firmware 7.00–13.60) runs llama.cpp's Vulkan backend on the console's RDNA 2 GPU via RADV (PS5_Vulkan port). Measured on the console: Qwen 3.5 0.8B Q4_K_M at 207 tok/s; Qwen 3.5 9B Q4_K_M with 64k context at 55 tok/s; Qwen 3.8 27B UD-Q2_K_XL at 20 tok/s ("the biggest that fits"). Uses 11.4 GB of shared memory for model + context; "About 11 GB of model and context" is the practical limit. — [PS5LM](https://ps5lm.cobanov.dev/)
- PS5LM grew from PS5LM Plus (the native GPU app by abhi245y); a separate report (wccftech, ~2 days before Oct 16 2026) describes a jailbroken PS5 running Qwen 3.8 27B at ~20 tok/s as a local AI server, with smaller models up to 124 tok/s. — [wccftech](https://wccftech.com/jailbroken-ps5-local-ai-server-27b-llm/); [back2gaming](https://www.back2gaming.com/news/ps5lm-qwen-3-8-27b-jailbroken-ps5-gpu/)
- llama.cpp Discussion #28226 (Sep 2, 2026): five AMD BC-250 boards (the "PS5 APU on a blade" crypto cards, gfx1013, 16 GB unified, no matrix cores) on a 2.5GbE switch, orchestrated with llama.cpp's Vulkan backend + RPC pipeline parallelism (#18626, b10727), served Qwen3.5-122B-A10B (IQ3_S, 43 GiB) across 4 nodes: prefill 96→162 t/s (+68% with shmem_staging delta) and generation 19.0→29.5 t/s (+55%). "As far as we can tell from the published record, no BC-250 cluster has served anything bigger than ~35B before." — [llama.cpp Discussion #28226](https://github.com/ggml-org/llama.cpp/discussions/28226)
- Price/context table from #28226 (community numbers): 5x BC-250 cluster ~$500 in boards → 29.5 t/s on Qwen3.5-122B; Ryzen AI Max 395/Strix Halo 128GB ~$2,000 → 28.5 t/s (FP4); Mac Studio M3 Ultra 256GB ~$5,600 → 57 t/s (mxfp4); Mac Mini M4 Pro 64GB ~$2,000 → ~2 t/s ("does not really fit"). — [llama.cpp Discussion #28226](https://github.com/ggml-org/llama.cpp/discussions/28226)
- Single BC-250 board (akandr/bc250) achieves 78.7 tok/s MoE @ 40 CU and 129.2 tok/s hybrid-Mamba @ 40 CU with 64K filled ctx — but these are dense/MoE models that fit the 16 GB, not 100B+ streaming. — [akandr/bc250](https://github.com/akandr/bc250)

### Inferences
- A single PS5 console has ~11 GB usable for model+context, so a 100B+ MoE (e.g., 122B-A10B at ~43–74 GB) cannot fit and must be SSD-streamed; no one has published a working single-console SSD-streaming 100B+ result as of Oct 2026 — this is an open gap the PS5 build spec could fill.
- The 5-board BC-250 cluster proves the gfx1013 silicon (same family as PS5) can serve 122B-A10B at ~29.5 tok/s when experts/weights are distributed across nodes — evidence that PS5-silicon-class compute is capable of 100B+ MoE inference, but only via multi-node or streaming.
- The BC-250 cluster used Vulkan + RPC and `LLAMA_ARG_LOAD_MODE=dio` (direct I/O), which is relevant to how a PS5 SSD-streaming engine might be built.

### Gaps
- No source was found for a single PS5 console running a 100B+ MoE (any loader: llama.cpp, custom, etc.). The strongest single-console evidence is the 27B 2-bit result. This should be stated as "no published result found" rather than "impossible."

---

## Key Question 5: New measurements for Qwen3.5-122B-A10B / gpt-oss-120b on gfx1013 / BC-250 / Radeon silicon (llama.cpp Discussion #28226, TechMakesArt/llama.cpp-bc250)

### Takeaway
The BC-250 cluster (#28226) published first gfx1013 shmem_staging numbers: Qwen3.5-122B-A10B (IQ3_S) at 29.5 tok/s generation across 4 nodes, with prefill 162 t/s. TechMakesArt/llama.cpp-bc250 is a Vulkan-optimization fork for gfx1013 with measured single-node gains on 9B Q4_K_M (37→54.99 tok/s, +48.6%) driven mostly by a community SMU governor (+29.7%), not by llama.cpp changes. No published gpt-oss-120b measurement on gfx1013/BC-250 was found; gpt-oss-120b on Radeon silicon is documented on Radeon 890M/Strix Halo (Vulkan, ~19–20 tok/s tg) and Ryzen AI Max 395 (ROCm, 50.2 tok/s tg).

### Cited Findings
- #28226 (MelonUsk1337, Sep 2 2026): 5x BC-250 cluster served Qwen3.5-122B-A10B IQ3_S (43 GiB) across 4 nodes + dirigent: FA off → FA on + forced shmem_staging delta prefill (3.6k-token prompt): 96→162 t/s prefill (+68%), generation 19.0→29.5 t/s (+55%). Context 28k, `LLAMA_ARG_LOAD_MODE=dio`, -np 1, prompt cache warm. Production synthesis job cut 803 s → ~440 s (−45%). shmem_staging is the NVIDIA-gated FA path from #19625 forced on; single-node A/B (gemma-4-26B MoE IQ3_S): +12% pp at d0, +34% at d4096, +50% at d8192 (issue #25207). Smaller models: gemma-4-26B MoE over RPC across 2 boards pp 508 t/s / tg 73.5 t/s; Qwen3.6-35B-A3B MoE across 3 boards pp 559 t/s / tg 68 t/s (IQ4_XS). — [llama.cpp Discussion #28226](https://github.com/ggml-org/llama.cpp/discussions/28226)
- TechMakesArt/llama.cpp-bc250 (fork pinned to upstream f772f6e43 / b8819): three Vulkan changes — rm_kq=4 on RDNA1 (+3% tg), Q4_K smin algebraic restructure (+3% tg), fused gate+up+SWIGLU_SPLIT Vulkan kernel for Q4_K (+5.8% tg). Measured on Qwen 3.5-9B Q4_K_M: single-stream tg 37.00→54.99 tok/s (+48.6%), pp 199.66→~306 (+53%), batched-32 aggregate ~100→151 tok/s (+51%), batched-64 →175 tok/s. Contribution stack: baseline 37.00; +fused gate+up 39.15 (+5.8%); +Q4_K smin 41.08 (+5.2%); +RDNA1 rm_kq=4 42.33 (+3.0%); +SMU governor 54.99 (+29.7%, biggest single win, third-party work). Perplexity unchanged within noise (±0.025). — [TechMakesArt/llama.cpp-bc250 README](https://github.com/TechMakesArt/llama.cpp-bc250)
- BC-250 hardware context (TechMakesArt): "Cyan Skillfish" silicon, cut-down PS5 APU, Zen 2 + gfx1013 iGPU, 16 GB GDDR6 unified, 448 GB/s peak; effective memory bandwidth ~290 GB/s (~65% of peak) across quants; SMU governor unlocks shader clock from 1500 MHz (driver-pinned) to 2300+ MHz; community BIOS required for dynamic VRAM; TTM pages_limit tuning required for models over ~5 GB. Featured MoE model: Gemma 4 26B-A4B UD-Q3_K_M (11.65 GiB, 25.2B total / ~4B active) at 69.66 tok/s tg128, 449.76 tok/s pp512, 24k context with KV q8_0. Speculative decoding does not deliver on BC-250/Vulkan (draft acceptance collapses to ~0.8% for Qwen3.5 9B; Llama 3.1 8B+1B draft 72% acceptance but identical wall-clock throughput). — [TechMakesArt/llama.cpp-bc250 README](https://github.com/TechMakesArt/llama.cpp-bc250)
- gpt-oss-120b on Radeon 890M (Strix Halo, Vulkan, RADV GFX1150, ceroma benchmark in #15396): 59.02 GiB / 116.83 B params, tg128 18.98–19.58 tok/s, pp2048 74.58–92.62 tok/s across ub sizes; later build (Sep 15) pp1024 165.48, tg512 19.64. — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- gpt-oss-120b on Ryzen AI Max 395 (ROCm, gfx1151, quozul.dev): pp512 707.09 tok/s, tg128 50.20 tok/s (--mmap 0). — [quozul.dev](https://quozul.dev/posts/2025-10-27-gpt-oss-120b-on-ryzen-ai-max-395/)
- gpt-oss-120b on AMD 7800X3D + 7900XTX (Vulkan, community): ~11.23 tok/s (oobabooga llama.cpp). — [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1n1oz10/gptoss120b_running_on_an_amd_7800x3d_cpu_and_a/)

### Inferences
- The gfx1013/BC-250 silicon can serve 122B-A10B at ~29.5 tok/s when spread across 4–5 nodes, and a single node can do ~55–70 tok/s on models that fit (9B Q4_K_M, Gemma 4 26B-A4B). This brackets what a single PS5 (same silicon family, 16 GB unified, but console firmware/homebrew constraints) could plausibly achieve for SSD-streamed 100B+ MoE: likely well below 29.5 tok/s single-node, but non-trivial.
- The SMU governor (clock unlock) is the single biggest gfx1013 performance lever (+29.7%), a hardware-level lesson that matters if the PS5's own clock governor limits shader clocks during inference.
- gpt-oss-120b MXFP4 is native to llama.cpp and runs on Vulkan/Radeon, but no gfx1013/BC-250 measurement was published; the closest Radeon data is Strix Halo (890M) at ~19–20 tok/s tg and AI Max 395 (ROCm) at 50 tok/s tg.

### Gaps
- No published gpt-oss-120b measurement specifically on gfx1013 / BC-250 silicon was found — only the 122B-A10B cluster result (#28226) and single-node small-model results.
- TechMakesArt/llama.cpp-bc250 is a snapshot fork pinned to b8819, not actively maintained; the gpt-oss-120b question on BC-250 remains unanswered.

---

## Key Question 6: Current state of llama.cpp's Vulkan backend and any gpt-oss GGUF support

### Takeaway
llama.cpp's Vulkan backend is actively used and production-capable (BC-250 cluster orchestration, PS5LM, Strix Halo, RX 9000 series); it supports native MXFP4 and gpt-oss. gpt-oss GGUF support is confirmed and official: llama.cpp supports gpt-oss in native MXFP4 format across CUDA, Vulkan, Metal, and CPU backends (announced by ggerganov, built with OpenAI/HuggingFace/NVIDIA AI PC). Known Vulkan-specific issues exist (Jinja fails on gpt-oss-120b with Vulkan; speculative decode path is weak on Vulkan).

### Cited Findings
- "Llama.cpp supports the new gpt-oss model in native MXFP4 format... can run the new gpt-oss model with all major backends, including CUDA, Vulkan, Metal and CPU at exceptional performance." Work done in collaboration with OpenAI, HuggingFace, and NVIDIA AI PC. — [ggerganov on X](https://x.com/ggerganov/status/1952779751736627627)
- Official ggml-org GGUF repos exist for gpt-oss-20b and gpt-oss-120b (used in guides: `-hf ggml-org/gpt-oss-120b-GGUF`), and the running guide #15396 documents NVIDIA, AMD, and low-end configs for gpt-oss via llama.cpp. — [llama.cpp Discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396); [DebuggerCafe](https://debuggercafe.com/gpt-oss-inference-with-llama-cpp/)
- Vulkan backend is the orchestration layer for the BC-250 cluster (#28226, b10727, RPC pipeline parallelism #18626) and for PS5LM (PS5's RADV port). Vulkan is a first-class compile target (`GGML_VULKAN=ON`, builds against stock Vulkan SDK packages, no ROCm/AMD driver needed). — [llama.cpp Discussion #28226](https://github.com/ggml-org/llama.cpp/discussions/28226); [agledger.ai Strix Halo guide](https://agledger.ai/guides/local-llm-strix-halo-ubuntu-26-04/)
- Vulkan-specific bugs: "Jinja fails on gpt-oss-120b when using Vulkan" (with both HIP and Vulkan backends, ub sizes 512–8192, all llama.cpp versions) — issue #15274. Speculative decoding throughput gap on Vulkan (BC-250): 72% draft acceptance but identical wall-clock throughput (~55 tok/s both), "the Vulkan backend's batched verify path" blamed. — [llama.cpp Issue #15274](https://github.com/ggml-org/llama.cpp/issues/15274); [TechMakesArt/llama.cpp-bc250 README](https://github.com/TechMakesArt/llama.cpp-bc250)
- Vulkan performance on gpt-oss-20b (NVIDIA forum, RTX 5090-class): Vulkan0 tg128 56.78–68.00 tok/s vs CUDA0 66.65–73.20 tok/s; Vulkan pp512 ~1778–2764 tok/s vs CUDA ~3088–4155 tok/s — Vulkan roughly 60–80% of CUDA on the same GPU. — [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/vulkan-as-alternative-backend-for-llama-cpp/363516)
- Phoronix end-of-year 2025 review: with GPT-OSS 20B, RX 9070 series came out slightly ahead of RTX 5070 in the TG128 benchmark on Vulkan; "Radeon RX 9070 performance-per-Watt" noted. — [Phoronix](https://www.phoronix.com/review/llama-cpp-vulkan-eoy2025/4)

### Inferences
- For a PS5 engine, llama.cpp's Vulkan backend is the proven path (PS5LM already runs it on the console's RDNA 2 via RADV), and gpt-oss GGUF/MXFP4 is natively supported — so both Qwen3.5-122B-A10B and gpt-oss-120b are viable targets with existing GGUF artifacts.
- Vulkan decode on gfx1013-class hardware is bandwidth-bound and roughly 60–80% of CUDA on the same silicon; a PS5 engine should expect similar or worse relative to NVIDIA, and should plan around the Vulkan-specific weaknesses (Jinja on gpt-oss-120b, speculative decode).

### Gaps
- The exact current build number/feature set of the Vulkan backend as of Oct 2026 (e.g., whether gpt-oss Jinja failure #15274 is fixed) was not confirmed; the issue was still open/active in the sources found. gpt-oss MXFP4 native support is confirmed but the precise merged PR/commit was not pinned.
