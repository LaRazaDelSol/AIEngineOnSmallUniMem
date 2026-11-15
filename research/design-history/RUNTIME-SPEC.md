# RUNTIME-SPEC — PS5LM custom streaming inference engine (16 GB unified, Zen 2 + RDNA2)

> **Superseded by `docs/ENGINE-SPEC.md` (rev 3, 2026-11-07)**, which is built from the console measurements: Vulkan on GFX1013 (no ROCm, no accelerated DP4A), EXL3 experts, measured I/O physics.

> Status: ACTIVE DESIGN (rev 2, 2026-11-03). **Design point = theoretical PS5 spec** (directive
> 2026-11-03: design for the theoretical hardware until it can be tested; always the custom,
> maximum-efficiency path, never the lazy one). Measured console values are kept as a
> **parameterized fallback column**, not the design point. Tags: **[T]** theoretical spec,
> **[M]** measured on our console (cite register), **[D]** derived, **[C]** implemented in code
> and tested in this repo, **[A]** assumed, with a named gate that converts it.

> **Known gaps (S12 handoff):** this rev does NOT yet include speculative decoding + the R2
> acceptance gate, n-gram-table streaming (51B table, every token), the trained prerouter,
> co-activation pack ordering, wave prefill, or the disk KV cache, even though the research measured
> large gains for each. Gap table and sources: `research/lab-notebook/2026-11-03-s12-custom-runtime-handoff.md` §3.

## 0. Design point vs. measured fallback

| Quantity | Design point [T] | Measured fallback [M] | Where it is a parameter |
|---|---|---|---|
| GPU ISA | gfx1030, RDNA2, 36 CU, wave32, dot1-insts (`v_dot4_i32_i8`) | RADV reports GFX1013 (S8 G0) | `PS5LM_GPU_ARCH`; `PS5LM_SOFT_DOT4=ON` emulates dot4 bit-exactly |
| GDDR6 bandwidth | 448 GB/s | ~290–314 GB/s effective | ledger |
| NVMe (Gen4 x4) | 5.5–7.0 GB/s | NAND-true seq 3.75–3.83, rand 1–2.6 MiB 1.59–2.45 GB/s (S11) | `stream_bench`, ledger BW |
| Usable RAM | 16 GiB, OS ≤ 200 MB | 14 GiB free (S9) | arena budget, hugepage counts |
| CPU | 8C/16T Zen 2, 2 CCX | same | core map (§1.3) |

**Not a parameter: WMMA.** RDNA2 has no WMMA in its ISA (it starts at RDNA3/gfx11), so no test can change that. The RDNA2 int8 matrix path is DP4A (`v_dot4_i32_i8`), and that is what the kernels use (§3.2).

## 1. System stripping & kernel tuning — `ps5/appliance/` [M] (S15: rewritten to what was measured)

`ps5lm-min` (runtime `on`/`off`/`status`, boot params via `cmdline`) plus two boot modes gated on the kernel flag
`ps5lm.ai=1`: `ps5lm-min.service` (strip at boot), `ps5lm-server.service` (API on :8081), and a getty@tty1 drop-in
that keeps the gaming session off. See `ps5/appliance/README.md`. The S12 drafts listed here before (`cmdline.conf`
with 1 GiB hugepages + `isolcpus`, sysctl/udev files, `ps5lm-strip.sh`, CPU-layout + `ps5lm-runtime` units) were
removed: wrong SMT map, nonexistent binary, and they contradicted the measured `ps5lm-min`.

**Scheduler:** EEVDF and BORE only arbitrate `SCHED_OTHER`. Every engine thread is `SCHED_FIFO` on an isolated core, so the choice affects housekeeping only. Use stock `linux-cachyos` and no sched_ext.

### 1.3 Core map (`platform.hpp: cores::`) [C]

| CPU | Role | Policy |
|---|---|---|
| 0 | io_uring **SQPOLL kernel thread** (+ IOPOLL completion polling) | kernel |
| 1 | **streamer**: router mailbox → slot binding → READ_FIXED → reap → `ready[L]` release | FIFO 80 |
| 2 | **GPU queue submission**: enqueues a whole token's graph once | FIFO 70 |
| 3 | sampler + token feedback (logits → next token → enqueue) | FIFO 60 |
| 4–7 | tokenizer, HTTP endpoint, prefill staging, offline pack tooling | OTHER, nice −10 |
| 15 | housekeeping, all IRQs (sibling of 7) | — |
| 8–14 | offline | — |

Cores 2–7 are the GPU host side. Cores 0–1 are I/O, and both I/O cores are always busy: CPU 0 polls the NVMe completion queues, and CPU 1 runs the cache. There is no interrupt and no syscall on the hot path (SQPOLL + IOPOLL + registered files and buffers).

### 1.5 Memory locking, HugeTLB, zero-copy [C]

- `platform::lock_all_memory()` calls `mlockall(MCL_CURRENT|MCL_FUTURE|MCL_ONFAULT)` after raising RLIMIT_MEMLOCK.
- `platform::map_pinned()` tries **1 GiB hugetlb**, then 2 MiB hugetlb, then THP plus `mlock`. Every mapping is `MAP_POPULATE`, so the hot path never takes a page fault. Hugetlb pages are never swapped, migrated or compacted.
- `Arena` (`arena.hpp`) is the zero-copy tensor manager. It holds one pinned mapping carved into named regions: `[resident weights | shared experts | KV | activations | staging | experts]`. Every tensor is an offset, and nothing calls malloc after load.
- The experts region is registered once with **io_uring** (`io_uring_register_buffers`, in ≤ 1 GiB chunks with no slot straddling a chunk) and once with **HIP** (`hipHostRegister(Mapped|ReadOnly)` → `hipHostGetDevicePointer`). NVMe DMA writes the bytes and the GPU kernels read them in place, with no copy on unified GDDR6.
- Control memory (router mailboxes, `ready[]`/`done[]` sequence words, slot-id tables) is `hipHostMallocCoherent|Mapped`. The CPU and GPU synchronize by polling these words, not by syscalls or interrupts.
- [A→gate M2] GPU read bandwidth through host-registered memory vs `hipMalloc` memory. If it is more than 15% lower, add an on-GPU promote-copy of hot slots into a device arena (~400 GB/s, which is negligible next to the SSD).

## 2. 100B+ execution: the streaming engine

### 2.1 Physics

`tok/s = SSD_BW / (misses_per_token × slot_bytes)`, with `misses_per_token = reads × (1 − h)`.

- **Dense 100B @ INT4:** every token reads ~50 GB. At 7.0 GB/s that is **0.14 tok/s**, and pipeline parallelism cannot reduce bytes. Dense streaming is infeasible, so the engine streams **fine-grained MoE experts**.
- **Coarse MoE (8 × top-2):** experts of ~170 MB at 4-bit mean a 10 GiB pool holds about 60 of them, so hit rates are low. Fine-grained MoE (512 experts, top-10, ~1.5 MiB each at Q2) is what makes a 16 GB pool viable.

Design-point ledger. Hit rates are [C], from `trace_replay --policy clock` on the synthetic agent trace (2000 tokens). On-console Zipf hit rates (S11: 86–89%) are lower, and real routing dumps decide.

| Pack | Pool | Hit | Misses/token | MB/token | @7.0 GB/s [T] | @5.5 GB/s [T] | @2.3 GB/s [M] |
|---|---|---|---|---|---|---|---|
| **PS5Q2** (2.5 bpw, 1,556,480 B slot) | 10 GiB | 90.7% | 44.4 | 69.1 | **101** | 80 | 33 |
| PS5Q2 | 11.6 GiB | 91.4% | 41.3 | 64.3 | **109** | 86 | 36 |
| PS5Q4 (4.5 bpw, 2,785,280 B slot) | 10 GiB | 86.4% | 65.5 | 182 | 38 | 30 | 13 |
| PS5Q4 | 11.6 GiB | 88.5% | 55.4 | 154 | 45 | 36 | 15 |

GPU-side cost per token at Q2: 480 × 1.56 MB ≈ 0.75 GB of expert weights read from GDDR6 (1.7 ms at 448 GB/s), plus the dense part. **At the design point the SSD, not the GPU, is the bound.**

**Queue-depth requirement.** At 90% hit there are ~0.9 misses per layer visit. If routing is layer-serial, that is qd ≈ 1 and the rated bandwidth is unreachable. The engine therefore **predicts K layers ahead on the GPU** (§2.4). `stream_bench --lookahead K` measures the oracle ceiling of that predictor on real hardware.

### 2.2 Pack format — PS5Q (`quant.hpp`) [C]

- **Blocks:** 32 weights + an fp16 scale. Q4 is 4.5 bpw and Q2 is 2.5 bpw. Quantization is offline with an MSE-optimal scale search plus a least-squares refit. Rel-RMS on Gaussian weights is 0.078 for Q4 and 0.323 for Q2, which is at the floor for a 4-level scalar quantizer. Going lower at the same bpw needs vector/trellis codes (EXL3 track, Eli).
- **Rows** are SoA (scales padded to 16 B, then quants). A wave32 reads 16 B per lane, fully coalesced.
- **Slots** are exactly 680 × 4 KiB (Q4) and 380 × 4 KiB (Q2), so O_DIRECT needs no padding. Pack layout: `[layer][expert]`, hot experts adjacent per layer (arena pinning).
- **Tests** (`test_quant`): fp16 round-trip is exact over all finite halves. The integer DP4A path equals the dequantized float dot. The full expert FFN through the Q8 activation path is within 1.1–1.2% rel-RMS of f32.

### 2.3 Memory budget (16 GiB design point)

| Region | Size |
|---|---|
| OS userspace | ≤ 0.2 GiB |
| Kernel + firmware carve-outs | measure (`16 − MemTotal`) [A→M1] |
| Resident non-expert weights | ~4.0 B params [D] → 2.1 GiB @ Q4 / 1.2 GiB @ Q2 |
| Router weights (fp16, also used by the GPU predictor) | 48 × 512 × 2560 × 2 B = 0.12 GiB |
| Shared experts | 0.13 GiB (Q4) |
| KV (12 full-attention layers) + activations + staging | ~1 GiB |
| **Expert slab** | **10 GiB (10 × 1 GiB hugetlb) → 6,898 Q2 slots (28% of the 24,576 experts)** |

F29's 5.5–6.5 GB non-expert estimate still disagrees with the [D] subtraction. Gate M1 settles it, and the slab is sized from the result.

### 2.4 Decode pipeline (`moe_pipeline.hpp`) [C, untested on GPU]

```
CPU 2 (once per token)          GPU stream                                  CPU 1 streamer (pump loop)
enqueue_token():                pre_moe(L)  [DenseOps hook]
  for L in 0..47                quantize_act_q8(x) → xq
                                router_topk(L): ids(L) + predict(L+1..L+K) ─► mailbox[L].seq
                                WaitValue32(ready[L] ≥ seq) ◄──────────────  ids → request+pin (demand)
                                                                             predictions → request (speculative)
                                                                             all slots ready → slot_ids[L], route_w[L]
                                                                             release ready[L] = seq
                                gate_up_swiglu → quantize(h) → down_combine
                                WriteValue32(done[L] = seq) ──────────────►  unpin L's slots
                                post_moe(L) [DenseOps hook]
```

- **The GPU predicts its own prefetch.** `router_topk` launches with `grid.y = 1 + K`. Block y applies layer L+y's router weights to layer L's hidden state, which is the Pre-gated-MoE idea fused into the existing router launch. The streamer issues those reads while layer L computes.
- **The host never submits per layer.** The whole token is enqueued once, and the GPU stalls in hardware on `hipStreamWaitValue32` only when a layer's experts are not yet resident.
- **Invariants:** `ready[L]` is released only when every slot of L is resident. Slots stay pinned from bind until `done[L]`. Retire runs before new-arrival handling, so pins never leak across tokens. Failed reads invalidate the slot and are re-requested.
- **Mailbox publication:** the last router block to finish (by atomic counter) publishes `seq` with system-scope release, so the host never sees ids or predictions from an older token.

## 3. Compute engine & driver stack

### 3.1 ROCm/HIP vs Vulkan — decision: **HIP primary (gfx1030), Vulkan secondary**

| | HIP (chosen) | Vulkan/SPIR-V |
|---|---|---|
| ISA control | `__builtin_amdgcn_sdot4`, wave32 shuffles, `__launch_bounds__`, inline asm if needed | through the SPIR-V → ACO compiler; `VK_KHR_shader_integer_dot_product` only |
| Zero-copy | `hipHostRegister` + device pointer: kernels dereference slab bytes directly | `VK_EXT_external_memory_host` (measured present) |
| GPU waits on host memory | `hipStreamWaitValue32` / `WriteValue32` (whole-token enqueue) | timeline semaphores, host-signalled per layer |
| Kompute | — | wrapper with no advantage over raw Vulkan; rejected |
| Risk | ROCm support for the console's GPU ID [A→gate C1] | already runs on the console (PS5LM) |

The kernels go through a C-ABI launch layer (`ps5lm_launch_*`), so a Vulkan backend can implement the same three entry points if gate C1 fails.

### 3.2 Matrix math on RDNA2 (`kernels/moe_kernels.hip`) [C, needs GPU]

- **DP4A everywhere on the weight path.** One `v_dot4_i32_i8` covers 4 weights × 4 activations. Q4 unpacks with one AND (+ shift) per 4 weights, and Q2 with one shift+AND per 4. The block math is `dw·da·(Σdot4 − zero·Σqa)`, with the activation sum precomputed during quantization.
- `gate_up_swiglu<B>`: 8 waves per workgroup, one output row per wave. Gate and up share every LDS activation read, and SwiGLU is applied in-register.
- `down_combine<B>`: one row per wave, with the top_k × blocks (expert, block) pairs spread over the 32 lanes for full lane utilization. The routed-weight combine happens in-register, so there are **no atomics** and the result is deterministic.
- `router_topk`: fp16 router GEMV (16 B loads), then k rounds of block argmax, softmax over the top-k, and the fused predictor.
- **Correctness:** `test_kernels_hip` compares against `expert_ffn_ref` (same integer path, so it should match to f32 summation order, < 1e-4 rel) and reports µs/layer and effective weight-stream GB/s.

## 4. Code map (`runtime/`)

| Path | What | Status |
|---|---|---|
| `include/ps5lm/slot_cache.hpp` | shared-pool slot cache, LRU/CLOCK, pins, in-flight states | [C] unit tests + **gate S1 PASS** (exact match with the Python gate, 15 points) |
| `include/ps5lm/quant.hpp` | PS5Q format, fp16, Q8 activations, CPU reference | [C] `test_quant` |
| `include/ps5lm/arena.hpp` | zero-copy tensor memory manager | [C] compiles; used by `stream_bench` |
| `include/ps5lm/platform.hpp` | pinning, FIFO, mlockall, 1G/2M hugetlb | Linux; not yet compiled |
| `include/ps5lm/expert_streamer.hpp` | io_uring SQPOLL+IOPOLL, registered files/buffers, READ_FIXED | Linux; not yet compiled |
| `include/ps5lm/moe_pipeline.hpp` | GPU-driven per-token pipeline | HIP; not yet compiled |
| `kernels/moe_kernels.hip` | DP4A kernels + router/predictor | HIP; not yet compiled |
| `src/trace_replay.cpp` | cache-only replay (gate S1) | [C] |
| `src/stream_bench.cpp` | real-I/O decode replay, oracle lookahead (gates S2/R1) | Linux; not yet compiled |
| `tools/trace_to_txt.py`, `tools/gate_s1.py` | trace bridge, S1 gate | [C] |

Build: `cmake -S runtime -B build && cmake --build build && ctest --test-dir build`. On Linux with ROCm, add `-DPS5LM_HIP=ON` (and `-DPS5LM_SOFT_DOT4=ON` for gfx1013).

## 5. Roadmap (desktop → console), each phase ends in a PASS/FAIL gate

**P0 — any Linux x86_64 box with an RDNA2 card (RX 6700/6800 = gfx1030-class: exact design-point ISA)**
1. Build with `-DPS5LM_HIP=ON`. Gate **K1**: `test_kernels_hip` PASS (rel < 1e-4). Record µs/layer.
2. Gate **S2**: `stream_bench --make-pack`, let the pack fold, then sweep `--lookahead 0,1,2,4,8` × `--compute-us 0,35` at a 10 GiB pool. This gives the layer-serial floor vs the oracle ceiling at the drive's real bandwidth.

**P1 — model integration (custom, no llama.cpp)**
3. Pack converter: HF safetensors → PS5Q pack + resident section + router fp16 table. Gate **Q1**: perplexity at Q4/Q2 vs bf16 on a fixed eval set.
4. `DenseOps`: attention (12 full + linear-attention layers), norms, shared expert, lm_head in HIP. Gate **E1**: logits match a reference implementation token-for-token (greedy).
5. Gate **R1**: real routing dumps measure GPU-predictor precision at K = 1..6 and the achieved tok/s vs the oracle from step 2.

**P2 — console**
6. `ps5lm-strip.sh` + `cmdline.conf`. Gate **O1**: userspace ≤ 200 MB; hugepages reserved; isolated = 0–7.
7. Gate **C1**: GPU ID/ISA. If ROCm will not run, use `PS5LM_SOFT_DOT4` or the Vulkan backend behind the same C ABI.
8. Gate **M1**: real resident weights loaded, then pool sized from what is left. Gate **M2**: GPU read bandwidth from registered host memory.
9. **WD SN810** bench with the S2 protocol on folded data. `nvme id-ctrl` HMB fields test the DRAM-less hypothesis against the Micron.
10. Gate **E2**: end-to-end tok/s within ±10% of the ledger prediction from that run's measured hit rate and bandwidth.

**P3 — hardening:** NVMe heatsink and thermal logging (S11: 25–45% throttle after 50 min), 24 h soak, release.
