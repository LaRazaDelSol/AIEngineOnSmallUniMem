# PROTOTYPE PLAN (S13, 2026-11-06): streamed-MoE decode on the PS5 GPU, Vulkan backend

**Goal:** the shortest path to a running prototype. Qwen3.8-Flash-Next's expert path (which is ≥95% of
streamed bytes) decodes on the PS5 GPU with experts streamed from the SN810 through io_uring. The design
point is the theoretical maximum; each stage ends with a measured number on the console, not a projection.

## Hardware facts this plan is built on (measured S13, handoff §3c)
GFX1013 (RADV, Mesa 26.2.5, Vulkan 1.4), **no accelerated DP4A**, fp16 + int8 storage, wave32/64,
64 KiB LDS, BDA, `VK_EXT_external_memory_host`. No ROCm/hipcc. g++, glslc, liburing 2.15.
SN810 Gen4 TLC with DRAM. 14.1 GiB visible, GTT 14 GiB. SMT siblings = (2k, 2k+1).

## Architecture decisions
1. **Backend = Vulkan compute** (GLSL → SPIR-V via glslc → RADV/ACO). The HIP files stay as reference only.
   One backend for the PC sim and the console.
2. **No descriptor sets:** every kernel takes buffer device addresses in push constants
   (`GL_EXT_buffer_reference`), so an expert slot binding is a 64-bit pointer write.
3. **Kernels (no DP4A):** weights PS5Q Q4/Q2 dequantized in-register to fp32 FMA. One wave per row
   group with x held in registers across rows (x is read once per wave, not per row). Fused gate+up+SwiGLU
   (+48.6% measured on gfx1013 in llama.cpp-bc250). Down + routed-weight combine in one deterministic
   kernel, no atomics. Router matvec + top-10 + renormalise on the GPU, written to a host-coherent mailbox.
4. **Zero-copy bytes path:** expert slab = anonymous (hugetlb when reserved) host memory, registered with
   io_uring (READ_FIXED, O_DIRECT) AND imported into Vulkan via `VK_EXT_external_memory_host`, so the
   NVMe DMA lands where the GPU reads. **Measured, not assumed:** imported-userptr read bandwidth vs a
   native Vulkan allocation (K1b). If snooped GTT is slow, the fallback order is: (a) native host-visible BO
   + plain O_DIRECT `pread` into its mapping, if the kernel accepts it; (b) bounce + GPU copy.
5. **Pipeline v0 (host-driven):** per layer: router → host reads the mailbox → SlotCache lookup →
   io_uring misses → bind slot addresses → expert kernels. The next layer's router is in the same submit.
   v1 = cross-layer prefetch (oracle lookahead first, to measure the ceiling; then a predictor) and a
   GPU-side wait to remove per-layer submit latency.
6. **Workload realism:** synthetic weights at exact Flash-Next geometry (48 × 512 × [3 × 640 × 2560]),
   with routing either from the GPU router (≈ uniform, worst case) or **forced from a trace** (agent
   trace / Zipf) to reproduce realistic locality through the real pipeline.

## Stages and gates (each a console measurement)
| Stage | Deliverable | Gate (PASS = number recorded + correctness) |
|---|---|---|
| E0 | console toolchain (cmake/ninja/vulkan-headers), work dir, sync script | builds |
| E1 | **SN810 NAND-true battery** on old stores (seq/rand 1M/1.5M/2.7M, qd1/4/8/32) | replaces the QLC constants |
| K1 | `vk_kernel_test`: Q4/Q2 gate_up + down_combine + router vs CPU double reference; µs/layer; GB/s | rel-err < 1e-3; GB/s reported |
| K1b | memory-path bench: imported userptr vs native BO read GB/s | picks the slab memory path |
| S2 | `stream_bench` on the SN810 (Q2 pack, agent trace, lookahead 0/1/2/4) | tok/s I/O ceiling per lookahead |
| P3 | **`moe_decode`**: 48-layer expert stack, GPU router, cache + io_uring + kernels end to end | **prototype tok/s** (expert path) + dense-floor estimate |
| P4 | real weights: GGUF/HF → pack converter; dense ops (GDN, QSA, HC, PLE, MTP) | first real token |

## Self-review (adversarial pass on this plan)
- *Risk: the snooped userptr path halves GPU bandwidth.* → K1b measures it before P3 depends on it.
  Fallbacks listed in decision 4.
- *Risk: per-layer host round-trips (~50 µs × 48 ≈ 2.4 ms/token) cap tok/s near 400.* Not binding at the
  target (40–100 tok/s), but P3 reports it separately (`sync_ms/token`) so v1 knows the cost.
- *Risk: synthetic routing flatters or punishes hit rates.* → both modes reported; the trace mode uses the
  same traces as the S1/N7 gates, so the numbers join the existing ledger.
- *Risk: fresh pack = SLC-assisted reads (S11 lesson).* → E1 runs on old cloned stores. The P3 pack is
  re-benched after it folds and labelled FRESH until then.
- *Risk: 10 s DEVICE_LOST.* → the GPU never waits on I/O in v0 (the host binds only resident slots).
- *Scope cut if time runs short:* P4 dense ops are the long pole. P3 is the prototype; P4 starts with the
  converter + PLE/router on real weights.
- *What the plan deliberately does NOT do yet:* MTP/PLD, wave prefill, basis-delta, OEA. They need the
  P3 pipeline as their substrate; their slots in `moe_decode` are reserved.
