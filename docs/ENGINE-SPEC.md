# ENGINE-SPEC — PS5LM inference engine for the PS5 (Oberon: Zen 2 + RDNA 2, 16 GB unified GDDR6)

> Rev 4, 2026-11-07 (S18: 40 tok/s at Q4-class quality reached). Rev 3, 2026-11-07. Supersedes `RUNTIME-SPEC.md` rev 2 wherever they disagree: rev 2 was written before the hardware ran
> and chose HIP + DP4A. The console has since been measured (S8-S16) and the engine runs on it. Every number carries a tag:
> **[M]** measured on the console (ledger entry), **[C]** implemented in `runtime/` and tested, **[D]** derived from [M],
> **[A]** assumption with the gate that settles it. R-numbers (`[M R53]`) refer to entries in the development ledger.

## 0. Summary

| | |
|---|---|
| Model | Qwen3.8-Flash-Next: 48 layers (36 GatedDeltaNet + 12 gated attention), hyper-connections (4 streams x 2560), 512 routed experts + 1 shared per layer (top-10), PLE n-gram table (51B, on disk), MTP head (1 layer, own 512 experts) |
| Quality point | routed experts EXL3 2.68 bpw (served default since S17; GPQA-D 91.4 = 3.18, ERQA 66.5 vs 68.3) or 3.18 bpw (GPQA-D 91.4 vs 91.7 BF16), dense PS5Q4 4.5 bpw, router bf16 |
| Execution | dense weights resident in GDDR6; routed experts streamed from NVMe into a 12 GiB pinned slot cache; one GPU pass per layer, host binds experts between layers |
| Backend (console) | **Vulkan 1.4 compute on RADV/ACO** (BDA pointers in push constants, no descriptors) |
| Backend (PC simulation) | **HIP** on a gfx1030-class dGPU, same C++ engine, same kernels' math, behind one backend interface (§6.3) |
| Measured today | **41.0-42.9 tok/s** cold with EXL3-2.68 experts + MTP speculative decoding (stage-C verify), **41.9 served**; output identical to plain greedy decoding (md5) [M R65-R70]. Plain decoding: 35.7 |
| Target | 40 tok/s at the same quality: **reached** (S18) |

## 1. The hardware, as measured

| Component | Spec sheet | Measured on this console | Consequence |
|---|---|---|---|
| CPU | 8C/16T Zen 2, 3.5 GHz | "AMD Custom Zen 2 (Oberon)", 2 CCX x 4 MiB L3, SMT siblings (2k, 2k+1) [M] | host work per token is ~0.1 ms [M R56]; the CPU is not a bottleneck |
| GPU ISA | RDNA 2 | **GFX1013** (RADV), **no accelerated DP4A**, no WMMA [M] | no ROCm; int8 dot products are emulated -> kernels dequantize in registers and use fp32/fp16 FMA |
| CUs | 36 | 36 + 4 harvested = **40** with `amdgpu.ps5_cu_unlock=3`, AI mode only, bit-exact [M R53] | +10% FMA, +3% tok/s |
| GPU clock | 2.23 GHz | SMU caps the target at 2230; **~2100 under load**; 1962 MHz x 40 CU effective [M R48] | no clock lever left |
| Memory | 16 GB GDDR6, 448 GB/s | MemTotal 14.35 GiB; **512 MiB VRAM carve** (CPU-visible); dense matvecs stream ~270 GB/s [D R52] | the pool is what is left after the OS and the dense weights |
| Storage | 825 GB custom SSD | engine boots from a **WD SN810** (M.2): 6.5 GB/s from qd4; **5.2 GB/s while busy in decode; ~0.5 ms per 2 MB expert miss, linear in the number of misses** [M R56] | expert I/O is throughput-bound |
| USB | | `/dev/sdb` = a PS5-formatted SSD: **never used**; boot stick enumerates at USB 2.0 [M R53] | no second storage tier |

## 2. Performance model (the physics the design follows)

> **S18 operating point (2.68 pack, `--spec 1`, 40 CUs):** per verify cycle (T = 2 tokens, 1.85-1.89 accepted on
> average) ~43 ms = verify GPU-pure 29.2 + exposed demand-miss wait ~10.5-11 + MTP draft 2.7 (1.1 of it MTP expert
> misses) + MTP KV fill 0.2 -> **~23.3 ms/token = ~42.9 tok/s** [M R65-R68]. The rest of this section is the S16
> derivation that led there (EXL3-3.18, plain decoding).

Per token, measured at the operating point (EXL3-3.18, 40 CUs, `--predict 1`, 300-token answer, cold) [M R56]:

| term | ms/token | what it is |
|---|---|---|
| GPU-pure | 16.6 | dense body ~9-10 (2.67 GiB of resident weights read per token) + routed MoE ~7 (EXL3 trellis decode) |
| submit/sync | 3.9 | ~97 queue submits + fence waits per token (GPU+sync 20.5 minus GPU-pure) |
| exposed I/O wait | 19.4 | 94% of it in the ~22 layers/token that have a **demand** miss (36 demand misses/token x ~0.5 ms) |
| host | 0.1 | embedding row, n-gram hashing, slot binding |
| **total** | **~37.7** | **26.5 tok/s** |

SSD accounting: 64 expert reads/token (36 demand + 28 prefetch) x ~2 MB = 128 MB/token; SSD busy **63% of wall time at
5.2 GB/s** -> ~24 ms of SSD work per token, of which only ~4 ms overlaps GPU work [M R56].

```
tok/s = 1 / (GPU_pure + sync + max(0, SSD_work - overlap) + host)
```

Floors this implies [D]:
- SSD work at hit 0.855 with ~2 MB experts: ~24 ms/token. Perfect overlap -> ~24 ms/token -> **~41 tok/s**.
- Without overlap the path to 40 (25 ms/token) needs GPU+sync **and** exposed I/O cut together.
- Speculative decoding (MTP + batched verify) removes GPU time but **adds** reads for rejected positions: it only pays once
  GPU time exceeds exposed I/O (measured: k=1 22.75, k=3 18.1 tok/s while I/O dominates) [M R56].

## 3. Linux / kernel configuration blueprint

Base: CachyOS `linux-cachyos` **7.2.3**, PREEMPT_DYNAMIC (booted `preempt=full`), HZ=1000, NO_HZ_FULL=y,
RCU_NOCB_CPU=y, IO_URING=y, UDMABUF=y, DMABUF_HEAPS=y, HUGETLBFS=y + HUGETLB_PAGE_OPTIMIZE_VMEMMAP=y, CMA_SIZE_MBYTES=0,
LRU_GEN=y, AMD_IOMMU passthrough, X86_AMD_PSTATE, SCHED_CLASS_EXT available (unused) [M: `/boot/config-7.2.3`].

### 3.1 Kernel command line (`/boot/efi/cmdline.txt`; kexec soft reboots read it, cold boots read the loader stick)

| parameter | value | why [tag] |
|---|---|---|
| `mitigations=off` | | syscall/I/O path cost; appliance with no untrusted code [M in use] |
| `preempt=full` | | low wake-up latency for the engine thread [M in use] |
| `idle=halt` | | Oberon idle-state stability [M in use] |
| `kho=off` | | KHO kept 1,512 MiB as MIGRATE_CMA scratch that pinned memory cannot use [M S14] |
| `ttm.pages_limit=4194304 ttm.page_pool_size=4194304 amdgpu.gttsize=14336` | 16 GiB / 14 GiB | GPU-visible system memory large enough for dense weights + the imported slab [M in use] |
| `udmabuf.size_limit_mb=16384` | | one udmabuf per 1 GiB slab chunk [M in use] |
| `init_on_alloc=0 init_on_free=0` | | no page zeroing of the 12 GiB slab at every start [M in use] |
| `swiotlb=4096` | 8 MiB | default 64 MiB bounce pool peaked at 0.5 MiB [M S15] |
| `nowatchdog nmi_watchdog=0` | | no periodic NMIs on the engine cores [M in use] |
| `amdgpu.ps5_cu_unlock=3` | AI mode only | 40 CUs [M R53] |
| `ps5lm.ai=1` | | gates the AI-mode units (strip at boot, server) [M] |
| `nvme.poll_queues=2` | **proposed** | enables IORING_SETUP_IOPOLL (today `poll_queues=0`, `iopoll=false`) [A -> gate K1: demand-miss wait per read, §8] |
| `nohz_full=2-3 rcu_nocbs=0-15 irqaffinity=4-15` | **proposed** | keep the engine core (CPU 2, sibling 3) free of ticks and IRQs [A -> gate K2: per-layer host jitter p99] |

### 3.2 Runtime state set by `ps5lm-min on` (measured values) [M]
- Services stopped/masked for the boot: display manager, tty1 gaming session (gamescope + Steam ~1.1 GB), bluetooth,
  audio, udisks/upower/polkit, homed/userdb, networkd (NetworkManager owns the links), timesyncd, coredump, cups, avahi,
  power-profiles, ananicy, oomd. Kept: sshd, NetworkManager, wpa_supplicant, journald, udevd, logind, dbus, fan/boost
  daemons, the serial getty.
- Modules unloaded: bluetooth stack, HDA audio, kvm.
- sysctls: `vm.min_free_kbytes=16384 vm.watermark_boost_factor=0 vm.watermark_scale_factor=10 vm.compaction_proactiveness=0
  vm.swappiness=0 vm.overcommit_memory=1 vm.zone_reclaim_mode=0 kernel.numa_balancing=0 kernel.sched_autogroup_enabled=0
  kernel.timer_migration=0 vm.stat_interval=10 vm.hugetlb_optimize_vmemmap=1`.
- Block: `nvme0n1` scheduler `none`, `max_sectors_kb=512`, `nr_requests=1023`; THP `madvise`.
- Result: ~14 GiB available to the engine at start; 2.67 GiB dense + **12.0 GiB slab** in the served configuration [M].

### 3.3 Kconfig deltas for a dedicated appliance kernel [A, gate K3: boot + full regression]
`DRM_AMDGPU=y` and `BLK_DEV_NVME=y` built in (no initramfs module waits), `HSA_AMD=n` (no ROCm on GFX1013, saves memory),
`DEBUG_*`/`KASAN`/`FTRACE` off, `CMA_SIZE_MBYTES=0` kept, `KEXEC_HANDOVER=n` (removes the KHO trap entirely),
`ZSWAP=n` (swappiness 0, no swap), `TRANSPARENT_HUGEPAGE_MADVISE=y`. Expected gain is memory (tens of MiB) and boot time,
not tok/s: treat it as hardening, not a performance lever.

### 3.4 Safety rules baked into the appliance [M]
- The engine sets `oom_score_adj=1000` (the kernel kills the engine, never sshd/NetworkManager).
- Pool ceiling = MemAvailable - CmaFree - 192 MiB floor, also for an explicit size (a pool 130 MiB past the edge hung
  the console in S14).
- hugetlb pages are released on SIGTERM/HUP/INT and on crashes (SEGV/BUS/ABRT/FPE), so a crash cannot leave 11 GiB reserved.
- Soft reboots only (kexec); never touch `/dev/sdb`.

## 4. Memory architecture

### 4.1 Budget (served configuration, AI mode) [M]
| region | size | placement |
|---|---|---|
| dense weights (PS5Q4 4.5 bpw: attention/GDN projections, HC matrices, shared experts, lm_head 248,320 x 2560) | 2.67 GiB (2.84 with the MTP head) | first fills the **512 MiB VRAM carve**, then 128 MiB GTT chunks, first-fit |
| recurrent state (36 GDN layers: 128x128 fp32 x 48 heads + conv; PLE conv) | 113 MiB | one contiguous region (snapshot/restore/reset = one copy) |
| KV cache (12 attention layers + MTP, 2 KV heads x 256, fp32 K and V, max ctx 2048) | 96 MiB (+8 MTP) | dense alloc |
| activations (per-token blocks, `Act`) | ~1.5 MiB x batch | one allocation, stride `act_stride_` |
| snapshots (prefix reuse) | 113 MiB each, **host heap** (movable) | staged through a 16 MiB HOST_CACHED buffer |
| verify checkpoints (only with `--spec k`) | k x 113 MiB | dense alloc |
| **expert slab** | **12.0 GiB = 5,934 slots** | 2 MiB hugetlb pages -> udmabuf (<= 1 GiB per buffer) -> `vkImportMemory` dmabuf + `io_uring_register_buffers` |

### 4.2 Zero-copy expert path [C, M]
`NVMe DMA -> registered io_uring buffer (READ_FIXED, O_DIRECT) -> the same physical pages imported into Vulkan (dmabuf)
-> kernels read the slot by device address`. No CPU copy, no staging. Slots are 4 KiB-aligned spans of the pack.

### 4.3 Slot cache (`slot_cache.hpp`) [C]
- Two **size classes** for EXL3 (K=3 layers: 1.86 MB slots; K=3.5 layers 0-3/44-47: 2.17 MB), each its own LRU; class
  share = layer share. +13% slots vs one size, hit 0.838 -> 0.856 [M R52].
- Pins are reference counts (a slot used by two tokens of a verify batch is pinned twice).
- Prefetches can enter cold (LRU tail); measured worse than hot on this model [M S14].
- Residency persists across restarts (`save_residency` / `warm`): the server starts warm in ~2 s.

## 5. Model execution: per-token dataflow [C]

```
host: embedding row (pread bf16) -> H[4 x 2560]; 16 n-gram rows: one io_uring batch, reaped before layer 1
for layer l in 0..47 (49 GPU steps):
  GPU step l   = POSTMOE(l-1) + PRE(l)
     PRE(l)    = [PLE @ l=1] -> HC pre (norm, down+inject, up+mix) -> GDN (in_proj, conv, recurrent step, out_proj)
                 or gated attention (qkv, RoPE/qk-norm, KV append, flash-decoding split+combine, o_proj)
                 -> HC post -> HC pre (mlp) -> router (bf16) -> top-10 mailbox [+ trained predictor for l+1]
  host         = read the mailbox -> request/pin the 10 experts (resident first) -> prefetch predicted l+1 experts
                 -> start the resident experts' gate/up on the GPU (hits_ pre-pass) -> wait for the misses
                 -> write slot table + routing weights
POSTMOE(47) -> HC mixer -> lm_head -> argmax (logits HOST_CACHED for sampling)
```
EXL3 expert = `y = H128(H128(x * suh) . W) * svh` per projection; weights are 3-bit trellis words decoded in-register
(`mul1` codebook: one 32-bit multiply + a 4-byte sum per weight), checked against an f64 reference at rel. err ~7e-6 [M R51].

## 6. Software architecture (C++20)

### 6.1 Modules (`runtime/include/ps5lm/`)
| module | responsibility | status |
|---|---|---|
| `platform.hpp` | CPU pinning, RT priority, mlock, hugetlb reservation, udmabuf slab creation | [C] |
| `vk.hpp` | minimal Vulkan: device, BDA buffers, compute pipelines with push constants, command buffers, timestamps, dmabuf import | [C] |
| `quant.hpp` | PS5Q formats, fp16, CPU reference quantizers | [C] |
| `slot_cache.hpp` | bookkeeping half of the cache (portable, unit-tested) | [C] |
| `expert_streamer.hpp` | io_uring SQPOLL, registered file + buffers, READ_FIXED, per-layer slot geometry | [C] |
| `flash_engine.hpp` | `FlashEngine`: weights, activation blocks, recording, `feed()`, `verify()`, MTP head, snapshots, residency | [C] |
| `src/flash_decode.cpp` | CLI (benchmarks, md5-checked output) and the stdin server protocol | [C] |
| `tools/ps5lm_server.py` | HTTP front end: OpenAI + Anthropic APIs, tokenizer, SSE | [C] |
| `kernels/moe_kernels.hip`, `moe_pipeline.hpp` | HIP backend for PC simulation | [C, PC only] |

### 6.2 Engine interface
```cpp
namespace ps5lm::flash {
struct EngineOptions { std::string model_dir; int bits = 3; double pool_gib = 0; uint32_t predict = 1;
                       bool overlap = true, mtp = false; uint32_t spec_k = 0; uint32_t max_ctx = 2048; /* ... */ };
class FlashEngine {
 public:
  explicit FlashEngine(const EngineOptions&);             // load dense, size + map the slab, record command buffers once
  uint32_t feed(int64_t tok, bool want_logits);            // one token at pos(); returns greedy argmax
  uint32_t verify(const int64_t* v, uint32_t T, uint32_t* out);  // T tokens in one pass; returns accepted drafts
  uint32_t mtp_draft(int64_t next, uint32_t index, uint32_t blk = 0);
  void     mtp_fill (int64_t next, uint32_t index, uint32_t blk = 0);
  void snapshot(uint32_t slot); bool restore(uint32_t slot); void reset();   // prefix reuse
  bool save_residency(const std::string&); uint32_t warm(const std::string&); // warm pool across restarts
  const float* logits() const; EngineStats& stats();
};
}
```
Rules the implementation follows: command buffers are recorded once at load (no per-token recording); everything a
kernel touches is a device address in a push constant; per-token activations live in block `t` so a batch of T tokens
reuses the same recording helpers; the recurrent-state kernels take separate `state_in`/`state_out` pointers so a batch can
checkpoint after every token.

### 6.3 Backend abstraction (console Vulkan / PC HIP)
```cpp
struct Backend {                       // one per process
  virtual DevPtr alloc(size_t, MemKind) = 0;            // VRAM carve / GTT / host-coherent control words
  virtual DevPtr import_slab(int dmabuf_fd, size_t) = 0;  // Vulkan: VkImportMemoryFdInfo; HIP: hipHostRegister(Mapped)
  virtual void   launch(KernelId, const void* pc, size_t pc_bytes, Grid) = 0;
  virtual void   barrier() = 0;
  virtual void   submit(Recorded, Fence*) = 0;
  virtual void   wait_host_flag(DevPtr flag, uint32_t value, uint32_t spin_limit) = 0;  // §8 one-cb-per-token
};
```
- **Console: Vulkan.** ROCm does not support GFX1013, and DP4A is not accelerated there, so HIP's main advantages (dot4
  intrinsics, `hipStreamWaitValue32`) do not exist on the target. RADV/ACO produces the ISA we tune against
  (`RADV_DEBUG=shaders`).
- **PC: HIP on gfx1030/1031** (RX 6700 XT: 40 CU RDNA 2, 384 GB/s, close to the console). The HIP kernels must use the
  same math as the Vulkan shaders (`PS5LM_SOFT_DOT4=ON`, no DP4A), so PC results transfer.
- Kernel catalog (Vulkan, measured per layer at 40 CUs): EXL3 routed MoE 158 us (K=3) / ~180 us (K=3.5); HC pre ~31 us;
  dense attention/GDN body ~200-225 us; shared expert down 7.4 us [M R51-R53].

### 6.4 Threads and cores [M]
| CPU | role |
|---|---|
| 0 | io_uring SQPOLL kernel thread (idle 2 s) |
| 2 | engine thread: submit, mailbox read, slot binding, reap (pinned, `platform::pin_to_cpu(2)`) |
| any | HTTP server (Python), tokenizer |
CPU 1 is the SMT sibling of 0 and CPU 3 is the sibling of 2: keep them free (§3.1 proposal).

## 7. Speculative decoding subsystem [C, M] (S18: stage C built, default on in the appliance)

**Stage C (S18):** the verify of T = 1 + k tokens is ONE command buffer: every layer's kernels for the T tokens run
between the same barriers; large dense matrices are read once for all T columns (`matvec_xrb` K=2560, `matvec_ldsb`
K=6144, `matvecb`, `hc_upmixb`, `router2b`); the recurrent kernels process the T tokens in one dispatch with the
state in registers and a checkpoint per token (`gdn_convb`, `gdn_stepb`); experts shared by several tokens are decoded
once (`exl3_xmvb`, entries in GPU-uncached control memory); gates A/B release the union of the tokens' experts. Every
batched kernel is bit-identical per column to the single-token kernel, so the output equals plain greedy decoding.
Drafts carry a confidence (MTP softmax top-1, computed in the argmax kernel); a draft below 0.5 is fed as a plain
token (acceptance break-even 0.51). Sampling is exact too: the server samples every verified row with the request's
sampler and accepts while the sample equals the deterministic draft.
Measured: k=1 1.85-1.89 tokens/cycle, acceptance 0.85-0.90 (greedy code), 41.0-42.9 tok/s; k=2/3 within 1% [M R65-R68].

- MTP head: SGLang forward (fusion `fc_hidden(RMSNorm_10240(H)_c) + fc_embedding(RMSNorm(emb))`, one attention layer
  with its own KV and 512 EXL3 experts, own mixer, shared lm_head). **Acceptance 0.79 at depth 1, 2.88 tokens/verify at
  depth 3; ~2 ms per draft** [M R54].
- Verify: T = k+1 tokens per pass, all tokens' experts requested together, recurrent state chain through k checkpoints,
  greedy exact acceptance, bit-exact output [M R56].
- **Gate (activation rule):** enable `--spec k` only when measured `GPU+sync > exposed I/O` at the operating point; today
  it loses (§2). Stage C (batched dense matvecs, weights read once per batch) is specified but deferred.

## 8. Roadmap to 40 tok/s

> **S17 status (rev 3.1):** R1 one cb per token DONE (+4.3%; the real sync was 1.6 ms, not 3.9); R3 fewer bytes DONE
> (2.68 pack, +20%); R4 cache policy DONE (decayed-frequency LFU, -9% misses); predictor refit DONE (live precision
> 0.80); EXL3 decode software pipelining DONE (-15% MoE). Now GPU (~18 ms) > demand-miss wait (~9.7 ms): R6 (spec decode
> stage C) is the top lever.
 (ordered by expected ms/token saved; each step is gated by an md5-identical A/B)

| step | mechanism | expected | gate |
|---|---|---|---|
| R1 one command buffer per token | the host polls mailbox `seq` instead of fences; the GPU waits on a host-written ready word with a **bounded** spin shader (PS5 winsys: 10 s without progress = DEVICE_LOST) | -3.9 ms sync -> ~29-30 tok/s | GPU+sync - GPU-pure < 0.5 ms |
| R2 I/O-GPU overlap | earlier prefetch: predictor applied to the attention input (one PRE earlier) and to more context; issue order demand > prefetch | up to -14 ms exposed I/O (floor = SSD work 24 ms) | SSD busy share up, exposed wait down |
| R3 fewer bytes per token | EXL3 2.68 SC pack (GPQA-D 91.4): smaller experts AND more slots | ~-15-20% SSD work | task-level eval vs 3.18 on a reference server |
| R4 cache policy | frequency-aware admission/eviction vs LRU (trace replay first: `trace_replay`, Belady bound) | hit 0.855 -> ~0.87 [A] | replay, then live |
| R5 dense kernels | dense body at ~270 GB/s of 448: tune matvec occupancy / x staging | -2 to -3 ms | per-kernel GB/s |
| R6 spec decode stage C | batched matvecs + batched EXL3 decode for shared experts | **DONE S18: 35.7 -> 41-42.9 tok/s** | md5 identical |
| R7 (next) batched K=6144 matvec | out_proj/o_proj at T=2 cost 1.9x single (K=2560: 1.18x) | ~-1 ms/cycle | VPROF per kernel |
| R8 (next) K=2.5 trellis decode | cooperative tile loads (4 word loads/lane/tile today) | MoE ~-10% | exl3_moe_test |
| R9 (next) predictor data + draft-column predictor | 30k+ tokens of agentic/long chat | gate B ~11 ms/cycle down | live precision |
| K1/K2 kernel params | `nvme.poll_queues` + IOPOLL; `nohz_full` on the engine core | small, measured | per-miss wait, host p99 |

## 9. PC-simulation phase

1. **Functional parity:** the same CLI on a PC with an RDNA 2 dGPU runs the Vulkan path unchanged (RADV, gfx103x) and
   the HIP path; both must print the console's md5 for the benchmark prompt (`run1.sh`) at the same pack.
2. **Memory realism:** cap the engine with the same pool formula (`--pool-gib 12`) and a cgroup `memory.max` of 14.4 GiB.
3. **I/O realism:** throttle the NVMe with cgroup v2 `io.max rbps=5200000000` and replay real routing traces
   (`--trace-out` from the console) with `stream_bench`; the model must reproduce ~0.5 ms per 2 MB miss and the
   63% busy share before PC numbers are trusted.
4. **Compute realism:** scale GPU timings by measured kernel ratios (console/PC per-kernel us from `--profile`), not by
   spec-sheet TFLOPs.

## 10. Verification and observability [C]
- Bit-exactness: every optimization must print the same tokens (md5 of the 300-token benchmark: `9c8d86dd397c`);
  EXL3 kernels vs f64 reference (`exl3_moe_test`); residual-stream golden dumps (`--dump`) vs the HF reference
  (`tools/ref_qwen4.py`).
- Per-run metrics (CLI stderr): tok/s, GPU+sync, GPU-pure (timestamps), I/O wait, hit rate, reads/token, predictor
  precision, **SSD busy % and GB/s while busy** (`/sys/block/nvme0n1/stat`), **wait anatomy** (layers and ms by number
  of demand misses), MTP/verify acceptance histograms.
- Debug switches: `PS5LM_NANCHECK=1` (non-finite activations per verify step + slot-binding assertions), invalid
  expert ids abort instead of streaming past the pack.
