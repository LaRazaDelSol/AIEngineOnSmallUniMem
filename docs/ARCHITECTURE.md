# Architecture and repository map

The system serves Qwen3.8-Flash-Next (125B parameters, 48 layers x 512 experts, top-10 routing, about 6B active per
token) from a PS5 with 16 GB of unified memory. Dense weights live in memory; the 24,576 experts live on the M.2 SSD
and are streamed into a slot pool that the GPU reads in place.

## 1. Data path

```
 M.2 NVMe (WD SN810, 6.4 GB/s)          unified GDDR6 (14.45 GiB visible to Linux + 0.5 GiB VRAM carve)
 experts_exl3_268.pack  40.5 GB    -->  expert slot pool, 12.07 GiB of 2 MiB hugetlb pages (udmabuf)
   EXL3 trellis codes, 4 KiB slots        6,952-7,250 slots in two size classes, decayed-LFU eviction
   io_uring READ_FIXED, O_DIRECT          the same pages are imported into Vulkan as a dma-buf: zero copies
 ngram_bf16.bin 102 GB (16 rows/token)  dense weights 2.85 GiB resident (attention, GDN, routers, lm_head, MTP head)
                                        KV cache: int8, sparse buffer, grows page by page out of retired slots
```

Per token the GPU runs one pre-recorded command buffer. The host reads each layer's routing from a mailbox in
GPU-uncached memory, binds slots, and releases the layer through flags the GPU waits on in bounded gate kernels.
Experts that are resident run while the missing ones stream in.

Speed comes from four mechanisms, each measured on the console:

| mechanism | where | effect |
|---|---|---|
| small experts at Q4-class quality (EXL3 2.68 bpw trellis, decoded on the GPU) | `shaders/exl3_*.comp` | fewer bytes per miss and 60% more experts resident than a 4.5 bpw scalar pack |
| trained cross-layer router predictor + top-8 prefetch | `router2.comp`, `model/pred_s17/` | reads start one layer early (precision 0.80) |
| MTP draft + batched verify in one command buffer | `record_vonecb`, `mtp_draft` | 1.89 tokens per pass, dense weights read once per pass |
| layer-major, expert-major prefill | `prefill_lm`, `wide_moe_xm`, `exl3_xmvl.comp` | prompts at 320-340 tok/s; each expert read once per 4,096-token chunk |

## 2. Repository layout

| path | contents | status |
|---|---|---|
| `runtime/` | the engine: C++20 headers, GLSL compute shaders, CLI, server, tools | **the product** |
| `runtime/include/ps5lm/` | `flash_engine.hpp` (engine), `session_store.hpp` (conversations on the SSD, S25), `expert_streamer.hpp` (io_uring streamer), `slot_cache.hpp` (pool policy, portable), `vk.hpp`, `platform.hpp`, `quant.hpp` | |
| `runtime/src/flash_decode.cpp` | CLI and the `--serve` line protocol (prefill, decode, speculative loop, sampling) | |
| `runtime/shaders/`, `runtime/cmake/Shaders.cmake` | 73 source files, 191 SPIR-V modules; the manifest names every variant | |
| `runtime/tools/ps5lm_server.py`, `ps5lm_dash.py`, `ps5lm_chat.html` | OpenAI + Anthropic HTTP front end (:8081), dashboard (:8080), chat UI | |
| `runtime/tools/sNN/` | tools written in development session NN | index below |
| `ps5/appliance/` | boot modes, systemd units, `/etc/ps5lm/server.env` | deployed |
| `scripts/` | PC-side: `ps5-sync-build.sh`; console-side (`scripts/console/`): `preflight.sh`, `smoke.sh`, `tok.py`, `kconfig-check.sh`, benchmarks, hardware probes, kernel modules (`s28/`, `s31/`) | |
| `engine/` | the Python bandwidth and hit-rate model, pack manifest tool and acceptance gate (the original feasibility study) | maintained, CI |
| `tools/ps5probe/`, `tools/smoke_evaluate.py` | the pre-engine probe battery (bandwidth, hit curves) and its regression test | frozen, CI |
| `docs/` | this file, `ENGINE-SPEC.md`, `OPEN-ITEMS-100B.md`, `CACHE-AWARE-ROUTING.md`, `PERSISTENT-CONTEXT.md`, `KV-FOOTPRINT-RESEARCH.md`, `HARDWARE-DECONSTRUCTION.md`, `SCALING-500B-1T.md`, `S26`-`S33` studies, `evidence/` (feasibility-study charts) | |
| `research/` | `reports/` and `notes/` (literature studies before the build), `design-history/` (superseded plans and specs) | historical |
| `recon/` | raw console output: `s25/` hardware dumps, `ps5probe/` probe JSON, `gates/` probe-battery reports | raw data |

### Session tool index (`runtime/tools/`)

| dir | tools | used for |
|---|---|---|
| `s15/` | `pack_exl3.py`, `pack_exl3_mtp.py`, `exl3_verify.py`, `qstudy.py`, `predictor.py`, `export_pred.py`, `basis_test.py` | building the EXL3 pack byte-for-byte, quantizer study, router predictor |
| `s16/` | `run1.sh`, `mtp_measure.sh` | the 300-token reference run (superseded by `scripts/console/smoke.sh`) |
| `s17/` | `fetch_pack_exl3.py`, `fit_pred.py`, `pred_data.sh`, `policy_sim.py` | range-fetching a pack from Hugging Face, predictor refit, eviction simulation |
| `s18/` | `score_docs.py`, `ref_logprobs.py`, `score_report.py`, `serve_spec_smoke.py` | the quality instrument against a vLLM reference server |
| `s19/` | `policy_sim2.py`, `policy_sim3.py`, `tok_route.py` | eviction headroom (Belady bound), token-route study |
| `s20/` | `pf_ab.sh`, `think_smoke.py`, `long_smoke.py`, `score_pf.sh`, `render_ids.py` | prefill A/B, thinking and long-context server checks |
| `s22/` | `pos_sweep.sh` | decode speed versus prompt length |

S25 probes live in `scripts/console/`: `session_probe.py` (store against `kill -9`), `session_e2e.py` (the same through
HTTP), `hw_probe.py` (read-only register survey), `reclaim_lowmem.sh` (staged: +1 GiB of RAM).

## 3. Inside `flash_engine.hpp`

One class, `ps5lm::flash::FlashEngine`. Sections in file order, which are also the seams for a future split into
separate headers (each depends only on the ones above it):

| lines | section | would become |
|---|---|---|
| 70-145 | push-constant structs shared with the shaders | `engine/abi.hpp` |
| 147-285 | `EngineOptions`, `EngineStats` (every flag carries its measured result) | `engine/options.hpp` |
| 286-400 | weights, n-gram hashing | `engine/weights.hpp` |
| 460-860 | state snapshots, `feed()` (one token, one command buffer) | `engine/decode.hpp` |
| 860-1775 | `verify()` (batched speculative verify) | `engine/verify.hpp` |
| 1775-2183 | MTP draft head | `engine/mtp.hpp` |
| 2183-2530 | pool persistence, loading | `engine/load.hpp` |
| 2530-3202 | elastic KV, pool mapping, hugepages | `engine/pool.hpp` |
| 3202-3933 | command recording (token, verify) | `engine/record.hpp` |
| 3933-4670 | wide and layer-major prefill | `engine/prefill.hpp` |

Rule for any such split, and for every other change to the engine: `scripts/console/smoke.sh` must print PASS.

## 4. Build

```
cd runtime && cmake --preset ps5        # console: Release, -O3 -march=znver2, Ninja, Vulkan + io_uring
cmake --preset ps5-serve && cmake --build --preset ps5-serve   # engine + shaders only
cmake --preset host && cmake --build --preset host && ctest --preset host   # any machine, no GPU
```

| item | value on the console (verified 2026-11-09) |
|---|---|
| compiler | GCC 16.2.1, `-O3 -DNDEBUG -march=znver2 -mtune=znver2 -Wall -Wextra -Wpedantic`, C++20 |
| shaders | `glslc -O --target-env=vulkan1.3` (shaderc 2026.4), 191 modules |
| libraries | liburing 2.15, Vulkan headers 1.4.363, Mesa/RADV 26.2.5 |
| kernel | `linux-ps5 7.2.3` (ps5-linux tree, built on the console); requirements in `scripts/console/kconfig-check.sh` |
| boot parameters | `kho=off udmabuf.size_limit_mb=16384 ttm.pages_limit=4194304 ttm.page_pool_size=4194304 amdgpu.gttsize=14336 swiotlb=4096 nowatchdog nmi_watchdog=0 init_on_alloc=0 init_on_free=0 mitigations=off preempt=full idle=halt ps5lm.ai=1 amdgpu.ps5_cu_unlock=3` |

**ROCm/HIP is not part of this build.** The PS5 GPU reports as GFX1013; ROCm has no target for it and the part has
no DP4A. The `hip-legacy` preset builds the S12 design-point kernels for gfx1030 on a machine that has ROCm; nothing
on the console uses them.

## 5. Daily workflow

```
cp scripts/console.env.example scripts/console.env         # once: host, key, paths
scripts/ps5-sync-build.sh --stop-server                    # sync + build (stops the server: never build next to an engine)
ssh ... 'bash ~/ps5lm/src/scripts/console/smoke.sh mytag [flags]'   # 45 s, PASS = md5 18179c0e1f51
ssh ... 'sudo systemctl start ps5lm-server'                # preflight runs first
```

New console from scratch: the Setup section of the top-level `README.md`.
