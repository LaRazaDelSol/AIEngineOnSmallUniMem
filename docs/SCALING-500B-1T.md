# Scaling blueprint: 500B to 1T+ parameters at 4-bit-class quality on the PS5 (S23, 2026-11-09)

Everything here is a projection unless marked measured. The projections are anchored on the measured 125B operating
point and state their formulas so they can be re-derived when a constant changes. Nothing in this document has been
run at 500B+; section 7 lists the measurements that would confirm or kill each part.

## 1. The physics that decides it

Three budgets, all measured on this console:

| budget | value | source |
|---|---|---|
| memory | 14.9 GiB usable (14.45 Linux + 0.5 VRAM carve); today 12.07 pool + 2.85 resident | this session |
| storage | 6.4 GB/s per M.2 drive, 5.6-5.85 GB/s delivered while busy; +0.88 GB/s per USB 10 Gbps device, additive | R76, R6, this session |
| GPU | 386 G weights/s trellis decode-matvec (K=2.5), about 600 G weights/s dense 4.5-bit matvec, 3.3 TFMA/s batched trellis GEMM | R75, R74, S22 |

Interactive decode speed is

```
tok/s = 1000 / max( MB_per_token / (0.875 x GB/s) + 2.5 ,  GPU_ms_per_token )
MB_per_token = reads_per_token x miss(c) x slot_MB x 1.15        c = slots_in_pool / experts_in_model
miss(c) = 0.0414 x c^-0.8      (fit to this model: c 0.10 -> 0.26, 0.233 -> 0.14, 0.283 -> 0.11)
```

The fit is three points on one model's routing and is extrapolated below c = 0.10; treat hit rates under that as
the least certain numbers in this document.

**What decides feasibility is expert granularity, not parameter count.** A token pays for the bytes of the experts
it routes to and misses. Flash-Next routes to 480 experts of 4.9M parameters (0.75 GB at 2.5 bits); DeepSeek-V3 and
Kimi-K2 route to 464-480 experts of 44M parameters (6.5-6.7 GB). The second kind costs nine times more per miss
and its dense trunk alone (16-17B parameters) takes half of this machine's memory.

## 2. Throughput projections, interactive decode (one stream)

Quantization: EXL3-class trellis at 2.5 bits per weight in the experts (the format the engine decodes today),
dense weights as noted. Pool = 14.9 GiB minus resident dense minus 0.6 GiB for KV and working buffers.

| model class | pack on disk | slot | pool | slots | share resident | hit | MB/token | tok/s at 6.4 GB/s | 7.3 | 9.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Flash-Next 125B (**measured: 33-37 prose**) | 39 GB | 1.55 MB | 12.1 | 7,100 | 28.3% | 0.89 | 150 | 34 | 38 | 47 |
| same geometry, 250B (2x experts) | 75 GB | 1.55 | 12.0 | 7,060 | 14.4% | 0.80 | 258 | 21 | 23 | 29 |
| same geometry, 500B (4x) | 151 GB | 1.55 | 12.0 | 7,060 | 7.2% | 0.66 | 449 | **12** | 14 | **17** |
| same geometry, 1T (8x) | 302 GB | 1.55 | 12.0 | 7,060 | 3.6% | 0.41 | 782 | **7** | 8 | **10** |
| DeepSeek-V3-class 671B, dense at 4.5 bits | 204 GB + 9 | 13.9 | 5.3 | 348 | 2.3% | 0.17 | 9,500 | 0.6 | 0.7 | 0.8 |
| DeepSeek-V3-class 671B, dense at 3.5 bits | 204 GB + 7 | 13.9 | 7.3 | 479 | 3.2% | 0.35 | 7,400 | 0.8 | 0.9 | 1.1 |
| Kimi-K2-class 1T, dense at 3.5 bits | 317 GB + 7 | 13.9 | 7.8 | 512 | 2.2% | 0.13 | 10,300 | 0.5 | 0.6 | 0.8 |

Reading: a fine-grained MoE scales to 1T on this console at 7-10 tok/s interactive with no new mechanism, only a
larger drive. The coarse-expert models that exist today at 671B-1T run at about 1 tok/s interactive however the
engine is tuned, because each token must move 7-10 GB through a 6.4 GB/s drive. Sections 3 to 5 are the three
mechanisms that change that, in order of how much they change it.

GPU ceilings (everything resident, not reachable here but they cap the other modes): fine-grained 13 ms/token (77
tok/s); DeepSeek-V3-class 20.4B routed / 386 G/s + 16B dense / 600 G/s = 80 ms (12.5 tok/s); K2-class about 75 ms.

## 3. Mechanism A: wavefront batch decode (throughput mode)

Turn the prefill machinery into a decoder for many independent streams. The S20-S22 prefill already processes one
layer at a time over thousands of token positions, reads each expert once per chunk and applies it to all its
positions with a tiled trellis GEMM (`prefill_lm`, `wide_moe_xm`, `exl3_xmvl`). A wavefront step does the same with
the "chunk" being the current token of B streams:

```
for each step:                      # one new token for each of B streams
  for layer l in 0..L-1:
      attention / dense body of layer l for all B streams       (GEMMs; KV of layer l paged in per stream group)
      route all B streams; union of routed experts              (B x top-k draws over E experts)
      read the union once, sequentially, expert-major GEMM over each expert's streams
      carry H (4 x hidden floats per stream) to layer l+1
  sample B tokens
```

At B of a few hundred the union is every expert of the layer, so a step reads the whole pack once, sequentially
(the best case for the drive, and trivially striped across mirrors).

| model | pack sweep at 6.4 GB/s | GPU per step at 3.3 TFMA/s | B = 256 | B = 512 | B = 1024 |
|---|---|---|---|---|---|
| Flash-Next 125B | 6.1 s | B x 6 G FMA: 0.5-1.9 s | 32 tok/s | 51 | 72 |
| fine-grained 1T | 47 s | same per stream | 5.2 | 10 | 18 |
| DeepSeek-V3-class 671B | 32 s | B x 74 G FMA: 5.7-23 s | 6.3 | 10 | 15 |
| Kimi-K2-class 1T | 50 s | B x 64 G FMA: 5-20 s | 4.4 | 7.8 | 12 |

Step time = pack sweep + KV read for 2,048-token contexts (section 6.3: 1.5 / 3.1 / 6.2 s for the 125B geometry,
3.1 / 6.2 / 12.4 s for the MLA models) + GPU, with no credit for overlapping them.
Aggregate tokens per second; each stream advances one token per step, so this is a batch mode (evaluation sets,
synthetic data, overnight agent swarms), about 0.4-1.6 million tokens per day at 1T. For the 125B model it is a
second product: 51-72 tok/s aggregate against 35-45 interactive.

Memory at B = 512: carry 512 x 28 KB = 15 MB; one layer's routed experts are streamed through the pool, never all
resident (a 1.5 GiB window is enough); dense weights of the coarse models are streamed per layer as well (0.28B
parameters per layer), so nothing but the current layer and the KV window has to be resident. That is what makes
the 671B-1T trunk fit at all.

KV in this mode is the binding constraint, see section 6.

Build cost: the kernels exist. New code is the step driver (`wavefront_step`, about the size of `prefill_lm`), a
per-stream sampler loop, per-stream KV addressing in the wide attention kernels (`pos0` becomes a per-row table),
and the server's batch endpoint.

## 4. Mechanism B: remote expert execution on a networked GPU server (cluster mode, interactive)

A miss costs bytes only if the expert has to come to the PS5. Send the activation to the expert instead. Per layer
the PS5 sends one hidden vector and the routing (14 KB at hidden 7168 in fp16), a server that holds the cold experts
in memory returns the weighted sum of their outputs (14 KB). The PS5 keeps the trunk, the routers, KV and the hot
experts it has room for.

```
ms/token = dense_PS5 + L x (RTT + server_compute_per_layer)       hot experts run locally under the round trip
DeepSeek-V3-class:  27 + 58 x (0.2 + 0.6) = 73 ms  ->  about 13 tok/s        (versus 0.8 streaming)
Kimi-K2-class:      27 + 60 x (0.2 + 0.6) = 75 ms  ->  about 13 tok/s
```

Requirements, each a hard one:
- **Same wired LAN.** The round trip is paid 58-60 times per token. A routed VPN hop measured 41-97 ms, which
  would be 3 s per token.
  On one switch the console's 1 GbE is ample (28 KB per layer, under 2 MB/s).
- **Memory on the server side.** 671B at 2.5 bits is 204 GB; 1T is 317 GB (for example three 128 GB unified-memory
  machines).
- Honest accounting: in this mode the servers do most of the arithmetic. The PS5 contributes the trunk and the
  session state. It is the right design if the goal is "the console serves a 1T model interactively"; it is not
  evidence that the console runs one alone.

Protocol sketch (UDP or a persistent TCP stream, one message pair per layer):

```cpp
struct ExpertCall {            // PS5 -> server, then server -> PS5 with y[] in place of x[]
    uint32_t seq, layer;       // seq = token x L + layer; the reply echoes it
    uint16_t n, pad;           // experts to run remotely (the ones not resident on the PS5)
    uint16_t expert[16];
    float    weight[16];       // routing weights, so the server returns one combined vector
    uint16_t x[];              // hidden-size fp16 input (the MoE input after the layer's norm)
};
// engine: gate B waits on (local misses == 0) && (remote reply for this layer landed); the reply is added by the
// existing combine kernel as one more "expert" with weight 1. A lost reply after 5 ms falls back to streaming the experts.
```

Remote results come from the same trellis codes decoded on a different GPU, so outputs are not bit-identical to a
local run (fp32 summation order). The exactness gate for this mode is the quality instrument, not the md5.

## 5. Mechanism C: neuron-major partial reads for coarse experts (research, with a kill criterion)

For a 44M-parameter expert, a miss reads 13.9 MB to compute `down( silu(gate x) * (up x) )`. After the gate
product, most of the 2,048 hidden units contribute little. Store each expert as `[gate | per-unit (up row, down
column) records]`, read the gate matrix first (one third of the bytes), compute `silu(gate x)`, then read only the
records of the units that carry the output.

```
bytes per miss = 1/3 + 2/3 x rho          rho = share of hidden units read
rho = 0.25  ->  0.50 of today's bytes  ->  about 2x interactive tok/s on coarse models (0.8 -> 1.6)
```

Why it is research: (1) it is lossy, so it has to pass the quality instrument at the chosen rho (kill if KL versus
the full expert exceeds +0.01 nats at rho <= 0.35); (2) the trellis format rotates rows in 128-blocks, so the
per-unit records need the hidden dimension left unrotated on the gate output and down input, which means our own
quantization run (section 6.2), not a repack; (3) the second read is latency-bound at 2,048 small records, so
records must be grouped (64 units per 4 KiB-aligned extent). The offline measurement that decides it needs no
engine work: decode real experts with `tools/s15/exl3_verify.py`, zero all but the top-rho units per token, report
output error and KL through `flash_decode --score`. Even at its best this leaves coarse models near 2 tok/s on one
console, so it ranks below A and B.

## 6. Memory budgets and KV management

### 6.1 Budgets (GiB)

| item | 125B today (measured) | fine-grained 1T, interactive | K2-class 1T, wavefront B = 512 | K2-class 1T, cluster |
|---|---|---|---|---|
| usable | 14.9 | 14.9 | 14.9 | 14.9 |
| dense weights resident | 2.25 | 2.25 | 0.5 (current layer + lm_head) | 6.5 (3.5-bit trunk) |
| KV + state + working buffers | 0.6 (8k backed, grows) | 0.6 | 6.0 (KV window, 6.3) | 1.5 (32k context) |
| expert pool | 12.07 | 12.0 | 8.0 (layer stream window + hot set) | 6.5 (hot experts) |
| expert metadata (16 B per expert) | 0.0004 | 0.003 | 0.0004 | 0.0004 |
| on disk | 40.5 + 102 (n-gram) | 302 + n-gram | 317 + 7 | 7 on the PS5 |

A 1T pack needs a 512 GB+ free M.2 volume; the SN810 has 54 GiB free. A 1 TB (fine-grained without the n-gram
table) or 2 TB Gen4 drive is a prerequisite for sections 2 and 3.

### 6.2 Quantization at 4-bit-class quality

The quality bar already met at 125B is the template: trellis codes (QTIP family, LDLQ error feedback, 128-block
Hadamard rotation) at 2.5 bits in the experts, 3 bits in the first and last four layers, dense at 4.5 bits, scored
0.932 top-1 agreement against a 4.5-bit reference and published at the original model's level on GPQA-Diamond. For
a new model there is no ready-made pack, so the pipeline is ours to run (on a large-memory GPU machine, layer by layer, about 250
calibration rows of 2,048 tokens as exllamav3 does):

1. Calibrate with activations from the already-quantized prefix (error feedback across layers).
2. Per-layer bit allocation: start from 2.5 bits, raise layers to 3 or 3.5 by measured KL sensitivity (first and
   last layers first, as the published 2.68 and 3.18 packs do).
3. Per-expert bit allocation under two budgets. With `f_e` the routing frequency of expert `e`, `s_e(K)` its
   size and `d_e(K)` its measured output error at `K` bits:
   ```
   minimize   sum_e f_e x d_e(K_e)                                   (quality loss, weighted by use)
   subject to sum_e p_miss(e) x f_e x s_e(K_e)  <=  SSD bytes/token  (cold experts cost bandwidth)
              sum_{e in hot set} s_e(K_e)       <=  pool bytes        (hot experts cost memory)
   ```
   Greedy by marginal gain per byte over K in {2, 2.5, 3, 3.5}. Frequently used experts end at higher K (they are
   resident, so they cost pool, not bandwidth); rarely used ones at 2.5 (each miss is cheap). Accept only if the
   whole model passes the instrument: top-1 agreement >= 0.93 and KL <= 0.16 nats against the 4-bit reference on the
   scoring documents, the levels the 125B pack meets.
4. Pack byte-for-byte into 4 KiB-aligned slots with per-layer geometry (`tools/s15/pack_exl3.py`); size classes in
   the pool per distinct slot size (the engine has two today; generalize the class table to four).

Kernels: K = 2, 2.5, 3, 3.5 decode variants exist except K = 2 (`exl3_common.glsl` is parameterized by half-K;
add `-DKH=4` to the manifest and validate with `exl3_moe_test`).

### 6.3 KV cache

| model | per token | format |
|---|---|---|
| Flash-Next 125B (measured) | 19 KB | int8 K/V, 13 attention layers, sparse attention with a block indexer |
| DeepSeek-V3 / K2 class | about 35 KB | MLA latent (512 + 64 values) x 61 layers at int8 |

Interactive and cluster modes use what exists: int8 KV in a sparse buffer backed page by page from retired expert
slots and returned on reset (S20). At 35 KB/token a 32k context is 1.1 GiB and 128k is 4.5 GiB, which on a coarse
model leaves almost no pool; cap those at 32k.

Wavefront mode needs a tiered KV, because B streams x context does not fit:

```
B = 512 streams x 2,048 tokens x 35 KB = 37 GiB  ->  on NVMe, read layer-major
per step, per layer:  read that layer's KV for all streams (37 GiB / 61 = 0.6 GiB), attend, append B new rows (0.3 MB)
extra I/O per step:   37 GiB read (adds 18% to a 204 GB pack sweep), 18 MB written
resident:             one layer's KV window for the stream group in flight: 512 x 2,048 x 576 B = 0.6 GiB, x2 for overlap
```

File layout: one append-only extent per (layer, stream group of 64), 2 MiB-aligned, so a layer's read is sequential
and the write is one extent tail per group. Writes total 18 MB per step: under 50 GB per day, negligible wear.
Models with block-sparse attention (Flash-Next) read only the indexer's selected blocks, which cuts the KV read by
the selection ratio; dense-attention models read the full window.

Snapshots (prefix reuse) stay as they are for interactive modes; a disk tier for them is a 113 MiB write per
snapshot and worth adding only when conversations outnumber the two in-memory slots.

## 7. Implementation steps and their gates

| step | work | gate (measurement that passes or kills it) |
|---|---|---|
| 0 | 1-2 TB Gen4 M.2 drive; clone the system; optional USB 10 Gbps NVMe mirrors | fio at slot size: >= 6.4 GB/s primary, mirrors additive (`--pack-mirror`, built) |
| 1 | generalize constants (`L`, `E`, `K`, `HID` in `flash_engine.hpp` are compile-time for Flash-Next) into a model descriptor read from `config.json`; four slot size classes | 125B still prints md5 `18179c0e1f51` |
| 2 | wavefront decode on the 125B model first (kernels exist) | B = 256 aggregate >= 30 tok/s; per-stream tokens identical to single-stream greedy |
| 3 | hit-versus-capacity curve below 10% resident: replay real 125B routing traces through `trace_replay` with the pool cut to 1/2, 1/4, 1/8 | confirms or replaces `miss(c)`; this decides the fine-grained 500B-1T rows of section 2 |
| 4 | quantization pipeline on a large-memory GPU machine for the target model, one layer end to end | per-layer output error within 1.5x of the 4.5-bit scalar code (the 125B pack's ratio, R50) |
| 5 | a coarse-expert model in wavefront mode: MLA attention kernels, per-layer dense streaming, tiered KV | whole-model quality instrument passes; aggregate tok/s within 30% of section 3 |
| 6 | cluster mode, if the servers and the console share a LAN | round trip under 0.5 ms per layer measured before any engine work |
| 7 | neuron-major reads | offline rho sweep passes the KL bound |

Steps 0-3 use only hardware and code that exist. Step 3 is a one-hour replay with the tools in the tree and is the
cheapest way to remove the largest uncertainty in this document.

## 8. What this blueprint does not claim

- No 500B+ model has been loaded on the console. Every number above the 125B row is derived.
- Interactive use of today's 671B-1T open models on one console is about 1 tok/s. No software technique in this
  document changes that by more than 2x; only moving the arithmetic (cluster) or giving up latency (wavefront) does.
- "4-bit quality" at 2.5-3 bits is measured for one model and one quantizer. A new model has to pass the same
  instrument before the claim transfers.
