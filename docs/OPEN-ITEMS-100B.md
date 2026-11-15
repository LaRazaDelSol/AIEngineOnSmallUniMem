# Open items for sustained high-speed 100B+ execution (S23, 2026-11-09)

State of the 125B model on the console, then every remaining bottleneck with its measured size, the engineering
solution, and what was built or probed in this session. "Built" means in the tree, compiled on the console and
checked against the reference md5; "design" means specified here with code, not built.

> **S24 update (2026-11-09):** the SSD wall described below holds for EXACT routing only. Cache-aware routing
> (`CACHE-AWARE-ROUTING.md`) cuts expert reads per token by a third and puts decode at 53-58 tok/s with no
> measurable quality change on the S18 instrument; the GPU is the bound again, so 2.6 moves up and 2.1 / 2.3 down.

> **S25 update (2026-11-10):** contexts are kept on the SSD (`PERSISTENT-CONTEXT.md`): the restart / switch half of
> 2.5 and the "KV is lost on restart" item are closed. Prefill anatomy is in `PERSISTENT-CONTEXT.md`;
> one unused hardware resource was found (1,008 MiB of RAM, `HARDWARE-DECONSTRUCTION.md` section 3).

## 1. Where it stands (all measured on the console, serving flags, cold pool)

| workload | tok/s | what bounds it |
|---|---|---|
| reference C++ prompt, 300 tokens | **45.2** | GPU 25 ms + miss wait 12 ms per 1.89-token pass |
| prose, 256 tokens (S22) | 33-37 | SSD: about 150 MB/token of expert reads |
| copy-heavy code edit, 500 tokens after a 784-token prompt (this session) | 40.5 | SSD 81% busy, 28 demand misses/token |
| decode after an 8,000-token prompt (S22) | 28.7 | SSD, plus KV pages taken from the pool |
| prefill, 8,000 tokens | 323-341 | GPU (MoE 0.9 ms, GDN 1.1 ms, attention per token) |
| served, short prose answer (this session, after restart) | 34.7 | SSD |

Quality (S18/S20, versus an unmodified W4A16 reference served by vLLM): top-1 agreement 0.932, KL 0.153 nats on 3,532
positions; the pack's own published GPQA-Diamond is 91.4 versus 91.7 for the original model.

**The single remaining wall for decode is SSD bandwidth.** This session's copy-heavy probe makes that unambiguous:

| run (same output, md5 `5f515f20c1e7` in all three) | tokens/pass | GPU-pure ms/token | demand misses/token | SSD busy | tok/s |
|---|---|---|---|---|---|
| MTP drafts, k=1 | 2.00 | 12.8 | 28.5 | 81% at 5.62 GB/s | 40.5 |
| + prompt-lookup drafts, k=1 | 1.93 | 12.8 | 31.1 | 84% | 39.1 |
| + prompt-lookup drafts, k=3 | **3.37** | **11.4** | 34.8 | **92% at 5.85 GB/s** | 38.8 |

Tripling the drafted depth cut GPU time and changed nothing, because every verified token still needs its roughly 30
missing experts from a drive that is already saturated. Any lever that does not reduce bytes per token or add bytes
per second is now worth at most a few percent.

## 2. Bottlenecks and solutions

### 2.1 SSD bandwidth during miss bursts: mirrored packs (BUILT, needs a second device to pay)

`ExpertStreamer` now accepts byte-identical copies of the pack on other block devices and sends each miss to the
device that will finish it first:

```cpp
// expert_streamer.hpp, request(): the device that finishes this read first
uint32_t dev = 0;
if (fds_.size() > 1) {
    double best = (double(inflight_[0]) + total) / bw_[0];
    for (uint32_t d = 1; d < fds_.size(); ++d) {
        const double t = (double(inflight_[d]) + total) / bw_[d];
        if (t < best) { best = t; dev = d; }
    }
}
slot_dev_[r.slot] = uint8_t(dev);
inflight_[dev] += total;
io_uring_prep_read_fixed(sqe, int(dev), slab_ + slot_offset(r.slot) + pb, len, off + pb, slot_buf(r.slot));
```

Use: `--pack-mirror /mnt/usb1/experts_exl3_268.pack:880` (repeatable, at most 7; `--pack-mbps` sets the primary's
weight, default 6400). Probe: a mirror pointing at the same file with weight 3200 gave md5 PASS, 45.57 tok/s, reads
split 16,347 / 4,981, so the selection and the completion accounting are correct. A same-device mirror cannot add
bandwidth; the gain needs hardware:

| storage | bandwidth | source | projected prose tok/s |
|---|---|---|---|
| SN810 alone (today) | 6.4 GB/s | measured (R76) | 34-35 (measured 33-37) |
| + one USB 10 Gbps NVMe enclosure | 7.3 | measured additive, +881 MB/s (R6) | 38 |
| + three (two Type-A on bus 2, one Type-C on bus 4) | 9.1 | per-controller sharing unmeasured | 47 |
| a 7.3 GB/s-class M.2 drive instead of the SN810 | 7.3 | drive specification | 38 |

Projection: `ms/token = MB/token / (0.875 x GB/s) + 2.5`, anchored on the measured prose point (150 MB/token, 5.6
GB/s while busy on a 6.4 GB/s drive, 35 tok/s). The rear-port drive already attached is PS5-formatted storage and
is not used. The boot stick enumerates at USB 2.0 (43 MB/s) and is useless for this. Setup once a device exists:
copy the pack, `cmp` it, add the flag to `PS5LM_ARGS` in `/etc/ps5lm/server.env`. Address devices by
`/dev/disk/by-id`, never `sdX`.

### 2.2 Prompt-lookup drafts (BUILT, exact, measured null today: off by default)

`--pld N` drafts from the context when the last N tokens occurred before (`Pld` in `flash_decode.cpp`), skips the
MTP draft, and goes through the same verify. Exact for greedy and sampled decoding. Measured: reference prompt md5
PASS at k=1 and k=3 (45.11 / 44.78 versus 45.21 tok/s: neutral, the acceptance gate switches it off within four
cycles on text that is not copied); copy-heavy prompt in the table above: no gain. It is kept because it becomes
the cheapest lever the moment 2.1 lifts the SSD wall: it already delivers 3.37 tokens per pass at 11.4 ms GPU per
token. Re-measure with `--spec 3 --pld 4` after a mirror device is installed. Running at `--spec 3` costs two more
verify checkpoints of pool, which is why `--spec 1` stays the serving default until then.

The hypothesis from the early research notes that copied tokens reuse cached experts is refuted for this model:
routing below layer 4 follows the token, above it follows the context, and the misses did not fall.

### 2.3 Idle GPU during miss waits: lookahead routing on the partial residual (DESIGN)

Today the GPU sits in gate B for 12 ms (reference) to 21 ms (copy-heavy) per pass waiting for demand misses, and the
next layer's reads start only from a linear predictor (precision 0.80, top-8 prefetched, so about 20% of prefetched
bytes are wasted and 20% of the needed experts are found late). While layer `l` waits for `m` of its 10 experts, the
GPU can compute layer `l+1`'s routing from the residual that already contains the other `10-m` contributions plus
the shared expert, and the host can issue those reads at once. The speculative pass is discarded; the exact pass
runs when the misses land, so output stays bit-identical.

Where it goes: `record_onecb` / `record_vonecb`, between gate A and gate B of each layer.

```cpp
// After gate A of layer l (resident experts' gate/up/act/down have run into the partial routed sum Yp):
//   1. w_post-style kernel into SCRATCH (not H):  Hs = H + inject(Yp + shared)        (hc_post.comp, outp = Hs)
//   2. layer l+1's dense body on Hs, writing only scratch outputs and NO recurrent state:
//        attention layers: hc pre-step, qkv, attn_prep with a "no KV write" flag, attention, o_proj
//        GDN layers:       conv + step with state_out = scratch (PcConv/PcGdn already take state_out, R56)
//   3. router2 + topk2 on the scratch MoE input, into mailbox slot (L+1) + l   (a second mailbox row per layer)
// Host, in the gate-B wait loop of layer l:
if (const uint32_t* la = poll_mail_lookahead(l)) {          // seq number changed
    for (uint32_t k = 0; k < K; ++k)
        st_->request(l + 1, la[k], /*cold=*/false, /*spec=*/true);   // spec: does not count as an LFU use
    st_->submit();
    stats_.la_issued += K;
}
// Gate B of layer l is unchanged. Layer l+1's own router still runs on the exact residual and is what binds.
```

Cost: one dense-body pass per layer that has a demand miss (about 22 layers per token, 225 us each = 5 ms of GPU
time that is idle today). Skip it for layers whose wait class is 0 (everything resident). Expected: precision
toward the share of routing weight already present (measure first, see the gate), prefetch waste down, demand
misses of layer `l+1` already in flight for a full layer time. Upper bound from the S19 anatomy: gate B 12 ms to
7-8 ms per pass, about +10% on the reference prompt and more where the SSD is busier.

Gate before building: an offline measurement on the existing MoE-input dumps (`--dump-moe`): recompute layer
`l+1`'s top-10 with each of layer `l`'s experts removed in turn and report precision. Build only if precision with
one or two experts missing is at least 0.9. Earlier cheap variants failed for a known reason (router on the
attention input 0.60, depth-2 linear predictor 0.71): they lacked the layer's own MoE output, which this has.

### 2.4 Gate/up before down: staged reads inside one expert (DESIGN)

A slot is `[gate | up | down | scales]`. The GPU needs gate and up (two thirds of the bytes) first; down is needed
about 70 us later. Issue each miss as two reads and release the gate/up kernel when the first lands:

```cpp
// expert_streamer.hpp: two completion levels per slot (replaces the single mark_ready)
io_uring_prep_read_fixed(sqe, dev, slot_ptr,           gu_bytes,          off,            buf);  // part 0: gate|up
io_uring_sqe_set_data64(sqe, (uint64_t(gu_bytes) << 32) | PART_GU | r.slot);
io_uring_prep_read_fixed(sqe2, dev, slot_ptr + gu_bytes, total - gu_bytes, off + gu_bytes, buf); // part 1: down|scales
// reap(): PART_GU done -> cache_.mark_gu_ready(s);  both done -> cache_.mark_ready(s)
// engine: gate B splits into B1 (all gu_ready: run exl3_had_in + exl3_xmv gate/up + exl3_act) and B2 (all ready: down + combine)
```

The scales (`suh`/`svh`) sit at the end of the slot and the gate/up kernels read `suh`; move the 19 KB of scales to
the front in the pack layout (a repack, `tools/s15/pack_exl3.py`) or read them with part 0 as a third small read.
Bounded gain: the missed experts' gate/up time (about 70 us x 22 layers = 1.5 ms per token) hidden under the last
third of the I/O, about 3-4%. Do it after 2.3; it shares the gate plumbing.

### 2.5 Long conversations: KV pages come out of the expert pool (PARTLY BUILT, remainder is DESIGN)

Built in S20: int8 KV on a sparse buffer, grown from retired expert slots (19 KB/token: 128k context = 2.5 GB of
pool, about 15% of decode speed at 3% per 0.5 GiB). Remaining:

1. **rk ring.** The raw indexer keys (`rk`, 256 B per token per attention layer, 13 layers = 3.3 KB/token, 17% of
   the per-token KV) are only read for positions inside the current batch. Keep a ring of the last `TB_ + 3`
   positions and store those rows in each snapshot:
   ```cpp
   // attn_prep / idx_prep push constants: rk index = pos & (RK_RING - 1) instead of pos; RK_RING = 16
   // snapshot(): memcpy the ring (16 x 256 B x 13 layers = 53 KB) next to the recurrent state; restore() copies it back
   ```
   Gives back 3.3 KB/token: 128k context 2.5 GB to 2.1 GB of pool, about +2.5% decode at that length.
2. **k8v4 above a position threshold.** Measured quality cost is small (KL +0.002 to +0.005, R85) and it saves
   another 3.3 KB/token. Serve `--kv q8` up to 32k and switch the format for pages bound beyond that; the kernels
   exist (`*_kq4*`), the per-page format flag does not.
3. **Partial shrink.** Pages return to the pool only on reset or restore. Returning the tail when a conversation is
   truncated (edit and regenerate) is a call to the existing `ek_return` path with a position bound.
4. **Quality past 6k tokens is unmeasured.** A vLLM reference server scores documents up to its own context:
   start it, then run `tools/s20/score_pf.sh` with 16k and 32k documents. About 20 minutes of reference-server time;
   not run yet.

### 2.6 GPU time in the verify pass (DESIGN, 29 ms per 2-token pass is the second-largest item)

MoE is 58% of verify GPU time and the resident-expert pass is most of that (229 us of a 535 us GDN layer at T=2,
R79). The prefill already has the answer: `exl3_xmvl.comp` (LDS-tiled trellis GEMM, 3.31 TFMA/s versus 1.48 for
the verify kernel). At T=2 a tile has too few columns to pay, but at T=4 (`--spec 3`, which 2.2 makes useful) it
does. Route verify batches with `T >= 4` through `wide_moe_xm` with the chunk set to the batch. The entry tables
and `exl3_g_combine` are reused as they are. Also still S18-style: the batched HC kernels (`hc_upmixb`, `matvecb`);
port the S19 `hc_norm2560` / `matvec_ldsx10240` / `hc_upmix4` forms to `NB = 2..4`.

### 2.7 Prefill (DESIGN, 323-341 tok/s today, GPU-bound)

Per token at a 4,096 chunk: MoE 0.91 ms, GDN pre-step 1.07 ms (in-projection GEMM 12.8 us per token-layer), attention
grows with position. In order: (1) GDN delta rule chunked over the slice instead of one step per token (4.1 us per
token-layer today); (2) attention tiled over queries (one workgroup per (head, 16 queries) instead of per query) so
K/V rows are read once per tile; (3) the in-projection through `gemm_q4` with 128x128 tiles. Prefill does not bound
interactive use at 320 tok/s (a 12k-token document in 39 s), so this ranks below everything above.

### 2.8 Robustness for sustained operation (BUILT)

| failure seen in S14-S22 | fix now in the tree |
|---|---|
| killed engine leaves hugepages reserved, the next one sizes a small pool | `preflight.sh --fix` as `ExecStartPre` |
| build next to a running engine gets the engine OOM-killed | `ps5-sync-build.sh` refuses unless `--stop-server` |
| console clock behind, ninja loops on future mtimes | clock set on every sync, `tar -m` |
| `pkill -f` kills its own SSH session | scripts match with `pgrep -x` |
| no quick way to know a change is lossless | `smoke.sh` prints PASS/DIFF against the reference md5 |
| tmpfs junk eats pool, boot parameter drift | preflight warns on every start |

Not changed: the pool reserve (256 MiB below the edge that hung the console in S14). There is no memory left to
give the pool; resident data is 2.85 GiB, the pool 12.07 GiB, Linux sees 14.45 GiB plus the 0.5 GiB VRAM carve.

## 3. Order of work from here

1. A second storage device, then `--pack-mirror`, then re-measure `--spec 3 --pld 4`.
2. The offline precision gate for 2.3; build 2.3 if it passes, then 2.4.
3. rk ring and k8v4 pages (2.5), the long-document quality run on the reference server.
4. Verify through `exl3_xmvl` at T=4 (2.6), only once item 1 has made T=4 worth running.

Without new storage hardware, items 2 and 3 are the remaining gains and they total roughly 10-15% on decode. The
engine is otherwise at the measured limit of this drive for lossless decoding of this pack.
