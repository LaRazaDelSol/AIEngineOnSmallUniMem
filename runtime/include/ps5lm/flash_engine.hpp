// flash_engine.hpp — the Qwen3.8-Flash-Next decode engine (Vulkan, GFX1013) as a reusable object.
//
// S14 refactor of the S13 flash_decode.cpp monolith so one process can serve many requests with the
// weights, the pipelines AND the warm expert pool kept resident:
//   FlashEngine e(opts);            // loads dense weights (dense_q4 cache), maps the udmabuf slab, records
//                                   // the per-layer command buffers once
//   e.feed(tok, want_logits);       // one token at position e.pos(); returns the greedy argmax if want_logits
//   e.snapshot() / e.restore()      // recurrent state (GDN S, conv, PLE conv) + position: prefix reuse
//   e.reset()                       // position 0, zero recurrent state (KV needs no reset: it is written
//                                   // before it is read)
//   e.save_residency(f) / e.warm(f) // persist / preload the expert pool across process restarts
//
// Per token: CPU embedding row + n-gram hash rows -> 48 decoder layers on the GPU:
//   [PLE @ layer 1] -> attn hyper-connection -> GatedDeltaNet (36) | gated QSA attention (12, dense while
//   ctx <= 2051, which is exactly what the indexer selects) -> inject -> mlp hyper-connection -> router/top-10
//   -> (host binds streamed expert slots) -> 10 routed + shared expert -> inject
// then the hyper-connection mixer -> lm_head -> greedy argmax. Semantics follow transformers
// modeling_qwen4_exp.py (runtime/tools/ref_qwen4.py is the golden reference built from that source).
#pragma once

#include "ps5lm/expert_streamer.hpp"
#include "ps5lm/platform.hpp"
#include "ps5lm/quant.hpp"
#include "ps5lm/vk.hpp"

#include <fcntl.h>
#include <liburing.h>
#include <malloc.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

namespace ps5lm::flash {

namespace qn = ps5lm::q;

// S15: hugetlb count to restore if the process is killed by a signal (systemd stop, kill). The destructor never runs then,
// and the slab's 2 MiB pages stayed reserved as FREE hugepages (measured: 5,572 pages = 10.9 GiB): invisible to
// MemAvailable, so the next start sized a near-empty pool. Written by the engine, used by hugepage_signal_restore().
inline volatile long g_hugepages_restore = -1;
inline void hugepage_signal_restore(int sig) {  // async-signal-safe: open/write/close/_exit only
    const long n = g_hugepages_restore;
    if (n >= 0) {
        char b[24];
        int k = 0;
        long v = n;
        char t[24];
        do { t[k++] = char('0' + v % 10); v /= 10; } while (v && k < 23);
        int m = 0;
        while (k) b[m++] = t[--k];
        b[m++] = '\n';
        const int fd = ::open("/proc/sys/vm/nr_hugepages", O_WRONLY);
        if (fd >= 0) { ssize_t w = ::write(fd, b, size_t(m)); (void)w; ::close(fd); }
    }
    ::_exit(128 + sig);
}
using Clock = std::chrono::steady_clock;
inline double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

inline constexpr uint32_t L = 48, E = 512, K = 10, HID = 2560, HC = 4, HCH = HC * HID, LOWR = 320, TAB = 16;
inline constexpr uint32_t VOCAB = 248320, PLE_LAYER = 1, EOS = 248044;

struct alignas(8) PcMatvec { uint64_t w, x, y; uint32_t K, N, rpw, mode; float alpha; uint32_t pad; };
struct alignas(8) PcNorm { uint64_t x, w, outp; uint32_t G; float eps; };
struct alignas(8) PcMix { uint64_t sig, normed, xout, injl, inj; uint32_t C, pad; };
struct alignas(8) PcHcDown { uint64_t h, nw, wdown, winj, t, injl, normed; uint32_t rows_per_wg, n_inj; float alpha, eps; };
struct alignas(8) PcHcUpMix { uint64_t w, t, normed, xout, injl, inj; };
struct alignas(8) PcPost { uint64_t h, y, inj; uint32_t C, pad; };
struct alignas(8) PcConv { uint64_t x, w, state, outp, state_out; uint32_t n, pad; };
struct alignas(8) PcGdn { uint64_t conv, proj, state, alog, dtb, normw, outp, state_out; float eps; uint32_t pad; };
struct alignas(8) PcPrep { uint64_t qkv, qnw, knw, qout, gout, kc, vc, params; float eps, theta; };
struct alignas(8) PcAttn { uint64_t q, g, kc, vc, outp, scores, params; };
struct alignas(8) PcAttnSplit { uint64_t q, kc, vc, part, params; uint32_t max_chunks, pad; uint64_t sel = 0; };  // S19 sel: QSA list
struct alignas(8) PcAttnComb { uint64_t part, g, outp, params; uint32_t max_chunks, pad; uint64_t sel = 0; };
struct alignas(8) PcIdxPrep { uint64_t iqk, qnw, qi, rk, params; float eps, theta; };                       // S19 QSA indexer
struct alignas(8) PcSelStat { uint64_t sel, params, last, cnt; uint32_t win, pad; };  // S32 tiering probe
struct alignas(8) PcIdxSel { uint64_t qi, rk, pk, knw, sel, params; uint32_t t; float eps, theta; uint32_t gmin; uint64_t scr; };
struct alignas(8) PcIdxScore { uint64_t qi0, pk, rk, knw, par0, scr0; uint32_t S, T, nwg; float eps, theta; uint32_t pad; };  // S30
struct alignas(8) PcIdxScoreW { uint64_t qi0, pk, scr0; uint32_t qis, scrs, pos0, T, nwg, pad; };  // S33 wide prefill scores
struct alignas(8) PcPleGate { uint64_t keyn, qn, value, gv; };
struct alignas(8) PcPleConv { uint64_t gvn, gv, w, state, h, state_out; uint32_t n, pad; };
struct alignas(8) PcArgmax { uint64_t logits, outp; uint32_t n, pad; };
struct alignas(8) PcGateUp { uint64_t x, h, slots; uint32_t rpw, pad; };
struct alignas(8) PcDown { uint64_t h, slots, wt, resid, outp; uint32_t n_exp, rpw; };
struct alignas(8) PcDownMoe { uint64_t h, slots, wt, hs, inj; uint32_t n_exp, rows_per_wave; };
struct alignas(8) PcRouter { uint64_t w, x, logits; uint32_t n_exp, pad; };
// S15b EXL3 routed experts (shaders/exl3_*.comp)
struct alignas(8) PcExHad { uint64_t x, slots, xt; };
struct alignas(8) PcExMv { uint64_t slots, xin, out; uint32_t kt, nt, tr0, tr1, xs_e, xs_m, os_e, os_m; };
struct alignas(8) PcExAct { uint64_t pre, slots, hd; };
struct alignas(8) PcExCmb { uint64_t yd, slots, wt, out; uint32_t nexp, pad; };
struct alignas(8) PcTopk { uint64_t logits, mail; uint32_t n_exp, k, seq, pad; };
struct alignas(8) PcRouter2 { uint64_t w, w2, x, logits, logits2; uint32_t n_exp, pad; };  // S17
struct alignas(8) PcTopk2 { uint64_t logits, mail; uint32_t n_exp, k, seq, pad; uint64_t logits2, mail2; uint32_t seq2, pad2; };
struct alignas(8) PcSumsq { uint64_t x, psum; };  // S17 hc_mode 3
struct alignas(8) PcHcDown2 { uint64_t w, winj, x, nw, psum, t, il; float alpha, eps; uint32_t n_inj, pad; };
struct alignas(8) PcHcUpMix2 { uint64_t w, t, x, nw, psum, xout, injl, inj; float eps; uint32_t pad; };
struct alignas(8) PcGate { uint64_t flag, err, src0, dst0, src1, dst1, src2, dst2; uint32_t n0, n1, n2, ntok, sstr, dstr01, dstr2, pad; };  // S17 (S18: ntok rows)
// S18 stage C: batched matvec / router for T token columns (x, y column strides in bytes)
struct alignas(8) PcMatvecX { uint64_t w, x, y; uint32_t K, N, rpw, mode; float alpha; uint32_t N2; uint64_t w2, y2; uint32_t mode2; float alpha2; };  // S19 matvec_ldsx
struct alignas(8) PcMatvecB { uint64_t w, x, y; uint32_t K, N, rpw, mode; float alpha; uint32_t xs, ys, nb; };
struct alignas(8) PcRouter2B { uint64_t w, w2, x, logits, logits2; uint32_t n_exp, xs, ls, pad; };
struct alignas(8) PcExMvB { uint64_t ents, xin, out; uint32_t kt, nt, tr0, tr1, xs_e, xs_m, os_e, os_m, tstride, pad; };
struct alignas(8) PcCarry { uint64_t carry, blk, gword; uint32_t stride, n, T, dir; };  // S20 carry.comp
struct VEnt { uint64_t slot; uint32_t n, tk[4], pad; };  // S18 unique-expert entry (exl3_xmvb.comp)
// S21 wide prefill (gemm_q4 / w_*.comp / exl3_w_*.comp / exl3_xmvw.comp)
struct alignas(8) PcGemm { uint64_t w, x, y; uint32_t K, N, T, mode; float alpha; uint32_t xs, ys, ks; uint64_t rmap; };
struct alignas(8) PcKsum { uint64_t part, y; uint32_t N, T, ks, mode; float alpha; uint32_t ps, ys, pad; };
struct alignas(8) PcWNorm { uint64_t x, w, outp; uint32_t xs, os; float eps; uint32_t pad; };
struct alignas(8) PcWMix { uint64_t u, n, x, il, inj; uint32_t us, ns, xs, ils, injs, pad; };
struct alignas(8) PcWPost { uint64_t h, y, inj; uint32_t hs, ys, is, pad; };
struct alignas(8) PcWTopk { uint64_t logits, mail; uint32_t n_exp, k, seq, ls, ms, pad; };
struct alignas(8) PcWConv { uint64_t x, w, state, outp; uint32_t n, xs, os, nt; };
struct alignas(8) PcWStep { uint64_t conv, proj, state, alog, dtb, normw, outp; float eps; uint32_t cs, ps, os, nt, pad; };
struct alignas(8) PcWSact { uint64_t gu, sg, h; uint32_t gs, sgs, hs, pad; };
struct alignas(8) PcWDot { uint64_t w, x, y; uint32_t K, N, mode; float alpha; uint32_t xs, ys; };
struct alignas(8) PcWHad { uint64_t x, slots, xt; uint32_t xs, pad; };
struct alignas(8) PcWCmb { uint64_t yd, slots, wt, out; uint32_t nexp, os; };
struct alignas(8) PcXmvw { uint64_t ents, xin, out; uint32_t kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride; };
struct WEnt { uint64_t slot; uint32_t n, pad, p[16]; };  // S21 exl3_xmvw entry (NB = 16)
// S22 chunk-wide expert-major MoE: exl3_xmvl entry (pairs plist[p0 .. p0 + n) of one expert), exl3_g_combine token entry
struct alignas(8) PcXmvl { uint64_t ents, xin, out, plist; uint32_t kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride; };
struct LEnt { uint64_t slot; uint32_t n, p0; };
struct TEnt { uint32_t tok, n, gq[10]; float w[10]; };
static_assert(sizeof(TEnt) == 88 && sizeof(LEnt) == 16, "S22 entry layouts");
struct alignas(8) PcWHadG { uint64_t x, slots, xt, tok; uint32_t xs, pad; };
struct alignas(8) PcGCmb { uint64_t yd, slots, tents, out; uint32_t os, pad; };
struct alignas(8) PcPrepW { uint64_t qkv, qnw, knw, qout, gout, kc, vc, params; float eps, theta; uint32_t qs, os, pos0, pad; };
struct alignas(8) PcIdxPrepW { uint64_t iqk, qnw, qi, rk, params; float eps, theta; uint32_t is, qis, pos0, pad; };
struct alignas(8) PcIdxSelW { uint64_t qi, rk, pk, knw, sel, params; uint32_t t; float eps, theta; uint32_t gmin; uint64_t scr;
                              uint32_t qis, ss, scrs, pos0, tok0, pad; };
struct alignas(8) PcIdxPool { uint64_t rk, pk, knw; uint32_t b0, pos0, T; float eps, theta; uint32_t pad; };
struct alignas(8) PcIdxPoolP { uint64_t rk, pk, knw, params; float eps, theta; };  // S30
struct alignas(8) PcWAttn { uint64_t q, kc, vc, g, outp, sel; uint32_t qs, os, ss, pos0; };
struct alignas(8) PcWEmbed { uint64_t emb, h; uint32_t hs, pad; };
static_assert(sizeof(VEnt) == 32, "VEnt layout");
struct alignas(8) PcHcUpMixB { uint64_t w, t, normed, xout, injl, inj; uint32_t st, pad; };
struct alignas(8) PcConvB { uint64_t x, w, state, outp, so[4]; uint32_t n, st, nt, pad; };
struct alignas(8) PcGdnB { uint64_t conv, proj, state, alog, dtb, normw, outp, so[4]; float eps; uint32_t st, nt; };

struct EngineOptions {
    std::string model_dir, shader_dir = "shaders", pack_name;
    // S23: byte-identical copies of the expert pack on other block devices, "path" or "path:MBps" (default 880 = a USB
    // 10 Gbps NVMe enclosure, R6). Misses go to the device that finishes them first (expert_streamer.hpp).
    std::vector<std::string> pack_mirrors;
    // S31 lowmem tier: CPU-only second expert tier in the firmware-reserved low gigabyte (scripts/console/s31,
    // module ps5lowmem). Empty = off. A missing module only warns. tier_bias: bonus of a tier expert in cache-aware
    // routing, as a fraction of route_bias (1 = same as resident: a tier fill costs a memcpy, not an SSD read).
    std::string tier_dev;
    uint64_t tier_mib = 0;          // 0 = everything the module exposes
    uint32_t tier_threads = 4, tier_window = 64;
    double tier_min_score = 0;      // admit only experts with at least this decayed LFU use count (0 = all)
    // admit only while decoding at >= this context (S31 exact routing: +1.6% at 300 tokens, +4.7% at 123k; the served-flag
    // short-context A/B is inside the +-3% text noise of biased routing). Never during a prefill (GPU-bound, huge churn).
    uint32_t tier_min_ctx = 0;
    float tier_bias = 1.0f;
    double pack_mbps = 6400;
    double pool_gib = 0;        // <= 0: auto-size from /proc/meminfo after the weights are resident
    uint32_t pool_floor_mib = 192;    // never pin past MemAvailable - CmaFree - this, even if asked (OOM = console hang)
    bool force_pool = false;
    bool slab_hugetlb = true;   // expert slab on 2 MiB hugetlb pages (reserved by the engine, released at exit). S14b:
                                // bit-exact; pool 10.88 -> 11.25 GiB (memmap returned), 26.30 -> 27.23 tok/s
    bool pool_auto_ = false;    // internal: pool size was auto-derived
    uint32_t pool_reserve_mib = 256;  // auto pool = MemAvailable - CmaFree - this. S14 with ps5lm-min on: 0 -> 9.77 GiB ran
                                      // clean, 9.9 GiB hung the console; 256 keeps ~0.4 GiB above that edge
    bool size_classes = true;   // S15b, bits 3: K=3 and K=3.5 layers get their own slot size + LRU (false: one size)
    int bits = 4;               // 4 = PS5Q4 pack, 2 = PS5Q2, 3 = EXL3 (turboderp mul1 trellis, experts_exl3.pack, S15b)
    uint32_t predict = 0;   // early-routing prefetch depth (router of l+d on layer l's MoE input)
    // S15 trained predictor: <pred_dir>/pred_d{d}_l{l}.bf16 = router_{l+d} + ridge correction fitted on real MoE inputs
    // (runtime/tools/s15/export_pred.py). Same shape/cost as the router; precision@10 at d=1 0.626 -> 0.732 held out.
    std::string pred_dir;
    uint32_t max_ctx = 2048;     // S19/S20: up to 262144 (--ctx); beyond 2051 positions the QSA indexer selects <= 2051 tokens
    int kv16 = -1;               // S19: fp16 K/V cache (-1 auto: on when max_ctx > 2048)
    uint32_t kv_base = 0;        // S20 elastic KV: positions backed at start (0 = all of max_ctx); beyond, KV pages are
                                 // taken from retired expert slots on demand and given back on reset / restore
    uint32_t kvq = 0;            // S20 quantized KV: 8 = int8 K/V, 4 = k8v4 (int8 K + rotated int4 V), 2 = k4v4 (rotated 4-bit K and V); 0 = off (kv16/fp32)
    bool indexer = true;         // S19: QSA indexer (the model's sparse attention) when max_ctx > 2051; false = dense
    uint32_t layers = L;    // validation: run only the first N layers
    bool profile = false;   // timestamp every barrier of layer-steps 3 (attention) and 4 (GDN), last token
    bool dump = false;      // snapshot H after every layer-step (golden comparison)
    bool dump_moe = false;  // S15: copy every layer's MoE input x (what router + experts read) per token (quantizer calibration)
    bool record_trace = false;
    bool attn_legacy = false;   // S13 one-WG-per-head attention (A/B only)
    // S15 hyper-connection pre-step: 0 = S14 (norm, down+inj, up, mix: 4 kernels), 1 = norm, down+inj, hc_upmix (default),
    // 2 = hc_down + hc_upmix. Measured per HC-pre (GDN layer-step): 33.9 / ~31 / 47.3 us. Full fusion LOSES: every hc_down
    // workgroup must recompute the 4 group norms and stage 40 KB in LDS before its rows (serial prelude), while matvec
    // hides the same L2 reads across 320 waves. All three are bit-exact.
    uint32_t hc_mode = 1;
    // S15 overlap: as soon as a layer's routing is known, the gate_up of its RESIDENT experts + the shared expert is
    // submitted (no fence) and runs on the GPU while the misses stream in; the misses' gate_up runs in the next step.
    // Bit-exact: every expert's h is computed by the same kernel independently (a slot address of 0 = skip).
    bool overlap = true;
    bool spin_wait = false;     // S15: busy-poll the per-layer fence. NULL result (17.0 vs 17.1 ms GPU+sync): the ~2.2 ms
                                // GPU+sync - GPU-pure gap is per-submit CS latency (~97 submits/token), not the wake-up
    // S15b: the 16 per-token n-gram rows (320 B each, random over the 102 GB table, never page-cached) as ONE io_uring
    // batch submitted with the token and reaped just before layer PLE_LAYER (=1), i.e. under layer 0. The old serial
    // pread path cost 1.11 ms/token on the critical path (16 x 69.5 us qd1, measured R40); same bytes either way.
    bool ngram_async = true;
    uint32_t hc_rows = 4;       // rows per hc_down workgroup (mode 2): 4 -> 35.3 us, 16 -> 45.6, 2 -> 54.6
    uint32_t down_rows = 0;     // >0: Q2 routed down via down_xr.comp (x in registers). S14: bit-exact but NOT faster
                                // (48.9 us at 4 rows vs 46 us down_combine); the down pass is not x-gather bound
    bool fused_moe = false;     // S14 down_moe.comp (routed+shared+inject, 1 dispatch): correct (2.5e-6) but SLOWER,
                                // 73 us vs ~56 us for the 3-kernel path (28 KB LDS/WG -> 2 WGs/CU, loads not hidden)
    uint32_t read_split = 1;    // SQEs per expert read. S14: 1/4/8/16 measured identical (decode is bandwidth-bound)
    float prefetch_min_w = 0;   // only prefetch predicted experts whose predicted routing weight >= this
    bool prefetch_max_set = false;  // internal: --prefetch-max was given
    uint32_t prefetch_max = 5;  // and at most this many per predicted layer (highest weight first). S14: 5 = best (23.24 tok/s, 26% fewer reads than 10)
    bool cold_prefetch = false; // true: prefetches enter at the LRU tail (S14: measured WORSE, 19.95 vs 21.46 tok/s)
    uint32_t snapshots = 2; // prefix-reuse slots (each = the recurrent state, ~113 MiB for 48 layers)
    // S16: load the MTP draft head (layer 48: BF16 dense -> PS5Q4 like the trunk; its 512 routed experts from the pack,
    // EXL3 K=3 appended by tools/s15/pack_exl3_mtp.py). bits 3 only.
    bool mtp = false;
    // S16: submit the predicted prefetches of layer s+1.. AFTER layer s's demand misses have landed (they used to share the
    // SSD with them during the wait, and the SSD then idled while the GPU computed: I/O wait = SSD busy time, R54)
    bool prefetch_late = false;
    // S16 speculative decoding: MTP draft depth k (verify batches of k + 1 tokens); 0 = off. Needs mtp.
    uint32_t spec_k = 0;
    // S17 prefetch hygiene: a prefetched expert its target layer did NOT use is demoted to the LRU tail when that layer
    // binds (a wrong guess used to stay MRU for the slot's whole ~85-token life: ~14% of the pool held wrong guesses)
    bool pref_expire = false;
    // S17 idle-SSD prefetch: predictions of depth >= defer_from (0 = off) are queued and issued only while fewer than
    // defer_qd expert reads are in flight, from the GPU-wait loop (the SSD idles ~37% of the wall time, R56)
    uint32_t defer_from = 0, defer_qd = 1;
    // S17 (bits 3, overlap): the hits_ pre-pass also runs act + down of the resident experts, so after the misses land
    // only the streamed experts' act/down + the combine remain on the critical path
    bool hits_down = true;
    // S17 one command buffer per token: the whole token (49 steps + the overlap pre-passes) is ONE submit; the host
    // polls the router mailboxes in GPU-uncached control memory and releases each layer with flags the GPU waits on
    // in bounded gate kernels (shaders/gate.comp). Needs overlap; not with dump/profile.
    bool onecb = true;   // S17: 27.6 vs 26.5 tok/s (EXL3-3.18), bit-exact
    // S17 eviction: decayed-frequency LFU (half-life in tokens) instead of LRU; 0 = LRU
    double lfu_half_life = 36;
    uint32_t pf_lfu = 1;  // S22 prefill uses in the LFU: 0 = all at the chunk end (S17-S21), 1 = at their token's time, 2 = not counted
    // S17 early routing (one-cb only): layer s's own router applied to its ATTENTION input (HC attn pre-step output),
    // ~150 us before the true routing; the host prefetches those experts as soon as the early mailbox lands
    bool early = false;
    // S18: reserve the slab's 2 MiB hugepages BEFORE the dense weights fragment memory (the weights' 4 KiB GTT pages
    // left only 5,277-5,350 of ~5,500 2 MiB blocks: -4% slots). Sized from this configuration's footprint measured on the
    // previous run (dense_q4/footprint_*.txt); the first run of a configuration measures it.
    bool hp_early = true;
    bool hp_vmemmap = true;
    bool mtp_class_share = true;
    uint32_t mtp_pref = 0;  // S18: prefetch this many MTP experts predicted at the last layer (precision 0.05 measured: off)
    bool mtp_lossy = false;  // S18: drafts skip non-resident MTP experts (renormalised) instead of waiting for them
    double mtp_lfu_w = 1.0;  // S18: LFU weight of an MTP-layer use
    float mtp_lossy_w = 1.f; // ... only if their routing weight share is <= this (S18: 1 = always: acceptance 0.892 -> 0.842, net -4%)  // S18: the MTP layer counts toward its size class's slot share
    bool vb_off = false;       // S18: no verify one-cb (stage B verify() instead)
    bool vb_nobatch = false;
    // S20 batched prefill: prompt tokens go through the one-cb verify path in chunks of up to prefill_tb (<= 4) tokens,
    // recorded without verify checkpoints (state written in place) and without the lm_head; 0/1 = token-serial
    uint32_t prefill_tb = 4;
    // S20 layer-major prefill: prompt chunks of up to prefill_chunk tokens (multiple of 4) are processed one LAYER at a
    // time over all their 4-token groups, so each layer's experts are read once per chunk instead of once per token
    // window (prefill was SSD-bound at 144 MB/token). Carry buffer: prefill_chunk x 40 KB. 0 = off.
    uint32_t prefill_chunk = 1024;
    // S21 wide prefill: inside the layer-major chunk, GDN layers' pre-MoE part and EVERY layer's MoE run over slices of
    // wide_slice tokens with GEMMs (gemm_q4: 128-row x 64-token tiles, ~68% of the fp32 FMA peak) and the routed experts
    // expert-major (one trellis decode per <= 16 tokens of the slice); attention / PLE layers keep the 4-token groups.
    bool wide = true;
    uint32_t wide_slice = 256;
    uint32_t wide_mtp = 1;  // S22: the prefill's MTP KV fill as GEMMs over slices
    uint32_t wide_xm = 1;  // S22: routed MoE of the wide prefill expert-major over the whole chunk (exl3_xmvl), X/I2 kept from the router step
    bool vb_nodedupe = false;
    uint32_t tok_route = 0;    // S19 (null, R78: opt-in --tok-route 8): token-id -> layer-0 / MTP-layer expert table (learned online), prefetched as soon as the
                               // token id is known (layer-0 routing is 76% token-determined, S19 R78); 0 = off
    uint32_t tok_route_mode = 7; // S19: bit 1 = layer 0 at feed/verify start, 2 = layer 0 during the draft, 4 = MTP layer
    bool hc_fast = true;       // S19: hc_norm2560 + matvec_ldsx10240 (HC down/inject) + hc_upmix4, bit-identical
    bool mv_xrs = true;        // S19: matvec_xrs (x in registers, saddr loads, next row pipelined) for K = 2560 / 6144, bit-identical
    uint32_t ldsb_rpw = 16;    // S18: rows per workgroup of the batched K=6144 matvec (4 waves; <= 64)  // S18 A/B: verify decodes an expert shared by several tokens once per token   // S18 A/B: verify one-cb with per-token dense dispatches (no weight sharing)
    // S24 cache-aware routing (opt-in). The router mailbox carries the top route_cand candidates; the host keeps the K
    // with the highest log(weight) + bias * [resident or in flight] and renormalises over the kept set (norm_topk_prob
    // on the substituted set). route_bias: trunk layers (LOSSY: changes the output; measure with --score). route_mtp:
    // the MTP draft layer only (the verify stays exact, so the output is unchanged). route_keep: the top ranks that are
    // never displaced.
    float route_bias = 0, route_mtp = 0;
    bool route_pf = false;  // also bias the prefill (token groups and the wide layer-major path); off = the prompt is routed exactly
    uint32_t route_cand = 16, route_keep = 0;
    // S24: a displaced true top-K expert is read in the BACKGROUND (counted as a use by the LFU) while fewer than
    // route_fetch expert reads are in flight, so the pool keeps following the true routing without a token ever
    // waiting for it. 0 = off (the pool then only follows the predictor's prefetches and the kept misses).
    // Measured S24: slower (53.1 vs 57.6 tok/s, the reads compete with the kept misses), no fidelity gain. Off.
    uint32_t route_fetch = 0;
    // S35 shadow use: a displaced true top-K expert earns an LFU use of route_shadow x (its weight / the true top-K
    // weight) x K without being read, so the pool ranks it by true demand (0 = off, the S24 behaviour). With
    // route_shadow_fetch T > 0, a displaced expert whose decayed count reaches T is also queued for a background read
    // (the route_fetch queue; route_fetch caps the reads in flight, 4 if unset): only experts the model keeps asking for.
    float route_shadow = 0, route_shadow_fetch = 0;
    // S24 A/B (--route-ready): only READY experts earn the bonus (a prefetch still in flight is kept only on its own weight, so a layer
    // never waits for a read when a ready near-tie exists); false = in-flight experts count as resident too
    // (default; measured S24: ready-only is no faster and displaces more weight)
    bool route_ready = false;
    // S25 chunk lookahead of the layer-major prefill: experts read ahead per layer (0 = off). See pf_lookahead().
    uint32_t pf_ahead = 512;
    bool persist = false;      // S25: keep the staging buffer for on-disk sessions (session_store.hpp) even with --snapshots 0
    bool router_fuse = true;   // S17: router + depth-1 predictor in one dispatch, both top-k in one (bit-exact)  // S17: 2.68 pack 33.86 vs 33.33 tok/s; trace replay 0.848 -> 0.862
};

struct EngineStats {
    // S25 layer-major prefill anatomy: GPU time by phase (0 MoE, 1 GDN pre-step, 2 attention pre-step, 3 MTP KV fill) and
    // host time by job (0 n-gram rows + setup, 1 MoE pair tables, 2 router mails -> requests, 3 embedding rows)
    double pf_gpu[4] = {0, 0, 0, 0}, pf_host[4] = {0, 0, 0, 0};
    uint64_t pfa_issued = 0, pfa_used = 0, pfa_layers = 0, pfa_uniq = 0;  // S25 chunk lookahead: reads issued / later routed, layer sweeps, unique experts routed
    uint64_t wide_ents = 0, wide_pairs = 0;  // S21: expert-major entries / (token, k) pairs of the wide MoE
    uint64_t tokens = 0, demand = 0, pred_total = 0, pred_correct = 0, pref_issued = 0;
    double t_gpu = 0, t_io = 0, t_gpu_pure = 0, t_host = 0, t_ngram = 0;  // t_ngram: wait for the n-gram batch (exposed part)
    uint64_t mtp_drafts = 0, mtp_demand = 0, mtp_pred_total = 0, mtp_pred_ok = 0, mtp_skipped = 0;
    uint64_t trt_issued = 0, trt_total = 0, trt_ok = 0;  // S19 token-route table: reads issued, guesses scored / right
    double t_mtp = 0, t_mtp_io = 0, t_mtp_fill = 0;  // S16: wall time of MTP draft steps (incl. their expert I/O) / KV fills
    uint64_t spec_cycles = 0, spec_drafted = 0, spec_accepted = 0;  // S16 verify
    uint64_t prefill_chunks = 0;  // S20 batched prefill chunks
    // S16 wait anatomy (feed): per layer, class 0 = all experts ready, 1 = only prefetches in flight, 2 = >= 1 demand miss
    double wait_cls[3] = {0, 0, 0}, wait_by_dem[8] = {};
    uint64_t layers_cls[3] = {0, 0, 0}, layers_by_dem[8] = {}, n_dem = 0, n_pin = 0;
    // S17 prefetch outcome at the target layer's bind: [0] immediate (depth < defer_from), [1] deferred (idle SSD)
    uint64_t pref_used[2] = {0, 0}, pref_unused[2] = {0, 0}, pref_def_issued = 0, pref_def_dropped = 0;
    uint64_t early_total = 0, early_correct = 0, early_issued = 0;  // S17 early routing
    uint64_t rr_sel = 0, rr_swaps = 0, rr_miss_kept = 0;  // S24 cache-aware routing: biased selections, substituted experts, non-resident experts kept
    double rr_mass = 0;                 // ... and the true top-K weight share that was displaced (sum over selections)
    uint64_t rr_shadow = 0;  // S35 shadow uses given to displaced experts
    uint64_t rr_bg = 0, rr_bg_dropped = 0;  // S24 background reads of displaced experts issued / dropped (queue full)
    double gate_t[3] = {0, 0, 0};  // S17 one-cb: GPU time spent waiting in gates A (resident), B (all ready), N (n-gram)
};

// ---------------------------------------------------------------- weights
struct Dev {
    // Host-visible device memory for the resident tensors (S14 rewrite, memory is the I/O lever):
    //  * the FIRST chunk fills the free VRAM carve (512 MiB on the PS5, invisible to Linux otherwise; the old
    //    allocator put one 256 MiB chunk there and left ~247 MiB idle);
    //  * later chunks are GTT (DEVICE_LOCAL avoided explicitly), 128 MiB, FIRST-FIT across all chunks, and
    //    tensors larger than a chunk get an exact-size chunk (the old bump allocator abandoned every chunk's
    //    tail: 2.85 GiB of tensors occupied 3.4 GiB).
    struct Chunk { vk::Buffer b; uint64_t top = 0; };
    vk::Ctx* ctx;
    std::vector<Chunk> chunks;
    uint64_t used = 0, allocated = 0, vram_bytes = 0;
    uint64_t vram_margin = 48ull << 20;  // left for RADV internals (shaders, command/upload buffers) + the console
    static constexpr uint64_t kChunk = 128ull << 20;
    std::function<void(uint64_t)> before_gtt;
    static uint64_t sysfs_u64(const char* path) {
        uint64_t v = 0;
        if (FILE* f = std::fopen(path, "r")) { if (std::fscanf(f, "%llu", reinterpret_cast<unsigned long long*>(&v)) != 1) v = 0; std::fclose(f); }
        return v;
    }
    std::pair<uint64_t, void*> zalloc(uint64_t bytes) { auto r = alloc(bytes); std::memset(r.second, 0, bytes); return r; }
    std::pair<uint64_t, void*> alloc(uint64_t bytes, uint64_t align = 256) {
        for (auto& c : chunks) {  // first fit
            const uint64_t off = (c.top + align - 1) / align * align;
            if (off + bytes <= c.b.size) {
                c.top = off + bytes;
                used += bytes;
                return {c.b.addr + off, static_cast<uint8_t*>(c.b.map) + off};
            }
        }
        Chunk c;
        if (chunks.empty()) {  // VRAM carve first
            const uint64_t tot = sysfs_u64("/sys/class/drm/card0/device/mem_info_vram_total");
            const uint64_t use = sysfs_u64("/sys/class/drm/card0/device/mem_info_vram_used");
            if (tot > use + vram_margin + bytes && tot - use - vram_margin >= (64ull << 20)) {
                const uint64_t sz = (tot - use - vram_margin) & ~((2ull << 20) - 1);
                try {
                    c.b = ctx->create(sz, true, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
                    if (!(ctx->memprops.memoryTypes[c.b.mem_type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                        ctx->destroy(c.b);  // no CPU-visible VRAM type: fall through to GTT
                    } else {
                        vram_bytes = sz;
                    }
                } catch (const std::exception&) { c.b = {}; }
            }
        }
        if (!c.b.buf) {
            const uint64_t sz = std::max<uint64_t>((bytes + 4095) & ~4095ull, kChunk);
            if (before_gtt) before_gtt(sz);  // S18: the engine may hand early-reserved hugepages back first
            c.b = ctx->create(sz, true, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);  // GTT
        }
        allocated += c.b.size;
        c.top = bytes;
        used += bytes;
        chunks.push_back(c);
        return {c.b.addr, c.b.map};
    }
    VkBuffer buf_of(uint64_t addr, uint64_t& off) const {
        for (auto& c : chunks) if (addr >= c.b.addr && addr < c.b.addr + c.b.size) { off = addr - c.b.addr; return c.b.buf; }
        return VK_NULL_HANDLE;
    }
};

// S18: read a file straight into `dst` (e.g. a host-visible GTT mapping): no transient heap buffer. Returns false on a
// short read.
inline bool read_into(const std::string& p, void* dst, size_t bytes) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return false;
    size_t got = 0;
    while (got < bytes) { ssize_t r = ::pread(fd, static_cast<uint8_t*>(dst) + got, bytes - got, off_t(got)); if (r <= 0) break; got += size_t(r); }
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    return got == bytes;
}
inline std::vector<uint8_t> read_file(const std::string& p) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("missing " + p);
    struct stat st;
    fstat(fd, &st);
    std::vector<uint8_t> b(size_t(st.st_size));
    size_t got = 0;
    while (got < b.size()) { ssize_t r = ::pread(fd, b.data() + got, b.size() - got, off_t(got)); if (r <= 0) break; got += size_t(r); }
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // S14: 2.6 GB of weight files must not linger as page cache
    ::close(fd);
    return b;
}

// ---------------------------------------------------------------- n-gram hashing (Qwen4ExpTextNGramEmbedding)
struct NGram {
    int64_t mult[3] = {0, 0, 0};
    std::vector<int64_t> vs, off;
    int fd = -1;
    static int64_t pymod(int64_t a, int64_t m) { int64_t r = a % m; return r < 0 ? r + m : r; }
    static int64_t wmul(int64_t a, int64_t b) { return int64_t(uint64_t(a) * uint64_t(b)); }
    // hist includes the 2 leading EOS of the HF previous_context; t = index of the current token in hist.
    void ids(const std::vector<int64_t>& hist, size_t t, int64_t out[16]) const {
        int64_t prev_eos = -1;
        for (size_t j = 0; j < t; ++j) if (hist[j] == EOS) prev_eos = int64_t(j);
        const int64_t pis = int64_t(t) - (prev_eos + 1);
        int64_t sh[3];
        for (int s = 0; s < 3; ++s) sh[s] = (pis >= s && int64_t(t) - s >= 0) ? hist[t - s] : int64_t(EOS);
        for (int n = 2; n <= 3; ++n) {
            int64_t mixed = wmul(sh[0], mult[0]);
            for (int p = 1; p < n; ++p) mixed ^= wmul(sh[p], mult[p]);
            for (int h = 0; h < 8; ++h) { const int hi = (n - 2) * 8 + h; out[hi] = pymod(mixed, vs[hi]) + off[hi]; }
        }
    }
};

struct HcW { uint64_t norm, down, up, inj; };
struct LayerW {
    bool attn = false;
    HcW hca, hcm;
    uint64_t inproj = 0, conv = 0, alog = 0, dtb = 0, gnorm = 0, outproj = 0;  // GDN
    uint64_t qkv = 0, qn = 0, kn = 0, oproj = 0;                              // attention
    uint64_t iqk = 0, iqn = 0, ikn = 0;                                       // S19 QSA indexer
    uint64_t router = 0, sgate = 0, shared_slot = 0;
};

class FlashEngine {
public:
    explicit FlashEngine(const EngineOptions& o) : o_(o), d_{&ctx_} {
        NL_ = std::min(o.layers, L);
        if (o_.max_ctx > 262144) o_.max_ctx = 262144;  // S20: the model's max_position_embeddings (idx_select spills to global > 8192 blocks)
        o_.max_ctx = (o_.max_ctx + 63) & ~63u;
        if (o_.kv16 < 0) o_.kv16 = o_.max_ctx > 2048 ? 1 : 0;
        if (o_.kvq && o_.kvq != 8 && o_.kvq != 4 && o_.kvq != 2) throw std::runtime_error("--kv: q8, k8v4 or k4v4");
        if (o_.kvq) o_.kv16 = 0;
        idx_on_ = o_.indexer && o_.max_ctx > 2051;
        if (o_.pack_name.empty()) o_.pack_name = o_.bits == 2 ? "experts_q2.pack" : o_.bits == 3 ? "experts_exl3.pack" : "experts_q4.pack";
        if (o_.mtp && (o_.bits != 3 || NL_ != L)) throw std::runtime_error("--mtp needs --bits 3 and all 48 layers");
        if (o_.spec_k && !o_.mtp) o_.mtp = true;  // speculative decoding drafts with the MTP head
        if (1 + o_.spec_k > kMaxTB) throw std::runtime_error("spec_k too large");
        if (o_.bits == 3) load_exl3_geometry();
        ::mkdir((o_.model_dir + "/dense_q4").c_str(), 0755);
        if (o_.slab_hugetlb && o_.hp_early) early_reserve();  // S18: also for an explicit --pool-gib (deterministic A/B pools)
        d_.before_gtt = [this](uint64_t b) { guard_early(b); };
        load_weights();
        alloc_activations();
        map_pool();
        ek_init_pool();
        load_pipelines();
        record();
        hist_ = {EOS, EOS};
        ::malloc_trim(0);  // give the loader's temporary buffers back to the kernel
    }
    ~FlashEngine() {
        vkDeviceWaitIdle(ctx_.dev);
        if (o_.slab_hugetlb && prev_hugepages_ >= 0) set_hugepages(prev_hugepages_);  // in-use pages free at exit as surplus
        if (ngr_ok_) io_uring_queue_exit(&ngr_);
    }
    FlashEngine(const FlashEngine&) = delete;
    FlashEngine& operator=(const FlashEngine&) = delete;

    uint32_t pos() const { return uint32_t(hist_.size() - 2); }
    std::vector<int64_t> tokens() const { return {hist_.begin() + 2, hist_.end()}; }
    int64_t token_at(uint32_t i) const { return hist_[size_t(i) + 2]; }  // S23: history without the copy (PLD index)
    uint32_t max_ctx() const { return o_.max_ctx; }
    bool full() const { return NL_ == L; }
    const float* logits() const { return static_cast<const float*>(lg_.map); }
    uint32_t vocab() const { return VOCAB; }
    double resident_gib() const { return double(d_.used) / double(1ull << 30); }
    double allocated_gib() const { return double(d_.allocated) / double(1ull << 30); }
    double vram_gib() const { return double(d_.vram_bytes) / double(1ull << 30); }
    double pool_gib() const { return double(n_slots_) * slot_bytes_ / double(1ull << 30); }
    uint32_t pool_slots() const { return n_slots_; }
    ExpertStreamer& streamer() { return *st_; }
    EngineStats& stats() { return stats_; }
    const std::vector<std::vector<uint16_t>>& trace() const { return trace_; }
    const float* dump_area() const { return static_cast<const float*>(pDump_); }
    size_t dump_floats() const { return size_t(NL_ + 1) * HCH; }
    const float* moe_in() const { return static_cast<const float*>(pMoe_); }
    size_t moe_in_floats() const { return size_t(NL_) * HID; }

    // ---- recurrent state management (prefix reuse across requests) ----
    // Snapshot slots hold the recurrent state (GDN S, GDN conv, PLE conv) at a position p plus the tokens
    // 0..p-1. KV entries are NOT copied: a slot is restorable only while the KV cache still holds the same
    // tokens at positions [0, p) (kv_hist_ tracks what each KV position was last written with).
    void reset() {
        ctx_.run(cb_reset_);
        hist_ = {EOS, EOS};
        carry_release();
        ek_shrink(0);
    }
    uint32_t n_snapshots() const { return uint32_t(snaps_.size()); }
    void snapshot(uint32_t slot) {
        auto& sn = snaps_.at(slot);
        // GPU state -> 16 MiB HOST_CACHED staging -> plain heap memory (S14: snapshots used to be 113 MiB of pinned
        // GPU memory each; heap pages are movable, so they can even live in the KHO/CMA scratch the pool cannot use)
        if (sn.host.size() != rstate_bytes_) sn.host.resize(rstate_bytes_);
        for (size_t k = 0; k < seg_save_.size(); ++k) {
            ctx_.run(seg_save_[k]);
            const uint64_t off = uint64_t(k) * kSeg, len = std::min<uint64_t>(kSeg, rstate_bytes_ - off);
            std::memcpy(sn.host.data() + off, stage_.map, len);
        }
        sn.hist = hist_;
        sn.valid = true;
        sn.last_use = ++use_clock_;
    }
    // Tokens 0..p-1 of a slot, or empty if the slot is unusable (never taken, or KV overwritten since).
    std::vector<int64_t> snapshot_tokens(uint32_t slot) const {
        const auto& sn = snaps_.at(slot);
        if (!sn.valid) return {};
        const size_t p = sn.hist.size() - 2;
        if (kv_hist_.size() < p || !std::equal(sn.hist.begin() + 2, sn.hist.end(), kv_hist_.begin())) return {};
        return {sn.hist.begin() + 2, sn.hist.end()};
    }
    void touch_snapshot(uint32_t slot) { snaps_.at(slot).last_use = ++use_clock_; }
    uint64_t snapshot_last_use(uint32_t slot) const { return snaps_.at(slot).valid ? snaps_.at(slot).last_use : 0; }
    bool restore(uint32_t slot) {
        auto& sn = snaps_.at(slot);
        if (!sn.valid) return false;
        for (size_t k = 0; k < seg_restore_.size(); ++k) {
            const uint64_t off = uint64_t(k) * kSeg, len = std::min<uint64_t>(kSeg, rstate_bytes_ - off);
            std::memcpy(stage_.map, sn.host.data() + off, len);
            ctx_.run(seg_restore_[k]);
        }
        hist_ = sn.hist;
        sn.last_use = ++use_clock_;
        carry_release();
        ek_shrink(pos());
        return true;
    }

    // ---- S25 persistent context: the per-position caches and the recurrent state to / from files ----
    // What a position costs to recompute is ~3 ms of GPU; what it costs to read back is 19 KB. session_store.hpp keeps
    // conversations on the SSD with these primitives. File layout of the KV rows: blocks of kKvBlock positions, inside a
    // block one run per cache (allocation order), so rows append in place and every transfer is 4 KiB aligned
    // (kKvGran positions of any cache are a multiple of 4096 bytes). All of them run with the GPU idle.
    static constexpr uint32_t kKvGran = 256, kKvBlock = 4096;
    bool persist_ok() const { return stage_.buf && !kvb_.empty() && o_.max_ctx % kKvGran == 0 && rstate_bytes_ % 4096 == 0; }
    size_t kv_caches() const { return kvb_.size(); }
    uint64_t kv_per_pos(size_t i) const { return kvb_[i].per_pos; }
    uint64_t kv_row_bytes() const { uint64_t t = 0; for (auto& k : kvb_) t += k.per_pos; return t; }
    uint64_t rstate_bytes() const { return rstate_bytes_; }
    const std::vector<int64_t>& kv_tokens() const { return kv_hist_; }
    uint64_t kv_rows_backed() const { return kv_backed(); }
    int bits() const { return o_.bits; }
    uint64_t kv_file_off(size_t i, uint64_t row) const {
        uint64_t pre = 0;
        for (size_t j = 0; j < i; ++j) pre += kvb_[j].per_pos;
        return row / kKvBlock * kv_row_bytes() * kKvBlock + pre * kKvBlock + (row % kKvBlock) * kvb_[i].per_pos;
    }
    static void io_full(int fd, bool wr, uint8_t* p, uint64_t len, uint64_t off) {
        for (uint64_t done = 0; done < len;) {
            const ssize_t r = wr ? ::pwrite(fd, p + done, len - done, off_t(off + done)) : ::pread(fd, p + done, len - done, off_t(off + done));
            if (r <= 0) throw std::runtime_error(std::string("session file ") + (wr ? "write" : "read") + " failed: " + (r < 0 ? std::strerror(errno) : "short"));
            done += uint64_t(r);
        }
    }
    uint8_t* bounce() {
        if (!bounce_ && ::posix_memalign(reinterpret_cast<void**>(&bounce_), 4096, kSeg) != 0) throw std::runtime_error("bounce buffer");
        return bounce_;
    }
    // rows [lo, hi) of every cache <-> fd (lo, hi multiples of kKvGran, hi <= max_ctx)
    void kv_io(int fd, bool save, uint64_t lo, uint64_t hi) {
        if (hi <= lo) return;
        if (lo % kKvGran || hi % kKvGran || hi > o_.max_ctx) throw std::runtime_error("kv_io: unaligned row range");
        if (!save) ek_ensure(hi);
        else if (hi > kv_backed()) throw std::runtime_error("kv_io: rows beyond the backed range");
        uint8_t* bb = bounce();
        struct Part { size_t i; uint64_t boff, soff, len, foff; };
        std::vector<Part> parts;
        uint64_t used = 0;
        auto flush = [&]() {
            if (parts.empty()) return;
            if (!save) {
                for (auto& p : parts) io_full(fd, false, bb + p.soff, p.len, p.foff);
                std::memcpy(stage_.map, bb, used);
            }
            VkCommandBuffer cb = ctx_.cmd();
            vk::Ctx::begin(cb);
            xbar(cb);
            for (auto& p : parts) {
                uint64_t off = 0;
                VkBuffer b = kvb_[p.i].sparse ? kvb_[p.i].sparse : d_.buf_of(kvb_[p.i].addr, off);
                const VkBufferCopy rg = save ? VkBufferCopy{off + p.boff, p.soff, p.len} : VkBufferCopy{p.soff, off + p.boff, p.len};
                vkCmdCopyBuffer(cb, save ? b : stage_.buf, save ? stage_.buf : b, 1, &rg);
            }
            xbar(cb);
            vk::Ctx::barrier(cb);
            vk::Ctx::end(cb);
            ctx_.run(cb);
            vkFreeCommandBuffers(ctx_.dev, ctx_.pool, 1, &cb);
            if (save) {
                std::memcpy(bb, stage_.map, used);
                for (auto& p : parts) io_full(fd, true, bb + p.soff, p.len, p.foff);
            }
            parts.clear();
            used = 0;
        };
        for (size_t i = 0; i < kvb_.size(); ++i) {
            const uint64_t pp = kvb_[i].per_pos;
            for (uint64_t r = lo; r < hi;) {
                const uint64_t room = (kSeg - used) / (pp * kKvGran) * kKvGran;
                if (!room) { flush(); continue; }
                const uint64_t n = std::min({hi, (r / kKvBlock + 1) * kKvBlock, r + room}) - r;
                parts.push_back({i, r * pp, used, n * pp, kv_file_off(i, r)});
                used += n * pp;
                r += n;
            }
        }
        flush();
    }
    // the live recurrent state <-> fd (offset 0)
    void rstate_io(int fd, bool save) {
        uint8_t* bb = bounce();
        for (size_t k = 0; k < seg_save_.size(); ++k) {
            const uint64_t off = uint64_t(k) * kSeg, len = std::min<uint64_t>(kSeg, rstate_bytes_ - off);
            if (save) {
                ctx_.run(seg_save_[k]);
                std::memcpy(bb, stage_.map, len);
                io_full(fd, true, bb, len, off);
            } else {
                io_full(fd, false, bb, len, off);
                std::memcpy(stage_.map, bb, len);
                ctx_.run(seg_restore_[k]);
            }
        }
    }
    // an in-memory snapshot's recurrent state -> fd (no GPU work); false if the slot is not usable
    bool snapshot_write(uint32_t slot, int fd) {
        const auto& sn = snaps_.at(slot);
        if (!sn.valid || sn.host.size() != rstate_bytes_) return false;
        uint8_t* bb = bounce();
        for (uint64_t off = 0; off < rstate_bytes_; off += kSeg) {
            const uint64_t len = std::min<uint64_t>(kSeg, rstate_bytes_ - off);
            std::memcpy(bb, sn.host.data() + off, len);
            io_full(fd, true, bb, len, off);
        }
        return true;
    }
    // After kv_io(load) + rstate_io(load): the engine is at position toks.size() of that conversation; kv_toks = what the
    // KV rows now hold from position 0 (>= toks; in-memory snapshots past the first difference stop matching by themselves).
    void adopt(const std::vector<int64_t>& toks, const std::vector<int64_t>& kv_toks) {
        hist_.assign({int64_t(EOS), int64_t(EOS)});
        hist_.insert(hist_.end(), toks.begin(), toks.end());
        kv_hist_ = kv_toks;
        carry_release();
        ek_shrink(pos());
    }

    // ---- one token ----
    // Feeds `tok` at position pos(). want_out: run the mixer + lm_head + argmax (logits() valid after).
    // Returns the argmax token when want_out, else UINT32_MAX.
    uint32_t feed(int64_t tok, bool want_out) {
        carry_release();
        st_->tier_admit_enable(pos() >= o_.tier_min_ctx);  // S31
        ek_ensure(uint64_t(pos()) + 1 + o_.spec_k + 2);
        st_->cache().tick();
        if (tok_cb_[0]) return feed_onecb(tok, want_out);
        if (pos() >= o_.max_ctx) throw std::runtime_error("context full");
        auto t_h0 = Clock::now();
        hist_.push_back(tok);
        // embedding row (bf16) -> 4 identical streams
        if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(tok) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
        auto* H = static_cast<float*>(pH_);
        for (uint32_t i = 0; i < HID; ++i) { uint32_t x = uint32_t(row_[i]) << 16; float f; std::memcpy(&f, &x, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + i] = f; }
        if (use_ple_) {
            int64_t nid[16];
            ng_.ids(hist_, hist_.size() - 1, nid);
            if (ng_pending_) ng_finish();  // a previous feed threw before reaping: drain before reusing ngbuf_
            if (ngr_ok_) {  // S15b: one batch, reaped before layer PLE_LAYER (ng_finish)
                for (int h = 0; h < 16; ++h) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ngr_);
                    ngoff_[h] = uint64_t(nid[h]) * 320;
                    io_uring_prep_read(sqe, ng_.fd, ngbuf_ + h * 160, 320, ngoff_[h]);
                    io_uring_sqe_set_data64(sqe, uint64_t(h));
                }
                if (io_uring_submit(&ngr_) != 16) throw std::runtime_error("ngram submit");
                ng_pending_ = 16;
                ng_tokens_ = 1;
            } else {
                for (int h = 0; h < 16; ++h)
                    if (::pread(ng_.fd, ngbuf_ + h * 160, 320, off_t(uint64_t(nid[h]) * 320)) != 320) throw std::runtime_error("ngram read");
                ng_convert();
            }
        }
        static_cast<uint32_t*>(pPar_)[0] = pos() - 1;
        if (kv_hist_.size() < pos()) kv_hist_.resize(pos());
        kv_hist_[pos() - 1] = tok;
        if (o_.record_trace) trace_.emplace_back();
        stats_.t_host += secs(t_h0, Clock::now());
        deferred_.clear();
        dq_head_ = 0;
        for (auto& v : pref_rec_) v.clear();
        for (uint32_t s = 0; s <= NL_; ++s) {
            VkCommandBuffer cb = s < NL_ ? steps_[s] : (want_out ? step_head_ : step_nohead_);
            if (s == PLE_LAYER && ng_pending_) ng_finish();
            auto a = Clock::now();
            ctx_.submit(cb, fence_);
            cur_layer_ = s;
            gpu_wait();
            stats_.t_gpu += secs(a, Clock::now());
            for (SlotId p : pinned_) st_->cache().unpin(p);
            pinned_.clear();
            if (s == NL_) break;
            const uint32_t* mail = reinterpret_cast<const uint32_t*>(static_cast<uint8_t*>(pMail_) + uint64_t(s) * 256);
            uint32_t mloc[32];
            if (route_on(s)) { std::memcpy(mloc, mail, sizeof mloc); route_select(s, mloc, o_.route_bias); mail = mloc; }
            const float* mw = reinterpret_cast<const float*>(mail + 16);
            auto c = Clock::now();
            SlotId sl[K];
            for (uint32_t k = 0; k < K; ++k) if (mail[k] >= E) throw std::runtime_error("invalid expert id " + std::to_string(mail[k]) + " at layer " + std::to_string(s));
            if (o_.record_trace) for (uint32_t k = 0; k < K; ++k) trace_.back().push_back(uint16_t(mail[k]));
            // Two passes: resident/in-flight experts first (pin + promote), then misses, so a miss can never
            // evict a (cold-inserted) prefetch of this same layer that has not been looked up yet.
            uint32_t n_dem = 0, n_pin = 0;  // S16 wait anatomy: demand misses / prefetches still in flight at bind time
            for (uint32_t k = 0; k < K; ++k) {
                const SlotId f = st_->cache().find(s, uint16_t(mail[k]));
                if (f == kNoSlot) ++n_dem; else if (!st_->cache().ready(f)) ++n_pin;
            }
            for (int pass = 0; pass < 2; ++pass)
                for (uint32_t k = 0; k < K; ++k) {
                    const bool present = st_->cache().find(s, uint16_t(mail[k])) != kNoSlot;
                    if (present != (pass == 0)) continue;
                    SlotCache::Lookup r;
                    r = request_wait(s, uint16_t(mail[k]));
                    st_->cache().pin(r.slot);
                    pinned_.push_back(r.slot);
                    sl[k] = r.slot;
                    ++stats_.demand;
                }
            expire_prefetches(s, mail);
            // score the predictions made for this layer, then prefetch the predicted experts of layers s+1..s+D
            for (uint32_t dd = 1; dd <= o_.predict && s >= dd; ++dd) {
                const uint32_t* pm = reinterpret_cast<const uint32_t*>(pmail_host_ + (uint64_t(dd) * L + s) * 256);
                for (uint32_t k = 0; k < K; ++k) {
                    ++stats_.pred_total;
                    for (uint32_t j = 0; j < K; ++j) if (pm[k] == mail[j]) { ++stats_.pred_correct; break; }
                }
            }
            if (!o_.prefetch_late) prefetch(s);
            st_->submit();
            const double io_before = stats_.t_io;
            wait_bind(s, sl, mail, mw, c);
            {
                const double w = stats_.t_io - io_before;
                const int cls = n_dem ? 2 : n_pin ? 1 : 0;
                stats_.wait_cls[cls] += w;
                ++stats_.layers_cls[cls];
                stats_.n_dem += n_dem;
                stats_.n_pin += n_pin;
                if (n_dem < 8) { stats_.wait_by_dem[n_dem] += w; ++stats_.layers_by_dem[n_dem]; }
            }
            if (o_.prefetch_late) { prefetch(s); st_->submit(); }  // S16: transfers while the GPU runs layer s's MoE + s+1's pre
        }
        {
            auto ts = ctx_.read_ts(tsq_, 2 * (NL_ + 1));
            for (uint32_t s2 = 0; s2 <= NL_; ++s2) stats_.t_gpu_pure += double(ts[2 * s2 + 1] - ts[2 * s2]) * ctx_.ts_period_ns * 1e-9;
        }
        ++stats_.tokens;
        if (st_->stats().io_errors > 64) throw std::runtime_error("expert I/O errors");
        return want_out && full() ? static_cast<uint32_t*>(arg_.map)[0] : UINT32_MAX;
    }

    // ---- S17: one token as ONE command buffer ----
    // Host side of record_onecb(): zero the control words, submit, then per layer: poll the router mailbox, bind (resident
    // first), publish the resident table (flag A: the GPU starts their gate/up/act/down), prefetch, wait the misses,
    // publish the full table (flag B). Same binding logic as feed(), so the tokens are identical.
    void service() {
        st_->reap(false);
        if (o_.defer_from) pump_deferred();
        if (!rr_q_.empty()) rr_pump();
    }
    void ctl_fail() {
        if (vkGetFenceStatus(ctx_.dev, fence_) == VK_SUCCESS) {  // the GPU finished without publishing: a gate timed out
            vkResetFences(ctx_.dev, 1, &fence_);
            throw std::runtime_error("one-cb: GPU gate timed out (control error " + std::to_string(*ctlw(kCtlErr)) + ")");
        }
    }
    void wait_word(uint64_t off, uint32_t want) {
        for (uint32_t n = 1; *ctlw(off) != want; ++n) {
            service();
            if ((n & 0xFFFF) == 0) ctl_fail();
            __builtin_ia32_pause();
        }
    }
    void ng_finish_ctl() {  // n-gram rows -> f32 into the control block, then release the PLE gate
        if (ng_pending_) {
            auto t = Clock::now();
            while (ng_pending_) {
                io_uring_cqe* cqe = nullptr;
                int r = io_uring_wait_cqe(&ngr_, &cqe);
                if (r == -EINTR) continue;
                if (r < 0) throw std::runtime_error("ngram wait");
                const int res = cqe->res;
                const uint64_t h = io_uring_cqe_get_data64(cqe);
                io_uring_cqe_seen(&ngr_, cqe);
                --ng_pending_;
                if (res != 320 && ::pread(ng_.fd, ngbuf_ + h * 160, 320, off_t(ngoff_[h])) != 320) throw std::runtime_error("ngram read " + std::to_string(res));
            }
            stats_.t_ngram += secs(t, Clock::now());
        }
        float tmp[16 * 160];
        for (int i = 0; i < 16 * 160; ++i) { uint32_t x = uint32_t(ngbuf_[i]) << 16; std::memcpy(&tmp[i], &x, 4); }
        std::memcpy(ctlp_ + kCtlE, tmp, sizeof tmp);
        __builtin_ia32_sfence();
        *ctlw(kCtlEFlag) = 1;
    }
    uint32_t feed_onecb(int64_t tok, bool want_out) {
        if (pos() >= o_.max_ctx) throw std::runtime_error("context full");
        auto t_h0 = Clock::now();
        hist_.push_back(tok);
        if (o_.tok_route_mode & 1) tok_prefetch(0, tok);  // S19: layer-0 experts this token used last time (before the submit)
        if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(tok) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
        auto* H = static_cast<float*>(pH_);
        for (uint32_t i = 0; i < HID; ++i) { uint32_t x = uint32_t(row_[i]) << 16; float f; std::memcpy(&f, &x, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + i] = f; }
        bool ng_now = false;
        if (use_ple_) {
            int64_t nid[16];
            ng_.ids(hist_, hist_.size() - 1, nid);
            if (ng_pending_) ng_finish();
            if (ngr_ok_) {
                for (int h = 0; h < 16; ++h) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ngr_);
                    ngoff_[h] = uint64_t(nid[h]) * 320;
                    io_uring_prep_read(sqe, ng_.fd, ngbuf_ + h * 160, 320, ngoff_[h]);
                    io_uring_sqe_set_data64(sqe, uint64_t(h));
                }
                if (io_uring_submit(&ngr_) != 16) throw std::runtime_error("ngram submit");
                ng_pending_ = 16;
                ng_tokens_ = 1;
            } else {
                for (int h = 0; h < 16; ++h)
                    if (::pread(ng_.fd, ngbuf_ + h * 160, 320, off_t(uint64_t(nid[h]) * 320)) != 320) throw std::runtime_error("ngram read");
                ng_now = true;
            }
        }
        static_cast<uint32_t*>(pPar_)[0] = pos() - 1;
        if (kv_hist_.size() < pos()) kv_hist_.resize(pos());
        kv_hist_[pos() - 1] = tok;
        if (o_.record_trace) trace_.emplace_back();
        // control words of this token: mailbox / prediction seqs, stage flags, the n-gram flag
        for (uint32_t s = 0; s < NL_; ++s) {
            *ctlw(kCtlMail + uint64_t(s) * 256 + 128) = 0;
            *ctlw(kCtlEarly + uint64_t(s) * 256 + 128) = 0;
            *ctlw(stage(s) + 448) = 0;
            *ctlw(stage(s) + 452) = 0;
            for (uint32_t dd = 1; dd <= o_.predict && dd < 4; ++dd) *ctlw(kCtlPmail + (uint64_t(dd) * L + s) * 256 + 128) = 0;
        }
        *ctlw(kCtlEFlag) = 0;
        if (*ctlw(kCtlErr)) throw std::runtime_error("one-cb: control error set");
        __builtin_ia32_sfence();
        if (ng_now) ng_finish_ctl();
        deferred_.clear();
        dq_head_ = 0;
        for (auto& v : pref_rec_) v.clear();
        pmail_host_ = ctlp_ + kCtlPmail;
        stats_.t_host += secs(t_h0, Clock::now());
        auto a = Clock::now();
        ctx_.submit(tok_cb_[want_out ? 1 : 0], fence_);
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        for (uint32_t s = 0; s < NL_; ++s) {
            cur_layer_ = s;
            uint32_t early_ids[K];
            bool have_early = false;
            if (o_.early) {  // early routing lands ~150 us before the true mailbox: prefetch what is not resident
                wait_word(kCtlEarly + uint64_t(s) * 256 + 128, s + 1);
                std::memcpy(early_ids, ctlp_ + kCtlEarly + uint64_t(s) * 256, sizeof early_ids);
                have_early = true;
                bool any = false;
                for (uint32_t k = 0; k < K; ++k) {
                    if (early_ids[k] >= E || st_->cache().find(s, uint16_t(early_ids[k])) != kNoSlot) continue;
                    if (st_->request(s, uint16_t(early_ids[k]), false, true).slot != kNoSlot) { ++stats_.early_issued; any = true; }
                }
                if (any) st_->submit();
            }
            wait_word(kCtlMail + uint64_t(s) * 256 + 128, s + 1);  // PRE(s) done (and POSTMOE(s-1): its pins can go)
            stats_.t_gpu += secs(a, Clock::now());
            for (SlotId p : pinned_) st_->cache().unpin(p);
            pinned_.clear();
            uint32_t mail[32];
            std::memcpy(mail, ctlp_ + kCtlMail + uint64_t(s) * 256, sizeof mail);  // ids[16] | w[16] (one uncached read)
            if (route_on(s)) route_select(s, mail, o_.route_bias);
            const float* mw = reinterpret_cast<const float*>(mail + 16);
            auto c = Clock::now();
            SlotId sl[K];
            for (uint32_t k = 0; k < K; ++k) if (mail[k] >= E) throw std::runtime_error("invalid expert id " + std::to_string(mail[k]) + " at layer " + std::to_string(s));
            if (o_.record_trace) for (uint32_t k = 0; k < K; ++k) trace_.back().push_back(uint16_t(mail[k]));
            if (s == 0) tok_learn(0, tok, mail);
            if (have_early)
                for (uint32_t k = 0; k < K; ++k) {
                    ++stats_.early_total;
                    for (uint32_t j = 0; j < K; ++j) if (early_ids[k] == mail[j]) { ++stats_.early_correct; break; }
                }
            uint32_t n_dem = 0, n_pin = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const SlotId f = st_->cache().find(s, uint16_t(mail[k]));
                if (f == kNoSlot) ++n_dem; else if (!st_->cache().ready(f)) ++n_pin;
            }
            for (int pass = 0; pass < 2; ++pass)
                for (uint32_t k = 0; k < K; ++k) {
                    const bool present = st_->cache().find(s, uint16_t(mail[k])) != kNoSlot;
                    if (present != (pass == 0)) continue;
                    SlotCache::Lookup r;
                    r = request_wait(s, uint16_t(mail[k]));
                    st_->cache().pin(r.slot);
                    pinned_.push_back(r.slot);
                    sl[k] = r.slot;
                    ++stats_.demand;
                }
            st_->submit();  // demand reads first
            expire_prefetches(s, mail);
            {  // flag A: resident experts start (gate/up/act/down) while the misses stream
                uint64_t ta[TAB] = {};
                for (uint32_t k = 0; k < K; ++k) ta[k] = st_->cache().ready(sl[k]) ? slot_addr(sl[k]) : 0;
                std::memcpy(ctlp_ + stage(s), ta, sizeof ta);
                __builtin_ia32_sfence();
                *ctlw(stage(s) + 448) = 1;
                for (uint32_t dd = 1; dd <= o_.predict && s >= dd; ++dd) {
                    const uint32_t* pm = reinterpret_cast<const uint32_t*>(pmail_host_ + (uint64_t(dd) * L + s) * 256);
                    for (uint32_t k = 0; k < K; ++k) {
                        ++stats_.pred_total;
                        const uint32_t e = pm[k];
                        for (uint32_t j = 0; j < K; ++j) if (e == mail[j]) { ++stats_.pred_correct; break; }
                    }
                }
                if (!o_.prefetch_late) {
                    for (uint32_t dd = 1; dd <= o_.predict && s + dd < NL_; ++dd) wait_word(kCtlPmail + (uint64_t(dd) * L + s + dd) * 256 + 128, s + dd + 1);
                    prefetch(s);
                    st_->submit();
                }
                for (;;) {  // the misses (same re-request logic as wait_bind)
                    bool all = true;
                    for (uint32_t k = 0; k < K; ++k) {
                        if (st_->cache().ready(sl[k])) continue;
                        all = false;
                        if (st_->cache().state(sl[k]) == SlotState::Free) {
                            SlotCache::Lookup r;
                            r = request_wait(s, uint16_t(mail[k]));
                            st_->cache().pin(r.slot);
                            pinned_.push_back(r.slot);
                            sl[k] = r.slot;
                        }
                    }
                    if (all) break;
                    st_->reap(true);
                }
                st_->reap(false);
                const double w = secs(c, Clock::now());
                stats_.t_io += w;
                const int cls = n_dem ? 2 : n_pin ? 1 : 0;
                stats_.wait_cls[cls] += w;
                ++stats_.layers_cls[cls];
                stats_.n_dem += n_dem;
                stats_.n_pin += n_pin;
                if (n_dem < 8) { stats_.wait_by_dem[n_dem] += w; ++stats_.layers_by_dem[n_dem]; }
                // flag B: full table + streamed-only table + routing weights
                uint64_t tsl[TAB] = {}, tb[TAB] = {};
                float twt[TAB] = {};
                for (uint32_t k = 0; k < K; ++k) {
                    tsl[k] = slot_addr(sl[k]);
                    tb[k] = ta[k] ? 0 : tsl[k];
                    twt[k] = mw[k];
                    slots_tab[s * TAB + k] = tsl[k];  // host copy (nancheck / debugging)
                }
                std::memcpy(ctlp_ + stage(s) + 128, tsl, sizeof tsl);
                std::memcpy(ctlp_ + stage(s) + 256, tb, sizeof tb);
                std::memcpy(ctlp_ + stage(s) + 384, twt, sizeof twt);
                __builtin_ia32_sfence();
                *ctlw(stage(s) + 452) = 1;
            }
            if (o_.prefetch_late) {
                for (uint32_t dd = 1; dd <= o_.predict && s + dd < NL_; ++dd) wait_word(kCtlPmail + (uint64_t(dd) * L + s + dd) * 256 + 128, s + dd + 1);
                prefetch(s);
                st_->submit();
            }
            if (s + 1 == PLE_LAYER && use_ple_ && !ng_now) ng_finish_ctl();
            a = Clock::now();
        }
        {  // tail: POSTMOE(47) [+ head]
            VkResult r;
            while ((r = vkGetFenceStatus(ctx_.dev, fence_)) == VK_NOT_READY) { service(); __builtin_ia32_pause(); }
            vk::check(r, "vkGetFenceStatus");
            vkResetFences(ctx_.dev, 1, &fence_);
        }
        stats_.t_gpu += secs(a, Clock::now());
        for (SlotId p : pinned_) st_->cache().unpin(p);
        pinned_.clear();
        if (*ctlw(kCtlErr)) throw std::runtime_error("one-cb: GPU gate timed out");
        {
            auto ts = ctx_.read_ts(tsq1_, tsq1_n_);
            double gates = 0;
            for (uint32_t q = 1, gi = 0; q + 1 < tsq1_n_; q += 2, ++gi) {
                const double d = double(ts[q + 1] - ts[q]);
                gates += d;
                if (gi < gate_kind_.size()) stats_.gate_t[gate_kind_[gi]] += d * ctx_.ts_period_ns * 1e-9;
            }
            stats_.t_gpu_pure += (double(ts[tsq1_n_ - 1] - ts[0]) - gates) * ctx_.ts_period_ns * 1e-9;
        }
        pmail_host_ = static_cast<uint8_t*>(pPmail_);
        ++stats_.tokens;
        if (st_->stats().io_errors > 64) throw std::runtime_error("expert I/O errors");
        return want_out && full() ? static_cast<uint32_t*>(arg_.map)[0] : UINT32_MAX;
    }

    // ---- speculative verify (S16) ----
    // Feeds v[0..T-1] at positions pos()..pos()+T-1 (v[0] = the last sampled token, v[1..] = drafts) in ONE pass over the
    // layers: every layer's routed experts of all T tokens are requested together (their reads overlap). out[t] = greedy
    // argmax after v[t]. Accepts m = the longest prefix with v[t+1] == out[t]; the sequence advances by v[0..m] (the
    // caller's new tokens are out[0..m]) and the recurrent state is the one after v[m]. KV entries past p+m are stale
    // (overwritten later). Returns m. Logits rows 0..T-1 hold each token's logits.
    // S18: decide(T, out) returns the number of accepted drafts m (the caller samples rows 0..T-1 of logits_row());
    // empty = greedy (longest prefix with v[t+1] == out[t])
    using Decide = std::function<uint32_t(uint32_t, const uint32_t*)>;
    uint32_t accept_m(const int64_t* v, uint32_t T, const uint32_t* out, const Decide& decide) const {
        if (decide) return std::min(decide(T, out), T - 1);
        uint32_t m = 0;
        while (m + 1 < T && int64_t(out[m]) == v[m + 1]) ++m;
        return m;
    }
    const float* logits_row(uint32_t t) const { return static_cast<const float*>(lg_.map) + uint64_t(t) * VOCAB; }
    uint32_t verify(const int64_t* v, uint32_t T, uint32_t* out, const Decide& decide = Decide()) {
        ek_ensure(uint64_t(pos()) + T + 2);
        st_->tier_admit_enable(pos() >= o_.tier_min_ctx);  // S31
        if (T < 2 || T > 1 + o_.spec_k || vsteps_.empty()) throw std::runtime_error("verify: bad batch size");
        if (has_vonecb(T)) return verify_onecb(v, T, out, decide);  // S18 stage C
        const uint32_t p = pos();
        if (p + T > o_.max_ctx) throw std::runtime_error("context full");
        auto t_h0 = Clock::now();
        std::vector<int64_t> h2 = hist_;
        for (uint32_t t = 0; t < T; ++t) h2.push_back(v[t]);
        if (ng_pending_) ng_finish();
        for (uint32_t t = 0; t < T; ++t) {
            uint8_t* blk = act_host_ + uint64_t(t) * act_stride_;
            if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(v[t]) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
            auto* H = reinterpret_cast<float*>(blk + act_off_.H);
            for (uint32_t i = 0; i < HID; ++i) { uint32_t x = uint32_t(row_[i]) << 16; float f; std::memcpy(&f, &x, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + i] = f; }
            *reinterpret_cast<uint32_t*>(blk + act_off_.Par) = p + t;
            if (use_ple_) {
                int64_t nid[16];
                ng_.ids(h2, hist_.size() + t, nid);
                for (int h = 0; h < 16; ++h) {
                    const uint32_t q = t * 16 + uint32_t(h);
                    ngoff_[q] = uint64_t(nid[h]) * 320;
                    if (ngr_ok_) {
                        io_uring_sqe* sqe = io_uring_get_sqe(&ngr_);
                        io_uring_prep_read(sqe, ng_.fd, ngbuf_ + q * 160, 320, ngoff_[q]);
                        io_uring_sqe_set_data64(sqe, q);
                    } else if (::pread(ng_.fd, ngbuf_ + q * 160, 320, off_t(ngoff_[q])) != 320) throw std::runtime_error("ngram read");
                }
            }
        }
        if (use_ple_) {
            if (ngr_ok_) {
                if (io_uring_submit(&ngr_) != int(16 * T)) throw std::runtime_error("ngram submit");
                ng_pending_ = 16 * T;
                ng_tokens_ = T;
            } else ng_convert(T);
        }
        if (kv_hist_.size() < p + T) kv_hist_.resize(p + T);
        for (uint32_t t = 0; t < T; ++t) kv_hist_[p + t] = v[t];
        if (o_.record_trace) trace_.emplace_back();
        stats_.t_host += secs(t_h0, Clock::now());
        const auto& vs = vsteps_[T - 2];
        for (uint32_t t = 0; t < T; ++t) st_->cache().tick();
        for (uint32_t s = 0; s <= NL_; ++s) {
            if (s == PLE_LAYER && ng_pending_) ng_finish();
            auto a = Clock::now();
            ctx_.submit(vs[s], fence_);
            if (o_.spin_wait) ctx_.wait_spin(fence_); else ctx_.wait(fence_);
            stats_.t_gpu += secs(a, Clock::now());
            for (SlotId q : pinned_) st_->cache().unpin(q);
            pinned_.clear();
            if (nancheck_)
                for (uint32_t t = 0; t < T; ++t) {
                    const float* H = reinterpret_cast<const float*>(act_host_ + uint64_t(t) * act_stride_ + act_off_.H);
                    bool bad = false;
                    for (uint32_t i = 0; i < HCH && !bad; ++i) bad = !std::isfinite(H[i]);
                    if (bad) {
                        std::fprintf(stderr, "NANCHECK: token %u/%u non-finite H after verify step %u (pos %u):", t, T, s, p + t);
                        const uint8_t* blk = act_host_ + uint64_t(t) * act_stride_;
                        const std::pair<const char*, std::pair<uint64_t, uint32_t>> bufs[] = {
                            {"Hm", {act_off_.Hm, TAB * 640}}, {"ExXt", {act_off_.ExXt, TAB * 2 * HID}}, {"ExPre", {act_off_.ExPre, TAB * 2 * 640}},
                            {"ExHd", {act_off_.ExHd, TAB * 640}}, {"ExYd", {act_off_.ExYd, TAB * HID}}, {"Ym2", {act_off_.Ym2, HID}}, {"Ym", {act_off_.Ym, HID}},
                            {"X", {act_off_.X, HID}}, {"P", {act_off_.P, 16480}}, {"C", {act_off_.C, 10240}}, {"Core", {act_off_.Core, 6144}}, {"Y", {act_off_.Y, HID}}};
                        for (auto& bb : bufs) {
                            const float* f = reinterpret_cast<const float*>(blk + bb.second.first);
                            uint32_t nb = 0, first = 0;
                            for (uint32_t i = 0; i < bb.second.second; ++i) if (!std::isfinite(f[i])) { if (!nb) first = i; ++nb; }
                            std::fprintf(stderr, " %s:%u(@%u)", bb.first, nb, first);
                        }
                        const uint64_t row = uint64_t(t) * (L + 1) + (s - 1);
                        std::fprintf(stderr, "\n  layer %u slots:", s - 1);
                        for (uint32_t k = 0; k <= K; ++k) std::fprintf(stderr, " %llx", (unsigned long long)static_cast<uint64_t*>(pSl_)[row * TAB + k]);
                        std::fprintf(stderr, "\n  token 0 slots:");
                        for (uint32_t k = 0; k <= K; ++k) std::fprintf(stderr, " %llx", (unsigned long long)static_cast<uint64_t*>(pSl_)[(s - 1) * TAB + k]);
                        std::fprintf(stderr, "\n");
                    }
                }
            if (s == NL_) break;
            bind_tokens(s, T);
        }
        for (uint32_t t = 0; t < T; ++t) out[t] = static_cast<uint32_t*>(arg_.map)[t * 4];
        const uint32_t m = accept_m(v, T, out, decide);
        if (m + 1 < T) ctx_.run(ck_restore_[m]);  // the state after v[m]; m == T-1 left it in place
        for (uint32_t t = 0; t <= m; ++t) hist_.push_back(v[t]);
        stats_.tokens += m + 1;
        ++stats_.spec_cycles;
        stats_.spec_drafted += T - 1;
        stats_.spec_accepted += m;
        if (st_->stats().io_errors > 64) throw std::runtime_error("expert I/O errors");
        return m;
    }

    // ---- S18 stage C: verify as ONE command buffer (record_vonecb) ----
    // Same contract as verify(). Host side mirrors feed_onecb() for T tokens: per layer it waits for the T router mailboxes,
    // binds the union of their experts (resident first, presence decided once per entry), publishes the T resident tables
    // (flag A), prefetches every token's depth-1 predictions, waits for the misses and publishes the full tables (flag B).
    void ng_finish_ctl_v(uint32_t T) {
        if (ng_pending_) {
            auto t = Clock::now();
            while (ng_pending_) {
                io_uring_cqe* cqe = nullptr;
                int r = io_uring_wait_cqe(&ngr_, &cqe);
                if (r == -EINTR) continue;
                if (r < 0) throw std::runtime_error("ngram wait");
                const int res = cqe->res;
                const uint64_t h = io_uring_cqe_get_data64(cqe);
                io_uring_cqe_seen(&ngr_, cqe);
                --ng_pending_;
                if (res != 320 && ::pread(ng_.fd, ngbuf_ + h * 160, 320, off_t(ngoff_[h])) != 320) throw std::runtime_error("ngram read " + std::to_string(res));
            }
            stats_.t_ngram += secs(t, Clock::now());
        }
        float tmp[16 * 160];
        for (uint32_t t = 0; t < T; ++t) {
            const uint16_t* b = ngbuf_ + t * 16 * 160;
            for (int i = 0; i < 16 * 160; ++i) { uint32_t x = uint32_t(b[i]) << 16; std::memcpy(&tmp[i], &x, 4); }
            std::memcpy(ctlp_ + kVE + uint64_t(t) * HID * 4, tmp, sizeof tmp);
        }
        __builtin_ia32_sfence();
        *ctlw(kVEFlag) = 1;
    }
    void prefetch_from(uint32_t layer, const uint32_t* pm) {
        const float* pw = reinterpret_cast<const float*>(pm + 16);
        for (uint32_t k = 0; k < std::min(K, o_.prefetch_max); ++k) {
            if (pw[k] < o_.prefetch_min_w || pm[k] >= E || st_->cache().find(layer, uint16_t(pm[k])) != kNoSlot) continue;
            const auto r = st_->request(layer, uint16_t(pm[k]), o_.cold_prefetch, true);
            if (r.slot == kNoSlot) continue;
            ++stats_.pref_issued;
        }
    }
    bool has_vonecb(uint32_t T) const { return T < 5 && vcb_[T] != VK_NULL_HANDLE; }
    // S20 batched prefill: the largest chunk the prefill cbs take (1 = token-serial)
    uint32_t prefill_max() const { uint32_t m = 1; for (uint32_t T = 2; T < 5; ++T) if (vcbp_[T]) m = T; return m; }
    // Feeds v[0..T-1] (prompt tokens, all kept) in one pass: dense weights read once per chunk, the chunk's experts bound
    // together (their reads overlap). No logits; activation block t holds token t's residual after it (for mtp_fill).
    void prefill_chunk(const int64_t* v, uint32_t T) {
        st_->tier_admit_enable(false);  // S31
        if (T == 1 || !vcbp_[T]) { for (uint32_t t = 0; t < T; ++t) feed(v[t], false); return; }
        uint32_t out[kMaxTB];
        ek_ensure(uint64_t(pos()) + T + 2);
        verify_onecb(v, T, out, [T](uint32_t, const uint32_t*) { return T - 1; }, true);
    }
    uint32_t lm_max() const { return o_.prefill_chunk; }
    // S21: the MoE of layer l for the whole chunk: wait for its experts (requested in sweep l, pinned), write the pair tables
    // and the per-slice unique-expert entries (<= 16 pairs each), run every slice in one command buffer
    // S22: LFU stamp of chunk token i's expert uses (the chunk's N ticks already happened: token i was fed at now - (N-1-i))
    void pf_use_time(uint32_t i, uint32_t N) {
        if (o_.pf_lfu == 1) st_->cache().set_use_time(st_->cache().now() - double(N - 1 - i));
    }
    void wide_moe(uint32_t l, uint32_t N, std::vector<SlotId>& sl, const std::vector<float>& wt, std::vector<SlotId>& pin_prev) {
        auto c = Clock::now();
        pf_phase_ = 0;
        for (;;) {
            bool all = true;
            for (uint64_t q = 0; q < uint64_t(N) * K; ++q) {
                if (st_->cache().ready(sl[q])) continue;
                all = false;
                if (st_->cache().state(sl[q]) == SlotState::Free) {  // failed read: request again
                    SlotCache::Lookup r;
                    r = request_wait(l, uint16_t(lm_mail_[uint64_t(l) * N * K + q]));
                    st_->cache().pin(r.slot);
                    pin_prev.push_back(r.slot);
                    sl[q] = r.slot;
                    st_->submit();
                }
            }
            if (all) break;
            st_->reap(true);
        }
        stats_.t_io += secs(c, Clock::now());
        pf_lookahead(l + 1);  // S25: the next layer's reads run under this MoE
        auto th = Clock::now();
        auto* tsl = reinterpret_cast<uint64_t*>(wctlp_ + kWSl_);
        auto* twt = reinterpret_cast<float*>(wctlp_ + kWWt_);
        std::vector<uint64_t> bsl(uint64_t(N) * K);
        for (uint64_t q = 0; q < uint64_t(N) * K; ++q) bsl[q] = slot_addr(sl[q]);
        if (xm_on()) { stats_.t_host += secs(th, Clock::now()); stats_.pf_host[1] += secs(th, Clock::now()); wide_moe_xm(l, N, bsl, wt); return; }
        std::memcpy(tsl, bsl.data(), bsl.size() * 8);
        std::memcpy(twt, wt.data(), uint64_t(N) * K * 4);
        if (const char* dp = std::getenv("PS5LM_WDUMP")) {  // S22: pairs per expert of this layer in the chunk (one line per layer)
            std::vector<uint32_t> cnt(E, 0);
            for (uint64_t q = 0; q < uint64_t(N) * K; ++q) ++cnt[lm_mail_[uint64_t(l) * N * K + q]];
            if (FILE* f = std::fopen(dp, "a")) {
                std::fprintf(f, "%u %u", l, N);
                for (uint32_t c : cnt) std::fprintf(f, " %u", c);
                std::fprintf(f, "\n");
                std::fclose(f);
            }
        }
        const uint32_t S = o_.wide_slice, ns = (N + S - 1) / S;
        wents_.clear();
        std::vector<std::pair<uint32_t, uint32_t>> se(ns);
        std::vector<int32_t> cur(E);
        for (uint32_t j = 0; j < ns; ++j) {
            const uint32_t a = j * S, T = std::min(S, N - a);
            std::fill(cur.begin(), cur.end(), -1);
            se[j].first = uint32_t(wents_.size());
            for (uint32_t i = a; i < a + T; ++i)
                for (uint32_t k = 0; k < K; ++k) {
                    const uint64_t q = uint64_t(i) * K + k;
                    const uint32_t id = lm_mail_[uint64_t(l) * N * K + q];
                    int32_t e = cur[id];
                    if (e < 0 || wents_[e].n == 16) {
                        e = int32_t(wents_.size());
                        WEnt w{};
                        w.slot = bsl[q];
                        wents_.push_back(w);
                        cur[id] = e;
                    }
                    wents_[e].p[wents_[e].n++] = (i - a) * K + k;
                }
            se[j].second = uint32_t(wents_.size()) - se[j].first;
        }
        std::memcpy(wctlp_ + kWEnt_, wents_.data(), wents_.size() * sizeof(WEnt));
        __builtin_ia32_sfence();
        stats_.wide_ents += wents_.size();
        stats_.wide_pairs += uint64_t(N) * K;
        stats_.t_host += secs(th, Clock::now());
        VkCommandBuffer cb = wide_begin(0);
        for (uint32_t j = 0; j < ns; ++j) {
            const uint32_t a = j * S, T = std::min(S, N - a);
            prof_begin(cb, lmprof_ && l == 3 && j == 0 && (!wattn_ok_ || lmprof2_) ? 0 : -1);  // S21 PS5LM_LMPROF: slot 0 = MoE of layer 3, slice 0
            W_MOE(cb, l, a, T, wctl_.addr + kWEnt_ + uint64_t(se[j].first) * sizeof(WEnt), se[j].second);
            prof_cur_ = -1;
        }
        wide_end(cb, 0);
        ctx_.submit(cb, wfence_[0]);
        wide_wait(cb, 0, wfence_[0]);
    }
    // S22 chunk-wide expert-major MoE of layer l: every pair of the chunk grouped by expert (groups of <= wide_slice * K
    // pairs = the S21 arena), entries of <= 64 / <= 32 pairs (exl3_xmvl, the trellis decoded once per entry into LDS) and
    // <= 16-pair tails (exl3_xmvw); the shared expert over the whole chunk (Z region, before the groups use it); each group's
    // routed output added into the token's Y row (exl3_g_combine, deterministic); post for the whole chunk. X / I2 come from
    // the router step (W_TAIL) or are re-derived here for layers whose PRE ran in 4-token groups (PLE layer).
    void wide_moe_xm(uint32_t l, uint32_t N, const std::vector<uint64_t>& bsl, const std::vector<float>& wt) {
        auto th = Clock::now();
        const auto& w = W_[l];
        const int k = kidx(l);
        const uint32_t S = o_.wide_slice, ns = (N + S - 1) / S, GP = S * K;
        const uint64_t NK = uint64_t(N) * K;
        const uint32_t* mail = lm_mail_.data() + uint64_t(l) * N * K;
        std::vector<uint32_t> st(E + 1, 0), ord(NK);
        for (uint64_t q = 0; q < NK; ++q) ++st[mail[q] + 1];
        for (uint32_t e = 0; e < E; ++e) st[e + 1] += st[e];
        {
            std::vector<uint32_t> c(st.begin(), st.end() - 1);
            for (uint64_t q = 0; q < NK; ++q) ord[c[mail[q]]++] = uint32_t(q);
        }
        struct Grp { uint32_t gp0, np, a64, n64, a32, n32, aw, nw, at, nt; };
        std::vector<Grp> gr;
        std::vector<LEnt> e64, e32;
        std::vector<WEnt> ew;
        std::vector<TEnt> te;
        std::vector<uint32_t> tq;  // pair index of each token-entry slot (sorting key = k)
        std::vector<uint64_t> hsl(NK);
        std::vector<uint32_t> htk(NK), hq(NK);
        if (wtix_.size() < N) wtix_.assign(N, -1);
        uint32_t pos = 0;
        Grp g{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto close = [&]() {
            if (!g.np) return;
            g.n64 = uint32_t(e64.size()) - g.a64; g.n32 = uint32_t(e32.size()) - g.a32; g.nw = uint32_t(ew.size()) - g.aw;
            g.at = uint32_t(te.size());
            for (uint32_t j = 0; j < g.np; ++j) {
                const uint32_t gp = g.gp0 + j, t = htk[gp], q = hq[gp];
                int32_t& x = wtix_[t];
                if (x < 0) { x = int32_t(te.size()); TEnt z{}; z.tok = t; te.push_back(z); tq.resize(te.size() * 10); }
                TEnt& z = te[x];
                tq[uint64_t(x) * 10 + z.n] = q;
                z.gq[z.n] = j; z.w[z.n] = wt[q]; ++z.n;
            }
            for (uint32_t x = g.at; x < te.size(); ++x) {  // each token's pairs in k order (q = t K + k)
                TEnt& z = te[x];
                uint32_t* kq = tq.data() + uint64_t(x) * 10;
                for (uint32_t a = 1; a < z.n; ++a)
                    for (uint32_t b = a; b > 0 && kq[b - 1] > kq[b]; --b) { std::swap(kq[b - 1], kq[b]); std::swap(z.gq[b - 1], z.gq[b]); std::swap(z.w[b - 1], z.w[b]); }
                wtix_[z.tok] = -1;
            }
            g.nt = uint32_t(te.size()) - g.at;
            gr.push_back(g);
            g = Grp{pos, 0, uint32_t(e64.size()), 0, uint32_t(e32.size()), 0, uint32_t(ew.size()), 0, 0, 0};
        };
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t i0 = st[e]; i0 < st[e + 1];) {
                if (g.np == GP) close();
                const uint32_t take = std::min(GP - g.np, st[e + 1] - i0);
                const uint64_t slot = bsl[ord[i0]];
                uint32_t p = g.np, left = take;
                while (left > 32) { const uint32_t n = std::min(left, 64u); e64.push_back({slot, n, p}); p += n; left -= n; }
                if (left > 16) { e32.push_back({slot, left, p}); p += left; left = 0; }
                if (left) { WEnt we{}; we.slot = slot; we.n = left; for (uint32_t j = 0; j < left; ++j) we.p[j] = p + j; ew.push_back(we); }
                for (uint32_t j = 0; j < take; ++j) { const uint32_t q = ord[i0 + j]; hsl[pos] = bsl[q]; htk[pos] = q / K; hq[pos] = q; ++pos; }
                g.np += take;
                i0 += take;
            }
        close();
        if (e64.size() > gcap_e_ || e32.size() > gcap_e_ || ew.size() > gcap_e_ || te.size() > NK) throw std::runtime_error("expert-major MoE: table overflow");
        std::memcpy(gctlp_ + kGSl_, hsl.data(), NK * 8);
        std::memcpy(gctlp_ + kGTk_, htk.data(), NK * 4);
        std::memcpy(gctlp_ + kGE64_, e64.data(), e64.size() * sizeof(LEnt));
        std::memcpy(gctlp_ + kGE32_, e32.data(), e32.size() * sizeof(LEnt));
        std::memcpy(gctlp_ + kGEw_, ew.data(), ew.size() * sizeof(WEnt));
        std::memcpy(gctlp_ + kGTe_, te.data(), te.size() * sizeof(TEnt));
        __builtin_ia32_sfence();
        stats_.wide_ents += e64.size() + e32.size() + ew.size();
        stats_.wide_pairs += NK;
        stats_.t_host += secs(th, Clock::now());
        stats_.pf_host[1] += secs(th, Clock::now());
        VkCommandBuffer cb = wide_begin(0);
        prof_begin(cb, lmprof_ && lmprof2_ && l == 3 ? 0 : -1);  // PS5LM_LMPROF=2: slot 0 = the whole MoE of layer 3
        if (!xc_ok_[l])  // X / I2 of a layer whose PRE ran in 4-token groups: re-derived from the carried H (S21 order)
            for (uint32_t j = 0; j < ns; ++j) {
                const uint32_t a = j * S, T = std::min(S, N - a);
                W_HCPRE(cb, w.hcm, aCarry_ + uint64_t(a) * HCH * 4, T, aI2c_ + uint64_t(a) * 16, aXY_ + uint64_t(a) * XYS, XYS);
            }
        {  // shared expert over the chunk (passes of TS tokens through region Z), stored into the Y rows
            const uint64_t zb = std::max<uint64_t>({uint64_t(S) * (2 * HCH + LOWR + 16 * LOWR + 4) * 4 + 1024, uint64_t(S) * (16480 + 10240 + 6144) * 4 + 1024,
                                                    uint64_t(S) * K * 2 * HID * 4});
            const uint32_t TS = std::max<uint32_t>(64, uint32_t(zb / ((1280 + 640 + 1) * 4)) & ~63u);
            const uint64_t zGu = wZ_;
            for (uint32_t a = 0; a < N; a += TS) {
                const uint32_t T = std::min(TS, N - a);
                const uint64_t zHs = zGu + uint64_t(T) * 1280 * 4, zSg = zHs + uint64_t(T) * 640 * 4;
                const uint64_t X = aXY_ + uint64_t(a) * XYS, Y = X + HID * 4;
                PcWDot sg{w.sgate, X, zSg, HID, 1, 3, 1.f, XYS, 4};
                vk::Ctx::dispatch(cb, pWd_, &sg, T);
                WG_GEMM(cb, w.shared_slot, X, XYS, zGu, 1280 * 4, HID, 1280, T);
                BAR(cb, __LINE__);
                PcWSact sa{zGu, zSg, zHs, 1280 * 4, 4, 640 * 4, 0};
                vk::Ctx::dispatch(cb, pWsa_, &sa, 640 / 128, T);
                BAR(cb, __LINE__);
                WG_GEMM(cb, w.shared_slot + qn::ExpertLayout<4>().down_off, zHs, 640 * 4, Y, XYS, 640, HID, T, 0);
                BAR(cb, __LINE__);
            }
        }
        const uint64_t gb = gctl_.addr;
        const uint32_t trg = ex_tr_gate(ex_k2_[l]), tru = ex_tr_up(ex_k2_[l]), trd = ex_tr_down();
        auto xl = [&](uint32_t nx, uint64_t ea, const vk::Pipeline& pp, bool gu) {
            if (!nx) return;
            PcXmvl m{ea, gu ? wXt_ : wHd_, gu ? wPre_ : wYd_, gb + kGPl_, gu ? HID / 16 : 640 / 16, gu ? 640 / 16 : HID / 16, gu ? trg : trd, gu ? tru : trd,
                     gu ? HID : 0, gu ? 640u : 0u, gu ? 2 * HID * 4 : 640 * 4, gu ? 1280 * 4 : HID * 4};
            vk::Ctx::dispatch(cb, pp, &m, (gu ? 640 : HID) / 128, gu ? 2 : 1, nx);
        };
        for (const Grp& G : gr) {
            const uint64_t gs = gb + kGSl_ + uint64_t(G.gp0) * 8, gt = gb + kGTk_ + uint64_t(G.gp0) * 4;
            const uint64_t a64 = gb + kGE64_ + uint64_t(G.a64) * sizeof(LEnt), a32 = gb + kGE32_ + uint64_t(G.a32) * sizeof(LEnt);
            const uint64_t aw = gb + kGEw_ + uint64_t(G.aw) * sizeof(WEnt);
            PcWHadG hd{aXY_, gs, wXt_, gt, XYS, 0};
            vk::Ctx::dispatch(cb, pWhG_[k], &hd, HID / 128, 2, G.np);
            BAR(cb, __LINE__);
            xl(G.n64, a64, pXl64_[k], true);
            xl(G.n32, a32, pXl32_[k], true);
            if (G.nw) {
                PcXmvw m{aw, wXt_, wPre_, HID / 16, 640 / 16, trg, tru, HID, 640, 2 * HID * 4, 1280 * 4};
                vk::Ctx::dispatch(cb, pXw_[k], &m, 640 / 32, 2, G.nw);
            }
            BAR(cb, __LINE__);
            PcExAct ac{wPre_, gs, wHd_};
            vk::Ctx::dispatch(cb, pExA_[k], &ac, 640 / 128, G.np);
            BAR(cb, __LINE__);
            xl(G.n64, a64, pXl64_[k], false);
            xl(G.n32, a32, pXl32_[k], false);
            if (G.nw) {
                PcXmvw m{aw, wHd_, wYd_, 640 / 16, HID / 16, trd, trd, 0, 0, 640 * 4, HID * 4};
                vk::Ctx::dispatch(cb, pXw_[k], &m, HID / 32, 1, G.nw);
            }
            BAR(cb, __LINE__);
            PcGCmb cm{wYd_, gs, gb + kGTe_ + uint64_t(G.at) * sizeof(TEnt), aXY_ + HID * 4, XYS, 0};
            vk::Ctx::dispatch(cb, pGc_[k], &cm, HID / 128, G.nt);
            BAR(cb, __LINE__);
        }
        PcWPost p{aCarry_, aXY_ + HID * 4, aI2c_, HCH * 4, XYS, 16, 0};
        vk::Ctx::dispatch(cb, pWp_, &p, HCH / 256, N);
        BAR(cb, __LINE__);
        prof_cur_ = -1;
        wide_end(cb, 0);
        ctx_.submit(cb, wfence_[0]);
        wide_wait(cb, 0, wfence_[0]);
    }
    // S21: the pre-MoE part of GDN layer s for the chunk, one command buffer per slice; slice j+1 is submitted before slice
    // j's router mails are turned into expert requests (pinned until the next sweep's MoE has used them)
    void wide_pre(uint32_t s, uint32_t N, std::vector<SlotId>& sl, std::vector<float>& wt, std::vector<SlotId>& pin_cur) {
        const uint32_t S = o_.wide_slice, ns = (N + S - 1) / S;
        pf_phase_ = W_[s].attn ? 2 : 1;
        if (xm_on()) xc_ok_[s] = 1;
        if (lm_mail_.size() < uint64_t(NL_) * N * K) lm_mail_.resize(uint64_t(NL_) * N * K);
        auto mails = [&](uint32_t j) {
            auto th = Clock::now();
            const uint32_t a = j * S, T = std::min(S, N - a);
            for (uint32_t t = 0; t < T; ++t) {
                uint32_t mail[32];
                std::memcpy(mail, wctlp_ + kWMail_ + uint64_t(a + t) * 256, sizeof mail);
                if (route_on(s)) route_select(s, mail, o_.route_pf ? o_.route_bias : 0.f);  // S24: earlier tokens' requests already count as resident
                const float* mw = reinterpret_cast<const float*>(mail + 16);
                const uint32_t i = a + t;
                pf_use_time(i, N);
                for (uint32_t k = 0; k < K; ++k) {
                    if (mail[k] >= E) throw std::runtime_error("wide prefill: invalid expert id " + std::to_string(mail[k]) + " layer " + std::to_string(s));
                    lm_mail_[(uint64_t(s) * N + i) * K + k] = mail[k];
                    SlotCache::Lookup r;
                    r = request_wait(s, uint16_t(mail[k]), false, o_.pf_lfu == 2);
                    st_->cache().pin(r.slot);
                    pin_cur.push_back(r.slot);
                    sl[uint64_t(i) * K + k] = r.slot;
                    wt[uint64_t(i) * K + k] = mw[k];
                    ++stats_.demand;
                }
            }
            st_->cache().set_use_time(std::numeric_limits<double>::quiet_NaN());
            st_->submit();
            stats_.t_host += secs(th, Clock::now());
            stats_.pf_host[2] += secs(th, Clock::now());
        };
        VkCommandBuffer prev = VK_NULL_HANDLE;
        for (uint32_t j = 0; j <= ns; ++j) {
            VkCommandBuffer cb = VK_NULL_HANDLE;
            if (j < ns) {
                const uint32_t a = j * S, T = std::min(S, N - a);
                cb = wide_begin(1 + (j & 1));
                if (s == 0 && pWe_.pipe) {  // embeddings of the slice: host stages the bf16 rows (half j & 1), the GPU expands them
                    auto th = Clock::now();
                    auto* st = static_cast<uint16_t*>(pWEmb_) + uint64_t(j & 1) * S * HID;
                    for (uint32_t t = 0; t < T; ++t)
                        if (::pread(emb_fd_, st + uint64_t(t) * HID, HID * 2, off_t(uint64_t(wv_[a + t]) * HID * 2)) != ssize_t(HID * 2))
                            throw std::runtime_error("embed read");
                    stats_.t_host += secs(th, Clock::now());
                    stats_.pf_host[3] += secs(th, Clock::now());
                    PcWEmbed pe{wEmb_ + uint64_t(j & 1) * S * HID * 2, aCarry_ + uint64_t(a) * HCH * 4, HCH * 4, 0};
                    vk::Ctx::dispatch(cb, pWe_, &pe, HCH / 256, T);
                    vk::Ctx::barrier(cb);
                }
                prof_begin(cb, lmprof_ && (s == 4 || (s == 7 && !lmprof2_)) && j == 0 ? (s == 4 ? 1 : 0) : -1);  // slot 1 = GDN layer 4, slot 0 = attention layer 7 (slice 0)
                if (W_[s].attn) W_PRE_ATTN(cb, s, a, T); else W_PRE_GDN(cb, s, a, T);
                prof_cur_ = -1;
                wide_end(cb, 1 + (j & 1));
                ctx_.submit(cb, wfence_[j & 1]);
            }
            if (j > 0) {
                wide_wait(prev, 1 + ((j - 1) & 1), wfence_[(j - 1) & 1]);
                mails(j - 1);
            }
            prev = cb;
        }
    }
    // S20 layer-major prefill of v[0..N-1] (N % 4 == 0, N <= lm_max()) at positions pos()..: sweep s = 0..NL_ runs lmcb_[s]
    // for every group in order (the recurrent state and the KV of layer s advance group by group, exactly as token-serial),
    // binding the group's layer s-1 experts (requested during sweep s-1, all pinned until sweep s ends) and requesting its
    // layer s experts from the mailboxes PRE(s) wrote. With mtp_next >= -1 the MTP KV of every position is filled afterwards
    // (v[j+1], and mtp_next for the last position when >= 0). progress(sweep) is called after every sweep.
    void prefill_lm(const int64_t* v, uint32_t N, bool mtp, int64_t mtp_next, const std::function<void(uint32_t)>& progress = {}) {
        if (!o_.prefill_chunk || N % 4 || N > o_.prefill_chunk || N == 0) throw std::runtime_error("prefill_lm: bad chunk");
        st_->tier_admit_enable(false);  // S31: no pool -> tier copies during a prefill
        const uint32_t T = 4, G = N / 4, p0 = pos();
        if (p0 + N > o_.max_ctx) throw std::runtime_error("context full");
        ek_ensure(uint64_t(p0) + N + 2);
        auto t_h0 = Clock::now();
        std::vector<int64_t> h2 = hist_;
        h2.insert(h2.end(), v, v + N);
        if (ng_pending_) ng_finish();
        std::vector<uint16_t> ng;
        if (use_ple_) {  // n-gram rows of every position (PLE input of layer PLE_LAYER): one deep io_uring batch (QD 256)
            ng.resize(uint64_t(N) * 16 * 160);
            std::vector<uint64_t> offs(uint64_t(N) * 16);
            for (uint32_t i = 0; i < N; ++i) {
                int64_t nid[16];
                ng_.ids(h2, hist_.size() + i, nid);
                for (int h = 0; h < 16; ++h) offs[uint64_t(i) * 16 + h] = uint64_t(nid[h]) * 320;
            }
            if (!lmr_init_) { lmr_ok_ = io_uring_queue_init(256, &lmr_, 0) == 0; lmr_init_ = true; }
            size_t q = 0, done = 0, inflight = 0;
            const size_t nq = offs.size();
            while (done < nq) {
                while (lmr_ok_ && q < nq && inflight < 256) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&lmr_);
                    if (!sqe) break;
                    io_uring_prep_read(sqe, ng_.fd, ng.data() + q * 160, 320, offs[q]);
                    io_uring_sqe_set_data64(sqe, q);
                    ++q; ++inflight;
                }
                if (!lmr_ok_) {  // no ring: synchronous
                    for (; q < nq; ++q) if (::pread(ng_.fd, ng.data() + q * 160, 320, off_t(offs[q])) != 320) throw std::runtime_error("ngram read");
                    break;
                }
                io_uring_submit_and_wait(&lmr_, 1);
                io_uring_cqe* cqe = nullptr;
                while (io_uring_peek_cqe(&lmr_, &cqe) == 0 && cqe) {
                    const uint64_t k = io_uring_cqe_get_data64(cqe);
                    const int res = cqe->res;
                    io_uring_cqe_seen(&lmr_, cqe);
                    --inflight; ++done;
                    if (res != 320 && ::pread(ng_.fd, ng.data() + k * 160, 320, off_t(offs[k])) != 320) throw std::runtime_error("ngram read");
                }
            }
        }
        if (kv_hist_.size() < p0 + N) kv_hist_.resize(p0 + N);
        for (uint32_t i = 0; i < N; ++i) { kv_hist_[p0 + i] = v[i]; st_->cache().tick(); }
        stats_.t_host += secs(t_h0, Clock::now());
        stats_.pf_host[0] += secs(t_h0, Clock::now());
        std::vector<SlotId> sl(uint64_t(N) * K);
        std::vector<float> wt(uint64_t(N) * K);
        std::vector<SlotId> pin_prev, pin_cur;
        pin_prev.reserve(uint64_t(N) * K);
        pin_cur.reserve(uint64_t(N) * K);
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        auto* tbB = static_cast<uint64_t*>(pSlB_);
        auto* wtab = static_cast<float*>(pWt_);
        auto* gword = static_cast<volatile uint32_t*>(pLmG_);
        const bool wide = o_.wide && wide_ok_;
        wpos0_ = p0;
        wv_ = v;
        if (xm_on()) std::fill(xc_ok_.begin(), xc_ok_.end(), 0);
        carry_grow(N);
        pf_release();
        pf_chunk_ = N;
        if (wide) pf_lookahead(0);
        if (wide && hCarry_ && !pWe_.pipe) {  // S21: every token's embedding straight into the carry (sweep 0 is a wide GDN sweep)
            auto th = Clock::now();
            auto* Hc = static_cast<float*>(hCarry_);
            for (uint32_t i = 0; i < N; ++i) {
                if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(v[i]) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
                float* H = Hc + uint64_t(i) * HCH;
                for (uint32_t x = 0; x < HID; ++x) { uint32_t b = uint32_t(row_[x]) << 16; float f; std::memcpy(&f, &b, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + x] = f; }
            }
            stats_.t_host += secs(th, Clock::now());
        }
        for (uint32_t s = 0; s <= NL_; ++s) {
            cur_layer_ = s;
            if (wide && s > 0) wide_moe(s - 1, N, sl, wt, pin_prev);
            if (wide && (s == NL_ || wide_pre_layer(s))) {
                if (s < NL_) { wide_pre(s, N, sl, wt, pin_cur); pf_note(s, N); }
                for (SlotId q : pin_prev) st_->cache().unpin(q);
                pin_prev.swap(pin_cur);
                pin_cur.clear();
                if (progress) progress(s);
                continue;
            }
            for (uint32_t g = 0; g < G; ++g) {
                auto th = Clock::now();
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t i = g * T + t;
                    uint8_t* blk = act_host_ + uint64_t(t) * act_stride_;
                    *reinterpret_cast<uint32_t*>(blk + act_off_.Par) = p0 + i;
                    if (s == 0) {
                        if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(v[i]) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
                        auto* H = reinterpret_cast<float*>(blk + act_off_.H);
                        for (uint32_t x = 0; x < HID; ++x) { uint32_t b = uint32_t(row_[x]) << 16; float f; std::memcpy(&f, &b, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + x] = f; }
                    }
                    if (s == PLE_LAYER && use_ple_) {
                        auto* e = reinterpret_cast<float*>(blk + act_off_.E);
                        const uint16_t* b = ng.data() + uint64_t(i) * 16 * 160;
                        for (int x = 0; x < 16 * 160; ++x) { uint32_t u = uint32_t(b[x]) << 16; std::memcpy(&e[x], &u, 4); }
                    }
                }
                stats_.t_host += secs(th, Clock::now());
                if (s > 0 && !wide) {  // bind the group's layer s-1 experts (requested in sweep s-1)
                    const uint32_t l = s - 1;
                    auto c = Clock::now();
                    for (;;) {
                        bool all = true;
                        for (uint32_t q = g * T * K; q < (g + 1) * T * K; ++q) {
                            if (st_->cache().ready(sl[q])) continue;
                            all = false;
                            if (st_->cache().state(sl[q]) == SlotState::Free) {  // failed read: request again
                                const uint32_t i = q / K;
                                const uint32_t* mail = lm_mail_.data() + (uint64_t(l) * N + i) * K;
                                SlotCache::Lookup r;
                                r = request_wait(l, uint16_t(mail[q % K]));
                                st_->cache().pin(r.slot);
                                pin_prev.push_back(r.slot);
                                sl[q] = r.slot;
                                st_->submit();
                            }
                        }
                        if (all) break;
                        st_->reap(true);
                    }
                    stats_.t_io += secs(c, Clock::now());
                    for (uint32_t t = 0; t < T; ++t) {
                        const uint64_t row = (uint64_t(t) * (L + 1) + l) * TAB;
                        for (uint32_t k = 0; k < K; ++k) {
                            const uint64_t q = uint64_t(g * T + t) * K + k;
                            slots_tab[row + k] = tbB[row + k] = slot_addr(sl[q]);
                            wtab[row + k] = wt[q];
                        }
                    }
                    if (lm_dedupe_) {  // unique-expert entries of the group (one trellis decode per expert per <= 4 pairs)
                        VEnt eb[4 * K] = {};
                        uint16_t idb[4 * K];
                        uint32_t nbe = 0;
                        for (uint32_t t = 0; t < T; ++t)
                            for (uint32_t k = 0; k < K; ++k) {
                                const uint64_t q = uint64_t(g * T + t) * K + k;
                                const uint16_t id = uint16_t(lm_mail_[(uint64_t(l) * N + g * T + t) * K + k]);
                                uint32_t j = 0;
                                while (j < nbe && (idb[j] != id || eb[j].n >= 4)) ++j;
                                if (j == nbe) { idb[nbe] = id; eb[nbe].slot = slot_addr(sl[q]); eb[nbe].n = 0; ++nbe; }
                                eb[j].tk[eb[j].n++] = (t << 8) | k;
                            }
                        std::memcpy(ctlp_ + vent(l, true), eb, sizeof(VEnt) * T * K);
                    }
                }
                *gword = g;
                __builtin_ia32_sfence();
                auto a = Clock::now();
                ctx_.submit(wide ? lmcbP_[s] : lmcb_[s], fence_);
                {
                    VkResult r;
                    while ((r = vkGetFenceStatus(ctx_.dev, fence_)) == VK_NOT_READY) { service(); __builtin_ia32_pause(); }
                    vk::check(r, "vkGetFenceStatus");
                    vkResetFences(ctx_.dev, 1, &fence_);
                }
                stats_.t_gpu += secs(a, Clock::now());
                {
                    VkQueryPool qp = lmtsq_;
                    uint64_t ts[2] = {};
                    vkGetQueryPoolResults(ctx_.dev, qp, 2 * s, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT);
                    stats_.t_gpu_pure += double(ts[1] - ts[0]) * ctx_.ts_period_ns * 1e-9;
                }
                if (s < NL_) {  // the group's layer s routing -> requests (pinned until sweep s+1 has used them)
                    if (lm_mail_.size() < uint64_t(NL_) * N * K) lm_mail_.resize(uint64_t(NL_) * N * K);
                    for (uint32_t t = 0; t < T; ++t) {
                        uint32_t mail[32];
                        std::memcpy(mail, ctlp_ + vmail(t, s), sizeof mail);
                        if (route_on(s)) route_select(s, mail, o_.route_pf ? o_.route_bias : 0.f);  // S24 (the mailbox holds kr(s) candidates)
                        const float* mw = reinterpret_cast<const float*>(mail + 16);
                        const uint32_t i = g * T + t;
                        pf_use_time(i, N);
                        for (uint32_t k = 0; k < K; ++k) {
                            if (mail[k] >= E) throw std::runtime_error("prefill_lm: invalid expert id " + std::to_string(mail[k]) + " layer " + std::to_string(s));
                            lm_mail_[(uint64_t(s) * N + i) * K + k] = mail[k];
                            SlotCache::Lookup r;
                            r = request_wait(s, uint16_t(mail[k]), false, o_.pf_lfu == 2);
                            st_->cache().pin(r.slot);
                            pin_cur.push_back(r.slot);
                            sl[uint64_t(i) * K + k] = r.slot;
                            wt[uint64_t(i) * K + k] = mw[k];
                            ++stats_.demand;
                        }
                    }
                    st_->cache().set_use_time(std::numeric_limits<double>::quiet_NaN());
                    st_->submit();
                }
            }
            if (wide) pf_note(s, N);
            for (SlotId q : pin_prev) st_->cache().unpin(q);
            pin_prev.swap(pin_cur);
            pin_cur.clear();
            if (progress) progress(s);
        }
        for (SlotId q : pin_prev) st_->cache().unpin(q);
        pf_release();
        if (std::getenv("PS5LM_LMSTAT")) {  // S20: expert reuse inside a 4-token group / 64 / 256 tokens / the chunk
            for (uint32_t W : {4u, 16u, 64u, 256u, 1024u}) {
                if (W > N) break;
                double uses = 0, uniq = 0;
                for (uint32_t l = 0; l < NL_; ++l)
                    for (uint32_t b = 0; b + W <= N; b += W) {
                        std::vector<uint8_t> seen(E, 0);
                        for (uint32_t i = b; i < b + W; ++i)
                            for (uint32_t k = 0; k < K; ++k) { uint32_t x = lm_mail_[(uint64_t(l) * N + i) * K + k]; uniq += !seen[x]; seen[x] = 1; uses += 1; }
                    }
                std::fprintf(stderr, "lmstat: window %4u tokens: %.2f uses per unique expert (%.1f unique / layer / window)\n", W, uses / uniq,
                             uniq / double(NL_) / double(N / W));
            }
        }
        if (lmprof_) { std::fprintf(stderr, "[lmprof: last 4-token group of a %u-token chunk; step3 = sweep 4 (MoE l3 + GDN l4), step4 = sweep 7 (MoE l6 + attention l7)]", N); print_profile(); }
        for (uint32_t i = 0; i < N; ++i) hist_.push_back(v[i]);
        stats_.tokens += N;
        stats_.prefill_chunks += G;
        if (*ctlw(kCtlErr)) throw std::runtime_error("prefill_lm: control error set");
        if (mtp && mtp_ok_ && wmtp_ok_ && o_.wide_mtp) wide_mtp_fill(v, N, p0, mtp_next);  // S22: as GEMMs over slices
        else if (mtp && mtp_ok_)  // the MTP head's KV entry of every position: each group's final residual back into the blocks
            for (uint32_t g = 0; g < G; ++g) {
                *gword = g;
                __builtin_ia32_sfence();
                ctx_.run(cb_carry_in_);
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t i = g * T + t;
                    const int64_t nx = i + 1 < N ? v[i + 1] : mtp_next;
                    if (nx >= 0) mtp_fill(nx, p0 + i, t);
                }
            }
        if (st_->stats().io_errors > 64) throw std::runtime_error("expert I/O errors");
    }
    uint32_t verify_onecb(const int64_t* v, uint32_t T, uint32_t* out, const Decide& decide, bool pf = false) {
        const uint32_t p = pos();
        if (p + T > o_.max_ctx) throw std::runtime_error("context full");
        auto t_h0 = Clock::now();
        if (o_.tok_route_mode & 1) for (uint32_t t = 0; t < T; ++t) tok_prefetch(0, v[t]);  // S19
        std::vector<int64_t> h2 = hist_;
        for (uint32_t t = 0; t < T; ++t) h2.push_back(v[t]);
        if (ng_pending_) ng_finish();
        bool ng_now = false;
        for (uint32_t t = 0; t < T; ++t) {
            uint8_t* blk = act_host_ + uint64_t(t) * act_stride_;
            if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(v[t]) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
            auto* H = reinterpret_cast<float*>(blk + act_off_.H);
            for (uint32_t i = 0; i < HID; ++i) { uint32_t x = uint32_t(row_[i]) << 16; float f; std::memcpy(&f, &x, 4); for (uint32_t c = 0; c < HC; ++c) H[c * HID + i] = f; }
            *reinterpret_cast<uint32_t*>(blk + act_off_.Par) = p + t;
            if (use_ple_) {
                int64_t nid[16];
                ng_.ids(h2, hist_.size() + t, nid);
                for (int h = 0; h < 16; ++h) {
                    const uint32_t q = t * 16 + uint32_t(h);
                    ngoff_[q] = uint64_t(nid[h]) * 320;
                    if (ngr_ok_) {
                        io_uring_sqe* sqe = io_uring_get_sqe(&ngr_);
                        io_uring_prep_read(sqe, ng_.fd, ngbuf_ + q * 160, 320, ngoff_[q]);
                        io_uring_sqe_set_data64(sqe, q);
                    } else if (::pread(ng_.fd, ngbuf_ + q * 160, 320, off_t(ngoff_[q])) != 320) throw std::runtime_error("ngram read");
                }
            }
        }
        if (use_ple_) {
            if (ngr_ok_) {
                if (io_uring_submit(&ngr_) != int(16 * T)) throw std::runtime_error("ngram submit");
                ng_pending_ = 16 * T;
                ng_tokens_ = T;
            } else ng_now = true;
        }
        if (kv_hist_.size() < p + T) kv_hist_.resize(p + T);
        for (uint32_t t = 0; t < T; ++t) kv_hist_[p + t] = v[t];
        if (o_.record_trace) trace_.emplace_back();
        for (uint32_t s = 0; s < NL_; ++s) {
            for (uint32_t t = 0; t < T; ++t) { *ctlw(vmail(t, s) + 128) = 0; *ctlw(vpmail(t, s) + 128) = 0; }
            *ctlw(vstage(0, s) + 448) = 0;
            *ctlw(vstage(0, s) + 452) = 0;
        }
        for (uint32_t t = 0; t < T; ++t) *ctlw(vpmail(t, L) + 128) = 0;  // S18: MTP-expert prediction rows
        *ctlw(kVEFlag) = 0;
        if (*ctlw(kCtlErr)) throw std::runtime_error("one-cb: control error set");
        __builtin_ia32_sfence();
        if (ng_now) ng_finish_ctl_v(T);
        for (uint32_t t = 0; t < T; ++t) st_->cache().tick();
        stats_.t_host += secs(t_h0, Clock::now());
        auto a = Clock::now();
        ctx_.submit(pf ? vcbp_[T] : vcb_[T], fence_);
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        for (uint32_t s = 0; s < NL_; ++s) {
            cur_layer_ = s;
            for (uint32_t t = 0; t < T; ++t) wait_word(vmail(t, s) + 128, s + 1);
            stats_.t_gpu += secs(a, Clock::now());
            for (SlotId q : pinned_) st_->cache().unpin(q);
            pinned_.clear();
            uint32_t mail[kMaxTB][32];
            for (uint32_t t = 0; t < T; ++t) std::memcpy(mail[t], ctlp_ + vmail(t, s), sizeof mail[t]);
            if (route_on(s)) {  // S24: prefill chunks keep the true top-K unless route_pf; a later token counts the earlier tokens' picks as resident
                uint32_t also[kMaxTB * K];
                for (uint32_t t = 0; t < T; ++t) {
                    route_select(s, mail[t], pf && !o_.route_pf ? 0.f : o_.route_bias, also, t * K);
                    for (uint32_t k = 0; k < K; ++k) also[t * K + k] = mail[t][k];
                }
            }
            auto c = Clock::now();
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k)
                    if (mail[t][k] >= E) throw std::runtime_error("verify: invalid expert id " + std::to_string(mail[t][k]) + " token " + std::to_string(t) + " layer " + std::to_string(s));
            if (o_.record_trace) for (uint32_t k = 0; k < K; ++k) trace_.back().push_back(uint16_t(mail[0][k]));
            if (s == 0) for (uint32_t t = 0; t < T; ++t) tok_learn(0, v[t], mail[t]);
            bool present[kMaxTB][K];
            SlotId sl[kMaxTB][K];
            uint32_t n_dem = 0, n_pin = 0;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    const SlotId f = st_->cache().find(s, uint16_t(mail[t][k]));
                    present[t][k] = f != kNoSlot;
                    bool dup = false;
                    for (uint32_t t2 = 0; t2 < t && !dup; ++t2) for (uint32_t k2 = 0; k2 < K; ++k2) dup |= mail[t2][k2] == mail[t][k];
                    if (dup) continue;
                    if (f == kNoSlot) ++n_dem; else if (!st_->cache().ready(f)) ++n_pin;
                }
            for (int pass = 0; pass < 2; ++pass)
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t k = 0; k < K; ++k) {
                        if (present[t][k] != (pass == 0)) continue;
                        SlotCache::Lookup r;
                        r = request_wait(s, uint16_t(mail[t][k]));
                        st_->cache().pin(r.slot);
                        pinned_.push_back(r.slot);
                        sl[t][k] = r.slot;
                        ++stats_.demand;
                    }
            st_->submit();  // demand reads first
            uint64_t ta[kMaxTB][TAB] = {};
            for (uint32_t t = 0; t < T; ++t) {  // flag A: every token's resident experts start
                for (uint32_t k = 0; k < K; ++k) ta[t][k] = st_->cache().ready(sl[t][k]) ? slot_addr(sl[t][k]) : 0;
                std::memcpy(ctlp_ + vstage(t, s), ta[t], sizeof ta[t]);
            }
            // S18 unique-expert entries: resident ones (A) now, streamed ones (B) with their final slots at flag B
            VEnt ea[4 * K] = {}, eb[4 * K] = {};
            uint16_t ida[4 * K], idb[4 * K];
            uint32_t na = 0, nbe = 0;
            if (vdedupe_[T]) {
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t k = 0; k < K; ++k) {
                        const bool res = ta[t][k] != 0;
                        VEnt* ev = res ? ea : eb;
                        uint16_t* iv = res ? ida : idb;
                        uint32_t& nv = res ? na : nbe;
                        uint32_t j = 0;
                        while (j < nv && (iv[j] != mail[t][k] || ev[j].n >= 4)) ++j;
                        if (j == nv) { iv[nv] = uint16_t(mail[t][k]); ev[nv].slot = res ? ta[t][k] : 1; ev[nv].n = 0; ++nv; }
                        ev[j].tk[ev[j].n++] = (t << 8) | k;
                    }
                std::memcpy(ctlp_ + vent(s, false), ea, sizeof(VEnt) * T * K);
            }
            __builtin_ia32_sfence();
            *ctlw(vstage(0, s) + 448) = 1;
            if (s >= 1)
                for (uint32_t t = 0; t < T; ++t) {  // score the depth-1 predictions made for this layer
                    const uint32_t* pm = reinterpret_cast<const uint32_t*>(ctlp_ + vpmail(t, s));
                    for (uint32_t k = 0; k < K; ++k) {
                        ++stats_.pred_total;
                        for (uint32_t j = 0; j < K; ++j) if (pm[k] == mail[t][j]) { ++stats_.pred_correct; break; }
                    }
                }
            if (s + 1 < NL_) {  // prefetch every token's predicted layer s+1 experts (token 0 first: it is always kept)
                for (uint32_t t = 0; t < T; ++t) {
                    wait_word(vpmail(t, s + 1) + 128, s + 2);
                    prefetch_from(s + 1, reinterpret_cast<const uint32_t*>(ctlp_ + vpmail(t, s + 1)));
                }
                st_->submit();
            } else if (mtp_ok_ && o_.mtp_pref && !pf) {  // S18: the next draft's experts (MTP layer), predicted on the last token
                wait_word(vpmail(T - 1, L) + 128, L + 1);
                const uint32_t* pm = reinterpret_cast<const uint32_t*>(ctlp_ + vpmail(T - 1, L));
                const uint32_t pmax = o_.prefetch_max;
                o_.prefetch_max = o_.mtp_pref;
                prefetch_from(L, pm);
                o_.prefetch_max = pmax;
                std::memcpy(mtp_pred_, pm, sizeof mtp_pred_);
                mtp_pred_valid_ = true;
                st_->submit();
            }
            for (;;) {
                bool all = true;
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t k = 0; k < K; ++k) {
                        if (st_->cache().ready(sl[t][k])) continue;
                        all = false;
                        if (st_->cache().state(sl[t][k]) == SlotState::Free) {
                            SlotCache::Lookup r;
                            r = request_wait(s, uint16_t(mail[t][k]));
                            st_->cache().pin(r.slot);
                            pinned_.push_back(r.slot);
                            sl[t][k] = r.slot;
                        }
                    }
                if (all) break;
                st_->reap(true);
            }
            st_->reap(false);
            {
                const double w = secs(c, Clock::now());
                stats_.t_io += w;
                const int cls = n_dem ? 2 : n_pin ? 1 : 0;
                stats_.wait_cls[cls] += w;
                ++stats_.layers_cls[cls];
                stats_.n_dem += n_dem;
                stats_.n_pin += n_pin;
                if (n_dem < 8) { stats_.wait_by_dem[n_dem] += w; ++stats_.layers_by_dem[n_dem]; }
            }
            for (uint32_t t = 0; t < T; ++t) {  // flag B: full tables, streamed-only tables, routing weights
                uint64_t tsl[TAB] = {}, tb[TAB] = {};
                float twt[TAB] = {};
                const float* mw = reinterpret_cast<const float*>(mail[t] + 16);
                for (uint32_t k = 0; k < K; ++k) {
                    tsl[k] = slot_addr(sl[t][k]);
                    tb[k] = ta[t][k] ? 0 : tsl[k];
                    twt[k] = mw[k];
                    slots_tab[(uint64_t(t) * (L + 1) + s) * TAB + k] = tsl[k];
                }
                std::memcpy(ctlp_ + vstage(t, s) + 128, tsl, sizeof tsl);
                std::memcpy(ctlp_ + vstage(t, s) + 256, tb, sizeof tb);
                std::memcpy(ctlp_ + vstage(t, s) + 384, twt, sizeof twt);
            }
            if (vdedupe_[T]) {
                for (uint32_t j = 0; j < nbe; ++j) { const uint32_t tk = eb[j].tk[0]; eb[j].slot = slot_addr(sl[tk >> 8][tk & 0xFF]); }
                std::memcpy(ctlp_ + vent(s, true), eb, sizeof(VEnt) * T * K);
            }
            __builtin_ia32_sfence();
            *ctlw(vstage(0, s) + 452) = 1;
            if (s + 1 == PLE_LAYER && use_ple_ && !ng_now) ng_finish_ctl_v(T);
            a = Clock::now();
        }
        {
            VkResult r;
            while ((r = vkGetFenceStatus(ctx_.dev, fence_)) == VK_NOT_READY) { service(); __builtin_ia32_pause(); }
            vk::check(r, "vkGetFenceStatus");
            vkResetFences(ctx_.dev, 1, &fence_);
        }
        stats_.t_gpu += secs(a, Clock::now());
        for (SlotId q : pinned_) st_->cache().unpin(q);
        pinned_.clear();
        if (*ctlw(kCtlErr)) throw std::runtime_error("verify one-cb: GPU gate timed out");
        {
            const uint32_t n = 2 + 2 * (2 * NL_ + 1);
            auto ts = ctx_.read_ts(pf ? vtsqp_[T] : vtsq_[T], n);
            double gates = 0;
            const auto& gk = pf ? vgate_kindsp_[T] : vgate_kinds_[T];
            for (uint32_t q = 1, gi = 0; q + 1 < n; q += 2, ++gi) {
                const double d = double(ts[q + 1] - ts[q]);
                gates += d;
                if (gi < gk.size()) stats_.gate_t[gk[gi]] += d * ctx_.ts_period_ns * 1e-9;
            }
            stats_.t_gpu_pure += (double(ts[n - 1] - ts[0]) - gates) * ctx_.ts_period_ns * 1e-9;
        }
        for (uint32_t t = 0; t < T; ++t) out[t] = static_cast<uint32_t*>(arg_.map)[t * 4];
        const uint32_t m = accept_m(v, T, out, decide);
        if (m + 1 < T) ctx_.run(ck_restore_[m]);  // the state after v[m]; m == T-1 left it in place
        for (uint32_t t = 0; t <= m; ++t) hist_.push_back(v[t]);
        stats_.tokens += m + 1;
        if (pf) ++stats_.prefill_chunks;
        else {
            ++stats_.spec_cycles;
            stats_.spec_drafted += T - 1;
            stats_.spec_accepted += m;
        }
        if (st_->stats().io_errors > 64) throw std::runtime_error("expert I/O errors");
        return m;
    }
    uint32_t batch_max() const { return 1 + o_.spec_k; }  // S20: verify batch limit (TB_ also counts prefill blocks)

    // ---- MTP draft head (S16) ----
    // The head consumes the 4-stream residual A_.H of position p (the trunk's after feed(), or its own output after a
    // previous draft) and the token x_{p+1}, writes its own KV entry at index p and returns its greedy guess for x_{p+2}.
    // Fusion (SGLang qwen4_exp_mtp.py): H'_c = fc_hidden(RMSNorm_10240(H)_c) + fc_embedding(RMSNorm_2560(emb(x))), c = 0..3.
    // The lm_head logits are overwritten (consume the trunk's logits before drafting).
    bool has_mtp() const { return mtp_ok_; }
    float mtp_conf() const { return mtp_conf_; }  // S18: softmax top-1 probability of the last draft
    uint32_t mtp_draft(int64_t next_tok, uint32_t index_pos, uint32_t blk = 0) {
        if (!mtp_ok_) throw std::runtime_error("MTP head not loaded");
        auto t0 = Clock::now();
        if (o_.tok_route_mode & 4) tok_prefetch(1, next_tok);  // S19: this draft's MTP-layer experts (needed after its attention)
        if (o_.tok_route_mode & 2) tok_prefetch(0, next_tok);  // S19: next verify's column 0 = next_tok: layer 0 during the draft
        mtp_prepare(next_tok, index_pos, blk);
        ctx_.submit(cb_mtp_pre_.at(blk), fence_);
        ctx_.wait(fence_);
        for (SlotId p : pinned_) st_->cache().unpin(p);
        pinned_.clear();
        const uint32_t s = L;
        const uint32_t* mail = reinterpret_cast<const uint32_t*>(static_cast<uint8_t*>(pMail_) + uint64_t(s) * 256);
        uint32_t mloc[32];
        if (route_on(s)) { std::memcpy(mloc, mail, sizeof mloc); route_select(s, mloc, o_.route_mtp); mail = mloc; }
        const float* mw = reinterpret_cast<const float*>(mail + 16);
        auto c = Clock::now();
        if (mtp_pred_valid_) {  // S18: precision of the MTP-expert prediction made in the previous verify
            for (uint32_t k = 0; k < K; ++k) {
                ++stats_.mtp_pred_total;
                for (uint32_t j = 0; j < K; ++j) if (mtp_pred_[k] == mail[j]) { ++stats_.mtp_pred_ok; break; }
            }
            mtp_pred_valid_ = false;
        }
        tok_learn(1, next_tok, mail);
        SlotId sl[K];
        for (int pass = 0; pass < 2; ++pass)
            for (uint32_t k = 0; k < K; ++k) {
                const bool present = st_->cache().find(s, uint16_t(mail[k])) != kNoSlot;
                if (present != (pass == 0)) continue;
                SlotCache::Lookup r;
                r = request_wait(s, uint16_t(mail[k]));
                st_->cache().pin(r.slot);
                pinned_.push_back(r.slot);
                sl[k] = r.slot;
                ++stats_.mtp_demand;
            }
        st_->submit();
        const double io0 = stats_.t_io;
        if (o_.mtp_lossy && o_.overlap) {
            // S18 lossy draft: the verify is exact whatever the draft, so the head runs with its RESIDENT experts only
            // (weights renormalised over them); the misses stream in for later drafts instead of stalling this one
            auto* slots_tab = static_cast<uint64_t*>(pSl_);
            auto* wt = static_cast<float*>(pWt_);
            uint64_t* ta = static_cast<uint64_t*>(pSlA_) + uint64_t(s) * TAB;
            uint64_t* tb = static_cast<uint64_t*>(pSlB_) + uint64_t(s) * TAB;
            float wsum = 0;
            uint32_t nres = 0;
            for (uint32_t k = 0; k < K; ++k) if (st_->cache().ready(sl[k])) { wsum += mw[k]; ++nres; }
            float wtot = 0;
            for (uint32_t k = 0; k < K; ++k) wtot += mw[k];
            if (nres == 0 || wtot - wsum > o_.mtp_lossy_w * wtot) wait_bind(s, sl, mail, mw, c, hits_mtp_.at(blk));  // too much missing: exact
            else {
                for (uint32_t k = 0; k < K; ++k) {
                    const bool r = st_->cache().ready(sl[k]);
                    ta[k] = r ? slot_addr(sl[k]) : 0;
                    tb[k] = 0;
                    slots_tab[s * TAB + k] = r ? slot_addr(sl[k]) : 0;
                    wt[s * TAB + k] = r ? mw[k] / wsum : 0.f;
                }
                stats_.mtp_skipped += K - nres;
                ctx_.submit(hits_mtp_.at(blk), VK_NULL_HANDLE);
            }
        } else
        wait_bind(s, sl, mail, mw, c, o_.overlap ? hits_mtp_.at(blk) : VK_NULL_HANDLE);
        stats_.t_mtp_io += stats_.t_io - io0;
        stats_.t_io = io0;
        ctx_.submit(cb_mtp_post_[blk], fence_);
        ctx_.wait(fence_);
        for (SlotId p : pinned_) st_->cache().unpin(p);
        pinned_.clear();
        ++stats_.mtp_drafts;
        stats_.t_mtp += secs(t0, Clock::now());
        {  // S18: softmax top-1 probability of the draft
            const float se = static_cast<const float*>(arg_.map)[2];
            mtp_conf_ = se > 0 ? 1.f / se : 0.f;
        }
        return static_cast<uint32_t*>(arg_.map)[0];
    }
    // S22: the MTP head's KV entry of every chunk position as GEMMs over slices (S20: one submit + fence per token, its
    // matvecs re-reading the MTP layer's weights: 0.18 ms/token). Per slice: next-token embeddings (bf16 staging, w_embed),
    // hidden norm over all 4 streams (w_norm10240) + embedding norm, fc_embedding GEMM, its output copied into the 4 streams,
    // + fc_hidden over the 4T stream rows, the MTP layer's attention HC pre-step, qkv (+ indexer qk) GEMMs, wide prep of the
    // K/V (and raw indexer key) at positions p0 + a ... Same ops as cb_mtp_fill_ (sum order of the GEMMs differs).
    static void xbar(VkCommandBuffer cb) {  // compute <-> transfer (vk::Ctx::barrier orders compute -> compute only)
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cb, &di);
    }
    void wide_mtp_fill(const int64_t* v, uint32_t N, uint32_t p0, int64_t mtp_next) {
        auto t0 = Clock::now();
        const uint32_t Nf = mtp_next >= 0 ? N : N - 1;
        if (!Nf) return;
        ek_ensure(uint64_t(p0) + Nf + 2);
        pf_phase_ = 3;
        const uint32_t S = o_.wide_slice, ns = (Nf + S - 1) / S;
        const auto& w = W_[L];
        const uint64_t aE = wY_, aEn = wHd_, aEf = wHd_ + uint64_t(S) * HID * 4, aHp = wPre_;  // scratch rows of the slice arena
        VkCommandBuffer prev = VK_NULL_HANDLE;
        for (uint32_t j = 0; j <= ns; ++j) {
            VkCommandBuffer cb = VK_NULL_HANDLE;
            if (j < ns) {
                const uint32_t a = j * S, T = std::min(S, Nf - a), pos0 = p0 + a;
                auto th = Clock::now();
                auto* st = static_cast<uint16_t*>(pWEmb_) + uint64_t(j & 1) * S * HID;
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t i = a + t;
                    const int64_t nx = i + 1 < N ? v[i + 1] : mtp_next;
                    if (::pread(emb_fd_, st + uint64_t(t) * HID, HID * 2, off_t(uint64_t(nx) * HID * 2)) != ssize_t(HID * 2))
                        throw std::runtime_error("embed read");
                }
                stats_.t_host += secs(th, Clock::now());
                cb = wide_begin(1 + (j & 1));
                PcWEmbed pe{wEmb_ + uint64_t(j & 1) * S * HID * 2, aE, HID * 4, 0};
                vk::Ctx::dispatch(cb, pWe_, &pe, HID / 256, T);
                PcWNorm nh{aCarry_ + uint64_t(a) * HCH * 4, mtp_nh_, wN_, HCH * 4, HCH * 4, 1e-6f, 0};
                vk::Ctx::dispatch(cb, pWn10k_, &nh, 1, T);
                BAR(cb, __LINE__);
                PcWNorm ne{aE, mtp_ne_, aEn, HID * 4, HID * 4, 1e-6f, 0};
                vk::Ctx::dispatch(cb, pWn_, &ne, 1, T);
                BAR(cb, __LINE__);
                WG_GEMM(cb, mtp_fce_, aEn, HID * 4, aEf, HID * 4, HID, HID, T);
                BAR(cb, __LINE__);
                xbar(cb);  // the copy reads what a compute shader wrote and the next GEMM reads what it wrote
                {
                    uint64_t so = 0, doff = 0;
                    VkBuffer sb = d_.buf_of(aEf, so), db = d_.buf_of(aHp, doff);
                    std::vector<VkBufferCopy> rg(uint64_t(T) * HC);
                    for (uint32_t t = 0; t < T; ++t)
                        for (uint32_t c = 0; c < HC; ++c) rg[uint64_t(t) * HC + c] = {so + uint64_t(t) * HID * 4, doff + (uint64_t(t) * HC + c) * HID * 4, HID * 4};
                    vkCmdCopyBuffer(cb, sb, db, uint32_t(rg.size()), rg.data());
                }
                xbar(cb);
                WG_GEMM(cb, mtp_fch_, wN_, HID * 4, aHp, HID * 4, HID, HID, T * HC, 1);
                BAR(cb, __LINE__);
                W_HCPRE(cb, w.hca, aHp, T, wI_);
                WG_GEMM(cb, w.qkv, wX_, HID * 4, wQkv_, 13312 * 4, HID, 13312, T);
                if (idx_on_) WG_GEMM(cb, w.iqk, wX_, HID * 4, wIqk_, 640 * 4, HID, 640, T);
                BAR(cb, __LINE__);
                PcPrepW pr{wQkv_, w.qn, w.kn, wQ_, wG_, kc_[L], vc_[L], 0, 1e-6f, 1e7f, 13312 * 4, 6144 * 4, pos0, 0};
                vk::Ctx::dispatch(cb, pWpr_, &pr, 26, T);
                if (idx_on_) { PcIdxPrepW ip{wIqk_, w.iqn, wQi_, rk_[L], 0, 1e-6f, 1e7f, 640 * 4, 512 * 4, pos0, 0}; vk::Ctx::dispatch(cb, pWip_, &ip, 5, T); }
                BAR(cb, __LINE__);
                if (idx_on_) {  // S30: the MTP layer's pooled block keys of the slice. Until S30 nothing pooled them in a wide
                                // prefill: every prefill block scored 0 for the MTP drafts (ties -> the first 2048 positions).
                    const uint32_t b0 = pos0 / 4, b1 = (pos0 + T) / 4;
                    PcIdxPool pl{rk_[L], pk_[L], w.ikn, b0, pos0, T, 1e-6f, 1e7f, 0};
                    vk::Ctx::dispatch(cb, pWpl_, &pl, b1 - b0 + 1);
                    BAR(cb, __LINE__);
                }
                wide_end(cb, 1 + (j & 1));
                ctx_.submit(cb, wfence_[j & 1]);
            }
            if (j > 0) wide_wait(prev, 1 + ((j - 1) & 1), wfence_[(j - 1) & 1]);
            prev = cb;
        }
        stats_.t_mtp_fill += secs(t0, Clock::now());
    }
    // KV entry only (prefill): fusion + attention hyper-connection + qkv + prep at index p. No MoE, no head.
    void mtp_fill(int64_t next_tok, uint32_t index_pos, uint32_t blk = 0) {
        if (!mtp_ok_) return;
        auto t0 = Clock::now();
        mtp_prepare(next_tok, index_pos, blk);
        ctx_.submit(cb_mtp_fill_.at(blk), fence_);
        ctx_.wait(fence_);
        stats_.t_mtp_fill += secs(t0, Clock::now());
    }

    // n-gram rows: bf16 -> f32 into the PLE input (block t's E), 16 heads x 160
    void ng_convert(uint32_t tokens = 1) {
        for (uint32_t t = 0; t < tokens; ++t) {
            auto* e = reinterpret_cast<float*>(act_host_ + uint64_t(t) * act_stride_ + act_off_.E);
            const uint16_t* b = ngbuf_ + t * 16 * 160;
            for (int i = 0; i < 16 * 160; ++i) { uint32_t x = uint32_t(b[i]) << 16; std::memcpy(&e[i], &x, 4); }
        }
    }
    void ng_finish() {
        auto t = Clock::now();
        while (ng_pending_) {
            io_uring_cqe* cqe = nullptr;
            int r = io_uring_wait_cqe(&ngr_, &cqe);
            if (r == -EINTR) continue;
            if (r < 0) throw std::runtime_error("ngram wait");
            const int res = cqe->res;
            const uint64_t h = io_uring_cqe_get_data64(cqe);
            io_uring_cqe_seen(&ngr_, cqe);
            --ng_pending_;
            if (res != 320 && ::pread(ng_.fd, ngbuf_ + h * 160, 320, off_t(ngoff_[h])) != 320)  // retry once, sync
                throw std::runtime_error("ngram read " + std::to_string(res));
        }
        ng_convert(ng_tokens_);
        stats_.t_ngram += secs(t, Clock::now());
    }

    // S19 token-route table: the last routing seen for this token id at layer 0 (which 0) / the MTP layer (which 1)
    void tok_prefetch(int which, int64_t tok) {
        if (!o_.tok_route || tok < 0 || uint64_t(tok) >= VOCAB || trt_[which].empty()) return;
        const uint32_t layer = which ? L : 0;
        if (which && !mtp_ok_) return;
        const uint16_t* e = trt_[which].data() + uint64_t(tok) * o_.tok_route;
        bool any = false;
        for (uint32_t k = 0; k < o_.tok_route; ++k) {
            if (e[k] >= E) break;
            if (st_->cache().find(layer, e[k]) != kNoSlot) continue;
            const auto r = st_->request(layer, e[k], o_.cold_prefetch, true);
            if (r.slot == kNoSlot) continue;
            any = true;
            ++stats_.trt_issued;
        }
        if (any) st_->submit();
    }
    void tok_learn(int which, int64_t tok, const uint32_t* mail) {
        if (!o_.tok_route || tok < 0 || uint64_t(tok) >= VOCAB || trt_[which].empty()) return;
        uint16_t* e = trt_[which].data() + uint64_t(tok) * o_.tok_route;
        if (e[0] < E)  // score the table's guess before overwriting it
            for (uint32_t k = 0; k < o_.tok_route && e[k] < E; ++k) {
                ++stats_.trt_total;
                for (uint32_t j = 0; j < K; ++j) if (e[k] == mail[j]) { ++stats_.trt_ok; break; }
            }
        for (uint32_t k = 0; k < o_.tok_route; ++k) e[k] = uint16_t(mail[k]);  // top-k is sorted by weight
    }
    // S24: a demand request that waits for a free slot. Slots free up when reads land (in flight -> evictable) or when a
    // sweep unpins. If NOTHING is in flight and the class is still full, every slot is pinned and no amount of waiting
    // helps (this used to spin forever: the 3.18 pack's small size class had 998 slots, and the wide prefill pins the
    // experts of two consecutive layers of a chunk, up to 2 x E). Fails loudly after 2 s instead.
    // ---- S25 chunk lookahead (--pf-ahead) ----
    // The layer-major prefill learns a layer's experts only when that layer's router has run, so the reads of layer s
    // started during PRE(s) and the MoE of layer s then waited for the rest: 0.5-0.9 ms of every 3.4-3.8 ms prefill
    // token, with the SSD idle through every MoE (measured S25: busy 56-70%). A 4,096-token chunk uses 400-475 of a
    // layer's 512 experts and consecutive chunks of one document use nearly the same ones, so when the MoE of layer
    // s - 1 is released to the GPU the experts layer s used in the previous chunk (every expert, for the first chunk)
    // are read ahead, pinned until PRE(s) has made its own requests. Speculative lookups: no LFU use is counted, and a
    // wrong guess is the next victim once unpinned. Routing and arithmetic are untouched (exact routing stays exact).
    // Chunks of fewer than kPfAheadMin tokens are left alone.
    void pf_release() {
        for (SlotId q : pf_pins_) st_->cache().unpin(q);
        pf_pins_.clear();
    }
    void pf_lookahead(uint32_t s) {
        // Only for chunks that use most of a layer's experts: a 128-token chunk uses about 200 of 512, and reading the
        // other 300 per layer made a short append after a restart take 8.4 s instead of 3 (measured S25).
        if (!o_.pf_ahead || s >= NL_ || pf_chunk_ < kPfAheadMin) return;
        auto th = Clock::now();
        auto& cc = st_->cache();
        if (pf_cnt_.empty()) { pf_cnt_.assign(uint64_t(L) * E, 0); pf_seen_.assign(L, 0); pf_mark_.assign(E, 0); }
        pf_release();
        std::fill(pf_mark_.begin(), pf_mark_.end(), 0);
        std::vector<std::pair<uint32_t, uint16_t>> cand;
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t c = pf_seen_[s] ? pf_cnt_[uint64_t(s) * E + e] : 1;
            if (!c) continue;
            const SlotId r = cc.find(s, e);
            if (r != kNoSlot) { cc.pin(r); pf_pins_.push_back(r); continue; }  // resident: must not be this batch's victim
            cand.push_back({c, uint16_t(e)});
        }
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        const uint32_t ev = cc.evictable_in_class(cc.layer_class(s));
        const uint32_t budget = std::min<uint32_t>(o_.pf_ahead, ev > 64 ? ev - 64 : 0);
        uint32_t n = 0;
        for (const auto& ce : cand) {
            if (n >= budget) break;
            const auto r = st_->request(s, ce.second, true, true);
            if (r.slot == kNoSlot) break;
            cc.pin(r.slot);
            pf_pins_.push_back(r.slot);
            pf_mark_[ce.second] = 1;
            ++n;
        }
        st_->submit();
        stats_.pfa_issued += n;
        stats_.pfa_layers += 1;
        stats_.t_host += secs(th, Clock::now());
    }
    // layer s of an N-token chunk has been routed (lm_mail_): remember its experts for the next chunk, drop the pins
    void pf_note(uint32_t s, uint32_t N) {
        if (!o_.pf_ahead || s >= NL_) return;
        if (N < kPfAheadMin) { pf_release(); return; }  // keep the last large chunk's sets
        if (pf_cnt_.empty()) { pf_cnt_.assign(uint64_t(L) * E, 0); pf_seen_.assign(L, 0); pf_mark_.assign(E, 0); }
        uint32_t* row = pf_cnt_.data() + uint64_t(s) * E;
        std::fill(row, row + E, 0u);
        const uint32_t* m = lm_mail_.data() + uint64_t(s) * N * K;
        for (uint64_t q = 0; q < uint64_t(N) * K; ++q) ++row[m[q]];
        for (uint32_t e = 0; e < E; ++e) { stats_.pfa_used += pf_mark_[e] && row[e]; stats_.pfa_uniq += row[e] != 0; pf_mark_[e] = 0; }
        pf_seen_[s] = 1;
        pf_release();
    }
    SlotCache::Lookup request_wait(uint32_t layer, uint32_t expert, bool cold = false, bool spec = false) {
        SlotCache::Lookup r;
        uint64_t idle = 0;
        Clock::time_point t0;
        while ((r = st_->request(layer, expert, cold, spec)).slot == kNoSlot) {
            if (!pf_pins_.empty()) { pf_release(); continue; }  // S25: lookahead pins never starve a demand request
            st_->reap(true);
            if (st_->in_flight()) { idle = 0; continue; }
            if (idle++ == 0) t0 = Clock::now();
            else if ((idle & 0x3FF) == 0 && secs(t0, Clock::now()) > 2.0)
                throw std::runtime_error("expert pool saturated at layer " + std::to_string(layer) + ": every slot of its size class is pinned (" +
                                         std::to_string(st_->cache().stats().saturated) + " refused lookups)");
        }
        return r;
    }
    // S24 cache-aware routing: see Opts::route_bias. mail = a HOST COPY of the mailbox row (ids[16] | w[16]) holding
    // kr(s) candidates sorted by weight; on return its first K entries are the kept experts in rank order with
    // renormalised weights. also[] = experts already chosen in this layer by earlier tokens of the batch (count as
    // resident: their reads are issued anyway). bias = 0 keeps the true top-K.
    bool route_on(uint32_t s) const { return (s == L ? o_.route_mtp : o_.route_bias) > 0; }
    uint32_t kr(uint32_t s) const { return route_on(s) ? o_.route_cand : K; }
    void route_select(uint32_t s, uint32_t* mail, float bias, const uint32_t* also = nullptr, uint32_t n_also = 0) {
        const uint32_t C = o_.route_cand;
        float* w = reinterpret_cast<float*>(mail + 16);
        float sc[16];
        bool keep[16] = {}, res[16] = {};
        for (uint32_t c = 0; c < C; ++c) {
            if (mail[c] >= E || !(w[c] > 0)) { sc[c] = -std::numeric_limits<float>::infinity(); continue; }
            sc[c] = std::log(w[c]);
            if (bias > 0) {
                const SlotId f = st_->cache().find(s, uint16_t(mail[c]));  // route_ready: an expert still in flight earns no bonus
                res[c] = f != kNoSlot && (!o_.route_ready || st_->cache().ready(f));
                for (uint32_t j = 0; j < n_also && !res[c]; ++j) res[c] = also[j] == mail[c];
            }
            if (c < o_.route_keep) sc[c] = std::numeric_limits<float>::infinity();
            else if (res[c]) sc[c] += bias;
            else if (bias > 0 && st_->tier_has(s, mail[c])) { sc[c] += bias * o_.tier_bias; res[c] = o_.tier_bias >= 1.0f; }  // S31
        }
        for (uint32_t n = 0; n < K; ++n) {  // the K best scores; ties go to the higher true rank
            uint32_t b = C;
            for (uint32_t c = 0; c < C; ++c) if (!keep[c] && (b == C || sc[c] > sc[b])) b = c;
            keep[b] = true;
        }
        uint32_t ids[K], n = 0;
        float ww[K], sum = 0, top = 0, lost = 0;
        float topK = 0;  // S35: the whole true top-K weight (top below accumulates during the loop)
        for (uint32_t c = 0; c < K && c < C; ++c) if (mail[c] < E && w[c] > 0) topK += w[c];
        for (uint32_t c = 0; c < C; ++c) {
            if (c < K) top += w[c];
            if (!keep[c]) {
                if (c < K) {
                    lost += w[c];
                    if (o_.route_shadow > 0 && s != L && mail[c] < E && topK > 0) {  // S35: the pool keeps the true demand
                        st_->cache().shadow_use(s, uint16_t(mail[c]), double(o_.route_shadow) * double(w[c]) / double(topK) * double(K));
                        ++stats_.rr_shadow;
                        if (o_.route_shadow_fetch > 0 && st_->cache().find(s, uint16_t(mail[c])) == kNoSlot &&
                            st_->cache().expert_score_now(s, uint16_t(mail[c])) >= o_.route_shadow_fetch) {
                            if (rr_q_.size() >= 96) { rr_q_.erase(rr_q_.begin(), rr_q_.begin() + 32); stats_.rr_bg_dropped += 32; }
                            rr_q_.push_back({s, uint16_t(mail[c])});
                        }
                    }
                    if (o_.route_fetch && !o_.route_shadow_fetch && s != L && mail[c] < E) {  // the pool should still learn this expert
                        if (rr_q_.size() >= 96) { rr_q_.erase(rr_q_.begin(), rr_q_.begin() + 32); stats_.rr_bg_dropped += 32; }
                        rr_q_.push_back({s, uint16_t(mail[c])});
                    }
                }
                continue;
            }
            ids[n] = mail[c]; ww[n] = w[c]; sum += w[c]; ++n;
            if (bias > 0) { if (c >= K) ++stats_.rr_swaps; if (!res[c]) ++stats_.rr_miss_kept; }
        }
        for (uint32_t k = 0; k < K; ++k) { mail[k] = ids[k]; w[k] = ww[k] / sum; }
        if (bias > 0) { ++stats_.rr_sel; stats_.rr_mass += top > 0 ? lost / top : 0; }
    }
    // S24: issue queued background reads (newest first) while the SSD has room; called from every wait loop
    void rr_pump() {
        bool any = false;
        const uint32_t cap = o_.route_fetch ? o_.route_fetch : 4;  // S35: shadow fetch without --route-fetch
        while (!rr_q_.empty() && st_->in_flight() < cap) {
            const Deferred c = rr_q_.back();
            rr_q_.pop_back();
            if (st_->cache().find(c.layer, c.expert) != kNoSlot) continue;
            if (st_->request(c.layer, c.expert).slot == kNoSlot) break;
            any = true;
            ++stats_.rr_bg;
        }
        if (any) st_->submit();
    }
    // prefetch the predicted experts of layers s+1..s+D (predictions made on layer s's MoE input)
    void prefetch(uint32_t s) {
        for (uint32_t dd = 1; dd <= o_.predict && s + dd < NL_; ++dd) {
            const uint32_t* pm = reinterpret_cast<const uint32_t*>(pmail_host_ + (uint64_t(dd) * L + s + dd) * 256);
            const float* pw = reinterpret_cast<const float*>(pm + 16);  // top-k is sorted by weight
            const bool defer = o_.defer_from && dd >= o_.defer_from;
            for (uint32_t k = 0; k < std::min(K, o_.prefetch_max); ++k) {
                if (pw[k] < o_.prefetch_min_w || pm[k] >= E || st_->cache().find(s + dd, uint16_t(pm[k])) != kNoSlot) continue;
                if (defer) { deferred_.push_back({s + dd, uint16_t(pm[k])}); continue; }
                const auto r = st_->request(s + dd, uint16_t(pm[k]), o_.cold_prefetch, true);
                if (r.slot == kNoSlot) continue;
                ++stats_.pref_issued;
                if (!r.hit && o_.pref_expire) pref_rec_[s + dd].push_back({r.slot, uint16_t(pm[k]), 0});
            }
        }
    }
    // S17: issue queued (deep) predictions while the SSD is nearly idle; stale ones (target layer bound) are dropped
    void pump_deferred() {
        bool any = false;
        while (dq_head_ < deferred_.size() && st_->in_flight() < o_.defer_qd) {
            const Deferred c = deferred_[dq_head_++];
            if (c.layer <= cur_layer_) { ++stats_.pref_def_dropped; continue; }
            if (st_->cache().find(c.layer, c.expert) != kNoSlot) continue;
            const auto r = st_->request(c.layer, c.expert, o_.cold_prefetch, true);
            if (r.slot == kNoSlot) break;
            any = true;
            ++stats_.pref_def_issued;
            if (!r.hit && o_.pref_expire) pref_rec_[c.layer].push_back({r.slot, c.expert, 1});
        }
        if (dq_head_ == deferred_.size()) { deferred_.clear(); dq_head_ = 0; }
        if (any) st_->submit();
    }
    // GPU step wait: plain fence wait, or (deferred prefetches queued) a fence poll that keeps the SSD fed meanwhile
    void gpu_wait() {
        if (!o_.defer_from) {
            if (o_.spin_wait) ctx_.wait_spin(fence_); else ctx_.wait(fence_);
            return;
        }
        VkResult r;
        while ((r = vkGetFenceStatus(ctx_.dev, fence_)) == VK_NOT_READY) {
            st_->reap(false);
            pump_deferred();
            __builtin_ia32_pause();
        }
        vk::check(r, "vkGetFenceStatus");
        vkResetFences(ctx_.dev, 1, &fence_);
    }
    // S17: layer s just bound its experts (mail): prefetches aimed at s that it did not use go to the LRU tail
    void expire_prefetches(uint32_t s, const uint32_t* mail) {
        auto& v = pref_rec_[s];
        for (const auto& r : v) {
            if (st_->cache().find(s, r.expert) != r.slot) continue;  // evicted / remapped since
            bool used = false;
            for (uint32_t k = 0; k < K; ++k) used |= mail[k] == r.expert;
            if (used) ++stats_.pref_used[r.def]; else { ++stats_.pref_unused[r.def]; st_->cache().demote(r.slot); }
        }
        v.clear();
    }
    // Layer s's routed experts are requested (sl[], pinned) and the reads submitted: start the resident ones on the GPU
    // (overlap), wait for the misses, then publish the slot table + routing weights for the layer's down pass.
    void wait_bind(uint32_t s, SlotId* sl, const uint32_t* mail, const float* mw, Clock::time_point c, VkCommandBuffer hits = VK_NULL_HANDLE) {
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        auto* wt = static_cast<float*>(pWt_);
        uint64_t* ta = static_cast<uint64_t*>(pSlA_) + uint64_t(s) * TAB;
        if (o_.overlap) {  // resident (Ready + pinned) experts start now, while the misses stream
            for (uint32_t k = 0; k < K; ++k) ta[k] = st_->cache().ready(sl[k]) ? slot_addr(sl[k]) : 0;
            ctx_.submit(hits ? hits : hits_[s], VK_NULL_HANDLE);
        }
        for (;;) {
            bool all = true;
            for (uint32_t k = 0; k < K; ++k) {
                if (st_->cache().ready(sl[k])) continue;
                all = false;
                if (st_->cache().state(sl[k]) == SlotState::Free) {
                    SlotCache::Lookup r;
                    r = request_wait(s, uint16_t(mail[k]));
                    st_->cache().pin(r.slot);
                    pinned_.push_back(r.slot);
                    sl[k] = r.slot;
                }
            }
            if (all) break;
            st_->reap(true);
        }
        st_->reap(false);
        stats_.t_io += secs(c, Clock::now());
        for (uint32_t k = 0; k < K; ++k) { slots_tab[s * TAB + k] = slot_addr(sl[k]); wt[s * TAB + k] = mw[k]; }
        if (o_.overlap) {
            uint64_t* tb = static_cast<uint64_t*>(pSlB_) + uint64_t(s) * TAB;
            for (uint32_t k = 0; k < K; ++k) tb[k] = ta[k] ? 0 : slot_addr(sl[k]);
        }
    }
    // Verify: layer s's routed experts of T tokens (mail rows (t, s)), resident first, all reads submitted together, then
    // wait for every one and publish the T slot/weight rows. No hits_ pre-pass (POSTMOE recorded with ov_ = false).
    void bind_tokens(uint32_t s, uint32_t T) {
        auto c = Clock::now();
        SlotId sl[kMaxTB][K];
        const uint32_t* mail[kMaxTB];
        uint32_t mloc[kMaxTB][32], also[kMaxTB * K];
        for (uint32_t t = 0; t < T; ++t) {
            mail[t] = reinterpret_cast<const uint32_t*>(static_cast<uint8_t*>(pMail_) + (uint64_t(t) * (L + 1) + s) * 256);
            if (route_on(s)) {
                std::memcpy(mloc[t], mail[t], sizeof mloc[t]);
                route_select(s, mloc[t], o_.route_bias, also, t * K);
                for (uint32_t k = 0; k < K; ++k) also[t * K + k] = mloc[t][k];
                mail[t] = mloc[t];
            }
            for (uint32_t k = 0; k < K; ++k)
                if (mail[t][k] >= E) {  // NaN router logits (topk returns ~0): never stream past the pack
                    const float* H = reinterpret_cast<const float*>(act_host_ + uint64_t(t) * act_stride_ + act_off_.H);
                    const float* X = reinterpret_cast<const float*>(act_host_ + uint64_t(t) * act_stride_ + act_off_.X);
                    throw std::runtime_error("verify: invalid expert id " + std::to_string(mail[t][k]) + " token " + std::to_string(t) + "/" +
                                             std::to_string(T) + " layer " + std::to_string(s) + " H[0]=" + std::to_string(H[0]) + " X[0]=" + std::to_string(X[0]));
                }
        }
        if (o_.record_trace) for (uint32_t k = 0; k < K; ++k) trace_.back().push_back(uint16_t(mail[0][k]));
        // Two passes (resident first, so a miss never evicts this layer's not-yet-looked-up prefetch). Presence is decided
        // ONCE per entry: a later token's copy of an expert an earlier token just requested must not be skipped in pass 1.
        bool present[kMaxTB][K];
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t k = 0; k < K; ++k) present[t][k] = st_->cache().find(s, uint16_t(mail[t][k])) != kNoSlot;
        for (int pass = 0; pass < 2; ++pass)
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    if (present[t][k] != (pass == 0)) continue;
                    SlotCache::Lookup r;
                    r = request_wait(s, uint16_t(mail[t][k]));
                    st_->cache().pin(r.slot);
                    pinned_.push_back(r.slot);
                    sl[t][k] = r.slot;
                    ++stats_.demand;
                }
        prefetch(s);  // token 0's predictions (PRE records the predictor for t_ == 0 only)
        st_->submit();
        for (;;) {
            bool all = true;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    if (st_->cache().ready(sl[t][k])) continue;
                    all = false;
                    if (st_->cache().state(sl[t][k]) == SlotState::Free) {
                        SlotCache::Lookup r;
                        r = request_wait(s, uint16_t(mail[t][k]));
                        st_->cache().pin(r.slot);
                        pinned_.push_back(r.slot);
                        sl[t][k] = r.slot;
                    }
                }
            if (all) break;
            st_->reap(true);
        }
        st_->reap(false);
        stats_.t_io += secs(c, Clock::now());
        if (nancheck_)
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t k = 0; k < K; ++k) {
                    const SlotId f = st_->cache().find(s, uint16_t(mail[t][k]));
                    if (f != sl[t][k] || !st_->cache().ready(sl[t][k]) || st_->cache().pins(sl[t][k]) == 0)
                        std::fprintf(stderr, "BINDCHECK layer %u token %u k %u expert %u: sl %u find %u ready %d pins %u\n", s, t, k, mail[t][k], sl[t][k], f,
                                     int(st_->cache().ready(sl[t][k])), st_->cache().pins(sl[t][k]));
                }
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        auto* wt = static_cast<float*>(pWt_);
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t row = uint64_t(t) * (L + 1) + s;
            const float* mw = reinterpret_cast<const float*>(mail[t] + 16);
            for (uint32_t k = 0; k < K; ++k) { slots_tab[row * TAB + k] = slot_addr(sl[t][k]); wt[row * TAB + k] = mw[k]; }
        }
    }
    // host side of one MTP step: KV index / RoPE position, the next token's embedding (bf16 -> f32)
    void mtp_prepare(int64_t next_tok, uint32_t index_pos, uint32_t blk) {
        if (index_pos >= o_.max_ctx) throw std::runtime_error("MTP index beyond context");
        ek_ensure(uint64_t(index_pos) + 2);
        if (::pread(emb_fd_, row_.data(), HID * 2, off_t(uint64_t(next_tok) * HID * 2)) != ssize_t(HID * 2)) throw std::runtime_error("embed read");
        auto* e = static_cast<float*>(pEmb_);
        for (uint32_t i = 0; i < HID; ++i) { uint32_t x = uint32_t(row_[i]) << 16; std::memcpy(&e[i], &x, 4); }
        *reinterpret_cast<uint32_t*>(act_host_ + uint64_t(blk) * act_stride_ + act_off_.Par) = index_pos;
    }

    void print_profile() {
        if (!o_.profile && !vprof_ && !lmprof_) return;
        for (int st2 = 0; st2 < 2; ++st2) {
            const uint32_t n = prof_n_[st2] + 1;  // only the written queries (WAIT on unwritten ones blocks forever)
            if (n < 2) continue;
            std::vector<uint64_t> ts(n);
            vkGetQueryPoolResults(ctx_.dev, profq_, uint32_t(st2) * 512, n, n * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            std::fprintf(stderr, "\n[profile step %d (%s)] total %.1f us\n", st2 + 3, st2 ? "GDN" : "attention", double(ts[n - 1] - ts[0]) * ctx_.ts_period_ns * 1e-3);
            for (uint32_t q = 0; q + 1 < n; ++q)
                std::fprintf(stderr, "  line %4d  %7.1f us\n", prof_line_[st2][q], double(ts[q + 1] - ts[q]) * ctx_.ts_period_ns * 1e-3);
        }
    }

    // ---- expert pool persistence ----
    // Text file: one "layer expert" per line, most valuable first (the order in which warm() loads them).
    bool save_residency(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return false;
        auto keys = st_->cache().resident_keys_mru();
        for (uint32_t k : keys) std::fprintf(f, "%u %u\n", k / E, k % E);
        std::fclose(f);
        return true;
    }
    // Preload up to pool_slots() experts listed in `path` (missing file = no-op). Returns slots loaded.
    // S30: up to the slots the KV has left (n_slots_live_); reset = false keeps the hit statistics (a session restore).
    uint32_t warm(const std::string& path, bool reset = true) {
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return 0;
        std::vector<std::pair<uint32_t, uint32_t>> want;
        unsigned l, e;
        while (std::fscanf(f, "%u %u", &l, &e) == 2) if (l < NL_ && e < E) want.push_back({l, e});
        std::fclose(f);
        const uint32_t n = std::min<uint32_t>(uint32_t(want.size()), n_slots_live_ ? n_slots_live_ : n_slots_);
        // Load in reverse so the first-listed (most valuable) ends up most recently used.
        uint32_t loaded = 0;
        for (uint32_t i = n; i-- > 0;) {
            SlotCache::Lookup r;
            r = request_wait(want[i].first, want[i].second);
            if (!r.hit) ++loaded;
            if (st_->in_flight() >= 64) st_->reap(true);
        }
        while (st_->in_flight()) st_->reap(true);
        if (reset) st_->cache().reset_stats();
        return loaded;
    }

private:
    // ------------------------------------------------------------ loading
    std::vector<float> bf16_file(const std::string& name, size_t expect) {
        auto b = read_file(o_.model_dir + "/dense/" + name + ".bin");
        if (b.size() != expect * 2) throw std::runtime_error(name + ": unexpected size");
        std::vector<float> f(expect);
        const auto* u = reinterpret_cast<const uint16_t*>(b.data());
        for (size_t i = 0; i < expect; ++i) { uint32_t x = uint32_t(u[i]) << 16; std::memcpy(&f[i], &x, 4); }
        return f;
    }
    std::vector<int64_t> i64_file(const std::string& name) {
        auto b = read_file(o_.model_dir + "/dense/" + name + ".bin");
        std::vector<int64_t> v(b.size() / 8);
        std::memcpy(v.data(), b.data(), b.size());
        return v;
    }
    // Quantise rows (optionally concatenated from several tensors) to PS5Q4, cached in dense_q4/<key>.q4.
    uint64_t q4_matrix(const std::string& key, const std::vector<std::pair<std::string, uint32_t>>& parts, uint32_t Kd) {
        uint32_t rows = 0;
        for (auto& p : parts) rows += p.second;
        const size_t rb = qn::row_bytes<4>(int(Kd)), bytes = size_t(rows) * rb;
        auto [addr, ptr] = d_.alloc(bytes);
        const std::string cache = o_.model_dir + "/dense_q4/" + key + ".q4";
        struct stat st;
        if (::stat(cache.c_str(), &st) == 0 && size_t(st.st_size) == bytes && read_into(cache, ptr, bytes)) return addr;
        std::vector<uint8_t> out(bytes);
        uint32_t r0 = 0;
        for (auto& p : parts) {
            auto f = bf16_file(p.first, size_t(p.second) * Kd);
            const unsigned nt = std::min(14u, std::max(1u, p.second / 64));
            std::vector<std::thread> th;
            for (unsigned t = 0; t < nt; ++t)
                th.emplace_back([&, t] {
                    for (uint32_t r = t; r < p.second; r += nt) qn::quantize_row_fast<4>(f.data() + size_t(r) * Kd, int(Kd), out.data() + size_t(r0 + r) * rb, 24);
                });
            for (auto& x : th) x.join();
            r0 += p.second;
        }
        std::memcpy(ptr, out.data(), bytes);
        int fd = ::open(cache.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { ssize_t w = ::write(fd, out.data(), bytes); (void)w; ::close(fd); }
        return addr;
    }
    uint64_t f32_vec(const std::string& name, size_t n) {
        auto f = bf16_file(name, n);
        auto [addr, ptr] = d_.alloc(n * 4);
        std::memcpy(ptr, f.data(), n * 4);
        return addr;
    }
    uint64_t bf16_vec(const std::string& name, size_t n) {
        auto b = read_file(o_.model_dir + "/dense/" + name + ".bin");
        if (b.size() != n * 2) throw std::runtime_error(name + ": unexpected size");
        auto [addr, ptr] = d_.alloc(n * 2);
        std::memcpy(ptr, b.data(), n * 2);
        return addr;
    }
    HcW load_hc(const std::string& p, bool combine = true) {
        HcW h{};
        h.norm = f32_vec(p + "hc_norm.weight", HCH);
        h.down = q4_matrix(p + "down", {{p + "input_mix_weight_down.weight", LOWR}}, HCH);
        h.up = q4_matrix(p + "up", {{p + "input_mix_weight_up.weight", HCH}}, LOWR);
        if (combine) h.inj = q4_matrix(p + "inj", {{p + "block_inject_weight.weight", HC}}, HCH);
        return h;
    }
    // Shared expert assembled as a PS5Q4 expert slot so it rides the routed-expert kernels as the 11th expert.
    uint64_t shared_slot(const std::string& p) {
        const qn::ExpertLayout<4> Lay;
        auto [addr, ptr] = d_.alloc(Lay.bytes, 4096);
        const std::string cache = o_.model_dir + "/dense_q4/" + p + "shared_slot.q4";
        struct stat st;
        if (::stat(cache.c_str(), &st) == 0 && size_t(st.st_size) == Lay.bytes && read_into(cache, ptr, Lay.bytes)) return addr;
        auto* s = static_cast<uint8_t*>(ptr);
        auto g = bf16_file(p + "shared_expert.gate_proj.weight", size_t(640) * HID);
        auto u = bf16_file(p + "shared_expert.up_proj.weight", size_t(640) * HID);
        auto dn = bf16_file(p + "shared_expert.down_proj.weight", size_t(HID) * 640);
        const size_t rg = qn::row_bytes<4>(HID), rd = qn::row_bytes<4>(640);
        for (int r = 0; r < 640; ++r) {
            qn::quantize_row_fast<4>(g.data() + size_t(r) * HID, HID, s + Lay.gate_off + r * rg, 24);
            qn::quantize_row_fast<4>(u.data() + size_t(r) * HID, HID, s + Lay.up_off + r * rg, 24);
        }
        for (uint32_t r = 0; r < HID; ++r) qn::quantize_row_fast<4>(dn.data() + size_t(r) * 640, 640, s + Lay.down_off + r * rd, 24);
        int fd = ::open(cache.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { ssize_t w = ::write(fd, s, Lay.bytes); (void)w; ::close(fd); }
        return addr;
    }

    // S19: QSA indexer weights + its per-position raw-key cache (fp16) and pooled block-key cache (fp32)
    void load_indexer(LayerW& w, const std::string& a, uint32_t l) {
        if (!idx_on_) return;
        w.iqk = q4_matrix(a + "iqk", {{a + "indexer.index_qk_proj.weight", 640}}, HID);
        w.iqn = f32_vec(a + "indexer.q_layernorm.weight", 128);
        w.ikn = f32_vec(a + "indexer.k_layernorm.weight", 128);
        // S30: raw keys are only read until their 4-token block is pooled (the batch, or a wide-prefill slice of <= 512):
        // a ring of kRkRing positions at the end of the recurrent state (snapshots / resume points carry it) instead of a
        // per-position cache: 256 B per token per attention layer (17% of the KV) back to the expert pool.
        rk_[l] = rk_cur_;
        rk_cur_ += uint64_t(kRkRing) * 256;
        pk_[l] = kv_alloc(64);  // pooled block keys fp16: 128 dims x 2 B per 4 positions (was fp32, 128 B per position)
    }
    void load_weights() {
        auto t_load = Clock::now();
        const std::string P = "model.language_model.";
        W_.resize(L + 1);  // [L] = the MTP layer (S16)
        gdn_state_.assign(L + 1, 0);
        conv_state_.assign(L + 1, 0);
        kc_.assign(L + 1, 0);
        vc_.assign(L + 1, 0);
        rk_.assign(L + 1, 0);
        pk_.assign(L + 1, 0);
        // All recurrent state in ONE contiguous region so snapshot/restore/reset is a single copy/fill.
        uint32_t n_gdn = 0;
        for (uint32_t l = 0; l < NL_; ++l) n_gdn += (l + 1) % 4 != 0;
        const uint64_t gdn_b = 48ull * 128 * 128 * 4, conv_b = 10240ull * 3 * 4, ple_b = uint64_t(HCH) * 9 * 4;
        rstate_core_ = n_gdn * (gdn_b + conv_b) + ple_b;
        uint32_t n_att = 1;  // + the MTP layer
        for (uint32_t l = 0; l < NL_; ++l) n_att += (l + 1) % 4 == 0;
        rstate_bytes_ = rstate_core_ + (idx_on_ ? uint64_t(n_att) * kRkRing * 256 : 0);  // S30 rk rings after the core
        rstate_ = d_.zalloc(rstate_bytes_).first;
        rk_cur_ = rstate_ + rstate_core_;
        // S16 verify checkpoints: the core only. The rings hold every position of the batch (a rejected draft's row is
        // rewritten when that position is computed again, as in the K/V cache); a checkpoint copy would roll them back.
        for (uint32_t j = 0; j < o_.spec_k; ++j) ck_.push_back(d_.zalloc(rstate_core_).first);
        snaps_.resize(o_.snapshots);  // host memory, allocated on first use
        uint64_t cur = rstate_;
        for (uint32_t l = 0; l < NL_; ++l) {
            const std::string p = P + "layers." + std::to_string(l) + ".";
            auto& w = W_[l];
            w.attn = (l + 1) % 4 == 0;
            w.hca = load_hc(p + "attn_hyper_connection.");
            w.hcm = load_hc(p + "mlp_hyper_connection.");
            if (!w.attn) {
                const std::string g = p + "linear_attn.";
                w.inproj = q4_matrix(g + "inproj", {{g + "in_proj_qkv.weight", 10240}, {g + "in_proj_z.weight", 6144}, {g + "in_proj_b.weight", 48}, {g + "in_proj_a.weight", 48}}, HID);
                w.conv = f32_vec(g + "conv1d.weight", 10240 * 4);
                w.alog = f32_vec(g + "A_log", 48);
                w.dtb = f32_vec(g + "dt_bias", 48);
                w.gnorm = f32_vec(g + "norm.weight", 128);
                w.outproj = q4_matrix(g + "out", {{g + "out_proj.weight", HID}}, 6144);
                gdn_state_[l] = cur; cur += gdn_b;
                conv_state_[l] = cur; cur += conv_b;
            } else {
                const std::string a = p + "self_attn.";
                w.qkv = q4_matrix(a + "qkv", {{a + "q_proj.weight", 12288}, {a + "k_proj.weight", 512}, {a + "v_proj.weight", 512}}, HID);
                w.qn = f32_vec(a + "q_norm.weight", 256);
                w.kn = f32_vec(a + "k_norm.weight", 256);
                w.oproj = q4_matrix(a + "o", {{a + "o_proj.weight", HID}}, 6144);
                kc_[l] = kv_alloc(2 * kv_row(false));
                vc_[l] = kv_alloc(2 * kv_row(true));
                load_indexer(w, a, l);
            }
            w.router = bf16_vec(p + "mlp.gate.weight", size_t(E) * HID);  // bf16 as in the checkpoint (lossless)
            w.sgate = q4_matrix(p + "sgate", {{p + "mlp.shared_expert_gate.weight", 1}}, HID);
            w.shared_slot = shared_slot(p + "mlp.");
            std::fprintf(stderr, "\rloaded layer %u/%u (%.0f s)", l + 1, NL_, secs(t_load, Clock::now()));
        }
        ple_state_ = cur;
        const std::string pl = P + "layers.1.ple.";
        use_ple_ = NL_ > PLE_LAYER;
        if (use_ple_) {
            ple_key_ = q4_matrix(pl + "key", {{pl + "key_proj.weight", HCH}}, HID);
            ple_val_ = q4_matrix(pl + "value", {{pl + "value_proj.weight", HID}}, HID);
            ple_nk_ = f32_vec(pl + "norm_key.weight", HCH);
            ple_nq_ = f32_vec(pl + "norm_query.weight", HCH);
            ple_nc_ = f32_vec(pl + "norm_conv.weight", HCH);
            ple_cw_ = f32_vec(pl + "conv1d.weight", HCH * 4);
            auto m = i64_file(pl + "ple_embedding.layer_multipliers");
            for (int i = 0; i < 3; ++i) ng_.mult[i] = m[i];
            ng_.vs = i64_file(pl + "ple_embedding.ngram_heads_vocab_sizes");
            ng_.off = i64_file(pl + "ple_embedding.ngram_heads_offsets");
            ng_.fd = ::open((o_.model_dir + "/ngram_bf16.bin").c_str(), O_RDONLY);
            if (ng_.fd < 0) throw std::runtime_error("n-gram table missing");
            ::posix_fadvise(ng_.fd, 0, 0, POSIX_FADV_RANDOM);
            if (o_.ngram_async) ngr_ok_ = io_uring_queue_init(16 * kMaxTB, &ngr_, 0) == 0;
        }
        if (o_.pred_dir.empty()) {  // S15 default: the trained predictor next to the model, if it was exported
            struct stat st;
            // S17: pred_s17 (refit on 11.7k tokens of EXL3-2.68 activations, held-out precision 0.698 -> 0.756) first
            if (::stat((o_.model_dir + "/pred_s17").c_str(), &st) == 0) o_.pred_dir = o_.model_dir + "/pred_s17";
            else if (::stat((o_.model_dir + "/pred_s15").c_str(), &st) == 0) o_.pred_dir = o_.model_dir + "/pred_s15";
        }
        if (o_.pred_dir == "none") o_.pred_dir.clear();
        if (!o_.pred_dir.empty() && o_.predict) {
            pred_w_.assign(o_.predict + 1, std::vector<uint64_t>(L, 0));
            uint32_t n = 0;
            for (uint32_t dd = 1; dd <= o_.predict; ++dd)
                for (uint32_t l = 0; l + dd < NL_; ++l) {
                    const std::string f = o_.pred_dir + "/pred_d" + std::to_string(dd) + "_l" + std::to_string(l) + ".bf16";
                    struct stat st;
                    if (::stat(f.c_str(), &st) != 0 || size_t(st.st_size) != size_t(E) * HID * 2) continue;
                    auto [addr, ptr] = d_.alloc(size_t(E) * HID * 2);
                    if (!read_into(f, ptr, size_t(E) * HID * 2)) throw std::runtime_error("predictor read " + f);
                    pred_w_[dd][l] = addr;
                    ++n;
                }
            std::fprintf(stderr, "trained predictor: %u matrices from %s\n", n, o_.pred_dir.c_str());
            // S15 live (300-token Q2 chat): router top-5 27.80 tok/s; trained top-5 28.21, top-7 28.98, top-10 28.89
            if (n && !o_.prefetch_max_set) o_.prefetch_max = 8;  // S17 with pred_s17: top-8 35.75 vs top-7 35.42 / top-10 35.64
        }
        if (full()) {
            mixer_ = load_hc(P + "hyper_connection_mixer.", false);
            lm_head_ = q4_matrix("lm_head", {{"lm_head.weight", VOCAB}}, HID);
        }
        if (o_.mtp) {  // S16: the MTP head = one full-attention layer + its own mixer; the trunk lm_head is shared
            const std::string p = "mtp.layers.0.", a = p + "self_attn.";
            auto& w = W_[L];
            w.attn = true;
            w.hca = load_hc(p + "attn_hyper_connection.");
            w.hcm = load_hc(p + "mlp_hyper_connection.");
            w.qkv = q4_matrix(a + "qkv", {{a + "q_proj.weight", 12288}, {a + "k_proj.weight", 512}, {a + "v_proj.weight", 512}}, HID);
            w.qn = f32_vec(a + "q_norm.weight", 256);
            w.kn = f32_vec(a + "k_norm.weight", 256);
            w.oproj = q4_matrix(a + "o", {{a + "o_proj.weight", HID}}, 6144);
            kc_[L] = kv_alloc(2 * kv_row(false));
            vc_[L] = kv_alloc(2 * kv_row(true));
            load_indexer(w, a, L);
            w.router = bf16_vec(p + "mlp.gate.weight", size_t(E) * HID);
            w.sgate = q4_matrix(p + "sgate", {{p + "mlp.shared_expert_gate.weight", 1}}, HID);
            w.shared_slot = shared_slot(p + "mlp.");
            mtp_mixer_ = load_hc("mtp.hyper_connection_mixer.", false);
            mtp_fce_ = q4_matrix("mtp.fc_embedding", {{"mtp.fc_embedding.weight", HID}}, HID);
            mtp_fch_ = q4_matrix("mtp.fc_hidden", {{"mtp.fc_hidden.weight", HID}}, HID);
            mtp_ne_ = f32_vec("mtp.pre_fc_norm_embedding.weight", HID);
            mtp_nh_ = f32_vec("mtp.pre_fc_norm_hidden.weight", HCH);
            mtp_ok_ = true;
            std::fprintf(stderr, "\nMTP head loaded (layer %u)", L);
        }
        if (o_.tok_route) {
            o_.tok_route = std::min<uint32_t>(o_.tok_route, K);
            trt_[0].assign(uint64_t(VOCAB) * o_.tok_route, 0xFFFF);
            if (o_.mtp) trt_[1].assign(uint64_t(VOCAB) * o_.tok_route, 0xFFFF);
        }
        emb_fd_ = ::open((o_.model_dir + "/dense/" + P + "embed_tokens.weight.bin").c_str(), O_RDONLY);
        if (emb_fd_ < 0) throw std::runtime_error("embedding missing");
        row_.resize(HID);
        std::fprintf(stderr, "\nweights resident: %.2f GiB in %zu chunks (%.2f GiB allocated, %.0f MiB in the VRAM carve), load %.0f s\n", double(d_.used) / double(1ull << 30), d_.chunks.size(), allocated_gib(), vram_gib() * 1024,
                     secs(t_load, Clock::now()));
    }

    void alloc_activations() {
        auto act = [&](uint64_t n) { return d_.zalloc(n * 4); };
        // S16: every per-token activation lives in a block; token t of a verify batch uses block t (stride act_stride_).
        // Block 0 is the single-token path.
        TB_ = 1 + o_.spec_k;
        if (o_.bits == 3 && o_.prefill_tb > TB_) TB_ = std::min<uint32_t>(o_.prefill_tb, 4);  // S20 prefill blocks
        attn_chunks_ = idx_on_ ? (2051 + 63) / 64 : (o_.max_ctx + 63) / 64;  // S19: the indexer bounds a query at 2051 tokens
        uint64_t off = 0;
        auto put = [&](uint64_t& field, uint64_t n) { field = off; off = (off + n * 4 + 255) & ~255ull; };
        Act o{};
        put(o.H, HCH); put(o.N, HCH); put(o.T, LOWR); put(o.S, HCH); put(o.Il, HC); put(o.I, HC); put(o.I2, HC); put(o.X, HID);
        put(o.P, 16480); put(o.C, 10240); put(o.Core, 6144); put(o.Y, HID); put(o.Qkv, 13312); put(o.Q, 6144); put(o.G, 6144);
        put(o.Ao, 6144); put(o.Part, uint64_t(24) * attn_chunks_ * 258); put(o.Key, HCH); put(o.Keyn, HCH); put(o.Qn, HCH);
        put(o.Val, HID); put(o.Gv, HCH); put(o.Gvn, HCH); put(o.R, E); put(o.Hm, TAB * 640); put(o.Ym, HID); put(o.Ym2, HID);
        if (o_.bits == 3) {  // EXL3: transformed inputs, pre-Hadamard gate/up, transformed down input, pre-Hadamard down
            put(o.ExXt, uint64_t(TAB) * 2 * HID); put(o.ExPre, uint64_t(TAB) * 2 * 640);
            put(o.ExHd, uint64_t(TAB) * 640); put(o.ExYd, uint64_t(TAB) * HID);
        }
        put(o.Psum, 64); put(o.R2, E);
        put(o.E, HID); put(o.Iqk, 640); put(o.Qi, 512); put(o.Sel, 2064); put(o.Par, 64); put(o.Scr, std::max<uint64_t>(64, o_.max_ctx / 4));
        act_stride_ = (off + 4095) & ~4095ull;
        auto [base, hp] = d_.zalloc(act_stride_ * TB_);
        act_base_ = base;
        act_host_ = static_cast<uint8_t*>(hp);
        act_off_ = o;
        A_ = act_at(0);
        pH_ = act_host_ + o.H; pE_ = act_host_ + o.E; pPar_ = act_host_ + o.Par;
        aSc_ = act(24 * 4096).first;
        if (o_.prefill_chunk >= 16 && TB_ >= 4 && o_.bits == 3) {  // S20 layer-major prefill: per-token residual carry
            o_.prefill_chunk &= ~3u;
            if (o_.prefill_chunk > o_.max_ctx) o_.prefill_chunk = o_.max_ctx & ~3u;
            const uint64_t cbytes = uint64_t(o_.prefill_chunk) * HCH * 4;
            if (o_.wide && ek_on() && ctx_.has_sparse && !ek_.empty() && o_.prefill_chunk > kCarryBase) {
                // S21 elastic carry: a sparse buffer over the whole chunk, the first kCarryBase tokens backed by its own memory,
                // the rest bound from retired expert slots while a long prompt is prefilled (carry_grow / carry_release)
                VkDeviceSize al = 0;
                uint32_t bits = 0;
                cek_.b = ctx_.create_sparse(cbytes, al, bits);
                if (al != ek_page_) throw std::runtime_error("elastic carry: sparse alignment differs from the KV buffers");
                ek_bits_ &= bits;
                cek_.per_pos = HCH * 4;
                cek_.pages = cek_base_ = (uint64_t(kCarryBase) * HCH * 4 + ek_page_ - 1) / ek_page_;
                // the base pages come out of a resident chunk's slack when its memory type allows it (the S20 carry was such a
                // sub-allocation: a separate allocation cost ~40 MiB of pool, measured), else from their own allocation
                VkDeviceMemory bm = VK_NULL_HANDLE;
                uint64_t boff = 0;
                for (const auto& ch : d_.chunks)
                    if (((bits >> ch.b.mem_type) & 1u) && ch.b.size - std::min(ch.b.size, (ch.top + ek_page_ - 1) / ek_page_ * ek_page_) >= cek_.pages * ek_page_) {
                        const uint64_t a = d_.alloc(cek_.pages * ek_page_, ek_page_).first;
                        for (const auto& c2 : d_.chunks)
                            if (a >= c2.b.addr && a < c2.b.addr + c2.b.size) { bm = c2.b.mem; boff = a - c2.b.addr; }
                        break;
                    }
                if (!bm) { uint32_t type = 0; guard_early(cek_.pages * ek_page_); cek_.base = bm = ctx_.alloc_raw(cek_.pages * ek_page_, bits, type); }
                VkSparseMemoryBind mb{0, cek_.pages * ek_page_, bm, boff, 0};
                ctx_.bind_sparse({VkSparseBufferMemoryBindInfo{cek_.b.buf, 1, &mb}});
                aCarry_ = cek_.b.addr;
                hCarry_ = nullptr;
                std::fprintf(stderr, "elastic prefill carry: %u tokens (%.0f MiB) reserved, %u backed (%.0f MiB)\n", o_.prefill_chunk,
                             double(cbytes) / 1048576.0, kCarryBase, double(cek_.pages * ek_page_) / 1048576.0);
            } else
                std::tie(aCarry_, hCarry_) = d_.zalloc(cbytes);
            std::tie(aLmG_, pLmG_) = d_.zalloc(256);
            if (o_.wide) alloc_wide();
            if (o_.wide && o_.wide_xm) alloc_xm();
        } else o_.prefill_chunk = 0;
        aZero_ = act(HID).first;
        // per-layer tables: rows (token t, layer l) = t * (L + 1) + l; layer L is the MTP layer (S16)
        const uint64_t rows = uint64_t(TB_) * (L + 1);
        std::tie(aMail_, pMail_) = act(rows * 64);
        std::tie(aPmail_, pPmail_) = act(uint64_t(L) * 64 * 4);  // predictions: [depth 0..3][layer] mailboxes
        mailA_ = aMail_; pmailA_ = aPmail_; pmail_host_ = static_cast<uint8_t*>(pPmail_);
        std::tie(aSl_, pSl_) = d_.alloc(rows * TAB * 8);
        std::tie(aSlA_, pSlA_) = d_.alloc(rows * TAB * 8);  // S15 overlap: resident experts (0 = skip)
        std::tie(aSlB_, pSlB_) = d_.alloc(rows * TAB * 8);  // S15 overlap: streamed experts (0 = skip)
        std::memset(pSlA_, 0, rows * TAB * 8);
        std::memset(pSlB_, 0, rows * TAB * 8);
        std::tie(aWt_, pWt_) = act(rows * TAB);
        if (o_.mtp) { std::tie(aEmb_, pEmb_) = act(HID); aEn_ = act(HID).first; aEf_ = act(HID).first; }
        if (o_.dump) std::tie(aDump_, pDump_) = act(uint64_t(L + 1) * HCH);
        if (o_.dump_moe) std::tie(aMoe_, pMoe_) = act(uint64_t(L) * HID);
        // Logits + argmax in HOST_CACHED memory: the host reads all 248,320 logits when sampling (a write-combined
        // mapping would make that a ~1 MB uncached read per token).
        lg_ = ctx_.create(uint64_t(VOCAB) * 4 * TB_, true, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        arg_ = ctx_.create(4096, true, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    // ---- S20 elastic KV ----
    // Every per-position cache (K, V, indexer raw keys rk, pooled keys pk) is a sparse buffer over max_ctx positions.
    // Positions [0, kv_base) are backed by their own memory at start; growth retires expert slots from the END of each
    // size class (LFU state of the pool is kept), cuts their bytes into chunk-aligned pages and binds them (Strata
    // --kv-grow / vAttention / kvcached: virtual range reserved, physical pages on demand). Addresses never move, so
    // the kernels are unchanged. ek_ensure() runs between tokens (GPU idle).
    struct EkBuf { vk::Buffer b; uint64_t per_pos = 0; uint64_t pages = 0; VkDeviceMemory base = VK_NULL_HANDLE; std::vector<std::pair<VkDeviceMemory, uint64_t>> ext; };
    bool ek_on() const { return o_.kv_base > 0 && o_.kv_base < o_.max_ctx; }
    uint64_t kv_alloc(uint64_t per_pos) {
        if (!ek_on() || !ctx_.has_sparse) {
            const uint64_t a = d_.zalloc(uint64_t(o_.max_ctx) * per_pos).first;
            kvb_.push_back({a, per_pos, VK_NULL_HANDLE});
            return a;
        }
        EkBuf e;
        VkDeviceSize al = 0;
        uint32_t bits = 0;
        e.b = ctx_.create_sparse(uint64_t(o_.max_ctx) * per_pos, al, bits);
        e.per_pos = per_pos;
        if (!ek_page_) { ek_page_ = al; ek_bits_ = bits; }
        if (al != ek_page_) throw std::runtime_error("elastic KV: sparse alignment differs between buffers");
        ek_bits_ &= bits;
        e.pages = (uint64_t(o_.kv_base) * per_pos + ek_page_ - 1) / ek_page_;
        uint32_t type = 0;
        guard_early(e.pages * ek_page_);
        e.base = ctx_.alloc_raw(e.pages * ek_page_, bits, type);
        ek_base_bytes_ += e.pages * ek_page_;
        VkSparseMemoryBind mb{0, e.pages * ek_page_, e.base, 0, 0};
        VkSparseBufferMemoryBindInfo bi{e.b.buf, 1, &mb};
        ctx_.bind_sparse({bi});
        ek_.push_back(std::move(e));
        ek_commit_ = o_.kv_base;
        kvb_.push_back({ek_.back().b.addr, per_pos, ek_.back().b.buf});
        return ek_.back().b.addr;
    }
    uint32_t ek_chunk_of(uint64_t g) const {  // slab chunk holding byte g
        uint32_t c = uint32_t(std::upper_bound(ex_chunk_start_.begin(), ex_chunk_start_.end(), g) - ex_chunk_start_.begin());
        return c ? c - 1 : 0;
    }
    // Pages from the retired region of class c down to slab offset lo (descending, chunk-aligned)
    void ek_harvest(int c, uint64_t lo) {
        uint64_t g = ek_hlo_[c];
        while (g > lo) {
            const uint32_t ch = ek_chunk_of(g - 1);
            const uint64_t cs = ex_chunk_start_[ch];
            const uint64_t top = (g - cs) / ek_page_ * ek_page_;
            if (top < ek_page_) { g = cs; continue; }
            const uint64_t q = cs + top - ek_page_;
            if (q < lo) break;
            ek_free_.push_back({sch_[ch].mem, q - cs});
            g = q;
        }
        ek_hlo_[c] = g;
    }
    bool ek_retire_one() {  // the class with the smaller retired fraction gives its last live slot
        auto& cc = st_->cache();
        const int nc = int(cc.n_classes());
        int best = -1;
        double bf = 2.0;
        for (int c = 0; c < nc; ++c) {
            const uint32_t b = cc.class_begin(c), e = cc.class_end(c);
            if (ek_rb_[c] <= b + (e - b) / 4) continue;  // never more than 3/4 of a class
            if (ek_rb_[c] <= b + kClassFloor) continue;  // S24: nor below what the wide prefill pins at once
            const double f = double(e - ek_rb_[c]) / double(e - b);
            if (f < bf) { bf = f; best = c; }
        }
        if (best < 0) return false;
        const SlotId s = ek_rb_[best] - 1;
        for (int tries = 0; !cc.retire(s); ++tries) {
            if (cc.pins(s) || tries > 100000) return false;
            st_->reap(true);
        }
        ek_rb_[best] = s;
        ++ek_retired_;
        ek_harvest(best, ex_off_[s]);
        return true;
    }
    void ek_init_pool() {  // after map_pool: retirement bookkeeping (slots per class, slab offsets)
        if (ek_.empty()) return;
        auto& cc = st_->cache();
        ek_grow_ok_ = !ex_off_.empty() && !sch_.empty();
        for (auto& ch : sch_) ek_grow_ok_ &= (ek_bits_ & (1u << ch.mem_type)) != 0;
        for (uint32_t c = 0; c < cc.n_classes() && c < 2; ++c) ek_rb_[c] = cc.class_end(int(c));
        if (ek_grow_ok_) ek_init_pool_offsets();
        std::fprintf(stderr, "elastic KV: %zu sparse caches, %u positions backed (%.0f MiB), page %llu KiB, growth from the expert pool %s\n",
                     ek_.size(), o_.kv_base, double(ek_base_bytes_) / 1048576.0, (unsigned long long)(ek_page_ >> 10),
                     ek_grow_ok_ ? "on" : "UNAVAILABLE (memory type)");
    }
    uint64_t slab_bytes_used() const { uint64_t t = 0; for (auto& ch : sch_) t += ch.size; return ex_chunk_start_.empty() ? t : ex_chunk_start_.back() + sch_.back().size; }
    // Back positions [0, pos_end) (+ headroom, 4096-position steps)
    void ek_ensure(uint64_t pos_end) {
        if (ek_.empty() || pos_end <= ek_commit_) return;
        if (!ek_grow_ok_) throw std::runtime_error("context full (elastic KV cannot grow: memory type)");
        const uint64_t target = std::min<uint64_t>(o_.max_ctx, (pos_end + 1024 + 4095) / 4096 * 4096);
        uint64_t need = 0;
        for (auto& e : ek_) need += (target * e.per_pos + ek_page_ - 1) / ek_page_ - e.pages;
        for (SlotId q : pinned_) st_->cache().unpin(q);  // GPU idle: the last layer's pins can go now
        pinned_.clear();
        auto t0 = Clock::now();
        const uint64_t r0 = ek_retired_;
        while (ek_free_.size() < need)
            if (!ek_retire_one()) throw std::runtime_error("context full (elastic KV: no more slots to retire)");
        std::vector<std::vector<VkSparseMemoryBind>> mbs(ek_.size());
        std::vector<VkSparseBufferMemoryBindInfo> bi;
        for (size_t i = 0; i < ek_.size(); ++i) {
            auto& e = ek_[i];
            const uint64_t want = (target * e.per_pos + ek_page_ - 1) / ek_page_;
            for (uint64_t p = e.pages; p < want; ++p) {
                const auto pg = ek_free_.back();
                ek_free_.pop_back();
                mbs[i].push_back({p * ek_page_, std::min<uint64_t>(ek_page_, e.b.size - p * ek_page_), pg.first, pg.second, 0});
                e.ext.push_back(pg);
            }
            e.pages = want;
            if (!mbs[i].empty()) bi.push_back({e.b.buf, uint32_t(mbs[i].size()), mbs[i].data()});
        }
        ctx_.bind_sparse(bi);
        ek_commit_ = uint32_t(target);
        n_slots_live_ = n_slots_ - uint32_t(ek_retired_);
        std::fprintf(stderr, "elastic KV: %llu positions backed (+%llu slots retired, %llu total = %.2f GiB of the pool), %.1f ms\n",
                     (unsigned long long)target, (unsigned long long)(ek_retired_ - r0), (unsigned long long)ek_retired_,
                     double(ek_retired_) * slot_bytes_ / double(1ull << 30), secs(t0, Clock::now()) * 1e3);
    }
    // Give the pages above max(kv_base, pos_keep) back; at kv_base every retired slot returns to the pool
    void ek_shrink(uint64_t pos_keep) {
        if (ek_.empty()) return;
        const uint64_t target = std::max<uint64_t>(o_.kv_base, std::min<uint64_t>(o_.max_ctx, (pos_keep + 1024 + 4095) / 4096 * 4096));
        if (target >= ek_commit_) return;
        std::vector<std::vector<VkSparseMemoryBind>> mbs(ek_.size());
        std::vector<VkSparseBufferMemoryBindInfo> bi;
        for (size_t i = 0; i < ek_.size(); ++i) {
            auto& e = ek_[i];
            const uint64_t base_pages = (uint64_t(o_.kv_base) * e.per_pos + ek_page_ - 1) / ek_page_;
            const uint64_t want = std::max(base_pages, (target * e.per_pos + ek_page_ - 1) / ek_page_);
            while (e.pages > want) {
                --e.pages;
                mbs[i].push_back({e.pages * ek_page_, std::min<uint64_t>(ek_page_, e.b.size - e.pages * ek_page_), VK_NULL_HANDLE, 0, 0});
                ek_free_.push_back(e.ext.back());
                e.ext.pop_back();
            }
            if (!mbs[i].empty()) bi.push_back({e.b.buf, uint32_t(mbs[i].size()), mbs[i].data()});
        }
        ctx_.bind_sparse(bi);
        ek_commit_ = uint32_t(target);
        if (kv_hist_.size() > target) kv_hist_.resize(target);  // snapshots past the kept range are invalid now
        if (target == o_.kv_base) ek_return_all();
    }
    // every extension page (KV and carry) unbound: the retired slots go back to the pool
    void ek_return_all() {
        if (!ek_retired_ || ek_commit_ != o_.kv_base || !cek_.ext.empty() || !xek_.ext.empty()) return;
        auto& cc = st_->cache();
        for (uint32_t c = 0; c < cc.n_classes() && c < 2; ++c) {
            for (SlotId s = ek_rb_[c]; s < cc.class_end(int(c)); ++s) cc.unretire(s);
            ek_rb_[c] = cc.class_end(int(c));
        }
        ek_free_.clear();
        ek_retired_ = 0;
        ek_init_pool_offsets();
        n_slots_live_ = n_slots_;
        std::fprintf(stderr, "elastic KV: back to %u positions, every slot returned to the pool\n", o_.kv_base);
    }
    // S21 elastic carry: back the residual carry for N tokens (pages from retired expert slots, GPU idle); S22: and the
    // [X | Y] rows of the chunk-wide expert-major MoE
    void carry_grow(uint32_t N) {
        const bool c = cek_.b.buf && (uint64_t(N) * HCH * 4 + ek_page_ - 1) / ek_page_ > cek_.pages;
        const bool x = xek_.b.buf && xm_on() && (uint64_t(N) * XYS + ek_page_ - 1) / ek_page_ > xek_.pages;
        if (!c && !x) return;
        if (!ek_grow_ok_) throw std::runtime_error("prefill carry cannot grow (memory type): use --prefill-chunk <= 1024");
        for (SlotId q : pinned_) st_->cache().unpin(q);
        pinned_.clear();
        auto t0 = Clock::now();
        const uint64_t r0 = ek_retired_;
        if (c) ek_grow_buf(cek_, uint64_t(N) * HCH * 4);
        if (x) ek_grow_buf(xek_, uint64_t(N) * XYS);
        n_slots_live_ = n_slots_ - uint32_t(ek_retired_);
        std::fprintf(stderr, "elastic carry: %u tokens backed (+%llu slots retired), %.1f ms\n", N, (unsigned long long)(ek_retired_ - r0),
                     secs(t0, Clock::now()) * 1e3);
    }
    void ek_grow_buf(EkBuf& e, uint64_t bytes) {
        const uint64_t want = (bytes + ek_page_ - 1) / ek_page_;
        if (want <= e.pages) return;
        while (ek_free_.size() < want - e.pages)
            if (!ek_retire_one()) throw std::runtime_error("prefill carry: no more expert slots to retire");
        std::vector<VkSparseMemoryBind> mb;
        for (uint64_t pg = e.pages; pg < want; ++pg) {
            const auto f = ek_free_.back();
            ek_free_.pop_back();
            mb.push_back({pg * ek_page_, std::min<uint64_t>(ek_page_, e.b.size - pg * ek_page_), f.first, f.second, 0});
            e.ext.push_back(f);
        }
        e.pages = want;
        ctx_.bind_sparse({VkSparseBufferMemoryBindInfo{e.b.buf, uint32_t(mb.size()), mb.data()}});
    }
    void ek_release_buf(EkBuf& e, uint64_t base) {
        if (e.ext.empty()) return;
        std::vector<VkSparseMemoryBind> mb;
        while (e.pages > base) {
            --e.pages;
            mb.push_back({e.pages * ek_page_, std::min<uint64_t>(ek_page_, e.b.size - e.pages * ek_page_), VK_NULL_HANDLE, 0, 0});
            ek_free_.push_back(e.ext.back());
            e.ext.pop_back();
        }
        ctx_.bind_sparse({VkSparseBufferMemoryBindInfo{e.b.buf, uint32_t(mb.size()), mb.data()}});
    }
    // ... and give those pages back (first token after a prefill, reset, restore)
    void carry_release() {
        if (cek_.ext.empty() && xek_.ext.empty()) return;
        ek_release_buf(cek_, cek_base_);
        ek_release_buf(xek_, xek_base_);
        ek_return_all();
    }
    // S22: back the first e.pages pages of a sparse buffer from a resident chunk's slack (or its own allocation)
    void ek_back_base(EkBuf& e, uint32_t bits) {
        VkDeviceMemory bm = VK_NULL_HANDLE;
        uint64_t boff = 0;
        for (const auto& ch : d_.chunks)
            if (((bits >> ch.b.mem_type) & 1u) && ch.b.size - std::min(ch.b.size, (ch.top + ek_page_ - 1) / ek_page_ * ek_page_) >= e.pages * ek_page_) {
                const uint64_t a = d_.alloc(e.pages * ek_page_, ek_page_).first;
                for (const auto& c2 : d_.chunks)
                    if (a >= c2.b.addr && a < c2.b.addr + c2.b.size) { bm = c2.b.mem; boff = a - c2.b.addr; }
                break;
            }
        if (!bm) { uint32_t type = 0; guard_early(e.pages * ek_page_); e.base = bm = ctx_.alloc_raw(e.pages * ek_page_, bits, type); }
        VkSparseMemoryBind mb{0, e.pages * ek_page_, bm, boff, 0};
        ctx_.bind_sparse({VkSparseBufferMemoryBindInfo{e.b.buf, 1, &mb}});
    }
    // S22: the chunk's [X | Y] rows (elastic when the carry is), I2 rows, the group tables
    void alloc_xm() {
        const uint64_t C = o_.prefill_chunk, xbytes = C * XYS;
        if (cek_.b.buf && C > kXyBase) {
            VkDeviceSize al = 0;
            uint32_t bits = 0;
            xek_.b = ctx_.create_sparse(xbytes, al, bits);
            if (al != ek_page_) throw std::runtime_error("elastic X/Y rows: sparse alignment differs from the KV buffers");
            ek_bits_ &= bits;
            xek_.per_pos = XYS;
            xek_.pages = xek_base_ = (uint64_t(kXyBase) * XYS + ek_page_ - 1) / ek_page_;
            ek_back_base(xek_, bits);
            aXY_ = xek_.b.addr;
        } else aXY_ = d_.zalloc(xbytes).first;
        aI2c_ = d_.zalloc(C * 16).first;
        const uint64_t NK = C * K, GP = uint64_t(o_.wide_slice) * K;
        gcap_e_ = NK / 16 + E + 64;
        uint64_t off = 0;
        auto put = [&](uint64_t& f, uint64_t bytes) { f = off; off = (off + bytes + 255) & ~255ull; };
        put(kGSl_, NK * 8); put(kGTk_, NK * 4); put(kGPl_, GP * 4); put(kGE64_, gcap_e_ * sizeof(LEnt)); put(kGE32_, gcap_e_ * sizeof(LEnt));
        put(kGEw_, gcap_e_ * sizeof(WEnt)); put(kGTe_, NK * sizeof(TEnt));
        gctl_ = ctx_.create_uncached(off);
        gctlp_ = static_cast<uint8_t*>(gctl_.map);
        std::memset(gctlp_, 0, off);
        auto* pl = reinterpret_cast<uint32_t*>(gctlp_ + kGPl_);
        for (uint64_t i = 0; i < GP; ++i) pl[i] = uint32_t(i);
        xc_ok_.assign(L + 1, 0);
        std::fprintf(stderr, "expert-major MoE: [X | Y] rows of %llu tokens (%.0f MiB, %s), group tables %.1f MiB\n", (unsigned long long)C,
                     double(xbytes) / 1048576.0, xek_.b.buf ? "elastic" : "resident", double(off) / 1048576.0);
    }
    void ek_init_pool_offsets() {
        auto& cc = st_->cache();
        for (uint32_t c = 0; c < cc.n_classes() && c < 2; ++c) {
            const SlotId last = cc.class_end(int(c)) - 1;
            ek_hlo_[c] = last + 1 < ex_off_.size() ? ex_off_[last + 1] : slab_bytes_used();
        }
    }
    uint32_t kv_backed() const { return ek_.empty() ? o_.max_ctx : ek_commit_; }
    // S20: bytes per (position, kv head) row of the K (v = false) / V cache
    uint64_t kv_row(bool v) const {
        if (o_.kvq == 2) return 132;  // k4v4: 32 words of codes + fp32 step (attn_prep.comp put4)
        if (o_.kvq) return (v && o_.kvq == 4) ? 144 : 272;
        return 256ull * (o_.kv16 ? 2 : 4);
    }
    std::string footprint_path() const {
        char b[160];
        std::snprintf(b, sizeof b, "/dense_q4/footprint_b%d_m%d_k%u_c%u_s%u%s%s.txt", o_.bits, int(o_.mtp), o_.spec_k, ek_on() ? o_.kv_base : o_.max_ctx, o_.snapshots,
                      o_.kvq ? ("_kq" + std::to_string(o_.kvq)).c_str() : "",
                      (o_.bits == 3 && o_.prefill_tb > 1 + o_.spec_k) ? ("_p" + std::to_string(std::min<uint32_t>(o_.prefill_tb, 4)) + "_lm" +
                      std::to_string(o_.prefill_chunk)).c_str() : "");
        return o_.model_dir + b;
    }
    // S22: amdgpu's TTM page pool can hold GTT pages freed earlier (213 MiB measured after a run, up to 4 MiB orders) that
    // MemAvailable does not count and drop_caches does not drain. Each read of debugfs ttm/page_pool_shrink frees one
    // shrinker batch: read until the pool is empty. (The low first run after a console BUILD is not this: that memory
    // trickles back over minutes, ~4 MiB/s, and only affects dev runs.)
    void ttm_drain(const char* when) {
        uint64_t first = 0, last = 0;
        for (int i = 0; i < 4096; ++i) {  // "node: count/scanned": count = pages in the pool before this batch
            FILE* f = std::fopen("/sys/kernel/debug/ttm/page_pool_shrink", "r");
            if (!f) return;
            unsigned long a = 0, b = 0;
            const int r = std::fscanf(f, "%*d: %lu/%lu", &a, &b);
            std::fclose(f);
            if (r < 1) break;
            if (a > (1ul << 60)) a = 0;  // SHRINK_EMPTY (-2) once the pool is empty
            if (i == 0) first = a;
            last = a;
            if (a == 0) break;
        }
        const uint64_t freed = first - std::min(first, last);
        if (freed >= 256) std::fprintf(stderr, "ttm pool (%s): %.0f MiB of cached GTT pages released\n", when, double(freed) * 4096.0 / 1048576.0);
    }
    void early_reserve() {
        ttm_drain("early");
        ::malloc_trim(0);
        ::sync();
        if (FILE* f = std::fopen("/proc/sys/vm/drop_caches", "w")) { std::fputs("3\n", f); std::fclose(f); }
        if (FILE* f = std::fopen("/proc/sys/vm/compact_memory", "w")) { std::fputs("1\n", f); std::fclose(f); }
        {  // leaked free hugepages (an engine killed before the S15 handler) go back first
            const long tot = hugepages(), fr = long(meminfo_kib("HugePages_Free:")), rs = long(meminfo_kib("HugePages_Rsvd:"));
            const long in_use = std::max(0L, tot - fr + rs);
            if (tot > in_use) set_hugepages(in_use);
        }
        avail0_kib_ = meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:");
        // S22: the file holds "largest last" (KiB). The largest-ever value under-reserved by ~1.1 GiB (2722 vs 1584 MiB
        // measured on the next run): every 2 MiB block not reserved here must come from memory the dense weights' 4 KiB
        // pages have fragmented (measured: 147-362 hugepages = 3-7% of the slots lost). guard_early hands pages back if a
        // load allocation runs short, so the last measurement + 64 MiB is used (never more than the largest); the measurement counts the pages handed back.
        double fp = 0, fpl = 0;
        if (FILE* f = std::fopen(footprint_path().c_str(), "r")) {
            const int r = std::fscanf(f, "%lf %lf", &fp, &fpl);
            if (r < 1) fp = 0;
            if (r < 2) fpl = fp;
            std::fclose(f);
        }
        if (fp <= 0) { std::fprintf(stderr, "hugetlb early: no footprint for this configuration yet (measured this run)\n"); return; }
        const double fpmax = fp;
        fp = std::min(fpmax, fpl + 64.0 * 1024);
        uint32_t n_gdn = 0;
        for (uint32_t l = 0; l < NL_; ++l) n_gdn += (l + 1) % 4 != 0;
        const double rst = double(n_gdn) * (48.0 * 128 * 128 * 4 + 10240.0 * 3 * 4) + double(HCH) * 9 * 4;
        const double cma = meminfo_kib("CmaFree:"), snaps = double(o_.snapshots) * rst / 1024.0;
        const double kib = avail0_kib_ - fp - double(o_.pool_reserve_mib) * 1024 - std::max(0.0, snaps - cma) - 64.0 * 1024;  // 64 MiB margin
        long n = long(kib / (2048.0 - (o_.hp_vmemmap ? 28.0 : 0.0)));  // vm.hugetlb_optimize_vmemmap returns 28 KiB per page
        if (o_.pool_gib > 0) n = std::min(n, long(o_.pool_gib * 1024.0 * 1024.0 / 2048.0) + 8);
        if (n < 256) return;
        prev_hugepages_ = hugepages();
        g_hugepages_restore = prev_hugepages_;
        const long got = set_hugepages(prev_hugepages_ + n) - prev_hugepages_;
        early_hp_ = std::max(0L, got);
        avail_e_kib_ = meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:");  // after the reservation (incl. returned vmemmap)
        std::fprintf(stderr, "hugetlb early: %ld of %ld x 2 MiB reserved before the weights (footprint %.0f MiB: last %.0f + 64, largest %.0f)\n",
                     early_hp_, n, fp / 1024, fpl / 1024, fpmax / 1024);
    }
    // S18: an allocation of `bytes` is about to come from ordinary memory while early hugepages are held: if it would leave
    // less than 192 MiB, give hugepages back first (the OOM killer would otherwise take the engine mid-load)
    void guard_early(uint64_t bytes) {
        if (early_hp_ <= 0) return;
        const double free_kib = meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:"), need_kib = double(bytes) / 1024.0 + 192.0 * 1024;
        if (free_kib >= need_kib) return;
        const long rel = std::min(early_hp_, long((need_kib - free_kib) / 2048.0) + 1);
        set_hugepages(hugepages() - rel);
        early_hp_ -= rel;
        early_back_ += rel;
        std::fprintf(stderr, "hugetlb early: memory short during load, %ld hugepages handed back\n", rel);
    }
    void map_pool() {
        ttm_drain("pool");
        ::malloc_trim(0);  // MemAvailable must not count the loader's freed buffers as used
        if (o_.slab_hugetlb) {  // S17: clean page cache + reclaimable slab (TTM page pool shrinker) out, then compact
            ::sync();
            if (FILE* f = std::fopen("/proc/sys/vm/drop_caches", "w")) { std::fputs("3\n", f); std::fclose(f); }
            if (FILE* f = std::fopen("/proc/sys/vm/compact_memory", "w")) { std::fputs("1\n", f); std::fclose(f); }
        }
        if (o_.slab_hugetlb) {
            // S15: free hugepages nobody has reserved are a leak (an engine killed by a signal before the S15 handler):
            // release them BEFORE sizing the pool so MemAvailable counts them; only in-use pages are restored at exit.
            const long tot = hugepages(), fr = long(meminfo_kib("HugePages_Free:")), rs = long(meminfo_kib("HugePages_Rsvd:"));
            const long in_use = std::max(0L, tot - fr + rs);
            if (tot > in_use + early_hp_) {
                std::fprintf(stderr, "hugetlb: released %ld leaked free hugepages (%ld in use by others)\n", tot - in_use - early_hp_, in_use);
                set_hugepages(in_use + early_hp_);
            }
            // S18: this configuration's memory footprint (weights, activations, KV, state) for the next early reservation
            if (avail0_kib_ > 0) {
                // pages guard_early handed back raised MemAvailable by ~2020 KiB each: they were consumed by the load too
                const double fp = (early_hp_ || early_back_ ? avail_e_kib_ : avail0_kib_) - (meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:"))
                                + double(early_back_) * (2048.0 - (o_.hp_vmemmap ? 28.0 : 0.0));
                double old = 0;
                if (FILE* f = std::fopen(footprint_path().c_str(), "r")) { if (std::fscanf(f, "%lf", &old) != 1) old = 0; std::fclose(f); }
                std::fprintf(stderr, "hugetlb footprint: measured %.0f MiB (file %.0f), early %ld pages, MemAvailable-CmaFree %.0f -> %.0f -> %.0f MiB%c",
                             fp / 1024, old / 1024, early_hp_, avail0_kib_ / 1024, avail_e_kib_ / 1024,
                             (meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:")) / 1024, 10);
                if (fp > 0)
                    if (FILE* f = std::fopen(footprint_path().c_str(), "w")) { std::fprintf(f, "%.0f %.0f\n", std::max(fp, old), fp); std::fclose(f); }
            }
        }
        slot_bytes_ = o_.bits == 3 ? *std::max_element(ex_slot_.begin(), ex_slot_.end())  // v0: one pool, largest slot
                                   : uint32_t(o_.bits == 4 ? qn::ExpertLayout<4>().bytes : qn::ExpertLayout<2>().bytes);
        spc_ = uint32_t((1ull << 30) / slot_bytes_);
        // Hard ceiling, also for an explicit --pool-gib (S14: a 9.9 GiB pool, 130 MiB past the edge, left the console
        // unreachable: the killed engine's pinned pages could not be released fast enough and userspace starved).
        const double cma0 = meminfo_kib("CmaFree:"), snaps0 = double(o_.snapshots) * double(rstate_bytes_) / 1024.0;
        const double ceiling_gib = (meminfo_kib("MemAvailable:") - cma0 - double(o_.pool_floor_mib) * 1024 - std::max(0.0, snaps0 - cma0) + double(early_hp_) * 2048.0) / (1024.0 * 1024.0);
        if (o_.pool_gib > 0 && o_.pool_gib > ceiling_gib && !o_.force_pool) {
            std::fprintf(stderr, "pool %.2f GiB exceeds what can be pinned safely (%.2f GiB): clamped (--force-pool overrides)\n", o_.pool_gib, ceiling_gib);
            o_.pool_gib = ceiling_gib;
        }
        o_.pool_auto_ = o_.pool_gib <= 0;
        if (o_.pool_gib <= 0) {
            // auto (S14): what the kernel can still give us AFTER the weights are resident. CmaFree is excluded:
            // udmabuf long-term-pins its pages and pinned pages may not live in CMA (a 9.5 GiB pool OOM-killed
            // NetworkManager/sshd although MemAvailable said it fit).
            // host snapshots are allocated later from movable memory: CMA-typed pages cover them first
            const double cma = meminfo_kib("CmaFree:"), snaps = double(o_.snapshots) * double(rstate_bytes_) / 1024.0;
            const double avail = meminfo_kib("MemAvailable:") - cma - double(o_.pool_reserve_mib) * 1024 - std::max(0.0, snaps - cma) + double(early_hp_) * 2048.0;
            o_.pool_gib = std::max(0.5, avail / (1024.0 * 1024.0));
            std::fprintf(stderr, "pool auto: %.2f GiB (MemAvailable - CmaFree - %u MiB)\n", o_.pool_gib, o_.pool_reserve_mib);
        }
        uint64_t want_slots = uint64_t(o_.pool_gib * double(1ull << 30)) / slot_bytes_;
        if (o_.slab_hugetlb) {
            // 2 MiB hugetlb slab (S14b): with vm.hugetlb_optimize_vmemmap=1 every hugepage returns 28 KiB of struct-page
            // memmap (~150 MiB on a 10.6 GiB pool) and the slab needs no 4 KiB page tables (~18 MiB). Reserve in two
            // steps so an auto-sized pool also absorbs the memmap the first step gave back.
            const uint64_t HP = 2ull << 20;
            if (!early_hp_) prev_hugepages_ = hugepages();  // S18: else set by early_reserve (our early pages are not "previous")
            g_hugepages_restore = prev_hugepages_;
            const bool autosz = o_.pool_auto_;
            long need = long((want_slots * slot_bytes_ + HP - 1) / HP);
            long got = set_hugepages(need);
            // S17: the dense weights (2.7 GiB of scattered 4 KiB GTT pages) were just allocated: the buddy lists can be
            // short of 2 MiB blocks (measured: 4,720 of 5,450 -> the whole slab fell back to 4 KiB pages, -3.4% slots).
            // Compact and retry a few times before falling back.
            for (int tries = 0; got < need && tries < 4; ++tries) {
                if (FILE* f = std::fopen("/proc/sys/vm/compact_memory", "w")) { std::fputs("1\n", f); std::fclose(f); }
                const long g2 = set_hugepages(need);
                std::fprintf(stderr, "hugetlb: compaction pass %d: %ld -> %ld of %ld hugepages\n", tries + 1, got, g2, need);
                if (g2 <= got && tries > 0) { got = g2; break; }
                got = g2;
            }
            if (autosz && got == need) {
                const double extra_kib = meminfo_kib("MemAvailable:") - meminfo_kib("CmaFree:") - double(o_.pool_reserve_mib) * 1024
                                       - std::max(0.0, double(o_.snapshots) * double(rstate_bytes_) / 1024.0 - meminfo_kib("CmaFree:"));
                if (extra_kib > 64.0 * 1024) { need += long(extra_kib * 1024.0 / double(HP)); got = set_hugepages(need); }
            }
            const long need0 = long((want_slots * slot_bytes_ + HP - 1) / HP);
            if (got < need0 * 9 / 10) {  // fragmented (long uptime): fall back to the 4 KiB slab at the computed size
                std::fprintf(stderr, "hugetlb slab: only %ld of %ld hugepages available, using 4 KiB pages\n", got, need0);
                set_hugepages(prev_hugepages_);
                o_.slab_hugetlb = false;
            } else {
                const uint64_t fit = uint64_t(got) * HP / slot_bytes_;
                if (autosz) want_slots = fit; else want_slots = std::min(want_slots, fit);
            }
            o_.pool_gib = double(want_slots * slot_bytes_) / double(1ull << 30);
            if (o_.slab_hugetlb) std::fprintf(stderr, "hugetlb slab: %ld x 2 MiB reserved -> %.2f GiB pool (%llu slots)\n", got, o_.pool_gib, (unsigned long long)want_slots);
            if (want_slots == 0) throw std::runtime_error("no hugepages could be reserved");
        }
        StreamerConfig cfg;
        uint64_t slab_bytes = want_slots * slot_bytes_;
        std::vector<size_t> chunk_lens;
        if (o_.bits == 3 && o_.size_classes) {
            // S15b: two slot classes (K=3 layers / K=3.5 layers) instead of the largest slot for every expert.
            // Class share of the slots = share of layers (a layer's experts only live in its class's LRU).
            uint32_t b[2] = {0, 0}, nl[2] = {0, 0};
            std::vector<uint8_t> lc(L + 1, 0);
            const uint32_t big = *std::max_element(ex_slot_.begin(), ex_slot_.begin() + L);  // class 1 = the larger slot size
            for (uint32_t l = 0; l < L; ++l) { lc[l] = ex_slot_[l] == big; b[lc[l]] = std::max(b[lc[l]], ex_slot_[l]); ++nl[lc[l]]; }
            if (ex_slot_[L]) {  // S18: the MTP layer joins the class of its slot size (2.68 pack + 3.18 MTP experts: K=3 = class 1)
                lc[L] = ex_slot_[L] == big;
                if (o_.mtp_class_share) ++nl[lc[L]];
                if (ex_slot_[L] > b[lc[L]]) throw std::runtime_error("MTP expert slot larger than its size class");
            }
            if (nl[0] && nl[1]) {
                const double f1 = double(nl[1]) / double(nl[0] + nl[1]);
                uint64_t n = uint64_t(double(slab_bytes) / (f1 * b[1] + (1 - f1) * b[0]));
                uint64_t n1 = uint64_t(std::llround(f1 * double(n))), n0 = n - n1;
                while (n0 * b[0] + n1 * b[1] > slab_bytes) { if (n0 > n1 * 5) --n0; else --n1; }
                // S24: the wide prefill pins the experts of two consecutive layers of a chunk (<= E each, same class for
                // layers 0-3 / 44-47): a class below 2 E + slack saturates and the prefill can never finish
                const uint64_t floor = kClassFloor;
                if (n1 < floor && slab_bytes > floor * (b[0] + b[1])) { n1 = floor; n0 = (slab_bytes - n1 * b[1]) / b[0]; }
                if (n0 < floor && slab_bytes > floor * (b[0] + b[1])) { n0 = floor; n1 = (slab_bytes - n0 * b[0]) / b[1]; }
                uint64_t off = 0;
                for (int c = 0; c < 2; ++c) {
                    const uint64_t per = (1ull << 30) / b[c], cnt = c ? n1 : n0;
                    for (uint64_t s = 0; s < cnt; s += per) {
                        const uint64_t k = std::min(per, cnt - s);
                        cfg.chunks.push_back({off, k * b[c]});
                        chunk_lens.push_back(size_t(k * b[c]));
                        for (uint64_t j = 0; j < k; ++j) {
                            cfg.slot_off.push_back(off + j * b[c]);
                            cfg.slot_chunk.push_back(uint16_t(cfg.chunks.size() - 1));
                        }
                        off += k * b[c];
                    }
                }
                cfg.layer_class = lc;
                cfg.n_class0 = uint32_t(n0);
                std::fprintf(stderr, "EXL3 size classes: %llu x %.2f MB (K=3 layers) + %llu x %.2f MB (K=3.5 layers) = %llu slots (uniform: %llu)\n",
                             (unsigned long long)n0, b[0] / 1e6, (unsigned long long)n1, b[1] / 1e6, (unsigned long long)(n0 + n1),
                             (unsigned long long)want_slots);
                slab_bytes = off;
            }
        }
        slab_ = platform::map_udmabuf(slab_bytes, uint64_t(spc_) * slot_bytes_, o_.slab_hugetlb, chunk_lens.empty() ? nullptr : &chunk_lens);
        cfg.pack_path = o_.model_dir + "/" + o_.pack_name;
        cfg.primary_mbps = o_.pack_mbps;
        for (const std::string& m : o_.pack_mirrors) {
            const size_t c = m.rfind(':');
            const bool has_bw = c != std::string::npos && c + 1 < m.size() && m.find_first_not_of("0123456789.", c + 1) == std::string::npos;
            cfg.mirrors.push_back({has_bw ? m.substr(0, c) : m, has_bw ? std::stod(m.substr(c + 1)) : 880.0});
            std::fprintf(stderr, "pack mirror: %s (%.0f MB/s)\n", cfg.mirrors.back().path.c_str(), cfg.mirrors.back().mbps);
        }
        cfg.layers = L + 1;
        cfg.experts = E;
        cfg.slot_bytes = slot_bytes_;
        cfg.iopoll = false;
        cfg.policy = o_.lfu_half_life > 0 ? Policy::LFU : Policy::LRU;
        cfg.split = o_.read_split;
        if (o_.bits == 3) {  // expert (l, e) at base[l] + e * slot[l]; layer 48 = MTP (S16, if appended to the pack)
            cfg.layer_base.assign(ex_base_.begin(), ex_base_.end());
            cfg.layer_bytes.assign(ex_slot_.begin(), ex_slot_.end());
            if (!ex_slot_[L]) { cfg.layer_base[L] = 0; cfg.layer_bytes[L] = slot_bytes_; }
        }
        st_ = std::make_unique<ExpertStreamer>(cfg, slab_.ptr, slab_bytes);
        if (o_.lfu_half_life > 0) st_->cache().lfu_set_half_life(o_.lfu_half_life);
        if (!o_.tier_dev.empty()) {  // S31
            try {
                st_->enable_tier(o_.tier_dev, o_.tier_mib << 20, o_.tier_threads, o_.tier_window, o_.tier_min_score);
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "lowmem tier OFF: %s\n", ex.what());
            }
        }
        if (o_.lfu_half_life > 0 && o_.mtp && o_.mtp_lfu_w != 1.0) st_->cache().lfu_set_layer_weight(L, o_.mtp_lfu_w);
        n_slots_ = st_->cache().n_slots();
        for (size_t c = 0; c < slab_.dmabufs.size(); ++c) {
            const uint64_t len = chunk_lens.empty() ? std::min<uint64_t>(slab_.chunk_bytes, slab_.bytes - c * slab_.chunk_bytes)
                                                    : (c + 1 == chunk_lens.size() ? slab_.bytes - cfg.chunks[c].first : chunk_lens[c]);
            sch_.push_back(ctx_.import_dmabuf(slab_.dmabufs[c], len));
        }
        if (!cfg.slot_off.empty()) {
            ex_off_ = cfg.slot_off;
            ex_chunk_ = cfg.slot_chunk;
            for (auto& ch : cfg.chunks) ex_chunk_start_.push_back(ch.first);
        }
        auto* slots_tab = static_cast<uint64_t*>(pSl_);
        for (uint32_t t = 0; t < TB_; ++t) {
            for (uint32_t l = 0; l < NL_; ++l) slots_tab[(uint64_t(t) * (L + 1) + l) * TAB + K] = W_[l].shared_slot;
            if (mtp_ok_) slots_tab[(uint64_t(t) * (L + 1) + L) * TAB + K] = W_[L].shared_slot;
        }
    }
    static long hugepages() {
        long n = -1;
        if (FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) { if (std::fscanf(f, "%ld", &n) != 1) n = -1; std::fclose(f); }
        return n;
    }
    static long set_hugepages(long n) {
        if (FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "w")) { std::fprintf(f, "%ld\n", n); std::fclose(f); }
        return hugepages();
    }
    long prev_hugepages_ = -1;
    long early_hp_ = 0;        // S18: hugepages reserved before the weights
    long early_back_ = 0;      // S22: of those, handed back by guard_early during the load
    double avail0_kib_ = 0, avail_e_kib_ = 0;
    static double meminfo_kib(const char* key) {
        FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return 0;
        char k[64];
        double v = 0, out = 0;
        while (std::fscanf(f, "%63s %lf kB", k, &v) == 2) if (!std::strcmp(k, key)) { out = v; break; }
        std::fclose(f);
        return out;
    }
    uint64_t slot_addr(SlotId s) const {
        if (!ex_off_.empty()) return sch_[ex_chunk_[s]].addr + (ex_off_[s] - ex_chunk_start_[ex_chunk_[s]]);
        return sch_[s / spc_].addr + uint64_t(s % spc_) * slot_bytes_;
    }
    // S15b: experts_exl3.txt (written by tools/s15/pack_exl3.py): "layer K2 slot_bytes base span_bytes" per layer
    void load_exl3_geometry() {
        // S17: geometry next to the pack: <pack stem>.txt (experts_exl3.pack -> experts_exl3.txt)
        std::string gname = o_.pack_name;
        if (gname.size() > 5 && gname.compare(gname.size() - 5, 5, ".pack") == 0) gname.resize(gname.size() - 5);
        gname = o_.model_dir + "/" + gname + ".txt";
        std::FILE* f = std::fopen(gname.c_str(), "r");
        if (!f) throw std::runtime_error(gname + " missing (run tools/s15/pack_exl3.py or tools/s17/fetch_pack_exl3.py)");
        ex_k2_.assign(L + 1, 0); ex_slot_.assign(L + 1, 0); ex_base_.assign(L + 1, 0);  // [L] = MTP (optional)
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#') continue;
            unsigned l, k2, sb; unsigned long long b, sp;
            if (std::sscanf(line, "%u %u %u %llu %llu", &l, &k2, &sb, &b, &sp) == 5 && l <= L) {
                if (k2 < 5 || k2 > 7 || sb % 4096 || b % 4096) throw std::runtime_error("experts_exl3.txt: bad layer line");
                ex_k2_[l] = k2; ex_slot_[l] = sb; ex_base_[l] = b;
            }
        }
        std::fclose(f);
        for (uint32_t l = 0; l < L; ++l) if (!ex_k2_[l]) throw std::runtime_error("experts_exl3.txt: layer missing");
        if (o_.mtp && !ex_k2_[L]) throw std::runtime_error("experts_exl3.txt: no MTP layer 48 (run tools/s15/pack_exl3_mtp.py)");
        if (!o_.mtp) { ex_k2_[L] = 0; ex_slot_[L] = 0; ex_base_[L] = 0; }
    }
    // slot byte offsets inside an EXL3 expert span (exl3_common.glsl)
    static uint32_t ex_tr_down() { return 6404; }
    static uint32_t ex_tr_gate(uint32_t k2) { return 2 * 6404 + 102400 * k2; }
    static uint32_t ex_tr_up(uint32_t k2) { return 3 * 6404 + 2 * 102400 * k2; }

    void load_pipelines() {
        const std::string& sd = o_.shader_dir;
        pMv_ = ctx_.pipeline(sd + "/matvec.spv", sizeof(PcMatvec));
        pXr2560_ = ctx_.pipeline(sd + "/matvec_xr2560.spv", sizeof(PcMatvec));
        pLds6144_ = ctx_.pipeline(sd + "/matvec_lds6144.spv", sizeof(PcMatvec));
        if (o_.mv_xrs) {  // S19
            pXs6_[1] = ctx_.pipeline(sd + "/matvec_xrs6144_1.spv", sizeof(PcMatvecB));
            pXs25_[1] = ctx_.pipeline(sd + "/matvec_xrs2560_1.spv", sizeof(PcMatvecB));
        }
        pHn_ = ctx_.pipeline(sd + "/hc_norm.spv", sizeof(PcNorm), 0);
        if (o_.hc_fast) {  // S19
            pHn25_ = ctx_.pipeline(sd + "/hc_norm2560.spv", sizeof(PcNorm), 0);
            pLdx_ = ctx_.pipeline(sd + "/matvec_ldsx10240.spv", sizeof(PcMatvecX));
            pHu4_ = ctx_.pipeline(sd + "/hc_upmix4.spv", sizeof(PcHcUpMix));
        }
        pMx_ = ctx_.pipeline(sd + "/hc_mix.spv", sizeof(PcMix), 0);
        pPo_ = ctx_.pipeline(sd + "/hc_post.spv", sizeof(PcPost), 0);
        pHd_ = ctx_.pipeline(sd + "/hc_down.spv", sizeof(PcHcDown));  // wave64 required (rows stride 64 lanes)
        pHu_ = ctx_.pipeline(sd + "/hc_upmix.spv", sizeof(PcHcUpMix));
        if (o_.hc_mode == 3) {
            pSq_ = ctx_.pipeline(sd + "/hc_sumsq.spv", sizeof(PcSumsq));
            pHd2_ = ctx_.pipeline(sd + "/hc_down2.spv", sizeof(PcHcDown2));
            pHu2_ = ctx_.pipeline(sd + "/hc_upmix2.spv", sizeof(PcHcUpMix2));
        }
        pCv_ = ctx_.pipeline(sd + "/gdn_conv.spv", sizeof(PcConv), 0);
        pGd_ = ctx_.pipeline(sd + "/gdn_step.spv", sizeof(PcGdn), 0);
        pPr_ = ctx_.pipeline(sd + (o_.kvq ? "/attn_prep_kq" + std::to_string(o_.kvq) + ".spv" : o_.kv16 ? "/attn_prep_k16.spv" : "/attn_prep.spv"),
                             sizeof(PcPrep), 0);
        pAt_ = ctx_.pipeline(sd + "/attn.spv", sizeof(PcAttn), 0);
        {   // S19: fp16 KV and/or the QSA token list
            const std::string v = std::string(o_.kv16 ? "_k16" : "") + (idx_on_ ? "s" : "");
            pAs_ = ctx_.pipeline(sd + "/attn_split" + (v.empty() ? "" : (o_.kv16 ? v : "_s")) + ".spv", sizeof(PcAttnSplit));
            pAc_ = ctx_.pipeline(sd + (idx_on_ ? "/attn_combine_s.spv" : "/attn_combine.spv"), sizeof(PcAttnComb), 0);
            if (o_.kvq) {  // S20
                pAs_ = ctx_.pipeline(sd + "/attn_split_kq" + std::to_string(o_.kvq) + (idx_on_ ? "s" : "") + ".spv", sizeof(PcAttnSplit));
                if (o_.kvq == 4 || o_.kvq == 2) pAc_ = ctx_.pipeline(sd + (idx_on_ ? "/attn_combine_vrs.spv" : "/attn_combine_vr.spv"), sizeof(PcAttnComb), 0);
                std::fprintf(stderr, "KV cache: %s (%llu + %llu B per position per kv head)\n", o_.kvq == 2 ? "k4v4 (rotated 4-bit K and V)" : o_.kvq == 4 ? "k8v4 (int8 K, rotated int4 V)" : "int8",
                             (unsigned long long)kv_row(false), (unsigned long long)kv_row(true));
            }
            if (idx_on_) {
                pIp_ = ctx_.pipeline(sd + "/idx_prep.spv", sizeof(PcIdxPrep), 0);
                // S30: block scores over the whole GPU (idx_score), then the exact selection on them (idx_select_ps).
                // PS5LM_IDX_OLD=1: the S19 single-workgroup path (scores inside idx_select).
                idx_ps_ = !std::getenv("PS5LM_IDX_OLD");
                // S33: the selection by radix digits (3 passes over the scores instead of 31); PS5LM_SEL_BISECT=1 keeps the bisection.
                pIs_ = ctx_.pipeline(sd + (idx_ps_ ? (std::getenv("PS5LM_SEL_BISECT") ? "/idx_select_ps.spv" : "/idx_select_psr.spv") : "/idx_select.spv"), sizeof(PcIdxSel), 0);
                if (std::getenv("PS5LM_SELSTAT")) {  // S32 tiering probe
                    pSst_ = ctx_.pipeline(sd + "/sel_stat.spv", sizeof(PcSelStat));
                    std::tie(sst_last_, sst_lastp_) = d_.zalloc(uint64_t(L / 4) * (o_.max_ctx / 4) * 4);
                    std::tie(sst_cnt_, sst_cntp_) = d_.zalloc(uint64_t(L / 4) * 64);
                    sst_on_ = true;
                }
                if (idx_ps_) {  // S33: one thread per block, keys straight from pk (PS5LM_SCORE_OLD=1: the S30 LDS-staged kernel)
                    const bool old = std::getenv("PS5LM_SCORE_OLD") != nullptr;
                    pIsc_ = ctx_.pipeline(sd + (old ? "/idx_score.spv" : "/idx_score_d.spv"), sizeof(PcIdxScore));
                    isc_wg_ = old ? kIdxScoreWg : uint32_t(std::max<uint64_t>(1, (o_.max_ctx / 4 + 255) / 256));
                }
                pIpp_ = ctx_.pipeline(sd + "/idx_pool_p.spv", sizeof(PcIdxPoolP));
            }
            if (o_.attn_legacy && (o_.kv16 || o_.kvq || idx_on_)) throw std::runtime_error("--attn-legacy needs fp32 KV and max_ctx <= 2048");
        }
        pPg_ = ctx_.pipeline(sd + "/ple_gate.spv", sizeof(PcPleGate), 0);
        pPc_ = ctx_.pipeline(sd + "/ple_conv.spv", sizeof(PcPleConv), 0);
        pAm_ = ctx_.pipeline(sd + "/argmax.spv", sizeof(PcArgmax), 0);
        pGU_ = ctx_.pipeline(sd + "/gate_up_q4.spv", sizeof(PcGateUp));
        if (o_.prefill_chunk) pCarry_ = ctx_.pipeline(sd + "/carry.spv", sizeof(PcCarry));
        pDN_ = ctx_.pipeline(sd + "/down_combine_q4.spv", sizeof(PcDown));
        const std::string rtag = o_.bits == 4 ? "q4" : "q2";
        pGUr_ = ctx_.pipeline(sd + "/gate_up_" + rtag + ".spv", sizeof(PcGateUp));
        pDNr_ = ctx_.pipeline(sd + "/down_combine_" + rtag + ".spv", sizeof(PcDown));
        pDM_ = ctx_.pipeline(sd + "/down_moe_" + rtag + ".spv", sizeof(PcDownMoe));
        if (o_.bits == 2 && o_.down_rows) pDX_ = ctx_.pipeline(sd + "/down_xr_q2.spv", sizeof(PcDown));
        pRt_ = ctx_.pipeline(sd + "/router.spv", sizeof(PcRouter));
        if (o_.bits == 3)
            for (int k = 0; k < 3; ++k) {
                bool used = false;
                for (uint32_t l = 0; l <= L; ++l) used |= ex_k2_[l] == uint32_t(5 + k);
                if (!used) continue;
                static const char* tags[3] = {"k25", "k3", "k35"};
                const std::string t = tags[k];
                pExH_[k] = ctx_.pipeline(sd + "/exl3_had_in_" + t + ".spv", sizeof(PcExHad), 0);
                pExM_[k] = ctx_.pipeline(sd + "/exl3_xmv_" + t + (std::getenv("PS5LM_XMV_SUFFIX") ? std::getenv("PS5LM_XMV_SUFFIX") : "") + ".spv", sizeof(PcExMv), 32);
                pExA_[k] = ctx_.pipeline(sd + "/exl3_act_" + t + ".spv", sizeof(PcExAct), 0);
                pExC_[k] = ctx_.pipeline(sd + "/exl3_combine_" + t + ".spv", sizeof(PcExCmb), 0);
            }
        pTk_ = ctx_.pipeline(sd + "/topk.spv", sizeof(PcTopk));
        if (TB_ >= 2 && o_.bits == 3)  // S18 stage C batched kernels (S20: also for the batched prefill)
            for (uint32_t nb = 2; nb <= 4 && nb <= std::max<uint32_t>(TB_, o_.spec_k >= 2 ? 4 : 2); ++nb) {
                pXrB_[nb] = ctx_.pipeline(sd + "/matvec_xrb2560_" + std::to_string(nb) + ".spv", sizeof(PcMatvecB));
                pRt2B_[nb] = ctx_.pipeline(sd + "/router2b_" + std::to_string(nb) + ".spv", sizeof(PcRouter2B));
                if (nb == 2) pLdsB2_ = ctx_.pipeline(sd + "/matvec_ldsb6144_2.spv", sizeof(PcMatvecB));
                if (o_.mv_xrs) {  // S19
                    if (nb == 2) pXs6_[2] = ctx_.pipeline(sd + "/matvec_xrs6144_2.spv", sizeof(PcMatvecB));
                    pXs25_[nb] = ctx_.pipeline(sd + "/matvec_xrs2560_" + std::to_string(nb) + ".spv", sizeof(PcMatvecB));
                }
                pMvB_[nb] = ctx_.pipeline(sd + "/matvecb_" + std::to_string(nb) + ".spv", sizeof(PcMatvecB));
                pHuB_[nb] = ctx_.pipeline(sd + "/hc_upmixb_" + std::to_string(nb) + ".spv", sizeof(PcHcUpMixB));
                if (nb == 2 || nb == 4)
                    for (int k = 0; k < 3; ++k) {
                        bool used = false;
                        for (uint32_t l = 0; l <= L; ++l) used |= ex_k2_[l] == uint32_t(5 + k);
                        static const char* tags[3] = {"k25", "k3", "k35"};
                        if (used) pExMB_[k][nb == 4] = ctx_.pipeline(sd + "/exl3_xmvb_" + std::string(tags[k]) + "_" + std::to_string(nb) + ".spv", sizeof(PcExMvB), 32);
                        exmb_ok_[nb == 4] = true;
                    }
                if (nb == 2) {
                    pCvB_ = ctx_.pipeline(sd + "/gdn_convb.spv", sizeof(PcConvB), 0);
                    pGdB_ = ctx_.pipeline(sd + "/gdn_stepb.spv", sizeof(PcGdnB), 0);
                }
            }
        pRt2_ = ctx_.pipeline(sd + "/router2.spv", sizeof(PcRouter2));
        pTk2_ = ctx_.pipeline(sd + "/topk2.spv", sizeof(PcTopk2));
    }

    // ------------------------------------------------------------ command recording
    void BAR(VkCommandBuffer cb, int line) {
        vk::Ctx::barrier(cb);
        if (prof_cur_ >= 0 && (o_.profile || vprof_ || lmprof_)) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profq_, uint32_t(prof_cur_) * 512 + 1 + prof_n_[prof_cur_]++);
            prof_line_[prof_cur_].push_back(line);
        }
    }
    void MV(VkCommandBuffer cb, uint64_t w, uint64_t x, uint64_t y, uint32_t Kd, uint32_t N, uint32_t mode = 0, float alpha = 1.f, bool bar = true) {
        // K=2560: x in registers (rpw rows per wave). K=6144: x in LDS, 4 waves x 4 rows per WG.
        if (Kd == 10240 && pLdx_.pipe) {  // S19: HC down / inject, x in LDS, saddr loads (bit-identical to matvec.comp)
            const uint32_t rpw = (N + 35) / 36;  // <= 36 workgroups (one per CU: 40 KB LDS)
            PcMatvecX p{w, x, y, Kd, N, rpw, mode, alpha, 0, 0, 0, 0, 1.f};
            vk::Ctx::dispatch(cb, pLdx_, &p, (N + rpw - 1) / rpw);
        } else if ((Kd == 6144 || Kd == 2560) && o_.mv_xrs) {  // S19: matvec_xrs, bit-identical to lds6144 / xr2560 (mv_bench)
            const uint32_t rpw = Kd == 6144 ? 16 : (N >= 65536 ? 16 : N >= 2048 ? 8 : 1);
            PcMatvecB p{w, x, y, Kd, N, rpw, mode, alpha, 0, 0, 1};
            vk::Ctx::dispatch(cb, Kd == 6144 ? pXs6_[1] : pXs25_[1], &p, (N + rpw - 1) / rpw);
        } else if (Kd == 6144) {
            PcMatvec p{w, x, y, Kd, N, 16, mode, alpha, 0};
            vk::Ctx::dispatch(cb, pLds6144_, &p, (N + 15) / 16);
        } else {
            const bool xr = Kd == 2560;
            const uint32_t rpw = xr ? (N >= 65536 ? 16 : N >= 8192 ? 4 : 1) : (N >= 4096 ? 4 : 1);
            PcMatvec p{w, x, y, Kd, N, rpw, mode, alpha, 0};
            vk::Ctx::dispatch(cb, xr ? pXr2560_ : pMv_, &p, (N + rpw - 1) / rpw);
        }
        if (bar) BAR(cb, __LINE__);
    }
    void NORM(VkCommandBuffer cb, uint64_t x, uint64_t w, uint64_t o, uint32_t groups) {
        PcNorm p{x, w, o, HID, 1e-6f};
        vk::Ctx::dispatch(cb, pHn25_.pipe ? pHn25_ : pHn_, &p, groups);
        BAR(cb, __LINE__);
    }
    void HCPRE(VkCommandBuffer cb, const HcW& h, uint64_t inj) {  // H -> aX (+ inj)
        if (o_.hc_mode == 3) {  // S17: sumsq partials -> fused norm+down(+inj) (4 waves/row) -> upmix with on-the-fly N
            PcSumsq q{A_.H, A_.Psum};
            vk::Ctx::dispatch(cb, pSq_, &q, HCH / 256);
            BAR(cb, __LINE__);
            PcHcDown2 d{h.down, inj ? h.inj : 0, A_.H, h.norm, A_.Psum, A_.T, A_.Il, 1.f / HC, 1e-6f, inj ? HC : 0, 0};
            vk::Ctx::dispatch(cb, pHd2_, &d, LOWR + (inj ? HC : 0));
            BAR(cb, __LINE__);
            PcHcUpMix2 u{h.up, A_.T, A_.H, h.norm, A_.Psum, A_.X, inj ? A_.Il : 0, inj, 1e-6f, 0};
            vk::Ctx::dispatch(cb, pHu2_, &u, HID / 4);
            BAR(cb, __LINE__);
            return;
        }
        if (o_.hc_mode == 2) {  // S15: 2 kernels + 2 barriers (bit-exact, slower: see EngineOptions::hc_mode)
            PcHcDown d{A_.H, h.norm, h.down, inj ? h.inj : 0, A_.T, A_.Il, A_.N, o_.hc_rows, inj ? HC : 0, 1.f / HC, 1e-6f};
            vk::Ctx::dispatch(cb, pHd_, &d, (LOWR + HC + o_.hc_rows - 1) / o_.hc_rows);
            BAR(cb, __LINE__);
            PcHcUpMix u{h.up, A_.T, A_.N, A_.X, inj ? A_.Il : 0, inj};
            vk::Ctx::dispatch(cb, pHu_, &u, HID / 4);
            BAR(cb, __LINE__);
            return;
        }
        NORM(cb, A_.H, h.norm, A_.N, HC);
        if (pLdx_.pipe) {  // S19: down + inject rows in ONE dispatch (the inject alone was a 1-workgroup critical path)
            // one 40 KB-LDS workgroup per CU: <= 36 workgroups so all run in one round (36 or 40 CUs): rpw = ceil(rows / 36)
            const uint32_t rows = LOWR + (inj ? HC : 0), rpw = (rows + 35) / 36;
            PcMatvecX p{h.down, A_.N, A_.T, HCH, LOWR, rpw, 2, 1.f / HC, inj ? HC : 0, inj ? h.inj : 0, A_.Il, 0, 1.f};
            vk::Ctx::dispatch(cb, pLdx_, &p, (rows + rpw - 1) / rpw);
            BAR(cb, __LINE__);
        } else {
        MV(cb, h.down, A_.N, A_.T, HCH, LOWR, 2, 1.f / HC, inj == 0);
        if (inj) MV(cb, h.inj, A_.N, A_.Il, HCH, HC);  // independent of down: shares its barrier
        }
        if (o_.hc_mode == 1) {  // up projection + stream mix in one kernel: sig never goes to memory (bit-exact)
            PcHcUpMix u{h.up, A_.T, A_.N, A_.X, inj ? A_.Il : 0, inj};
            if (pHu4_.pipe) vk::Ctx::dispatch(cb, pHu4_, &u, HID / 16);  // S19: 4 outputs per wave
            else vk::Ctx::dispatch(cb, pHu_, &u, HID / 4);
            BAR(cb, __LINE__);
            return;
        }
        MV(cb, h.up, A_.T, A_.S, LOWR, HCH, 3);
        PcMix m{A_.S, A_.N, A_.X, inj ? A_.Il : 0, inj, HC, 0};
        vk::Ctx::dispatch(cb, pMx_, &m, HID / 256);
        BAR(cb, __LINE__);
    }
    void POST(VkCommandBuffer cb, uint64_t y, uint64_t inj) {
        PcPost p{A_.H, y, inj, HC, 0};
        vk::Ctx::dispatch(cb, pPo_, &p, HCH / 256);
        BAR(cb, __LINE__);
    }
    void PRE(VkCommandBuffer cb, uint32_t l) {
        const auto& w = W_[l];
        const float eps = 1e-6f;
        if (l == PLE_LAYER) {
            MV(cb, ple_key_, A_.E, A_.Key, HID, HCH);
            MV(cb, ple_val_, A_.E, A_.Val, HID, HID);
            NORM(cb, A_.Key, ple_nk_, A_.Keyn, HC);
            NORM(cb, A_.H, ple_nq_, A_.Qn, HC);
            PcPleGate g{A_.Keyn, A_.Qn, A_.Val, A_.Gv};
            vk::Ctx::dispatch(cb, pPg_, &g, HC);
            BAR(cb, __LINE__);
            NORM(cb, A_.Gv, ple_nc_, A_.Gvn, HC);
            PcPleConv c{A_.Gvn, A_.Gv, ple_cw_, ple_state_ + sin_, A_.H, ple_state_ + sout_, HCH, 0};
            vk::Ctx::dispatch(cb, pPc_, &c, HCH / 256);
            BAR(cb, __LINE__);
        }
        HCPRE(cb, w.hca, A_.I);
        if (earlyA_ && l < L) {  // S17: router_l on the attention input -> early mailbox (only the one-cb recording)
            PcRouter r{w.router, A_.X, A_.R, E, 0};
            vk::Ctx::dispatch(cb, pRt_, &r, E);
            BAR(cb, __LINE__);
            PcTopk t{A_.R, earlyA_ + uint64_t(l) * 256, E, K, l + 1, 0};
            vk::Ctx::dispatch(cb, pTk_, &t, 1);
        }
        if (!w.attn) {
            MV(cb, w.inproj, A_.X, A_.P, HID, 16480);
            PcConv c{A_.P, w.conv, conv_state_[l] + sin_, A_.C, conv_state_[l] + sout_, 10240, 0};
            vk::Ctx::dispatch(cb, pCv_, &c, 10240 / 256);
            BAR(cb, __LINE__);
            PcGdn g{A_.C, A_.P, gdn_state_[l] + sin_, w.alog, w.dtb, w.gnorm, A_.Core, gdn_state_[l] + sout_, eps, 0};
            vk::Ctx::dispatch(cb, pGd_, &g, 48);
            BAR(cb, __LINE__);
            MV(cb, w.outproj, A_.Core, A_.Y, 6144, HID);
        } else {
            MV(cb, w.qkv, A_.X, A_.Qkv, HID, 13312, 0, 1.f, !idx_on_);
            if (idx_on_) MV(cb, w.iqk, A_.X, A_.Iqk, HID, 640);  // S19 indexer q | raw key (shares the barrier)
            PcPrep pr{A_.Qkv, w.qn, w.kn, A_.Q, A_.G, kc_[l], vc_[l], A_.Par, eps, 1e7f};
            vk::Ctx::dispatch(cb, pPr_, &pr, 26);
            if (idx_on_) { PcIdxPrep ip{A_.Iqk, w.iqn, A_.Qi, rk_[l], A_.Par, eps, 1e7f}; vk::Ctx::dispatch(cb, pIp_, &ip, 5); }
            BAR(cb, __LINE__);
            if (idx_on_) {
                if (idx_ps_) {
                    PcIdxScore sc{A_.Qi, pk_[l], rk_[l], w.ikn, A_.Par, A_.Scr, uint32_t(act_stride_), 1, isc_wg_, eps, 1e7f, 0};
                    vk::Ctx::dispatch(cb, pIsc_, &sc, isc_wg_);
                    BAR(cb, __LINE__);
                }
                PcIdxSel is{A_.Qi, rk_[l], pk_[l], w.ikn, A_.Sel, A_.Par, 0, eps, 1e7f, idx_gmin_, A_.Scr};
                vk::Ctx::dispatch(cb, pIs_, &is, 1);
                BAR(cb, __LINE__);
                if (sst_on_ && l < L) {  // S32 tiering probe: reads Sel only (attention reads it too: no barrier)
                    PcSelStat ss{A_.Sel, A_.Par, sst_last_ + uint64_t(l / 4) * (o_.max_ctx / 4) * 4, sst_cnt_ + uint64_t(l / 4) * 64, 4096, 0};
                    vk::Ctx::dispatch(cb, pSst_, &ss, 1);
                }
            }
            if (o_.attn_legacy) {
                PcAttn at{A_.Q, A_.G, kc_[l], vc_[l], A_.Ao, aSc_, A_.Par};
                vk::Ctx::dispatch(cb, pAt_, &at, 24);
                BAR(cb, __LINE__);
            } else {  // flash-decoding: chunks beyond the current position exit immediately
                PcAttnSplit as{A_.Q, kc_[l], vc_[l], A_.Part, A_.Par, attn_chunks_, 0, A_.Sel};
                vk::Ctx::dispatch(cb, pAs_, &as, attn_chunks_, 2);
                BAR(cb, __LINE__);
                PcAttnComb ac{A_.Part, A_.G, A_.Ao, A_.Par, attn_chunks_, 0, A_.Sel};
                vk::Ctx::dispatch(cb, pAc_, &ac, 24);
                BAR(cb, __LINE__);
            }
            MV(cb, w.oproj, A_.Ao, A_.Y, 6144, HID);
        }
        POST(cb, A_.Y, A_.I);
        HCPRE(cb, w.hcm, A_.I2);
        if (o_.dump_moe && l < L) {
            uint64_t so = 0, doff = 0;
            VkBuffer sb = d_.buf_of(A_.X, so), db = d_.buf_of(aMoe_ + uint64_t(l) * HID * 4, doff);
            VkBufferCopy rg{so, doff, HID * 4};
            vkCmdCopyBuffer(cb, sb, db, 1, &rg);
            BAR(cb, __LINE__);
        }
        // S17: router + depth-1 predictor fused (one dispatch each for the matvecs and the two top-k)
        const bool fuse = o_.router_fuse && o_.predict == 1 && l + 1 < NL_ && t_ == 0;
        if (fuse) {
            const uint64_t pw = pred_w_.size() > 1 && pred_w_[1][l] ? pred_w_[1][l] : W_[l + 1].router;
            PcRouter2 r{w.router, pw, A_.X, A_.R, A_.R2, E, 0};
            vk::Ctx::dispatch(cb, pRt2_, &r, 2 * E);
            BAR(cb, __LINE__);
            PcTopk2 t{A_.R, mailA_ + rw(l) * 256, E, kr(l), l + 1, 0, A_.R2, pmailA_ + (uint64_t(1) * L + l + 1) * 256, l + 2, 0};
            vk::Ctx::dispatch(cb, pTk2_, &t, 2);
            MV(cb, w.sgate, A_.X, aWt_ + (rw(l) * TAB + K) * 4, HID, 1, 3);
            return;
        }
        PcRouter r{w.router, A_.X, A_.R, E, 0};
        vk::Ctx::dispatch(cb, pRt_, &r, E);
        BAR(cb, __LINE__);
        PcTopk t{A_.R, mailA_ + rw(l) * 256, E, kr(l), l + 1, 0};
        vk::Ctx::dispatch(cb, pTk_, &t, 1);
        MV(cb, w.sgate, A_.X, aWt_ + (rw(l) * TAB + K) * 4, HID, 1, 3);
        // Early routing: apply the routers of layers l+1..l+D to THIS layer's MoE input (residual streams change
        // slowly), so the host can prefetch those experts while the GPU finishes layer l and runs layer l+1.
        for (uint32_t dd = 1; dd <= o_.predict && l + dd < NL_ && t_ == 0; ++dd) {  // S16 verify: token 0 predicts
            BAR(cb, __LINE__);
            const uint64_t pw = dd < pred_w_.size() && pred_w_[dd][l] ? pred_w_[dd][l] : W_[l + dd].router;
            PcRouter rp{pw, A_.X, A_.R, E, 0};
            vk::Ctx::dispatch(cb, pRt_, &rp, E);
            BAR(cb, __LINE__);
            PcTopk tp{A_.R, pmailA_ + (uint64_t(dd) * L + l + dd) * 256, E, K, l + dd + 1, 0};
            vk::Ctx::dispatch(cb, pTk_, &tp, 1);
        }
    }
    // Routed experts (pack precision) and the shared expert (always PS5Q4) in separate dispatches; the shared
    // expert's down pass accumulates onto the routed output (resid = routed sum).
    // S15b EXL3: input Hadamard + gate/up decode-matvec for the experts bound in `table` (0 = skip)
    void EX_GATEUP(VkCommandBuffer cb, uint32_t l, uint64_t table) {
        const int k = kidx(l);
        PcExHad h{A_.X, table, A_.ExXt};
        vk::Ctx::dispatch(cb, pExH_[k], &h, HID / 128, 2, K);
        vk::Ctx::barrier(cb);  // not BAR: this also records the hits_ command buffers (no profiling slots there)
        PcExMv m{table, A_.ExXt, A_.ExPre, HID / 16, 640 / 16, ex_tr_gate(ex_k2_[l]), ex_tr_up(ex_k2_[l]), 2 * HID, HID, 2 * 640, 640};
        vk::Ctx::dispatch(cb, pExM_[k], &m, 640 / 32, 2, K);
    }
    // ... then (all K experts) activation + down decode-matvec + output Hadamard / weights -> A_.Ym2 (routed sum)
    // S17: act + down decode-matvec of the experts in `table` (0 = skip) -> their ExYd rows; the hits_ pre-pass runs it
    // for the resident experts while the misses stream, POSTMOE for the streamed ones (bit-exact: per-expert kernels)
    void EX_ACTDOWN(VkCommandBuffer cb, uint32_t l, uint64_t table, bool prof) {
        const int k = kidx(l);
        PcExAct a{A_.ExPre, table, A_.ExHd};
        vk::Ctx::dispatch(cb, pExA_[k], &a, 640 / 128, K);
        if (prof) BAR(cb, __LINE__); else vk::Ctx::barrier(cb);
        PcExMv m{table, A_.ExHd, A_.ExYd, 640 / 16, HID / 16, ex_tr_down(), ex_tr_down(), 640, 0, HID, 0};
        vk::Ctx::dispatch(cb, pExM_[k], &m, HID / 32, 1, K);
        if (prof) BAR(cb, __LINE__); else vk::Ctx::barrier(cb);
    }
    void EX_DOWN(VkCommandBuffer cb, uint32_t l, uint64_t sl, uint64_t wt, uint64_t table = 0) {
        const int k = kidx(l);
        EX_ACTDOWN(cb, l, table ? table : sl, true);
        PcExCmb c{A_.ExYd, sl, wt, A_.Ym2, K, 0};
        vk::Ctx::dispatch(cb, pExC_[k], &c, HID / 128);
        BAR(cb, __LINE__);
    }
    void POSTMOE(VkCommandBuffer cb, uint32_t l) {
        const uint64_t sl = aSl_ + rw(l) * TAB * 8, wt = aWt_ + rw(l) * TAB * 4;
        if (o_.bits == 3) {
            if (ov_) EX_GATEUP(cb, l, aSlB_ + rw(l) * TAB * 8);  // resident ones ran in hits_[l]
            else {
                EX_GATEUP(cb, l, sl);
                PcGateUp gs{A_.X, A_.Hm + uint64_t(K) * 640 * 4, sl + uint64_t(K) * 8, 4, 0};
                vk::Ctx::dispatch(cb, pGU_, &gs, 640 / 4, 1);
            }
            BAR(cb, __LINE__);
            EX_DOWN(cb, l, sl, wt, ov_ && o_.hits_down ? aSlB_ + rw(l) * TAB * 8 : 0);  // S17: resident act+down ran in hits_
            PcDown ds{A_.Hm + uint64_t(K) * 640 * 4, sl + uint64_t(K) * 8, wt + uint64_t(K) * 4, A_.Ym2, A_.Ym, 1, 2};
            vk::Ctx::dispatch(cb, pDN_, &ds, HID / 2);
            BAR(cb, __LINE__);
            POST(cb, A_.Ym, A_.I2);
            return;
        }
        if (ov_) {  // resident experts + shared ran in hits_[l]; only the streamed ones here
            PcGateUp g{A_.X, A_.Hm, aSlB_ + rw(l) * TAB * 8, 4, 0};
            vk::Ctx::dispatch(cb, pGUr_, &g, 640 / 4, K);
        } else {
            PcGateUp g{A_.X, A_.Hm, sl, 4, 0};
            vk::Ctx::dispatch(cb, pGUr_, &g, 640 / 4, K);
            PcGateUp gs{A_.X, A_.Hm + uint64_t(K) * 640 * 4, sl + uint64_t(K) * 8, 4, 0};
            vk::Ctx::dispatch(cb, pGU_, &gs, 640 / 4, 1);
        }
        BAR(cb, __LINE__);
        if (o_.fused_moe) {  // routed + shared down + combine + inject in one kernel (S14, opt-in)
            PcDownMoe dm{A_.Hm, sl, wt, A_.H, A_.I2, K, 4};
            vk::Ctx::dispatch(cb, pDM_, &dm, HID / (4 * 4));
            BAR(cb, __LINE__);
            return;
        }
        if (pDX_.pipe) {  // activations in registers, rows streamed (S14)
            PcDown dn{A_.Hm, sl, wt, aZero_, A_.Ym2, K, o_.down_rows};
            vk::Ctx::dispatch(cb, pDX_, &dn, (HID + o_.down_rows - 1) / o_.down_rows);
        } else {
            PcDown dn{A_.Hm, sl, wt, aZero_, A_.Ym2, K, 2};
            vk::Ctx::dispatch(cb, pDNr_, &dn, HID / 2);
        }
        BAR(cb, __LINE__);
        PcDown ds{A_.Hm + uint64_t(K) * 640 * 4, sl + uint64_t(K) * 8, wt + uint64_t(K) * 4, A_.Ym2, A_.Ym, 1, 2};
        vk::Ctx::dispatch(cb, pDN_, &ds, HID / 2);
        BAR(cb, __LINE__);
        POST(cb, A_.Ym, A_.I2);
    }
    void SNAP(VkCommandBuffer cb, uint32_t idx) {
        if (!o_.dump) return;
        uint64_t so = 0, doff = 0;
        VkBuffer sb = d_.buf_of(A_.H, so), db = d_.buf_of(aDump_ + uint64_t(idx) * HCH * 4, doff);
        VkBufferCopy rg{so, doff, HCH * 4};
        vkCmdCopyBuffer(cb, sb, db, 1, &rg);
        BAR(cb, __LINE__);
    }
    void HEAD(VkCommandBuffer cb, const HcW* mixer = nullptr) {
        HCPRE(cb, mixer ? *mixer : mixer_, 0);
        MV(cb, lm_head_, A_.X, lg_.addr + uint64_t(t_) * VOCAB * 4, HID, VOCAB);
        PcArgmax am{lg_.addr + uint64_t(t_) * VOCAB * 4, arg_.addr + uint64_t(t_) * 16, VOCAB, mixer ? 1u : 0u};  // S18: MTP head -> confidence
        vk::Ctx::dispatch(cb, pAm_, &am, 1);
        BAR(cb, __LINE__);
    }
    VkCommandBuffer copy_cb(uint64_t src, uint64_t dst, uint64_t bytes) {
        VkCommandBuffer cb = ctx_.cmd();
        vk::Ctx::begin(cb, false);
        uint64_t so = 0, doff = 0;
        VkBuffer sb = d_.buf_of(src, so), db = d_.buf_of(dst, doff);
        VkBufferCopy rg{so, doff, bytes};
        vkCmdCopyBuffer(cb, sb, db, 1, &rg);
        vk::Ctx::barrier(cb);
        vk::Ctx::end(cb);
        return cb;
    }

    // Recording target of the helpers for token t of a T-token verify batch: its activation block, its table rows, and its
    // recurrent state chain live -> ck_[0] -> ... -> ck_[T-2] -> live (token t reads the state token t-1 wrote).
    void set_tok(uint32_t t, uint32_t T) {
        t_ = t;
        A_ = act_at(t);
        auto ck = [&](uint32_t j) { return ck_[j] - rstate_; };  // address delta (mod 2^64)
        sin_ = (t == 0 || nock_) ? 0 : ck(t - 1);
        sout_ = (T == 1 || t + 1 == T || nock_) ? 0 : ck(t);
    }
    // S16 stage B: verify steps for T = 2..TB_ tokens. Same per-token kernels as feed() (bit-exact per token), recorded
    // token after token inside each step; the routed experts of all T tokens are bound by the host between steps.
    void record_verify() {
        const uint32_t VT = 1 + o_.spec_k;  // S20: TB_ may exceed it (prefill blocks); checkpoints exist for T <= VT
        if (VT < 2) return;
        const bool ov = ov_;
        ov_ = false;  // no hits_ pre-pass in verify: POSTMOE runs every expert of every token
        vsteps_.assign(VT - 1, {});
        for (uint32_t T = 2; T <= VT; ++T) {
            auto& v = vsteps_[T - 2];
            v.resize(NL_ + 1);
            for (uint32_t s = 0; s <= NL_; ++s) {
                VkCommandBuffer cb = ctx_.cmd();
                vk::Ctx::begin(cb, false);
                if (s > 0) for (uint32_t t = 0; t < T; ++t) { set_tok(t, T); POSTMOE(cb, s - 1); }
                for (uint32_t t = 0; t < T; ++t) { set_tok(t, T); if (s < NL_) PRE(cb, s); else HEAD(cb); }
                vk::Ctx::end(cb);
                v[s] = cb;
            }
        }
        for (uint32_t m = 0; m + 1 < VT; ++m) ck_restore_.push_back(copy_cb(ck_[m], rstate_, rstate_core_));
        set_tok(0, 1);
        ov_ = ov;
    }

    // S15 overlap pre-pass of layer l: gate/up of the resident experts (table A) + the shared expert, block A_
    VkCommandBuffer record_hits(uint32_t l) {
        VkCommandBuffer cb = ctx_.cmd();
        vk::Ctx::begin(cb, false);
        HITS(cb, l);
        vk::Ctx::end(cb);
        return cb;
    }
    void HITS(VkCommandBuffer cb, uint32_t l) {
        if (o_.bits == 3) {
            EX_GATEUP(cb, l, aSlA_ + rw(l) * TAB * 8);
            if (o_.hits_down) {  // S17: also act + down of the resident experts (shared gate_up below is independent)
                vk::Ctx::barrier(cb);
                EX_ACTDOWN(cb, l, aSlA_ + rw(l) * TAB * 8, false);
            }
        } else {
            PcGateUp g{A_.X, A_.Hm, aSlA_ + rw(l) * TAB * 8, 4, 0};
            vk::Ctx::dispatch(cb, pGUr_, &g, 640 / 4, K);
        }
        PcGateUp gs{A_.X, A_.Hm + uint64_t(K) * 640 * 4, aSl_ + (rw(l) * TAB + K) * 8, 4, 0};
        vk::Ctx::dispatch(cb, pGU_, &gs, 640 / 4, 1);
        vk::Ctx::barrier(cb);
    }

    // ---------------------------------------------------------------- S17 one command buffer per token
    static constexpr uint64_t kCtlErr = 0, kCtlMail = 4096, kCtlPmail = kCtlMail + uint64_t(L + 1) * 256,
                              kCtlStage = kCtlPmail + uint64_t(4) * L * 256, kStageRow = 512,
                              kCtlE = kCtlStage + uint64_t(L + 1) * kStageRow, kCtlEFlag = kCtlE + uint64_t(HID) * 4,
                              kCtlEarly = kCtlEFlag + 4096, kCtlEnd1 = kCtlEarly + uint64_t(L + 1) * 256,
                              // S18 verify (one cb per batch): per-token mail rows (t, s), depth-1 predictions (t, s), stage
                              // rows (t, s) [flags A/B live in token 0's row], n-gram rows of T tokens, their flag. 8 = kMaxTB
                              kVMail = (kCtlEnd1 + 4095) & ~4095ull, kVPmail = kVMail + 8ull * (L + 1) * 256,
                              kVStage = kVPmail + 8ull * L * 256, kVE = kVStage + 8ull * (L + 1) * kStageRow,
                              kVEFlag = kVE + 8ull * HID * 4, kVEnt = kVEFlag + 4096, kCtlBytes = kVEnt + uint64_t(L) * 4096;
    uint64_t vent(uint32_t s, bool streamed) const { return kVEnt + uint64_t(s) * 4096 + (streamed ? 2048 : 0); }
    uint64_t vmail(uint32_t t, uint32_t s) const { return kVMail + (uint64_t(t) * (L + 1) + s) * 256; }
    uint64_t vpmail(uint32_t t, uint32_t s) const { return kVPmail + (uint64_t(t) * L + s) * 256; }
    uint64_t vstage(uint32_t t, uint32_t s) const { return kVStage + (uint64_t(t) * (L + 1) + s) * kStageRow; }
    // stage row of layer s: slA u64[16] @0 | sl u64[16] @128 | slB u64[16] @256 | wt f32[16] @384 | flagA @448 | flagB @452
    uint64_t stage(uint32_t s) const { return kCtlStage + uint64_t(s) * kStageRow; }
    volatile uint32_t* ctlw(uint64_t off) { return reinterpret_cast<volatile uint32_t*>(ctlp_ + off); }
    void GATE(VkCommandBuffer cb, uint64_t flag_off, uint64_t s0, uint64_t d0, uint32_t n0, uint64_t s1 = 0, uint64_t d1 = 0, uint32_t n1 = 0,
              uint64_t s2 = 0, uint64_t d2 = 0, uint32_t n2 = 0) {
        gate_kind_.push_back(flag_off == kCtlEFlag ? 2 : (flag_off - kCtlStage) % kStageRow == 448 ? 0 : 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsq1_, tsq1_n_++);
        const uint64_t c = ctl_.addr;
        PcGate g{c + flag_off, c + kCtlErr, s0 ? c + s0 : 0, d0, s1 ? c + s1 : 0, d1, s2 ? c + s2 : 0, d2, n0, n1, n2, 1, 0, 0, 0, 0};
        vk::Ctx::dispatch(cb, pGate_, &g, 1);
        vk::Ctx::barrier(cb);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsq1_, tsq1_n_++);
    }
    void record_onecb() {
        if (!o_.onecb) return;
        if (!o_.overlap || o_.dump || o_.profile || o_.dump_moe || !full()) {  // these modes keep the per-step path
            std::fprintf(stderr, "one command buffer per token: off (needs overlap, all layers, no dump/profile)\n");
            return;
        }
        ctl_ = ctx_.create_uncached(kCtlBytes);
        ctlp_ = static_cast<uint8_t*>(ctl_.map);
        std::memset(ctlp_, 0, kCtlBytes);
        pGate_ = ctx_.pipeline(o_.shader_dir + "/gate.spv", sizeof(PcGate), 64);
        const uint32_t nts = 2 + 2 * (2 * NL_ + 1);
        tsq1_ = ctx_.timestamps(nts);
        mailA_ = ctl_.addr + kCtlMail;
        pmailA_ = ctl_.addr + kCtlPmail;
        if (o_.early) earlyA_ = ctl_.addr + kCtlEarly;
        for (int head = 0; head < 2; ++head) {
            VkCommandBuffer cb = ctx_.cmd();
            vk::Ctx::begin(cb, false);
            vkCmdResetQueryPool(cb, tsq1_, 0, nts);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsq1_, 0);
            tsq1_n_ = 1;
            gate_kind_.clear();
            for (uint32_t s = 0; s <= NL_; ++s) {
                if (s > 0) {  // layer s-1: wait for its full binding (all experts ready), then the rest of its MoE
                    const uint64_t r = stage(s - 1);
                    GATE(cb, r + 452, r + 128, aSl_ + uint64_t(s - 1) * TAB * 8, 2 * K, r + 256, aSlB_ + uint64_t(s - 1) * TAB * 8, 2 * K,
                         r + 384, aWt_ + uint64_t(s - 1) * TAB * 4, K);
                    POSTMOE(cb, s - 1);
                }
                if (s == NL_) { if (head) HEAD(cb); break; }
                if (s == PLE_LAYER && use_ple_) GATE(cb, kCtlEFlag, kCtlE, A_.E, HID);  // the n-gram rows (host-converted)
                PRE(cb, s);
                vk::Ctx::barrier(cb);
                const uint64_t r = stage(s);
                GATE(cb, r + 448, r, aSlA_ + uint64_t(s) * TAB * 8, 2 * K);  // resident experts known
                HITS(cb, s);
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsq1_, tsq1_n_++);
            if (tsq1_n_ != nts) throw std::runtime_error("onecb timestamp count");
            vk::Ctx::end(cb);
            tok_cb_[head] = cb;
        }
        mailA_ = aMail_;
        pmailA_ = aPmail_;
        earlyA_ = 0;
        std::fprintf(stderr, "one command buffer per token: control block %llu KiB (GPU-uncached), %u gates/token\n",
                     (unsigned long long)(kCtlBytes >> 10), 2 * NL_ + 1);
    }


    // ---------------------------------------------------------------- S18 stage C: one command buffer per verify batch
    // T tokens (v[0] = last sampled, v[1..] = MTP drafts) through all 48 layers in ONE submit, like record_onecb() for one
    // token: per layer the T tokens' kernels are dispatched between the SAME barriers (they run concurrently), the large
    // dense matrices are read ONCE for all T columns (matvec_xrb / matvec_ldsb / router2b, bit-identical per column), the
    // recurrent kernels chain token after token through the verify checkpoints, and gates A/B release the T tokens'
    // resident / all experts of a layer together (the host binds the union of their experts).
    void GATEV(VkCommandBuffer cb, uint8_t kind, uint64_t flag_off, uint32_t ntok, uint32_t sstr, uint32_t dstr01, uint32_t dstr2, uint64_t s0,
               uint64_t d0, uint32_t n0, uint64_t s1 = 0, uint64_t d1 = 0, uint32_t n1 = 0, uint64_t s2 = 0, uint64_t d2 = 0, uint32_t n2 = 0) {
        vgate_kind_.push_back(kind);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsqv_, tsqv_n_++);
        const uint64_t c = ctl_.addr;
        PcGate g{c + flag_off, c + kCtlErr, s0 ? c + s0 : 0, d0, s1 ? c + s1 : 0, d1, s2 ? c + s2 : 0, d2, n0, n1, n2, ntok, sstr, dstr01, dstr2, 0};
        vk::Ctx::dispatch(cb, pGate_, &g, 1);
        vk::Ctx::barrier(cb);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsqv_, tsqv_n_++);
    }
    // y_t (+)= W x_t for t < T: weights read once (xrb: K = 2560, T <= 4; ldsb: K = 6144, pairs), else per-token dispatches
    void MVB(VkCommandBuffer cb, uint64_t w, uint64_t x0, uint64_t xs, uint64_t y0, uint64_t ys, uint32_t Kd, uint32_t N, uint32_t T,
             uint32_t mode = 0, float alpha = 1.f, bool bar = true) {
        if (T == 1) { MV(cb, w, x0, y0, Kd, N, mode, alpha, bar); return; }
        if (o_.mv_xrs && !o_.vb_nobatch && ((Kd == 2560 && T <= 4 && pXs25_[T].pipe) || (Kd == 6144 && pXs6_[2].pipe))) {  // S19
            if (Kd == 2560) {
                const uint32_t rpw = N >= 65536 ? 16 : N >= 2048 ? 8 : 1;
                PcMatvecB p{w, x0, y0, Kd, N, rpw, mode, alpha, uint32_t(xs), uint32_t(ys), T};
                vk::Ctx::dispatch(cb, pXs25_[T], &p, (N + rpw - 1) / rpw);
            } else {
                uint32_t t = 0;
                for (; t + 2 <= T; t += 2) {
                    PcMatvecB p{w, x0 + t * xs, y0 + t * ys, Kd, N, 16, mode, alpha, uint32_t(xs), uint32_t(ys), 2};
                    vk::Ctx::dispatch(cb, pXs6_[2], &p, (N + 15) / 16);
                }
                if (t < T) MV(cb, w, x0 + t * xs, y0 + t * ys, Kd, N, mode, alpha, false);
            }
        } else if (Kd == 2560 && T <= 4 && pXrB_[T].pipe && !o_.vb_nobatch) {
            const uint32_t rpw = N >= 65536 ? 16 : N >= 8192 ? 4 : 1;
            PcMatvecB p{w, x0, y0, Kd, N, rpw, mode, alpha, uint32_t(xs), uint32_t(ys), T};
            vk::Ctx::dispatch(cb, pXrB_[T].pipe ? pXrB_[T] : pXr2560_, &p, (N + rpw - 1) / rpw);
        } else if (Kd == 6144 && pLdsB2_.pipe && !o_.vb_nobatch && o_.ldsb_rpw) {  // ldsb_rpw 0: per-token dispatches
            uint32_t t = 0;
            for (; t + 2 <= T; t += 2) {
                const uint32_t rpw = o_.ldsb_rpw;  // S18: rows per WG (x staging traffic = N / rpw x 48 KB)
                PcMatvecB p{w, x0 + t * xs, y0 + t * ys, Kd, N, rpw, mode, alpha, uint32_t(xs), uint32_t(ys), 2};
                vk::Ctx::dispatch(cb, pLdsB2_, &p, (N + rpw - 1) / rpw);
            }
            if (t < T) MV(cb, w, x0 + t * xs, y0 + t * ys, Kd, N, mode, alpha, false);
        } else if (Kd != 2560 && Kd != 6144 && T <= 4 && pMvB_[T].pipe && !o_.vb_nobatch) {
            const uint32_t rpw = N >= 4096 ? 4 : 1;  // MV()'s choice for the generic kernel
            PcMatvecB p{w, x0, y0, Kd, N, rpw, mode, alpha, uint32_t(xs), uint32_t(ys), T};
            vk::Ctx::dispatch(cb, pMvB_[T], &p, (N + rpw - 1) / rpw);
        } else {
            for (uint32_t t = 0; t < T; ++t) MV(cb, w, x0 + t * xs, y0 + t * ys, Kd, N, mode, alpha, false);
        }
        if (bar) BAR(cb, __LINE__);
    }
    void NORM_NB(VkCommandBuffer cb, uint64_t x, uint64_t w, uint64_t o, uint32_t groups) {
        PcNorm p{x, w, o, HID, 1e-6f};
        vk::Ctx::dispatch(cb, pHn25_.pipe ? pHn25_ : pHn_, &p, groups);
    }
    // HC pre-step for T tokens (hc_mode 1: norm | down + inject | up + mix), every stage dispatched for all tokens.
    // which: 0 = no inject (mixer), 1 = inject into A.I (attention HC), 2 = into A.I2 (MLP HC)
    void HCPRE_T(VkCommandBuffer cb, const HcW& h, int which, uint32_t T) {
        for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); NORM_NB(cb, a.H, h.norm, a.N, HC); }
        BAR(cb, __LINE__);
        const Act a0 = act_at(0);
        const uint64_t S = act_stride_;
        MVB(cb, h.down, a0.N, S, a0.T, S, HCH, LOWR, T, 2, 1.f / HC, false);
        if (which) MVB(cb, h.inj, a0.N, S, a0.Il, S, HCH, HC, T, 0, 1.f, false);
        BAR(cb, __LINE__);
        if (T <= 4 && pHuB_[T].pipe && !o_.vb_nobatch) {
            const uint64_t injp = which == 1 ? a0.I : which == 2 ? a0.I2 : 0;
            PcHcUpMixB u{h.up, a0.T, a0.N, a0.X, which ? a0.Il : 0, injp, uint32_t(S), 0};
            vk::Ctx::dispatch(cb, pHuB_[T], &u, HID / 4);
        } else
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                const uint64_t injp = which == 1 ? a.I : which == 2 ? a.I2 : 0;
                PcHcUpMix u{h.up, a.T, a.N, a.X, which ? a.Il : 0, injp};
                vk::Ctx::dispatch(cb, pHu_, &u, HID / 4);
            }
        BAR(cb, __LINE__);
    }
    void PRE_T(VkCommandBuffer cb, uint32_t l, uint32_t T) {
        const auto& w = W_[l];
        const float eps = 1e-6f;
        const uint64_t S = act_stride_;
        const Act A0 = act_at(0);
        if (l == PLE_LAYER) {
            MVB(cb, ple_key_, A0.E, S, A0.Key, S, HID, HCH, T, 0, 1.f, false);
            MVB(cb, ple_val_, A0.E, S, A0.Val, S, HID, HID, T);
            for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); NORM_NB(cb, a.Key, ple_nk_, a.Keyn, HC); NORM_NB(cb, a.H, ple_nq_, a.Qn, HC); }
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); PcPleGate g{a.Keyn, a.Qn, a.Val, a.Gv}; vk::Ctx::dispatch(cb, pPg_, &g, HC); }
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); NORM_NB(cb, a.Gv, ple_nc_, a.Gvn, HC); }
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) {  // recurrent: token t reads the conv state token t-1 wrote
                set_tok(t, T);
                PcPleConv c{A_.Gvn, A_.Gv, ple_cw_, ple_state_ + sin_, A_.H, ple_state_ + sout_, HCH, 0};
                vk::Ctx::dispatch(cb, pPc_, &c, HCH / 256);
                BAR(cb, __LINE__);
            }
            set_tok(0, T);
        }
        HCPRE_T(cb, w.hca, 1, T);
        if (!w.attn) {
            MVB(cb, w.inproj, A0.X, S, A0.P, S, HID, 16480, T);
            if (pGdB_.pipe && T <= 4 && !o_.vb_nobatch) {  // one dispatch each: state read once, checkpoints written per token
                auto so = [&](uint64_t base, uint32_t t) { return (t + 1 == T || nock_) ? base : base + (ck_[t] - rstate_); };
                PcConvB c{A0.P, w.conv, conv_state_[l], A0.C, {so(conv_state_[l], 0), so(conv_state_[l], 1 % T), T > 2 ? so(conv_state_[l], 2 % T) : 0,
                          T > 3 ? so(conv_state_[l], 3) : 0}, 10240, uint32_t(S), T, 0};
                vk::Ctx::dispatch(cb, pCvB_, &c, 10240 / 256);
                BAR(cb, __LINE__);
                PcGdnB g{A0.C, A0.P, gdn_state_[l], w.alog, w.dtb, w.gnorm, A0.Core,
                         {so(gdn_state_[l], 0), so(gdn_state_[l], 1 % T), T > 2 ? so(gdn_state_[l], 2 % T) : 0, T > 3 ? so(gdn_state_[l], 3) : 0}, eps, uint32_t(S), T};
                vk::Ctx::dispatch(cb, pGdB_, &g, 48);
                BAR(cb, __LINE__);
                MVB(cb, w.outproj, A0.Core, S, A0.Y, S, 6144, HID, T);
            } else
            // conv(t) and gdn(t-1) are independent: T + 1 barriers instead of 2T
            for (uint32_t i = 0; i <= T; ++i) {
                if (i < T) {
                    set_tok(i, T);
                    PcConv c{A_.P, w.conv, conv_state_[l] + sin_, A_.C, conv_state_[l] + sout_, 10240, 0};
                    vk::Ctx::dispatch(cb, pCv_, &c, 10240 / 256);
                }
                if (i > 0) {
                    set_tok(i - 1, T);
                    PcGdn g{A_.C, A_.P, gdn_state_[l] + sin_, w.alog, w.dtb, w.gnorm, A_.Core, gdn_state_[l] + sout_, eps, 0};
                    vk::Ctx::dispatch(cb, pGd_, &g, 48);
                }
                BAR(cb, __LINE__);
            }
            if (!(pGdB_.pipe && T <= 4 && !o_.vb_nobatch)) {
                set_tok(0, T);
                MVB(cb, w.outproj, A0.Core, S, A0.Y, S, 6144, HID, T);
            }
        } else {
            MVB(cb, w.qkv, A0.X, S, A0.Qkv, S, HID, 13312, T, 0, 1.f, !idx_on_);
            if (idx_on_) MVB(cb, w.iqk, A0.X, S, A0.Iqk, S, HID, 640, T);  // S19 indexer q | raw key
            for (uint32_t t = 0; t < T; ++t) {  // every token's K/V entry lands before any token attends
                const Act a = act_at(t);
                PcPrep pr{a.Qkv, w.qn, w.kn, a.Q, a.G, kc_[l], vc_[l], a.Par, eps, 1e7f};
                vk::Ctx::dispatch(cb, pPr_, &pr, 26);
                if (idx_on_) { PcIdxPrep ip{a.Iqk, w.iqn, a.Qi, rk_[l], a.Par, eps, 1e7f}; vk::Ctx::dispatch(cb, pIp_, &ip, 5); }
            }
            BAR(cb, __LINE__);
            if (idx_on_) {  // S19: every raw key of the batch is written: each column selects its tokens
                if (idx_ps_) {  // S30: the T columns' scores in one pass over the keys
                    const Act a = act_at(0);
                    PcIdxScore sc{a.Qi, pk_[l], rk_[l], w.ikn, a.Par, a.Scr, uint32_t(act_stride_), T, isc_wg_, eps, 1e7f, 0};
                    vk::Ctx::dispatch(cb, pIsc_, &sc, isc_wg_);
                    BAR(cb, __LINE__);
                }
                for (uint32_t t = 0; t < T; ++t) {
                    const Act a = act_at(t);
                    PcIdxSel is{a.Qi, rk_[l], pk_[l], w.ikn, a.Sel, a.Par, t, eps, 1e7f, idx_gmin_, a.Scr};
                    vk::Ctx::dispatch(cb, pIs_, &is, 1);
                }
                BAR(cb, __LINE__);
            }
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcAttnSplit as{a.Q, kc_[l], vc_[l], a.Part, a.Par, attn_chunks_, 0, a.Sel};
                vk::Ctx::dispatch(cb, pAs_, &as, attn_chunks_, 2);
            }
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcAttnComb ac{a.Part, a.G, a.Ao, a.Par, attn_chunks_, 0, a.Sel};
                vk::Ctx::dispatch(cb, pAc_, &ac, 24);
            }
            BAR(cb, __LINE__);
            MVB(cb, w.oproj, A0.Ao, S, A0.Y, S, 6144, HID, T);
        }
        for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); PcPost p{a.H, a.Y, a.I, HC, 0}; vk::Ctx::dispatch(cb, pPo_, &p, HCH / 256); }
        BAR(cb, __LINE__);
        HCPRE_T(cb, w.hcm, 2, T);
        const uint64_t c = ctl_.addr;
        if (l + 1 < NL_) {  // router + depth-1 predictor for all T columns, then each token's two top-k
            const uint64_t pw = pred_w_.size() > 1 && pred_w_[1][l] ? pred_w_[1][l] : W_[l + 1].router;
            PcRouter2B r{w.router, pw, A0.X, A0.R, A0.R2, E, uint32_t(S), uint32_t(S), 0};
            vk::Ctx::dispatch(cb, pRt2B_[T], &r, 2 * E);
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcTopk2 tk{a.R, c + vmail(t, l), E, kr(l), l + 1, 0, a.R2, c + vpmail(t, l + 1), l + 2, 0};
                vk::Ctx::dispatch(cb, pTk2_, &tk, 2);
            }
        } else if (mtp_ok_ && o_.mtp_pref) {
            // S18: the MTP router on the last layer's MoE input = a prediction of the next draft's experts (row (t, L) of the
            // prediction mailboxes, which aliases the never-used row (t + 1, 0))
            PcRouter2B r{w.router, W_[L].router, A0.X, A0.R, A0.R2, E, uint32_t(S), uint32_t(S), 0};
            vk::Ctx::dispatch(cb, pRt2B_[T], &r, 2 * E);
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcTopk2 tk{a.R, c + vmail(t, l), E, kr(l), l + 1, 0, a.R2, c + vpmail(t, L), L + 1, 0};
                vk::Ctx::dispatch(cb, pTk2_, &tk, 2);
            }
        } else {
            for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); PcRouter r{w.router, a.X, a.R, E, 0}; vk::Ctx::dispatch(cb, pRt_, &r, E); }
            BAR(cb, __LINE__);
            for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); PcTopk tk{a.R, c + vmail(t, l), E, kr(l), l + 1, 0}; vk::Ctx::dispatch(cb, pTk_, &tk, 1); }
        }
        MVB(cb, w.sgate, A0.X, S, aWt_ + (uint64_t(l) * TAB + K) * 4, uint64_t(L + 1) * TAB * 4, HID, 1, T, 3);
    }
    bool dedupe(uint32_t T) const { return (!lm_rec_ || !std::getenv("PS5LM_LM_NODEDUPE")) && !o_.vb_nodedupe && T <= 4 && exmb_ok_[T > 2]; }
    const vk::Pipeline& exmb(uint32_t l, uint32_t T) const { return pExMB_[kidx(l)][T > 2]; }
    void EXMB_GU(VkCommandBuffer cb, uint32_t l, uint32_t T, bool streamed) {
        const Act a0 = act_at(0);
        PcExMvB m{ctl_.addr + vent(l, streamed), a0.ExXt, a0.ExPre, HID / 16, 640 / 16, ex_tr_gate(ex_k2_[l]), ex_tr_up(ex_k2_[l]), 2 * HID, HID, 2 * 640, 640,
                  uint32_t(act_stride_), 0};
        vk::Ctx::dispatch(cb, exmb(l, T), &m, 640 / 32, 2, T * K);
    }
    void EXMB_DN(VkCommandBuffer cb, uint32_t l, uint32_t T, bool streamed) {
        const Act a0 = act_at(0);
        PcExMvB m{ctl_.addr + vent(l, streamed), a0.ExHd, a0.ExYd, 640 / 16, HID / 16, ex_tr_down(), ex_tr_down(), 640, 0, HID, 0, uint32_t(act_stride_), 0};
        vk::Ctx::dispatch(cb, exmb(l, T), &m, HID / 32, 1, T * K);
    }
    // resident experts of the T tokens (tables A, rows (t, l)): gate/up -> act -> down; + every token's shared gate/up
    void HITS_T(VkCommandBuffer cb, uint32_t l, uint32_t T) {
        const int k = kidx(l);
        const uint64_t R = uint64_t(L + 1) * TAB * 8;  // table row stride per token
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            const uint64_t ta = aSlA_ + (uint64_t(t) * (L + 1) + l) * TAB * 8;
            PcExHad h{a.X, ta, a.ExXt};
            vk::Ctx::dispatch(cb, pExH_[k], &h, HID / 128, 2, K);
            PcGateUp gs{a.X, a.Hm + uint64_t(K) * 640 * 4, aSl_ + (uint64_t(t) * (L + 1) * TAB + uint64_t(l) * TAB + K) * 8, 4, 0};
            vk::Ctx::dispatch(cb, pGU_, &gs, 640 / 4, 1);
        }
        (void)R;
        vk::Ctx::barrier(cb);
        if (dedupe(T)) EXMB_GU(cb, l, T, false);
        else
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                const uint64_t ta = aSlA_ + (uint64_t(t) * (L + 1) + l) * TAB * 8;
                PcExMv m{ta, a.ExXt, a.ExPre, HID / 16, 640 / 16, ex_tr_gate(ex_k2_[l]), ex_tr_up(ex_k2_[l]), 2 * HID, HID, 2 * 640, 640};
                vk::Ctx::dispatch(cb, pExM_[k], &m, 640 / 32, 2, K);
            }
        vk::Ctx::barrier(cb);
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            const uint64_t ta = aSlA_ + (uint64_t(t) * (L + 1) + l) * TAB * 8;
            PcExAct ac{a.ExPre, ta, a.ExHd};
            vk::Ctx::dispatch(cb, pExA_[k], &ac, 640 / 128, K);
        }
        vk::Ctx::barrier(cb);
        if (dedupe(T)) EXMB_DN(cb, l, T, false);
        else
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                const uint64_t ta = aSlA_ + (uint64_t(t) * (L + 1) + l) * TAB * 8;
                PcExMv m{ta, a.ExHd, a.ExYd, 640 / 16, HID / 16, ex_tr_down(), ex_tr_down(), 640, 0, HID, 0};
                vk::Ctx::dispatch(cb, pExM_[k], &m, HID / 32, 1, K);
            }
        vk::Ctx::barrier(cb);
    }
    // after gate B: the streamed experts (tables B) gate/up -> act -> down, then combine (all experts) + shared down + POST
    void POSTMOE_T(VkCommandBuffer cb, uint32_t l, uint32_t T) {
        const int k = kidx(l);
        auto row = [&](uint32_t t) { return uint64_t(t) * (L + 1) + l; };
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            PcExHad h{a.X, aSlB_ + row(t) * TAB * 8, a.ExXt};
            vk::Ctx::dispatch(cb, pExH_[k], &h, HID / 128, 2, K);
        }
        vk::Ctx::barrier(cb);
        if (dedupe(T)) EXMB_GU(cb, l, T, true);
        else
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcExMv m{aSlB_ + row(t) * TAB * 8, a.ExXt, a.ExPre, HID / 16, 640 / 16, ex_tr_gate(ex_k2_[l]), ex_tr_up(ex_k2_[l]), 2 * HID, HID, 2 * 640, 640};
                vk::Ctx::dispatch(cb, pExM_[k], &m, 640 / 32, 2, K);
            }
        BAR(cb, __LINE__);
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            PcExAct ac{a.ExPre, aSlB_ + row(t) * TAB * 8, a.ExHd};
            vk::Ctx::dispatch(cb, pExA_[k], &ac, 640 / 128, K);
        }
        BAR(cb, __LINE__);
        if (dedupe(T)) EXMB_DN(cb, l, T, true);
        else
            for (uint32_t t = 0; t < T; ++t) {
                const Act a = act_at(t);
                PcExMv m{aSlB_ + row(t) * TAB * 8, a.ExHd, a.ExYd, 640 / 16, HID / 16, ex_tr_down(), ex_tr_down(), 640, 0, HID, 0};
                vk::Ctx::dispatch(cb, pExM_[k], &m, HID / 32, 1, K);
            }
        BAR(cb, __LINE__);
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            PcExCmb cm{a.ExYd, aSl_ + row(t) * TAB * 8, aWt_ + row(t) * TAB * 4, a.Ym2, K, 0};
            vk::Ctx::dispatch(cb, pExC_[k], &cm, HID / 128);
        }
        BAR(cb, __LINE__);
        for (uint32_t t = 0; t < T; ++t) {
            const Act a = act_at(t);
            const uint64_t sl = aSl_ + row(t) * TAB * 8, wt = aWt_ + row(t) * TAB * 4;
            PcDown ds{a.Hm + uint64_t(K) * 640 * 4, sl + uint64_t(K) * 8, wt + uint64_t(K) * 4, a.Ym2, a.Ym, 1, 2};
            vk::Ctx::dispatch(cb, pDN_, &ds, HID / 2);
        }
        BAR(cb, __LINE__);
        for (uint32_t t = 0; t < T; ++t) { const Act a = act_at(t); PcPost p{a.H, a.Ym, a.I2, HC, 0}; vk::Ctx::dispatch(cb, pPo_, &p, HCH / 256); }
        BAR(cb, __LINE__);
    }
    void HEAD_T(VkCommandBuffer cb, uint32_t T) {
        HCPRE_T(cb, mixer_, 0, T);
        const Act A0 = act_at(0);
        MVB(cb, lm_head_, A0.X, act_stride_, lg_.addr, uint64_t(VOCAB) * 4, HID, VOCAB, T);
        for (uint32_t t = 0; t < T; ++t) {
            PcArgmax am{lg_.addr + uint64_t(t) * VOCAB * 4, arg_.addr + uint64_t(t) * 16, VOCAB, 0};
            vk::Ctx::dispatch(cb, pAm_, &am, 1);
        }
        BAR(cb, __LINE__);
    }
    // ---- S21 wide prefill ----
    // Slice arena (reused by every slice): common rows X, I, I2, Y (= Ym), R, Rp (router split-K partials), Sg, Gu, Hs, Pre, Hd,
    // then region Z aliased by the phases that never overlap: HC pre (N, U, T, Tp, Il) | GDN (P, C, Core) | MoE (Xt, then Yd).
    void alloc_wide() {
        const uint64_t S = o_.wide_slice = std::max<uint32_t>(64, std::min<uint32_t>(o_.wide_slice, o_.prefill_chunk)) & ~63u;
        uint64_t off = 0;
        auto put = [&](uint64_t& f, uint64_t bytes) { f = off; off = (off + bytes + 255) & ~255ull; };
        put(wX_, S * HID * 4); put(wI_, S * 16); put(wI2_, S * 16); put(wY_, S * HID * 4); put(wR_, S * E * 4); put(wRp_, 8 * S * E * 4);
        put(wSg_, S * 4); put(wGu_, S * 1280 * 4); put(wHs_, S * 640 * 4); put(wPre_, S * K * 1280 * 4); put(wHd_, S * K * 640 * 4);
        put(wZ_, std::max<uint64_t>({S * (2 * HCH + LOWR + 16 * LOWR + 4) * 4 + 1024, S * (16480 + 10240 + 6144) * 4 + 1024, S * K * 2 * HID * 4}));
        const uint64_t base = d_.zalloc(off).first;
        for (uint64_t* f : {&wX_, &wI_, &wI2_, &wY_, &wR_, &wRp_, &wSg_, &wGu_, &wHs_, &wPre_, &wHd_, &wZ_}) *f += base;
        wN_ = wZ_; wU_ = wN_ + S * HCH * 4; wT_ = wU_ + S * HCH * 4; wTp_ = wT_ + S * LOWR * 4; wIl_ = wTp_ + 16 * S * LOWR * 4;
        wP_ = wZ_; wC_ = wP_ + S * 16480 * 4; wCore_ = wC_ + S * 10240 * 4;
        wXt_ = wZ_; wYd_ = wZ_;
        // attention layers (region Z, never live with the GDN / MoE uses): qkv, indexer qk, q, gate, indexer q, out, selection
        wQkv_ = wZ_; wIqk_ = wQkv_ + S * 13312 * 4; wQ_ = wIqk_ + S * 640 * 4; wG_ = wQ_ + S * 6144 * 4; wQi_ = wG_ + S * 6144 * 4;
        wAo_ = wQi_ + S * 512 * 4; wSel_ = wAo_ + S * 6144 * 4;
        if (wSel_ + S * 2064 * 4 > wZ_ + S * K * 2 * HID * 4) throw std::runtime_error("wide arena: attention rows do not fit");
        if (idx_on_) wScr_ = d_.zalloc(uint64_t(kWScrRows) * std::max<uint64_t>(64, o_.max_ctx / 4) * 4).first;
        std::tie(wEmb_, pWEmb_) = d_.zalloc(2 * S * HID * 2);  // bf16 embedding rows of two slices (sweep 0, ping-pong)
        // host-written tables (GPU-uncached control memory): pair slots / weights of the chunk, the entries, the router mails
        const uint64_t C = o_.prefill_chunk;
        kWSl_ = 0; kWWt_ = kWSl_ + C * K * 8; kWMail_ = (kWWt_ + C * K * 4 + 4095) & ~4095ull; kWEnt_ = kWMail_ + C * 256;
        const uint64_t ctl = kWEnt_ + C * K * sizeof(WEnt);
        wctl_ = ctx_.create_uncached(ctl);
        wctlp_ = static_cast<uint8_t*>(wctl_.map);
        std::memset(wctlp_, 0, ctl);
        std::fprintf(stderr, "wide prefill: slices of %llu tokens, arena %.0f MiB, tables %.1f MiB\n", (unsigned long long)S, double(off) / 1048576.0,
                     double(ctl) / 1048576.0);
    }
    bool wide_pre_layer(uint32_t s) const { return s < NL_ && !(s == PLE_LAYER && use_ple_) && (!W_[s].attn || wattn_ok_); }
    static constexpr uint32_t kClassFloor = 2 * E + 64;  // S24: live slots a size class must keep (see map_pool / ek_retire_one)
    static constexpr uint32_t kWScrRows = 32;  // idx_select scratch rows (queries per dispatch when scores spill; S33: 64 measured no faster)
    void WG_GEMM(VkCommandBuffer cb, uint64_t w, uint64_t x, uint32_t xs, uint64_t y, uint32_t ys, uint32_t Kd, uint32_t N, uint32_t T,
                 uint32_t mode = 0, float alpha = 1.f, uint32_t ks = 1, uint64_t part = 0, bool bf16 = false) {
        const vk::Pipeline& pg = bf16 ? pGb_ : pGq_;
        if (ks > 1) {
            PcGemm g{w, x, part, Kd, N, T, 0, 1.f, xs, N * 4, ks, 0};
            vk::Ctx::dispatch(cb, pg, &g, (N + 127) / 128, (T + 63) / 64, ks);
            vk::Ctx::barrier(cb);
            PcKsum r{part, y, N, T, ks, mode, alpha, N * 4, ys, 0};
            vk::Ctx::dispatch(cb, pKs_, &r, (N + 255) / 256, T);
        } else {
            PcGemm g{w, x, y, Kd, N, T, mode, alpha, xs, ys, 1, 0};
            vk::Ctx::dispatch(cb, pg, &g, (N + 127) / 128, (T + 63) / 64, 1);
        }
    }
    // HC pre-step for T tokens whose H rows start at h (stride HCH floats): norm -> down (split-K 16, silu, 1/HC) + inject
    // (w_dot) -> up (sigmoid) -> mix (X, and inj = 2 sigmoid(Il / HC) when inj != 0)
    void W_HCPRE(VkCommandBuffer cb, const HcW& hw, uint64_t h, uint32_t T, uint64_t inj, uint64_t xo = 0, uint32_t xos = HID * 4) {
        PcWNorm n{h, hw.norm, wN_, HCH * 4, HCH * 4, 1e-6f, 0};
        vk::Ctx::dispatch(cb, pWn_, &n, HC, T);
        BAR(cb, __LINE__);
        WG_GEMM(cb, hw.down, wN_, HCH * 4, wT_, LOWR * 4, HCH, LOWR, T, 2, 1.f / HC, 16, wTp_);
        if (inj) { PcWDot d{hw.inj, wN_, wIl_, HCH, HC, 0, 1.f, HCH * 4, 16}; vk::Ctx::dispatch(cb, pWd_, &d, T); }
        BAR(cb, __LINE__);
        WG_GEMM(cb, hw.up, wT_, LOWR * 4, wU_, HCH * 4, LOWR, HCH, T, 3, 1.f);
        BAR(cb, __LINE__);
        PcWMix m{wU_, wN_, xo ? xo : wX_, inj ? wIl_ : 0, inj, HCH * 4, HCH * 4, xos, 16, 16, 0};
        vk::Ctx::dispatch(cb, pWm_, &m, HID / 256, T);
        BAR(cb, __LINE__);
    }
    // GDN layer l, pre-MoE, tokens [a, a + T) of the chunk (H in the carry): HC -> inproj -> conv -> recurrence -> outproj ->
    // post -> HC (MoE input) -> router -> top-k mails (row a + t of the wide mail table)
    void W_PRE_GDN(VkCommandBuffer cb, uint32_t l, uint32_t a, uint32_t T) {
        const auto& w = W_[l];
        const uint64_t h = aCarry_ + uint64_t(a) * HCH * 4;
        W_HCPRE(cb, w.hca, h, T, wI_);
        WG_GEMM(cb, w.inproj, wX_, HID * 4, wP_, 16480 * 4, HID, 16480, T);
        BAR(cb, __LINE__);
        PcWConv c{wP_, w.conv, conv_state_[l], wC_, 10240, 16480 * 4, 10240 * 4, T};
        vk::Ctx::dispatch(cb, pWcv_, &c, 10240 / 256);
        BAR(cb, __LINE__);
        PcWStep g{wC_, wP_, gdn_state_[l], w.alog, w.dtb, w.gnorm, wCore_, 1e-6f, 10240 * 4, 16480 * 4, 6144 * 4, T, 0};
        vk::Ctx::dispatch(cb, pWst_, &g, 48);
        BAR(cb, __LINE__);
        WG_GEMM(cb, w.outproj, wCore_, 6144 * 4, wY_, HID * 4, 6144, HID, T);
        BAR(cb, __LINE__);
        W_TAIL(cb, l, a, T);
    }
    // attention layer l, pre-MoE, tokens [a, a + T) at positions wpos0_ + a ..: HC -> qkv (+ indexer qk) -> prep (K/V of every
    // token of the slice into the cache) + indexer prep -> pooled block keys of the slice -> selection per query -> attention
    // (online softmax, one workgroup per (kv head, query)) -> o_proj -> tail
    void W_PRE_ATTN(VkCommandBuffer cb, uint32_t l, uint32_t a, uint32_t T) {
        const auto& w = W_[l];
        const uint64_t h = aCarry_ + uint64_t(a) * HCH * 4;
        const uint32_t pos0 = wpos0_ + a;
        W_HCPRE(cb, w.hca, h, T, wI_);
        WG_GEMM(cb, w.qkv, wX_, HID * 4, wQkv_, 13312 * 4, HID, 13312, T);
        if (idx_on_) WG_GEMM(cb, w.iqk, wX_, HID * 4, wIqk_, 640 * 4, HID, 640, T);
        BAR(cb, __LINE__);
        PcPrepW pr{wQkv_, w.qn, w.kn, wQ_, wG_, kc_[l], vc_[l], 0, 1e-6f, 1e7f, 13312 * 4, 6144 * 4, pos0, 0};
        vk::Ctx::dispatch(cb, pWpr_, &pr, 26, T);
        if (idx_on_) { PcIdxPrepW ip{wIqk_, w.iqn, wQi_, rk_[l], 0, 1e-6f, 1e7f, 640 * 4, 512 * 4, pos0, 0}; vk::Ctx::dispatch(cb, pWip_, &ip, 5, T); }
        BAR(cb, __LINE__);
        if (idx_on_) {
            const uint32_t b0 = pos0 / 4, b1 = (pos0 + T) / 4;  // blocks whose completion position 4b+3 may lie in the slice
            PcIdxPool pl{rk_[l], pk_[l], w.ikn, b0, pos0, T, 1e-6f, 1e7f, 0};
            vk::Ctx::dispatch(cb, pWpl_, &pl, b1 - b0 + 1);
            BAR(cb, __LINE__);
            const bool spill = (pos0 + T) / 4 > std::min<uint32_t>(idx_gmin_, 8192);  // scores in the global scratch: <= 32 queries per dispatch
            const bool tiled = spill && pWisc_.pipe;  // S33: scores of kWScrRows queries per pass over the keys, then the radix selection
            const uint32_t scrs = uint32_t(std::max<uint64_t>(64, o_.max_ctx / 4) * 4);
            for (uint32_t t0 = 0; t0 < T; t0 += spill ? kWScrRows : T) {
                const uint32_t n = spill ? std::min(kWScrRows, T - t0) : T;
                if (tiled) {
                    PcIdxScoreW sc{wQi_ + uint64_t(t0) * 512 * 4, pk_[l], wScr_, 512 * 4, scrs, pos0 + t0, n, kIdxScoreWg, 0};
                    vk::Ctx::dispatch(cb, pWisc_, &sc, kIdxScoreWg);
                    vk::Ctx::barrier(cb);
                }
                PcIdxSelW is{wQi_, rk_[l], pk_[l], w.ikn, wSel_, 0, 0, 1e-6f, 1e7f, idx_gmin_, wScr_, 512 * 4, 2064 * 4,
                             scrs, pos0, t0, 0};
                vk::Ctx::dispatch(cb, tiled ? pWisp_ : pWis_, &is, 1, n);
                if (spill) vk::Ctx::barrier(cb);
            }
            BAR(cb, __LINE__);
        }
        PcWAttn at{wQ_, kc_[l], vc_[l], wG_, wAo_, wSel_, 6144 * 4, 6144 * 4, 2064 * 4, pos0};
        vk::Ctx::dispatch(cb, pWat_, &at, 2, T);
        BAR(cb, __LINE__);
        WG_GEMM(cb, w.oproj, wAo_, 6144 * 4, wY_, HID * 4, 6144, HID, T);
        BAR(cb, __LINE__);
        W_TAIL(cb, l, a, T);
    }
    // post (H += y x inj) -> HC (MoE input) -> router -> top-k mails
    void W_TAIL(VkCommandBuffer cb, uint32_t l, uint32_t a, uint32_t T) {
        const auto& w = W_[l];
        const uint64_t h = aCarry_ + uint64_t(a) * HCH * 4;
        PcWPost p{h, wY_, wI_, HCH * 4, HID * 4, 16, 0};
        vk::Ctx::dispatch(cb, pWp_, &p, HCH / 256, T);
        BAR(cb, __LINE__);
        const bool xm = xm_on();  // S22: X / I2 of the MoE input straight into the chunk rows (the MoE phase does not re-derive them)
        const uint64_t xr = xm ? aXY_ + uint64_t(a) * XYS : wX_;
        const uint32_t xrs = xm ? XYS : HID * 4;
        W_HCPRE(cb, w.hcm, h, T, xm ? aI2c_ + uint64_t(a) * 16 : wI2_, xm ? xr : 0, xrs);
        WG_GEMM(cb, w.router, xr, xrs, wR_, E * 4, HID, E, T, 0, 1.f, 8, wRp_, true);
        BAR(cb, __LINE__);
        PcWTopk tk{wR_, wctl_.addr + kWMail_ + uint64_t(a) * 256, E, kr(l), l + 1, E * 4, 256, 0};
        vk::Ctx::dispatch(cb, pWtk_, &tk, 1, T);
        BAR(cb, __LINE__);
    }
    // MoE of layer l for tokens [a, a + T) (H in the carry; the pair tables of the chunk and this slice's nent entries at
    // ent are host-written): HC (X, I2) -> shared gate/up + gate, routed input Hadamard -> routed gate/up (expert-major) +
    // shared act -> routed act -> routed down (expert-major) -> combine -> + shared down -> post
    void W_MOE(VkCommandBuffer cb, uint32_t l, uint32_t a, uint32_t T, uint64_t ent, uint32_t nent) {
        const auto& w = W_[l];
        const int k = kidx(l);
        const uint64_t h = aCarry_ + uint64_t(a) * HCH * 4;
        const uint64_t slots = wctl_.addr + kWSl_ + uint64_t(a) * K * 8, wts = wctl_.addr + kWWt_ + uint64_t(a) * K * 4;
        W_HCPRE(cb, w.hcm, h, T, wI2_);
        PcWDot sg{w.sgate, wX_, wSg_, HID, 1, 3, 1.f, HID * 4, 4};
        vk::Ctx::dispatch(cb, pWd_, &sg, T);
        WG_GEMM(cb, w.shared_slot, wX_, HID * 4, wGu_, 1280 * 4, HID, 1280, T);
        PcWHad hd{wX_, slots, wXt_, HID * 4, 0};
        vk::Ctx::dispatch(cb, pWh_[k], &hd, HID / 128, 2, T * K);
        BAR(cb, __LINE__);
        PcWSact sa{wGu_, wSg_, wHs_, 1280 * 4, 4, 640 * 4, 0};
        vk::Ctx::dispatch(cb, pWsa_, &sa, 640 / 128, T);
        if (nent) {
            PcXmvw m{ent, wXt_, wPre_, HID / 16, 640 / 16, ex_tr_gate(ex_k2_[l]), ex_tr_up(ex_k2_[l]), HID, 640, 2 * HID * 4, 1280 * 4};
            vk::Ctx::dispatch(cb, pXw_[k], &m, 640 / 32, 2, nent);
        }
        BAR(cb, __LINE__);
        PcExAct ac{wPre_, slots, wHd_};
        vk::Ctx::dispatch(cb, pExA_[k], &ac, 640 / 128, T * K);
        BAR(cb, __LINE__);
        if (nent) {
            PcXmvw m{ent, wHd_, wYd_, 640 / 16, HID / 16, ex_tr_down(), ex_tr_down(), 0, 0, 640 * 4, HID * 4};
            vk::Ctx::dispatch(cb, pXw_[k], &m, HID / 32, 1, nent);
        }
        BAR(cb, __LINE__);
        PcWCmb cm{wYd_, slots, wts, wY_, K, HID * 4};
        vk::Ctx::dispatch(cb, pWc_[k], &cm, HID / 128, T);
        BAR(cb, __LINE__);
        WG_GEMM(cb, w.shared_slot + qn::ExpertLayout<4>().down_off, wHs_, 640 * 4, wY_, HID * 4, 640, HID, T, 1);
        BAR(cb, __LINE__);
        PcWPost p{h, wY_, wI2_, HCH * 4, HID * 4, 16, 0};
        vk::Ctx::dispatch(cb, pWp_, &p, HCH / 256, T);
        BAR(cb, __LINE__);
    }
    void wide_pipelines() {
        const std::string& sd = o_.shader_dir;
        try {
            pGq_ = ctx_.pipeline(sd + "/gemm_q4.spv", sizeof(PcGemm), 0);
            pGb_ = ctx_.pipeline(sd + "/gemm_bf16.spv", sizeof(PcGemm), 0);
            pKs_ = ctx_.pipeline(sd + "/w_ksum.spv", sizeof(PcKsum), 0);
            pWn_ = ctx_.pipeline(sd + "/w_norm.spv", sizeof(PcWNorm));
            pWm_ = ctx_.pipeline(sd + "/w_mix.spv", sizeof(PcWMix), 0);
            pWp_ = ctx_.pipeline(sd + "/w_post.spv", sizeof(PcWPost), 0);
            pWtk_ = ctx_.pipeline(sd + "/w_topk.spv", sizeof(PcWTopk));
            pWcv_ = ctx_.pipeline(sd + "/w_gdn_conv.spv", sizeof(PcWConv), 0);
            pWst_ = ctx_.pipeline(sd + "/w_gdn_step.spv", sizeof(PcWStep));
            pWsa_ = ctx_.pipeline(sd + "/w_sact.spv", sizeof(PcWSact), 0);
            pWd_ = ctx_.pipeline(sd + "/w_dot.spv", sizeof(PcWDot));
            pWe_ = ctx_.pipeline(sd + "/w_embed.spv", sizeof(PcWEmbed), 0);
            const char* tags[3] = {"k25", "k3", "k35"};
            for (int k = 0; k < 3; ++k) {
                bool used = false;
                for (uint32_t l = 0; l < NL_ && l < L; ++l) used |= kidx(l) == k;
                if (!used) continue;
                pWh_[k] = ctx_.pipeline(sd + "/exl3_w_had_in_" + std::string(tags[k]) + ".spv", sizeof(PcWHad), 0);
                pWc_[k] = ctx_.pipeline(sd + "/exl3_w_combine_" + std::string(tags[k]) + ".spv", sizeof(PcWCmb), 0);
                pXw_[k] = ctx_.pipeline(sd + "/exl3_xmvw_" + std::string(tags[k]) + ".spv", sizeof(PcXmvw), 32);
                if (gctl_.map) {
                    try {  // S22 expert-major MoE
                        pXl64_[k] = ctx_.pipeline(sd + "/exl3_xmvl_" + std::string(tags[k]) + "_64b16.spv", sizeof(PcXmvl), 32);
                        pXl32_[k] = ctx_.pipeline(sd + "/exl3_xmvl_" + std::string(tags[k]) + "_32.spv", sizeof(PcXmvl), 32);
                        pWhG_[k] = ctx_.pipeline(sd + "/exl3_w_had_in_g_" + std::string(tags[k]) + ".spv", sizeof(PcWHadG), 0);
                        pGc_[k] = ctx_.pipeline(sd + "/exl3_g_combine_" + std::string(tags[k]) + ".spv", sizeof(PcGCmb), 0);
                        xm_ok_ = true;
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "expert-major MoE off (%s)\n", e.what());
                        gctl_ = {};
                        gctlp_ = nullptr;
                    }
                }
            }
            xm_ok_ = xm_ok_ && gctl_.map != nullptr;
            try { pWn10k_ = ctx_.pipeline(sd + "/w_norm10240.spv", sizeof(PcWNorm)); } catch (const std::exception&) {}
            wide_ok_ = wctl_.map != nullptr;
            try {  // S21 wide attention layers
                const std::string kv = o_.kvq ? "_kq" + std::to_string(o_.kvq) : o_.kv16 ? "_k16" : "";
                pWpr_ = ctx_.pipeline(sd + "/attn_prep" + kv + "w.spv", sizeof(PcPrepW));
                pWat_ = ctx_.pipeline(sd + "/w_attn" + kv + (idx_on_ ? "s" : "") + ".spv", sizeof(PcWAttn));
                if (idx_on_) {
                    pWip_ = ctx_.pipeline(sd + "/idx_prep_w.spv", sizeof(PcIdxPrepW), 0);
                    pWis_ = ctx_.pipeline(sd + "/idx_select_w.spv", sizeof(PcIdxSelW), 0);
                    if (!std::getenv("PS5LM_WIDX_OLD")) {  // S33: tiled scores (each key read once per 32 queries) + radix selection
                        pWisc_ = ctx_.pipeline(sd + "/idx_score_w.spv", sizeof(PcIdxScoreW));
                        pWisp_ = ctx_.pipeline(sd + "/idx_select_wps.spv", sizeof(PcIdxSelW), 0);
                    }
                    pWpl_ = ctx_.pipeline(sd + "/idx_pool.spv", sizeof(PcIdxPool));
                }
                wattn_ok_ = !o_.attn_legacy && !std::getenv("PS5LM_WIDE_NOATTN");
                wmtp_ok_ = mtp_ok_ && pWn10k_.pipe && pWe_.pipe && W_.size() > L && W_[L].attn;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "wide prefill: attention layers in 4-token groups (%s)\n", e.what());
                wattn_ok_ = false;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "wide prefill: off (%s)\n", e.what());
            wide_ok_ = false;
        }
        wtsq_ = ctx_.timestamps(2 * 64);
        wfence_[0] = ctx_.fence();
        wfence_[1] = ctx_.fence();
    }
    // Waits for cb (submitted with fence f) with service(); GPU-pure time from its two timestamps (query pair q); frees cb
    void wide_wait(VkCommandBuffer cb, uint32_t q, VkFence f) {
        auto a = Clock::now();
        VkResult r;
        while ((r = vkGetFenceStatus(ctx_.dev, f)) == VK_NOT_READY) { service(); __builtin_ia32_pause(); }
        vk::check(r, "vkGetFenceStatus");
        vkResetFences(ctx_.dev, 1, &f);
        stats_.t_gpu += secs(a, Clock::now());
        uint64_t ts[2] = {};
        vkGetQueryPoolResults(ctx_.dev, wtsq_, 2 * q, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT);
        stats_.t_gpu_pure += double(ts[1] - ts[0]) * ctx_.ts_period_ns * 1e-9;
        stats_.pf_gpu[pf_phase_] += double(ts[1] - ts[0]) * ctx_.ts_period_ns * 1e-9;
        vkFreeCommandBuffers(ctx_.dev, ctx_.pool, 1, &cb);
    }
    void prof_begin(VkCommandBuffer cb, int slot) {
        prof_cur_ = slot;
        if (slot < 0) return;
        prof_n_[slot] = 0;
        prof_line_[slot].clear();
        vkCmdResetQueryPool(cb, profq_, uint32_t(slot) * 512, 512);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profq_, uint32_t(slot) * 512);
    }
    VkCommandBuffer wide_begin(uint32_t q) {
        VkCommandBuffer cb = ctx_.cmd();
        vk::Ctx::begin(cb, true);
        vkCmdResetQueryPool(cb, wtsq_, 2 * q, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, wtsq_, 2 * q);
        return cb;
    }
    void wide_end(VkCommandBuffer cb, uint32_t q) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, wtsq_, 2 * q + 1);
        vk::Ctx::end(cb);
    }

    // ---- S20 layer-major prefill steps ----
    // lmcb_[s] for one 4-token group g (read by carry.comp from the group word): carry-in H; phase B of layer s-1 (X, I2
    // re-derived from the carried post-attention H by the same HCPRE kernels = bit-identical, shared gate/up, sgate, then
    // every routed expert from table B, combine, inject); phase A of layer s (PRE_T: attention/GDN + router -> mailboxes);
    // carry-out H. No gates: the host binds the group's layer s-1 experts (all ready) before the submit.
    void CARRY(VkCommandBuffer cb, uint32_t dir, uint32_t T) {
        PcCarry c{aCarry_, act_base_ + act_off_.H, aLmG_, uint32_t(act_stride_), HCH, T, dir};
        vk::Ctx::dispatch(cb, pCarry_, &c, HCH / 256, T);
    }
    void record_lm() {
        if (!o_.prefill_chunk || !pCarry_.pipe || !tok_cb_[0] || TB_ < 4 || o_.hc_mode != 1 || !o_.router_fuse || o_.vb_off ||
            !pXrB_[4].pipe)
            { o_.prefill_chunk = 0; return; }
        const uint32_t T = 4;
        const uint64_t S = act_stride_;
        nock_ = lm_rec_ = true;
        lm_dedupe_ = dedupe(T);
        prof_cur_ = -1;
        lmcb_.assign(NL_ + 1, VK_NULL_HANDLE);
        lmtsq_ = ctx_.timestamps(2 * (NL_ + 1));
        for (uint32_t s = 0; s <= NL_; ++s) {
            VkCommandBuffer cb = ctx_.cmd();
            vk::Ctx::begin(cb, false);
            vkCmdResetQueryPool(cb, lmtsq_, 2 * s, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, lmtsq_, 2 * s);
            prof_cur_ = (lmprof_ && (s == 4 || s == 7)) ? (s == 4 ? 0 : 1) : -1;  // S21: PS5LM_LMPROF = per-BAR timestamps of steps 4 (GDN) / 7 (attention)
            if (prof_cur_ >= 0) {
                prof_n_[prof_cur_] = 0;
                prof_line_[prof_cur_].clear();
                vkCmdResetQueryPool(cb, profq_, uint32_t(prof_cur_) * 512, 512);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profq_, uint32_t(prof_cur_) * 512);
            }
            if (s > 0) {
                CARRY(cb, 0, T);
                BAR(cb, __LINE__);
                const uint32_t l = s - 1;
                const auto& w = W_[l];
                HCPRE_T(cb, w.hcm, 2, T);
                for (uint32_t t = 0; t < T; ++t) {
                    const Act a = act_at(t);
                    PcGateUp gs{a.X, a.Hm + uint64_t(K) * 640 * 4, aSl_ + (uint64_t(t) * (L + 1) * TAB + uint64_t(l) * TAB + K) * 8, 4, 0};
                    vk::Ctx::dispatch(cb, pGU_, &gs, 640 / 4, 1);
                }
                MVB(cb, w.sgate, act_at(0).X, S, aWt_ + (uint64_t(l) * TAB + K) * 4, uint64_t(L + 1) * TAB * 4, HID, 1, T, 3);
                BAR(cb, __LINE__);
                POSTMOE_T(cb, l, T);
            }
            if (s < NL_) {
                PRE_T(cb, s, T);
                vk::Ctx::barrier(cb);
            }
            CARRY(cb, 1, T);
            BAR(cb, __LINE__);
            prof_cur_ = -1;
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, lmtsq_, 2 * s + 1);
            vk::Ctx::end(cb);
            lmcb_[s] = cb;
        }
        if (o_.wide && wctl_.map) {  // S21: PRE-only group steps (the wide MoE already ran; H comes from and goes to the carry)
            wide_pipelines();
            lmcbP_.assign(NL_, VK_NULL_HANDLE);
            for (uint32_t s = 0; s < NL_; ++s) {
                if (wide_pre_layer(s)) continue;
                VkCommandBuffer cb = ctx_.cmd();
                vk::Ctx::begin(cb, false);
                vkCmdResetQueryPool(cb, lmtsq_, 2 * s, 2);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, lmtsq_, 2 * s);
                CARRY(cb, 0, T);
                vk::Ctx::barrier(cb);
                PRE_T(cb, s, T);
                vk::Ctx::barrier(cb);
                CARRY(cb, 1, T);
                vk::Ctx::barrier(cb);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, lmtsq_, 2 * s + 1);
                vk::Ctx::end(cb);
                lmcbP_[s] = cb;
            }
            std::fprintf(stderr, "wide prefill: %s, attention layers %s\n", wide_ok_ ? "on (all MoE, GDN layers; PLE layer in 4-token groups)" : "OFF",
                         wattn_ok_ ? "wide" : "in 4-token groups");
        }
        cb_carry_in_ = ctx_.cmd();
        vk::Ctx::begin(cb_carry_in_, false);
        CARRY(cb_carry_in_, 0, T);
        vk::Ctx::barrier(cb_carry_in_);
        vk::Ctx::end(cb_carry_in_);
        nock_ = lm_rec_ = false;
        set_tok(0, 1);
        std::fprintf(stderr, "layer-major prefill: chunks of <= %u tokens (4-token groups, %u steps), carry %.0f MiB\n", o_.prefill_chunk,
                     NL_ + 1, double(o_.prefill_chunk) * HCH * 4 / 1048576.0);
    }
    void record_vonecb() {
        if (!tok_cb_[0] || TB_ < 2 || o_.bits != 3 || o_.hc_mode != 1 || o_.predict != 1 || !o_.router_fuse || o_.vb_off) return;
        const uint64_t S = act_stride_;
        const uint32_t rowS = uint32_t((L + 1) * kStageRow), rowT8 = uint32_t((L + 1) * TAB * 8), rowT4 = uint32_t((L + 1) * TAB * 4);
        const uint32_t nts = 2 + 2 * (2 * NL_ + 1);
        for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t T = 2; T <= TB_ && T <= 4; ++T) {
            const bool pf = pass == 1;  // S20: prefill variant
            if (!pf && T > 1 + o_.spec_k) continue;
            nock_ = pf;
            tsqv_ = ctx_.timestamps(nts);
            tsqv_n_ = 0;
            vgate_kind_.clear();
            VkCommandBuffer cb = ctx_.cmd();
            vk::Ctx::begin(cb, false);
            vkCmdResetQueryPool(cb, tsqv_, 0, nts);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsqv_, tsqv_n_++);
            for (uint32_t s = 0; s <= NL_; ++s) {
                prof_cur_ = (vprof_ && !pf && T == 2 && (s == 3 || s == 4)) ? int(s) - 3 : -1;
                if (prof_cur_ >= 0) {
                    prof_n_[prof_cur_] = 0;
                    prof_line_[prof_cur_].clear();
                    vkCmdResetQueryPool(cb, profq_, uint32_t(prof_cur_) * 512, 512);
                    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profq_, uint32_t(prof_cur_) * 512);
                }
                if (s > 0) {
                    const uint64_t r = vstage(0, s - 1);
                    GATEV(cb, 1, r + 452, T, rowS, rowT8, rowT4, r + 128, aSl_ + uint64_t(s - 1) * TAB * 8, 2 * K, r + 256, aSlB_ + uint64_t(s - 1) * TAB * 8, 2 * K,
                          r + 384, aWt_ + uint64_t(s - 1) * TAB * 4, K);
                    POSTMOE_T(cb, s - 1, T);
                }
                if (s == NL_) { if (!pf) HEAD_T(cb, T); break; }
                if (s == PLE_LAYER && use_ple_) GATEV(cb, 2, kVEFlag, T, HID * 4, uint32_t(S), 0, kVE, act_at(0).E, HID);
                PRE_T(cb, s, T);
                vk::Ctx::barrier(cb);
                const uint64_t r = vstage(0, s);
                GATEV(cb, 0, r + 448, T, rowS, rowT8, 0, r, aSlA_ + uint64_t(s) * TAB * 8, 2 * K);
                HITS_T(cb, s, T);
                if (prof_cur_ >= 0) BAR(cb, __LINE__);
                prof_cur_ = -1;
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsqv_, tsqv_n_++);
            if (tsqv_n_ != nts) throw std::runtime_error("verify one-cb timestamp count");
            vk::Ctx::end(cb);
            vdedupe_[T] = dedupe(T);
            (pf ? vcbp_ : vcb_)[T] = cb;
            (pf ? vtsqp_ : vtsq_)[T] = tsqv_;
            (pf ? vgate_kindsp_ : vgate_kinds_)[T] = vgate_kind_;
        }
        nock_ = false;
        set_tok(0, 1);
        std::fprintf(stderr, "verify one-cb (stage C): batches T = 2..%u recorded, batched dense kernels %s\n", std::min<uint32_t>(TB_, 4),
                     o_.vb_nobatch ? "OFF" : "on");
    }

    void record() {
        ov_ = o_.overlap;
        profq_ = ctx_.timestamps(1024);
        tsq_ = ctx_.timestamps(2 * (NL_ + 1));
        auto begin_step = [&](uint32_t s) {
            VkCommandBuffer cb = ctx_.cmd();
            vk::Ctx::begin(cb, false);
            vkCmdResetQueryPool(cb, tsq_, 2 * s, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tsq_, 2 * s);
            return cb;
        };
        auto end_step = [&](VkCommandBuffer cb, uint32_t s) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, tsq_, 2 * s + 1);
            vk::Ctx::end(cb);
        };
        steps_.resize(NL_);
        for (uint32_t s = 0; s < NL_; ++s) {
            VkCommandBuffer cb = begin_step(s);
            prof_cur_ = (s == 3 && NL_ > 5) ? 0 : (s == 4 && NL_ > 5) ? 1 : -1;  // 3 = attention layer, 4 = GDN layer
            if (prof_cur_ >= 0 && (o_.profile || vprof_)) {
                vkCmdResetQueryPool(cb, profq_, uint32_t(prof_cur_) * 512, 512);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profq_, uint32_t(prof_cur_) * 512);
            }
            if (s == 0) SNAP(cb, 0);
            if (s > 0) { POSTMOE(cb, s - 1); SNAP(cb, s); }
            PRE(cb, s);
            prof_cur_ = -1;
            end_step(cb, s);
            steps_[s] = cb;
        }
        for (int head = 0; head < 2; ++head) {
            VkCommandBuffer cb = begin_step(NL_);
            POSTMOE(cb, NL_ - 1);
            SNAP(cb, NL_);
            if (head && full()) HEAD(cb);
            end_step(cb, NL_);
            (head ? step_head_ : step_nohead_) = cb;
        }
        if (mtp_ok_) {
            // S16 MTP: fusion -> layer L pre-MoE (attention + router) | host binds layer L experts | MoE -> MTP mixer -> lm_head
            auto fusion = [&](VkCommandBuffer cb) {
                PcNorm nh{A_.H, mtp_nh_, A_.N, HCH, 1e-6f};  // one group over all 4 streams (SGLang: hidden_norm_size = hc * H)
                vk::Ctx::dispatch(cb, pHn_, &nh, 1);
                PcNorm ne{aEmb_, mtp_ne_, aEn_, HID, 1e-6f};
                vk::Ctx::dispatch(cb, pHn_, &ne, 1);
                BAR(cb, __LINE__);
                MV(cb, mtp_fce_, aEn_, aEf_, HID, HID);
                uint64_t so = 0, doff = 0;
                VkBuffer sb = d_.buf_of(aEf_, so), db = d_.buf_of(A_.H, doff);
                VkBufferCopy rg[HC];
                for (uint32_t c = 0; c < HC; ++c) rg[c] = {so, doff + uint64_t(c) * HID * 4, HID * 4};
                vkCmdCopyBuffer(cb, sb, db, HC, rg);
                BAR(cb, __LINE__);
                for (uint32_t c = 0; c < HC; ++c) MV(cb, mtp_fch_, A_.N + uint64_t(c) * HID * 4, A_.H + uint64_t(c) * HID * 4, HID, HID, 1, 1.f, false);
                BAR(cb, __LINE__);
            };
            // one set per activation block b: the head reads block b's residual H (a verify batch leaves token t's in block t)
            // and uses block b as scratch; tables/logits use row 0. Its own position is block b's Par.
            for (uint32_t b = 0; b < TB_; ++b) {
                A_ = act_at(b);
                VkCommandBuffer pre = ctx_.cmd(), post = ctx_.cmd(), fill = ctx_.cmd();
                vk::Ctx::begin(pre, false);
                fusion(pre);
                PRE(pre, L);
                vk::Ctx::end(pre);
                vk::Ctx::begin(post, false);
                POSTMOE(post, L);
                HEAD(post, &mtp_mixer_);
                vk::Ctx::end(post);
                vk::Ctx::begin(fill, false);  // KV entry only: fusion, attention hyper-connection, qkv, prep (kc/vc[L] at p)
                fusion(fill);
                HCPRE(fill, W_[L].hca, A_.I);
                MV(fill, W_[L].qkv, A_.X, A_.Qkv, HID, 13312, 0, 1.f, !idx_on_);
                if (idx_on_) MV(fill, W_[L].iqk, A_.X, A_.Iqk, HID, 640);
                PcPrep pr{A_.Qkv, W_[L].qn, W_[L].kn, A_.Q, A_.G, kc_[L], vc_[L], A_.Par, 1e-6f, 1e7f};
                vk::Ctx::dispatch(fill, pPr_, &pr, 26);
                if (idx_on_) {  // S19: the MTP layer's raw indexer key at p
                    PcIdxPrep ip{A_.Iqk, W_[L].iqn, A_.Qi, rk_[L], A_.Par, 1e-6f, 1e7f};
                    vk::Ctx::dispatch(fill, pIp_, &ip, 5);
                    // S30: and its pooled block key when p completes the block. A draft query pools only blocks completed
                    // inside its own batch: blocks completed by fills (accepted tokens, prefill) were never pooled.
                    BAR(fill, __LINE__);
                    PcIdxPoolP pp{rk_[L], pk_[L], W_[L].ikn, A_.Par, 1e-6f, 1e7f};
                    vk::Ctx::dispatch(fill, pIpp_, &pp, 1);
                }
                BAR(fill, __LINE__);
                vk::Ctx::end(fill);
                cb_mtp_pre_.push_back(pre);
                cb_mtp_post_.push_back(post);
                cb_mtp_fill_.push_back(fill);
            }
            A_ = act_at(0);
        }
        if (o_.overlap) {
            hits_.resize(L + 1);
            for (uint32_t l = 0; l < NL_ + (mtp_ok_ ? 1u : 0u); ++l) {  // mtp_ok_ implies NL_ == L: row L = MTP
                if (l == L) {  // S16: the MTP layer runs in any activation block -> one pre-pass per block
                    for (uint32_t b = 0; b < TB_; ++b) { A_ = act_at(b); hits_mtp_.push_back(record_hits(L)); }
                    A_ = act_at(0);
                    hits_[L] = hits_mtp_[0];
                    continue;
                }
                hits_[l] = record_hits(l);
            }
        }
        fence_ = ctx_.fence();
        if (!snaps_.empty() || o_.persist) {
            stage_ = ctx_.create(kSeg, true, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            uint64_t so = 0;
            VkBuffer sb = d_.buf_of(rstate_, so);
            for (uint64_t off = 0; off < rstate_bytes_; off += kSeg) {
                const uint64_t len = std::min<uint64_t>(kSeg, rstate_bytes_ - off);
                for (int dir = 0; dir < 2; ++dir) {
                    VkCommandBuffer cb = ctx_.cmd();
                    vk::Ctx::begin(cb, false);
                    VkBufferCopy rg = dir == 0 ? VkBufferCopy{so + off, 0, len} : VkBufferCopy{0, so + off, len};
                    vkCmdCopyBuffer(cb, dir == 0 ? sb : stage_.buf, dir == 0 ? stage_.buf : sb, 1, &rg);
                    vk::Ctx::barrier(cb);
                    vk::Ctx::end(cb);
                    (dir == 0 ? seg_save_ : seg_restore_).push_back(cb);
                }
            }
        }
        record_verify();
        record_onecb();
        record_vonecb();
        record_lm();
        cb_reset_ = ctx_.cmd();
        vk::Ctx::begin(cb_reset_, false);
        {
            uint64_t off = 0;
            VkBuffer b = d_.buf_of(rstate_, off);
            vkCmdFillBuffer(cb_reset_, b, off, rstate_bytes_, 0);
            vk::Ctx::barrier(cb_reset_);
        }
        vk::Ctx::end(cb_reset_);
    }

    EngineOptions o_;
    vk::Ctx ctx_;
    Dev d_;
    uint32_t NL_ = L;
    std::vector<LayerW> W_;
    std::vector<std::vector<uint64_t>> pred_w_;  // [depth][source layer] trained predictor (0 = use the router)
    std::vector<uint64_t> gdn_state_, conv_state_, kc_, vc_;
    uint64_t rstate_ = 0, rstate_bytes_ = 0, ple_state_ = 0;
    uint64_t rstate_core_ = 0, rk_cur_ = 0;   // S30: recurrent state without the rk rings; next ring
    static constexpr uint32_t kRkRing = 1024;  // S30: >= wide slice (512) + 3; power of two (shaders mask with RKR - 1)
    bool use_ple_ = false;
    uint64_t ple_key_ = 0, ple_val_ = 0, ple_nk_ = 0, ple_nq_ = 0, ple_nc_ = 0, ple_cw_ = 0, lm_head_ = 0;
    HcW mixer_{};
    // S16 MTP head
    bool mtp_ok_ = false;
    float mtp_conf_ = 0;
    uint32_t mtp_pred_[K] = {};
    bool mtp_pred_valid_ = false;
    HcW mtp_mixer_{};
    uint64_t mtp_fce_ = 0, mtp_fch_ = 0, mtp_ne_ = 0, mtp_nh_ = 0, aEmb_ = 0, aEn_ = 0, aEf_ = 0;
    void* pEmb_ = nullptr;
    std::vector<VkCommandBuffer> cb_mtp_pre_, cb_mtp_post_, cb_mtp_fill_;  // [activation block]
    NGram ng_;
    io_uring ngr_{};
    bool ngr_ok_ = false;
    uint32_t ng_pending_ = 0;
    static constexpr uint32_t kMaxTB = 8;  // S16: verify batch limit (n-gram buffers)
    uint32_t ng_tokens_ = 1;               // tokens whose rows the pending n-gram batch holds
    uint64_t ngoff_[kMaxTB * 16] = {};
    alignas(64) uint16_t ngbuf_[kMaxTB * 16 * 160] = {};
    int emb_fd_ = -1;
    std::vector<uint16_t> row_;

    // S16 per-token activation blocks (see alloc_activations); A_ = the block the recording helpers currently target
    struct Act {
        uint64_t H, N, T, S, Il, I, I2, X, P, C, Core, Y, Qkv, Q, G, Ao, Part, Key, Keyn, Qn, Val, Gv, Gvn, R, Hm, Ym, Ym2, ExXt, ExPre, ExHd,
            ExYd, Psum, R2, E, Iqk, Qi, Sel, Par, Scr;
    };
    Act act_at(uint32_t t) const {
        const uint64_t b = act_base_ + uint64_t(t) * act_stride_;
        Act a = act_off_;
        for (uint64_t* f = &a.H; f <= &a.Scr; ++f) *f += b;
        return a;
    }
    uint64_t rw(uint32_t l) const { return uint64_t(t_) * (L + 1) + l; }  // table row of (recording token, layer)
    Act A_{}, act_off_{};
    uint32_t TB_ = 1, t_ = 0;
    // S20: idx_select uses the global score scratch above this many blocks (always above 8192; PS5LM_IDX_GMIN=0 tests it)
    uint32_t idx_gmin_ = std::getenv("PS5LM_IDX_GMIN") ? uint32_t(std::stoul(std::getenv("PS5LM_IDX_GMIN"))) : 0xFFFFFFFFu;
    uint64_t sin_ = 0, sout_ = 0;  // S16: recurrent-state address deltas while recording (verify checkpoints); 0 = in place
    bool ov_ = true;
    bool nancheck_ = std::getenv("PS5LM_NANCHECK") != nullptr;
    // S21 wide prefill state
    bool wide_ok_ = false;
    uint64_t wX_ = 0, wI_ = 0, wI2_ = 0, wY_ = 0, wR_ = 0, wRp_ = 0, wSg_ = 0, wGu_ = 0, wHs_ = 0, wPre_ = 0, wHd_ = 0, wZ_ = 0;
    uint64_t wN_ = 0, wU_ = 0, wT_ = 0, wTp_ = 0, wIl_ = 0, wP_ = 0, wC_ = 0, wCore_ = 0, wXt_ = 0, wYd_ = 0;
    uint64_t kWSl_ = 0, kWWt_ = 0, kWMail_ = 0, kWEnt_ = 0;
    vk::Buffer wctl_;
    uint8_t* wctlp_ = nullptr;
    void* hCarry_ = nullptr;
    vk::Pipeline pGq_, pGb_, pKs_, pWn_, pWm_, pWp_, pWtk_, pWcv_, pWst_, pWsa_, pWd_, pWh_[3], pWc_[3], pXw_[3];
    vk::Pipeline pWpr_, pWip_, pWis_, pWpl_, pWat_, pWe_;
    EkBuf cek_;                       // S21 elastic prefill carry
    EkBuf xek_;                       // S22 chunk rows [X | Y] (routed MoE input / output of every chunk token), elastic
    uint64_t xek_base_ = 0, aXY_ = 0, aI2c_ = 0;
    static constexpr uint32_t XYS = 2 * HID * 4, kXyBase = 256;  // bytes per [X | Y] row, rows backed permanently
    vk::Buffer gctl_;                 // S22 group tables (host-written): pair slots / tokens, plist, entries, token entries
    uint8_t* gctlp_ = nullptr;
    uint64_t kGSl_ = 0, kGTk_ = 0, kGPl_ = 0, kGE64_ = 0, kGE32_ = 0, kGEw_ = 0, kGTe_ = 0, gcap_e_ = 0;
    vk::Pipeline pXl64_[3], pXl32_[3], pWhG_[3], pGc_[3];
    bool xm_ok_ = false;
    bool wmtp_ok_ = false;            // S22 wide MTP fill (needs the wide attention prep + w_norm10240)
    vk::Pipeline pWn10k_;
    std::vector<uint8_t> xc_ok_;      // S22: layer l's X / I2 rows were written by its wide PRE (else re-derived)
    std::vector<int32_t> wtix_;       // S22: token -> token entry of the group being built (-1)
    bool xm_on() const { return o_.wide_xm && xm_ok_; }
    uint64_t cek_base_ = 0, wEmb_ = 0;
    void* pWEmb_ = nullptr;
    const int64_t* wv_ = nullptr;
    static constexpr uint32_t kCarryBase = 1024;  // carry tokens backed permanently (the rest borrowed from the pool)
    bool wattn_ok_ = false;
    uint32_t wpos0_ = 0;
    uint64_t wQkv_ = 0, wIqk_ = 0, wQ_ = 0, wG_ = 0, wQi_ = 0, wAo_ = 0, wSel_ = 0, wScr_ = 0;
    std::vector<VkCommandBuffer> lmcbP_;
    std::vector<WEnt> wents_;
    VkQueryPool wtsq_ = VK_NULL_HANDLE;
    VkFence wfence_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool lmprof_ = std::getenv("PS5LM_LMPROF") != nullptr;  // S21: per-BAR profile of layer-major steps 4 (MoE l3 + GDN l4) / 5
    bool lmprof2_ = std::getenv("PS5LM_LMPROF") && std::getenv("PS5LM_LMPROF")[0] == '2';  // S22: slot 0 = MoE l3 (not attention l7)
    bool vprof_ = std::getenv("PS5LM_VPROF") != nullptr;  // S18: timestamp every BAR of verify (T = 2) steps 3 and 4               // POSTMOE: resident experts already ran in hits_ (false while recording verify steps)
    std::vector<uint64_t> ck_;     // S16: TB_ - 1 recurrent-state checkpoints (rstate_bytes_ each); batch token t < T-1 writes ck_[t]
    std::vector<std::vector<VkCommandBuffer>> vsteps_;  // [T - 2][step] verify command buffers
    std::vector<VkCommandBuffer> ck_restore_;           // [m]: checkpoint m -> live state
    uint64_t act_base_ = 0, act_stride_ = 0;
    uint8_t* act_host_ = nullptr;
    uint64_t aSc_ = 0, aZero_ = 0, aMail_ = 0, aPmail_ = 0, aSl_ = 0, aWt_ = 0, aDump_ = 0, aMoe_ = 0, aSlA_ = 0, aSlB_ = 0;
    void *pH_ = nullptr, *pE_ = nullptr, *pPar_ = nullptr, *pMail_ = nullptr, *pPmail_ = nullptr, *pSl_ = nullptr, *pWt_ = nullptr, *pDump_ = nullptr, *pMoe_ = nullptr, *pSlA_ = nullptr, *pSlB_ = nullptr;
    vk::Buffer lg_, arg_;

    platform::UdmabufSlab slab_;
    std::unique_ptr<ExpertStreamer> st_;
    std::vector<vk::Buffer> sch_;
    uint32_t slot_bytes_ = 0, spc_ = 0, n_slots_ = 0;
    std::vector<uint32_t> ex_k2_, ex_slot_;   // S15b EXL3 per-layer geometry
    std::vector<uint64_t> ex_base_, ex_off_, ex_chunk_start_;
    std::vector<uint16_t> ex_chunk_;
    vk::Pipeline pExH_[3], pExM_[3], pExA_[3], pExC_[3];  // [KH - 5]: K=2.5, K=3, K=3.5
    int kidx(uint32_t l) const { return int(ex_k2_[l]) - 5; }

    uint32_t attn_chunks_ = 0;
    vk::Pipeline pSq_, pHd2_, pHu2_;  // S17 hc_mode 3
    vk::Pipeline pRt2_, pTk2_;        // S17 fused router + predictor
    vk::Pipeline pHd_, pHu_, pDX_, pDM_, pAs_, pAc_, pMv_, pXr2560_, pLds6144_, pHn_, pMx_, pPo_, pCv_, pGd_, pPr_, pAt_, pPg_, pPc_, pAm_, pGU_, pDN_, pGUr_, pDNr_, pRt_, pTk_;
    VkQueryPool profq_ = VK_NULL_HANDLE, tsq_ = VK_NULL_HANDLE;
    uint32_t prof_n_[2] = {0, 0};
    std::vector<int> prof_line_[2];
    int prof_cur_ = -1;
    std::vector<VkCommandBuffer> steps_, hits_, hits_mtp_;
    VkCommandBuffer step_head_ = VK_NULL_HANDLE, step_nohead_ = VK_NULL_HANDLE, cb_reset_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    struct Snap {
        uint64_t last_use = 0;
        std::vector<uint8_t> host;  // recurrent state copy in ordinary (movable, unpinned) memory
        std::vector<int64_t> hist;
        bool valid = false;
    };
    static constexpr uint64_t kSeg = 16ull << 20;
    vk::Buffer stage_;
    std::vector<VkCommandBuffer> seg_save_, seg_restore_;
    std::vector<int64_t> hist_, kv_hist_;
    struct KvReg { uint64_t addr, per_pos; VkBuffer sparse; };  // S25: every per-position cache, in allocation order
    std::vector<KvReg> kvb_;
    uint8_t* bounce_ = nullptr;  // S25: kSeg bytes of ordinary page-aligned memory (O_DIRECT cannot pin a GTT mapping)
    std::vector<uint16_t> trt_[2];
    std::vector<uint64_t> rk_, pk_;  // S19 QSA indexer caches per attention layer (raw keys fp16 per position, pooled keys fp32 per block)
    bool idx_on_ = false;  // S19 token-route tables (layer 0, MTP layer), 0xFFFF = unknown
    std::vector<Snap> snaps_;
    uint64_t use_clock_ = 0;
    std::vector<SlotId> pinned_;
    std::vector<uint32_t> pf_cnt_;   // S25 chunk lookahead: [layer][expert] pairs routed in the last layer-major chunk
    std::vector<uint8_t> pf_seen_, pf_mark_;  // [layer] routed before; [expert] read ahead for the layer being swept
    std::vector<SlotId> pf_pins_;
    uint32_t pf_phase_ = 0, pf_chunk_ = 0;
    static constexpr uint32_t kPfAheadMin = 1024;
    // S17 one cb per token
    vk::Buffer ctl_;
    uint8_t* ctlp_ = nullptr;
    vk::Pipeline pGate_;
    VkQueryPool tsq1_ = VK_NULL_HANDLE;
    uint32_t tsq1_n_ = 0;
    std::vector<uint8_t> gate_kind_;  // per recorded gate: 0 = A, 1 = B, 2 = N
    VkCommandBuffer tok_cb_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    // S18 verify one-cb: [T] command buffer, its timestamp pool, its gate kinds
    VkCommandBuffer vcb_[5] = {};
    VkCommandBuffer vcbp_[5] = {};                    // S20 batched prefill: in place, no lm_head
    VkQueryPool vtsqp_[5] = {};
    std::vector<uint8_t> vgate_kindsp_[5];
    bool nock_ = false;                               // S20: recording in place (no verify checkpoints)
    std::vector<EkBuf> ek_;                           // S20 elastic KV
    std::vector<std::pair<VkDeviceMemory, uint64_t>> ek_free_;
    uint64_t ek_page_ = 0, ek_base_bytes_ = 0, ek_retired_ = 0, ek_hlo_[2] = {0, 0};
    uint32_t ek_bits_ = ~0u, ek_commit_ = 0, ek_rb_[2] = {0, 0}, n_slots_live_ = 0;
    bool ek_grow_ok_ = false;
    bool lm_rec_ = false, lm_dedupe_ = false;         // S20: recording the layer-major prefill steps / they use unique-expert entries
    std::vector<VkCommandBuffer> lmcb_;               // S20 layer-major prefill: [layer step 0..NL_]
    VkCommandBuffer cb_carry_in_ = VK_NULL_HANDLE;
    uint64_t aCarry_ = 0, aLmG_ = 0;
    void* pLmG_ = nullptr;
    vk::Pipeline pCarry_;
    std::vector<uint32_t> lm_mail_;                   // [layer][token][k] routed expert ids of the chunk
    VkQueryPool lmtsq_ = VK_NULL_HANDLE;
    io_uring lmr_{};                                  // S20: n-gram reads of a layer-major chunk
    bool lmr_init_ = false, lmr_ok_ = false;
    bool vdedupe_[5] = {};  // S18: per batch size (a T = 3 recording without the 4-column kernel must not switch T = 2 off)
    VkQueryPool vtsq_[5] = {}, tsqv_ = VK_NULL_HANDLE;
    uint32_t tsqv_n_ = 0;
    std::vector<uint8_t> vgate_kind_, vgate_kinds_[5];
    vk::Pipeline pHn25_, pLdx_, pHu4_;  // S19 HC chain
    vk::Pipeline pIp_, pIs_;            // S19 QSA indexer
    vk::Pipeline pIsc_, pIpp_, pWisc_, pWisp_;  // S33 pWisc_/pWisp_: wide tiled scores + radix selection
             // S30 indexer block scores (whole GPU); MTP fill pooling
    vk::Pipeline pSst_;                 // S32 tiering probe
    bool sst_on_ = false;
    uint64_t sst_last_ = 0, sst_cnt_ = 0;
    void* sst_lastp_ = nullptr;
    void* sst_cntp_ = nullptr;
public:
    // S32 tiering probe report: per attention layer, the share of selected blocks by how long ago they were last selected
    void selstat_report() const {
        if (!sst_on_) return;
        const char* nm[8] = {"never", "prev", "<=4", "<=16", "<=64", "<=256", ">256", "window"};
        uint64_t tot[8] = {0};
        uint64_t toks = 0;
        for (uint32_t a = 0; a < L / 4; ++a) {
            const uint32_t* c = static_cast<const uint32_t*>(sst_cntp_) + a * 16;
            uint64_t s = 0;
            for (int k = 0; k < 8; ++k) { s += c[k]; tot[k] += c[k]; }
            toks = std::max<uint64_t>(toks, c[8]);
            std::fprintf(stderr, "selstat layer %2u: %u tokens, %.1f blocks/token:", a * 4 + 3, c[8], c[8] ? double(s) / c[8] : 0.0);
            for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %s %.4f", nm[k], s ? double(c[k]) / s : 0.0);
            std::fprintf(stderr, "\n");
        }
        uint64_t s = 0;
        for (int k = 0; k < 8; ++k) s += tot[k];
        std::fprintf(stderr, "selstat all: %llu tokens,", (unsigned long long)toks);
        for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %s %.4f", nm[k], s ? double(tot[k]) / s : 0.0);
        std::fprintf(stderr, " | misses/token/layer if only the previous selection + window are resident: %.1f\n",
                     toks ? double(s - tot[1] - tot[7]) / toks / (L / 4) : 0.0);
    }
private:
    bool idx_ps_ = false;
    uint32_t isc_wg_ = 80;                      // S33 idx_score_d grid: one thread per block of max_ctx
    static constexpr uint32_t kIdxScoreWg = 80;  // S30 idx_score grid (2 per CU; chunks of 64 blocks, grid-stride)
    vk::Pipeline pXs6_[3], pXs25_[5];  // S19 matvec_xrs (K = 6144: NB 1-2, K = 2560: NB 1-4)
    vk::Pipeline pXrB_[5], pLdsB2_, pRt2B_[5], pMvB_[5], pHuB_[5], pCvB_, pGdB_;
    vk::Pipeline pExMB_[3][2];  // S18 unique-expert decode [K-1 class][nb 2 / 4]
    bool exmb_ok_[2] = {false, false};
    uint64_t mailA_ = 0, pmailA_ = 0, earlyA_ = 0;
    uint8_t* pmail_host_ = nullptr;
    struct Deferred { uint32_t layer; uint16_t expert; };
    struct PrefRec { SlotId slot; uint16_t expert; uint8_t def; };
    std::vector<Deferred> deferred_;
    std::vector<Deferred> rr_q_;  // S24 displaced experts waiting for a background read
    size_t dq_head_ = 0;
    uint32_t cur_layer_ = 0;
    std::vector<PrefRec> pref_rec_[L + 1];
    std::vector<std::vector<uint16_t>> trace_;
    EngineStats stats_;
};

}  // namespace ps5lm::flash
