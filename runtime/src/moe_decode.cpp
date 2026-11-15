// moe_decode — P3 prototype: Qwen3.8-Flash-Next expert path, end to end on the PS5.
//
// Per token, 48 MoE layers: rmsnorm -> [dense stand-in] -> router (GPU) -> top-10 mailbox ->
// SlotCache + io_uring READ_FIXED (O_DIRECT) into the slab -> the SAME slab bytes are read by the
// GPU (VK_EXT_external_memory_host import, zero copy) -> fused gate+up+SwiGLU -> down+combine+residual.
//
// Command buffers are recorded ONCE (49 layer-steps) and replayed every token: expert binding is
// a write of 64-bit slot addresses + routing weights into host-coherent tables the kernels read
// through BDA. One submit + one fence per layer (v0 host-driven pipeline).
//
//   moe_decode --pack FILE --bits 2|4 --pool-gib G --shaders DIR [--tokens N] [--warm N]
//              [--route gpu|trace] [--trace FILE] [--lookahead K] [--dense-mb-per-token M]
//              [--make-pack] [--sqpoll-cpu C] [--policy lru|clock]
//
// --route trace: expert ids come from a routing trace (realistic locality) instead of the
//   synthetic random router (~uniform = worst case). --lookahead K (trace mode) = oracle
//   prefetch depth in layer-steps: the ceiling of any cross-layer predictor.
// --dense-mb-per-token: GPU reads this many MB of resident weights per token (spread over the
//   layers) as an honest stand-in for attention/GDN/shared-expert/lm_head traffic.
#include "ps5lm/arena.hpp"
#include "ps5lm/expert_streamer.hpp"
#include "ps5lm/platform.hpp"
#include "ps5lm/quant.hpp"
#include "ps5lm/trace_txt.hpp"
#include "ps5lm/vk.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ps5lm;
namespace qn = ps5lm::q;
using Clock = std::chrono::steady_clock;

struct alignas(8) PcGateUp { uint64_t x, h, slots; uint32_t rpw, pad; };
struct alignas(8) PcDown { uint64_t h, slots, wt, resid, outp; uint32_t n_exp, rpw; };
struct alignas(8) PcRouter { uint64_t w, x, logits; uint32_t n_exp, pad; };
struct alignas(8) PcTopk { uint64_t logits, mail; uint32_t n_exp, k, seq, pad; };
struct alignas(8) PcNorm { uint64_t x, gamma, outp; uint32_t n; float eps; };
struct alignas(8) PcMembw { uint64_t src, outp; uint32_t n, magic; };

static constexpr uint32_t L = 48, E = 512, K = 10, HID = 2560, FFN = 640, TAB = 16;

static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Valid PS5Q slots: random codes, small fixed fp16 scales (finite, stable through rmsnorm).
template <int B> static void fill_slot(uint8_t* s, uint64_t& rng) {
    const qn::ExpertLayout<B> Lay;
    auto* w = reinterpret_cast<uint64_t*>(s);
    for (size_t i = 0; i < Lay.bytes / 8; ++i) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; w[i] = rng; }
    const uint16_t d = qn::f32_to_f16(0.02f);
    auto put_scales = [&](size_t off, int K_, int rows) {
        const size_t rb = qn::row_bytes<B>(K_), sb = qn::scales_bytes<B>(K_);
        for (int r = 0; r < rows; ++r) {
            auto* sc = reinterpret_cast<uint16_t*>(s + off + size_t(r) * rb);
            for (size_t i = 0; i < sb / 2; ++i) sc[i] = d;
        }
    };
    put_scales(Lay.gate_off, HID, FFN);
    put_scales(Lay.up_off, HID, FFN);
    put_scales(Lay.down_off, FFN, HID);
}

template <int B> static void make_pack(const std::string& path) {
    const size_t slot = qn::ExpertLayout<B>().bytes;
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { std::perror("open pack"); std::exit(1); }
    const uint32_t per = 32;  // slots per write
    auto m = platform::map_pinned(slot * per, false, true);
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    const uint64_t total = uint64_t(L) * E;
    auto t0 = Clock::now();
    for (uint64_t s = 0; s < total; s += per) {
        const uint32_t n = uint32_t(std::min<uint64_t>(per, total - s));
        for (uint32_t i = 0; i < n; ++i) fill_slot<B>(reinterpret_cast<uint8_t*>(m.ptr) + size_t(i) * slot, rng);
        if (::pwrite(fd, m.ptr, size_t(n) * slot, off_t(s * slot)) != ssize_t(size_t(n) * slot)) { std::perror("pwrite"); std::exit(1); }
    }
    ::fsync(fd);
    ::close(fd);
    platform::unmap(m);
    std::fprintf(stderr, "wrote %.2f GiB PS5Q%d pack in %.1f s (FRESH: SLC-assisted until folded)\n",
                 double(total * slot) / (1ull << 30), B, secs(t0, Clock::now()));
}

int main(int argc, char** argv) {
    std::string pack, sdir = "shaders", trace_path, route = "gpu";
    int bits = 2;
    double pool_gib = 10, dense_mb = 0;
    uint32_t tokens = 200, warm = 30, lookahead = 0;
    bool make = false;
    StreamerConfig cfg;
    cfg.iopoll = false;
    cfg.policy = Policy::LRU;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--pack") pack = nx();
        else if (a == "--shaders") sdir = nx();
        else if (a == "--bits") bits = std::stoi(nx());
        else if (a == "--pool-gib") pool_gib = std::stod(nx());
        else if (a == "--tokens") tokens = uint32_t(std::stoul(nx()));
        else if (a == "--warm") warm = uint32_t(std::stoul(nx()));
        else if (a == "--route") route = nx();
        else if (a == "--trace") trace_path = nx();
        else if (a == "--lookahead") lookahead = uint32_t(std::stoul(nx()));
        else if (a == "--dense-mb-per-token") dense_mb = std::stod(nx());
        else if (a == "--make-pack") make = true;
        else if (a == "--sqpoll-cpu") cfg.sqpoll_cpu = std::stoi(nx());
        else if (a == "--policy") cfg.policy = nx() == "clock" ? Policy::CLOCK : Policy::LRU;
        else if (a == "--iopoll") cfg.iopoll = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (pack.empty() || (bits != 2 && bits != 4)) { std::fprintf(stderr, "need --pack and --bits 2|4\n"); return 2; }
    const uint32_t slot_bytes = uint32_t(bits == 4 ? qn::ExpertLayout<4>().bytes : qn::ExpertLayout<2>().bytes);
    if (make) { if (bits == 4) make_pack<4>(pack); else make_pack<2>(pack); }

    Trace tr;
    const bool use_trace = route == "trace";
    if (use_trace) {
        tr = load_trace_txt(trace_path);
        if (tr.layers != L || tr.experts != E) { std::fprintf(stderr, "trace geometry mismatch\n"); return 1; }
    }

    platform::pin_to_cpu(2);
    try { platform::set_fifo(70); } catch (...) {}

    // ---- slab: pinned host memory, registered with io_uring AND imported into Vulkan ----
    // udmabuf slab: memfd pages (O_DIRECT-able, io_uring-registered) exported as dma-buf chunks
    // the GPU imports pinned-once. Chunk = whole slots so no slot straddles two Vulkan buffers.
    const uint64_t pool = uint64_t(pool_gib * double(1ull << 30));
    const uint32_t spc = uint32_t((1ull << 30) / slot_bytes);  // must match the streamer's 1 GiB chunking
    const uint64_t want_slots = pool / slot_bytes;
    auto map = platform::map_udmabuf(want_slots * slot_bytes, uint64_t(spc) * slot_bytes);
    cfg.pack_path = pack;
    cfg.layers = L;
    cfg.experts = E;
    cfg.slot_bytes = slot_bytes;
    ExpertStreamer st(cfg, map.ptr, want_slots * slot_bytes);
    const uint32_t n_slots = st.cache().n_slots();
    struct { const char* kind_name() const { return "udmabuf-shmem"; } } map_kind;
    struct { uint64_t bytes; } slab{uint64_t(n_slots) * slot_bytes};

    vk::Ctx ctx;
    std::vector<vk::Buffer> chunks;
    for (size_t c = 0; c < map.dmabufs.size(); ++c)
        chunks.push_back(ctx.import_dmabuf(map.dmabufs[c], std::min<uint64_t>(map.chunk_bytes, map.bytes - c * map.chunk_bytes)));
    auto slot_addr = [&](SlotId s) { return chunks[s / spc].addr + uint64_t(s % spc) * slot_bytes; };

    // ---- resident buffers ----
    std::mt19937_64 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bR0 = ctx.create(HID * 4), bR1 = ctx.create(HID * 4), bXn = ctx.create(HID * 4), bGamma = ctx.create(HID * 4);
    auto bLogits = ctx.create(E * 4), bH = ctx.create(TAB * FFN * 4), bSink = ctx.create(64);
    auto bRouter = ctx.create(uint64_t(L) * E * HID * 4);
    auto bMail = ctx.create(uint64_t(L) * 256), bSlots = ctx.create(uint64_t(L) * TAB * 8), bWt = ctx.create(uint64_t(L) * TAB * 4);
    for (uint32_t i = 0; i < HID; ++i) { bR0.as<float>()[i] = nd(rng); bGamma.as<float>()[i] = 1.f; }
    {
        float* w = bRouter.as<float>();
        for (uint64_t i = 0; i < uint64_t(L) * E * HID; ++i) w[i] = nd(rng) * 0.02f;
    }
    const uint64_t dense_bytes = uint64_t(dense_mb * 1e6) / (16 * L) * 16 * L;
    vk::Buffer bDense{};
    if (dense_bytes) { bDense = ctx.create(dense_bytes, false); }
    std::fprintf(stderr, "slab %.2f GiB (%s, %zu Vulkan chunks), %u slots of %u B; dense stand-in %.0f MB/token; route %s lookahead %u\n",
                 double(slab.bytes) / (1ull << 30), map_kind.kind_name(), chunks.size(), n_slots, slot_bytes, dense_bytes / 1e6,
                 route.c_str(), lookahead);

    const std::string tag = bits == 4 ? "q4" : "q2";
    auto pNorm = ctx.pipeline(sdir + "/rmsnorm.spv", sizeof(PcNorm), 0);
    auto pRouter = ctx.pipeline(sdir + "/router_f32.spv", sizeof(PcRouter));
    auto pTopk = ctx.pipeline(sdir + "/topk.spv", sizeof(PcTopk));
    auto pGU = ctx.pipeline(sdir + "/gate_up_" + tag + ".spv", sizeof(PcGateUp));
    auto pDN = ctx.pipeline(sdir + "/down_combine_" + tag + ".spv", sizeof(PcDown));
    auto pMem = ctx.pipeline(sdir + "/membw.spv", sizeof(PcMembw), 0);
    const uint32_t rpw_gu = 4, rpw_dn = 2;
    vk::Buffer* R[2] = {&bR0, &bR1};

    // ---- record the 49 layer-steps once: step s = experts(s-1) + head(s) ----
    std::vector<VkCommandBuffer> steps(L + 1);
    for (uint32_t s = 0; s <= L; ++s) {
        VkCommandBuffer cb = ctx.cmd();
        steps[s] = cb;
        vk::Ctx::begin(cb, false);
        if (s > 0) {
            const uint32_t l = s - 1;
            PcGateUp pg{bXn.addr, bH.addr, bSlots.addr + uint64_t(l) * TAB * 8, rpw_gu, 0};
            vk::Ctx::dispatch(cb, pGU, &pg, (FFN + rpw_gu - 1) / rpw_gu, K);
            vk::Ctx::barrier(cb);
            PcDown pd{bH.addr, bSlots.addr + uint64_t(l) * TAB * 8, bWt.addr + uint64_t(l) * TAB * 4, R[l % 2]->addr,
                      R[(l + 1) % 2]->addr, K, rpw_dn};
            vk::Ctx::dispatch(cb, pDN, &pd, (HID + rpw_dn - 1) / rpw_dn);
            vk::Ctx::barrier(cb);
        }
        if (s < L) {
            const uint32_t l = s;
            PcNorm pn{R[l % 2]->addr, bGamma.addr, bXn.addr, HID, 1e-6f};
            vk::Ctx::dispatch(cb, pNorm, &pn, 1);
            if (dense_bytes) {
                PcMembw pm{bDense.addr + dense_bytes / L * l, bSink.addr, uint32_t(dense_bytes / L / 16), 0xDEADBEEFu};
                vk::Ctx::dispatch(cb, pMem, &pm, 36 * 8);
            }
            vk::Ctx::barrier(cb);
            PcRouter pr{bRouter.addr + uint64_t(l) * E * HID * 4, bXn.addr, bLogits.addr, E, 0};
            vk::Ctx::dispatch(cb, pRouter, &pr, E);
            vk::Ctx::barrier(cb);
            PcTopk pt{bLogits.addr, bMail.addr + uint64_t(l) * 256, E, K, l + 1, 0};
            vk::Ctx::dispatch(cb, pTopk, &pt, 1);
            vk::Ctx::barrier(cb);
        }
        vk::Ctx::end(cb);
    }
    VkFence fence = ctx.fence();

    // ---- decode loop ----
    auto* slots_tab = bSlots.as<uint64_t>();
    auto* wt_tab = bWt.as<float>();
    double t_gpu = 0, t_io = 0;
    uint64_t demand = 0, ready_on_arrival = 0, bytes0 = 0, ios0 = 0;
    uint64_t next_prefetch = 1;
    std::vector<SlotId> pinned;
    Clock::time_point t0 = Clock::now();
    const uint32_t total_tokens = warm + tokens;
    for (uint32_t t = 0; t < total_tokens; ++t) {
        if (t == warm) {
            st.cache().reset_stats();
            t_gpu = t_io = 0;
            demand = ready_on_arrival = 0;
            bytes0 = st.stats().bytes_read;
            ios0 = st.stats().reads_issued;
            t0 = Clock::now();
        }
        for (uint32_t s = 0; s <= L; ++s) {
            auto a = Clock::now();
            ctx.submit(steps[s], fence);
            ctx.wait(fence);
            auto b = Clock::now();
            t_gpu += secs(a, b);
            for (SlotId p : pinned) st.cache().unpin(p);  // layer s-1 experts are done
            pinned.clear();
            if (s == L) break;
            const uint32_t l = s;
            const uint32_t* mail = reinterpret_cast<const uint32_t*>(bMail.as<uint8_t>() + uint64_t(l) * 256);
            const float* mw = reinterpret_cast<const float*>(mail + 16);
            uint16_t ids[TAB];
            float w[TAB];
            const uint32_t tt = use_trace ? t % tr.tokens : 0;
            for (uint32_t k = 0; k < K; ++k) {
                ids[k] = use_trace ? tr.begin(tt, l)[k] : uint16_t(mail[k]);
                w[k] = use_trace ? 1.f / K : mw[k];
            }
            // demand reads
            SlotId sl[TAB];
            for (uint32_t k = 0; k < K; ++k) {
                SlotCache::Lookup r;
                while ((r = st.request(l, ids[k])).slot == kNoSlot) st.reap(true);
                if (st.cache().ready(r.slot)) ++ready_on_arrival;
                st.cache().pin(r.slot);
                pinned.push_back(r.slot);
                sl[k] = r.slot;
                ++demand;
            }
            // oracle prefetch (trace mode)
            if (use_trace && lookahead) {
                const uint64_t g = uint64_t(t) * L + l;
                for (; next_prefetch <= g + lookahead; ++next_prefetch) {
                    const uint32_t pt = uint32_t((next_prefetch / L) % tr.tokens), pl = uint32_t(next_prefetch % L);
                    for (const uint16_t* p = tr.begin(pt, pl); p != tr.end(pt, pl); ++p)
                        if (st.cache().find(pl, *p) == kNoSlot) st.request(pl, *p);
                }
            }
            st.submit();
            auto c = Clock::now();
            for (;;) {  // the GPU never waits on I/O: bind only resident slots
                bool all = true;
                for (uint32_t k = 0; k < K; ++k) {
                    if (st.cache().ready(sl[k])) continue;
                    all = false;
                    if (st.cache().state(sl[k]) == SlotState::Free) {  // failed read: re-request
                        st.cache().unpin(sl[k]);
                        SlotCache::Lookup r;
                        while ((r = st.request(l, ids[k])).slot == kNoSlot) st.reap(true);
                        st.cache().pin(r.slot);
                        pinned.push_back(r.slot);
                        sl[k] = r.slot;
                    }
                }
                if (all) break;
                st.reap(true);
            }
            st.reap(false);
            t_io += secs(c, Clock::now());
            for (uint32_t k = 0; k < K; ++k) {
                slots_tab[l * TAB + k] = slot_addr(sl[k]);
                wt_tab[l * TAB + k] = w[k];
            }
        }
        if (st.stats().io_errors > 64) { std::fprintf(stderr, "too many I/O errors\n"); return 1; }
    }
    const double wall = secs(t0, Clock::now());
    const uint64_t ios = st.stats().reads_issued - ios0;
    const double gib = double(st.stats().bytes_read - bytes0) / double(1ull << 30);
    const float* out = R[0]->as<float>();
    bool finite = true;
    for (uint32_t i = 0; i < HID; ++i) finite &= std::isfinite(out[i]);

    std::printf("P3 moe_decode  PS5Q%d  pool %.2f GiB (%u slots)  route %s  lookahead %u  dense %.0f MB/tok\n", bits,
                double(slab.bytes) / (1ull << 30), n_slots, route.c_str(), lookahead, dense_bytes / 1e6);
    std::printf("tok/s               %.2f   (%u timed tokens, %.1f ms/token)\n", tokens / wall, tokens, wall / tokens * 1e3);
    std::printf("SSD reads/token     %.1f   effective hit %.4f   resident-on-arrival %.4f\n", double(ios) / tokens,
                1.0 - double(ios) / double(demand), double(ready_on_arrival) / double(demand));
    std::printf("SSD bandwidth       %.0f MB/s (%.2f GiB/s)\n", gib * 1073.741824 / wall, gib / wall);
    std::printf("per token: GPU+sync %.2f ms | I/O wait %.2f ms | other %.2f ms\n", t_gpu / tokens * 1e3,
                t_io / tokens * 1e3, (wall - t_gpu - t_io) / tokens * 1e3);
    std::printf("io errors %llu   output finite: %s\n", (unsigned long long)st.stats().io_errors, finite ? "yes" : "NO");
    vkDeviceWaitIdle(ctx.dev);
    return 0;
}
