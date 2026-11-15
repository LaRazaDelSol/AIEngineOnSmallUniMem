// stream_bench — I/O-bound decode replay through the real io_uring path (gate S2 / R1-ceiling).
//
//   stream_bench --pack FILE --trace t.txt --slot-bytes B --pool-gib G
//                [--lookahead K]     oracle prefetch depth in layer-steps (0 = layer-serial)
//                [--compute-us U]    simulated GPU time per layer (spin) to model overlap
//                [--warm-tokens N]   excluded from timing (cache warm-up)
//                [--policy lru|clock] [--sqpoll-cpu C|-1] [--no-iopoll] [--no-hugetlb]
//                [--make-pack]       write the pack (layers*experts*slot) with O_DIRECT first
//
// Oracle lookahead reads the trace's FUTURE routing, i.e. a predictor with
// precision 1.0: it is the ceiling for any cross-layer predictor (RUNTIME-SPEC
// §2.4). K=0 reproduces the layer-serial qd~1 regime.
//
// Packs written by --make-pack are FRESH data: SLC-assisted. For NAND-true
// numbers write once, then let the drive fold (hours, or fill > SLC size)
// before benching (S11 lesson).
#include "ps5lm/arena.hpp"
#include "ps5lm/expert_streamer.hpp"
#include "ps5lm/platform.hpp"
#include "ps5lm/trace_txt.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ps5lm;
using Clock = std::chrono::steady_clock;

static void usage() {
    std::fprintf(stderr,
                 "usage: stream_bench --pack FILE --trace FILE --slot-bytes B --pool-gib G [--lookahead K]\n"
                 "       [--compute-us U] [--warm-tokens N] [--policy lru|clock] [--sqpoll-cpu C]\n"
                 "       [--no-iopoll] [--no-hugetlb] [--make-pack]\n");
    std::exit(2);
}

static void make_pack(const std::string& path, uint64_t bytes) {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { std::perror("open pack for write"); std::exit(1); }
    const size_t chunk = 16u << 20;
    auto m = platform::map_pinned(chunk, false, true);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        auto* w = reinterpret_cast<uint64_t*>(m.ptr);
        for (size_t i = 0; i < chunk / 8; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; w[i] = x; }  // incompressible
        const size_t n = size_t(std::min<uint64_t>(chunk, bytes - off));
        if (::pwrite(fd, m.ptr, (n + 4095) / 4096 * 4096, off_t(off)) < 0) { std::perror("pwrite"); std::exit(1); }
    }
    ::fsync(fd);
    ::ftruncate(fd, off_t(bytes));
    ::close(fd);
    platform::unmap(m);
    std::fprintf(stderr, "wrote %.2f GiB pack (FRESH data: SLC-assisted until folded)\n", double(bytes) / (1ull << 30));
}

static void spin_us(double us) {
    if (us <= 0) return;
    const auto end = Clock::now() + std::chrono::nanoseconds(int64_t(us * 1000));
    while (Clock::now() < end) {}
}

int main(int argc, char** argv) {
    std::string pack, trace_path;
    uint64_t slot_bytes = 0;
    double pool_gib = 0, compute_us = 0;
    uint32_t lookahead = 0, warm = 50;
    bool make = false, hugetlb = true;
    StreamerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        auto need = [&]() { if (i + 1 >= argc) usage(); return std::string(argv[++i]); };
        std::string a = argv[i];
        if (a == "--pack") pack = need();
        else if (a == "--trace") trace_path = need();
        else if (a == "--slot-bytes") slot_bytes = std::stoull(need());
        else if (a == "--pool-gib") pool_gib = std::stod(need());
        else if (a == "--lookahead") lookahead = uint32_t(std::stoul(need()));
        else if (a == "--compute-us") compute_us = std::stod(need());
        else if (a == "--warm-tokens") warm = uint32_t(std::stoul(need()));
        else if (a == "--policy") { auto p = need(); cfg.policy = p == "lru" ? Policy::LRU : Policy::CLOCK; }
        else if (a == "--sqpoll-cpu") cfg.sqpoll_cpu = std::stoi(need());
        else if (a == "--no-iopoll") cfg.iopoll = false;
        else if (a == "--no-hugetlb") hugetlb = false;
        else if (a == "--make-pack") make = true;
        else usage();
    }
    if (pack.empty() || trace_path.empty() || !slot_bytes || pool_gib <= 0) usage();

    Trace tr = load_trace_txt(trace_path);
    cfg.pack_path = pack;
    cfg.layers = tr.layers;
    cfg.experts = tr.experts;
    cfg.slot_bytes = uint32_t(slot_bytes);
    if (make) make_pack(pack, uint64_t(tr.layers) * tr.experts * slot_bytes);

    platform::pin_to_cpu(platform::cores::kStreamer);
    try { platform::set_fifo(80); } catch (const std::exception& e) { std::fprintf(stderr, "note: %s (running SCHED_OTHER)\n", e.what()); }

    const uint64_t pool = uint64_t(pool_gib * double(1ull << 30));
    auto map = platform::map_pinned(pool, hugetlb, true);
    Arena arena(map.ptr, map.bytes);
    auto& slab = arena.carve_rest("experts", slot_bytes);
    ExpertStreamer st(cfg, arena.host(slab.offset), slab.bytes);
    std::fprintf(stderr, "slab %.2f GiB (%s), %u slots, sqpoll cpu %d, iopoll %d, lookahead %u\n",
                 double(slab.bytes) / (1ull << 30), map.kind_name(), st.cache().n_slots(), cfg.sqpoll_cpu,
                 int(cfg.iopoll), lookahead);

    const uint64_t steps = uint64_t(tr.tokens) * tr.layers;
    std::vector<SlotId> demand;
    uint64_t demand_reads = 0, demand_ready_on_arrival = 0, stall_ns = 0, inflight_samples = 0, inflight_sum = 0;
    uint64_t ios_at_warm = 0, bytes_at_warm = 0;
    uint64_t next_prefetch = 1;  // first step not yet prefetched
    Clock::time_point t0 = Clock::now();

    for (uint64_t g = 0; g < steps; ++g) {
        const uint32_t t = uint32_t(g / tr.layers), l = uint32_t(g % tr.layers);
        if (t == warm && l == 0) {
            st.cache().reset_stats();
            demand_reads = demand_ready_on_arrival = stall_ns = inflight_samples = inflight_sum = 0;
            ios_at_warm = st.stats().reads_issued;
            bytes_at_warm = st.stats().bytes_read;
            t0 = Clock::now();
        }
        // demand for this step (pinned until "GPU" finishes it)
        demand.clear();
        for (const uint16_t* p = tr.begin(t, l); p != tr.end(t, l); ++p) {
            SlotCache::Lookup r;
            while ((r = st.request(l, *p)).slot == kNoSlot) st.reap(true);  // saturated: drain and retry
            if (st.cache().ready(r.slot)) ++demand_ready_on_arrival;
            st.cache().pin(r.slot);
            demand.push_back(r.slot);
            ++demand_reads;
        }
        // oracle prefetch of steps (g, g+K]
        for (; next_prefetch <= g + lookahead && next_prefetch < steps; ++next_prefetch) {
            const uint32_t pt = uint32_t(next_prefetch / tr.layers), pl = uint32_t(next_prefetch % tr.layers);
            for (const uint16_t* p = tr.begin(pt, pl); p != tr.end(pt, pl); ++p)
                if (st.cache().find(pl, *p) == kNoSlot) st.request(pl, *p);
        }
        st.submit();
        inflight_sum += st.in_flight();
        ++inflight_samples;
        // stall until every demand slot is resident (the GPU wait in the real pipeline)
        const auto ws = Clock::now();
        for (;;) {
            bool all = true;
            for (size_t i = 0; i < demand.size(); ++i) {
                if (st.cache().ready(demand[i])) continue;
                if (st.cache().state(demand[i]) == SlotState::Free) {  // failed read: re-request
                    st.cache().unpin(demand[i]);
                    const uint16_t e = tr.begin(t, l)[i];
                    SlotCache::Lookup r;
                    while ((r = st.request(l, e)).slot == kNoSlot) st.reap(true);
                    st.cache().pin(r.slot);
                    demand[i] = r.slot;
                }
                all = false;
            }
            if (all) break;
            st.reap(true);
        }
        stall_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - ws).count());
        spin_us(compute_us);  // GPU time; prefetches keep landing via the kernel poller meanwhile
        st.reap(false);
        for (SlotId s : demand) st.cache().unpin(s);
        if (st.stats().io_errors > 64) { std::fprintf(stderr, "too many I/O errors, aborting\n"); return 1; }
    }
    st.reap(false);
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    const uint32_t timed_tokens = tr.tokens > warm ? tr.tokens - warm : 0;
    if (!timed_tokens) { std::fprintf(stderr, "trace shorter than warm-up\n"); return 1; }
    const uint64_t ios = st.stats().reads_issued - ios_at_warm;
    const double gib = double(st.stats().bytes_read - bytes_at_warm) / double(1ull << 30);
    std::printf("tokens %u (timed)  lookahead %u  compute %.0f us/layer  policy %s  pages %s\n", timed_tokens,
                lookahead, compute_us, cfg.policy == Policy::LRU ? "lru" : "clock", map.kind_name());
    std::printf("tok/s                %.2f\n", timed_tokens / secs);
    std::printf("SSD reads / token    %.2f   (effective hit %.4f)\n", double(ios) / timed_tokens,
                1.0 - double(ios) / double(demand_reads));
    std::printf("demand resident on arrival %.4f\n", double(demand_ready_on_arrival) / double(demand_reads));
    std::printf("read bandwidth       %.3f GiB/s  (%.0f MB/s)\n", gib / secs, gib * 1073.741824 / secs);
    std::printf("mean in-flight       %.2f\n", double(inflight_sum) / double(inflight_samples));
    std::printf("stall share          %.1f%%\n", 100.0 * double(stall_ns) * 1e-9 / secs);
    std::printf("io errors            %llu\n", (unsigned long long)st.stats().io_errors);
    platform::unmap(map);
    return 0;
}
