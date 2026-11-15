// trace_replay — portable cache-only replay (gate S1; no I/O, no GPU).
//
//   trace_replay --trace t.txt --slot-bytes B (--pool-gib G | --slots N)
//                [--policy lru|clock] [--json]
//
// Policy lru must match `python -m engine gate replay` hit-for-hit
// (runtime/tools/gate_s1.py checks it). It also reports the per-layer miss
// shape, which is what sets I/O queue depth when routing is layer-serial:
// misses_per_layer_visit ~1.2 at h=88% is the S11-review qd1 problem.
#include "ps5lm/slot_cache.hpp"
#include "ps5lm/trace_txt.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace ps5lm;

static void usage() {
    std::fprintf(stderr,
        "usage: trace_replay --trace FILE --slot-bytes B (--pool-gib G | --slots N)\n"
        "                    [--policy lru|clock] [--json]\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    std::string trace_path;
    uint64_t slot_bytes = 0, slots = 0;
    double pool_gib = 0;
    Policy policy = Policy::LRU;
    bool json = false;
    double half_life = 36;           // S17 --policy lfu
    uint64_t c0 = 0, c1 = 0;         // S17 --classes N0,N1: EXL3 size classes (class 1 = layers 0-3, 44-47)
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* f) { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", f); usage(); } return argv[++i]; };
        if (!std::strcmp(argv[i], "--trace")) trace_path = need("--trace");
        else if (!std::strcmp(argv[i], "--slot-bytes")) slot_bytes = std::strtoull(need("--slot-bytes"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--pool-gib")) pool_gib = std::strtod(need("--pool-gib"), nullptr);
        else if (!std::strcmp(argv[i], "--slots")) slots = std::strtoull(need("--slots"), nullptr, 10);
        else if (!std::strcmp(argv[i], "--policy")) {
            std::string p = need("--policy");
            if (p == "lru") policy = Policy::LRU; else if (p == "clock") policy = Policy::CLOCK; else if (p == "lfu") policy = Policy::LFU; else usage();
        } else if (!std::strcmp(argv[i], "--half-life")) half_life = std::strtod(need("--half-life"), nullptr);
        else if (!std::strcmp(argv[i], "--classes")) {
            const char* v = need("--classes");
            c0 = std::strtoull(v, nullptr, 10);
            c1 = std::strtoull(std::strchr(v, ',') + 1, nullptr, 10);
            slots = c0 + c1;
        } else if (!std::strcmp(argv[i], "--json")) json = true;
        else usage();
    }
    if (trace_path.empty()) usage();
    if (!slots) {
        if (!slot_bytes || pool_gib <= 0) usage();
        // Same rounding as engine/replay.py: int(cache_gib * 1024**3 // slot)
        slots = uint64_t(std::floor(pool_gib * 1073741824.0 / double(slot_bytes)));
    }
    if (!slots || slots >= UINT32_MAX) { std::fprintf(stderr, "bad slot count %llu\n", (unsigned long long)slots); return 2; }

    Trace tr;
    try { tr = load_trace_txt(trace_path); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }

    SlotCache cache(tr.layers, tr.experts, uint32_t(slots), policy);
    if (policy == Policy::LFU) cache.lfu_set_half_life(half_life);
    if (c0 && c1) {
        std::vector<uint8_t> lc(tr.layers, 0);
        for (uint32_t l = 0; l < tr.layers; ++l) lc[l] = (l < 4 || l + 4 >= tr.layers);
        cache.set_classes(lc, uint32_t(c0));
    }
    uint64_t layer_visits = 0, visits_with_miss = 0, max_misses = 0;
    for (uint32_t t = 0; t < tr.tokens; ++t) {
        cache.tick();
        for (uint32_t l = 0; l < tr.layers; ++l) {
            uint64_t m = 0;
            for (const uint16_t* p = tr.begin(t, l); p != tr.end(t, l); ++p) {
                auto r = cache.lookup_or_assign(l, *p);
                if (r.slot == kNoSlot) { std::fprintf(stderr, "saturated (cannot happen without pins)\n"); return 1; }
                if (!r.hit) { cache.mark_ready(r.slot); ++m; }
            }
            ++layer_visits;
            visits_with_miss += m > 0;
            if (m > max_misses) max_misses = m;
        }
    }
    const CacheStats& s = cache.stats();
    const double mpl = double(s.misses) / double(layer_visits);
    const double mpt = double(s.misses) / double(tr.tokens);
    if (json) {
        std::printf("{\"policy\":\"%s\",\"slots\":%llu,\"reads\":%llu,\"hits\":%llu,\"misses\":%llu,"
                    "\"hit_rate\":%.6f,\"misses_per_token\":%.3f,\"misses_per_layer_visit\":%.4f,"
                    "\"frac_layer_visits_with_miss\":%.4f,\"max_misses_in_a_layer\":%llu}\n",
                    policy == Policy::LRU ? "lru" : policy == Policy::LFU ? "lfu" : "clock", (unsigned long long)slots,
                    (unsigned long long)tr.reads(), (unsigned long long)s.hits, (unsigned long long)s.misses,
                    s.hit_rate(), mpt, mpl, double(visits_with_miss) / double(layer_visits),
                    (unsigned long long)max_misses);
    } else {
        std::printf("policy %s  slots %llu  reads %llu\n", policy == Policy::LRU ? "lru" : policy == Policy::LFU ? "lfu" : "clock",
                    (unsigned long long)slots, (unsigned long long)tr.reads());
        std::printf("hit rate            %.4f  (%llu hits / %llu misses)\n", s.hit_rate(),
                    (unsigned long long)s.hits, (unsigned long long)s.misses);
        std::printf("misses / token      %.2f\n", mpt);
        std::printf("misses / layer      %.3f   (layer-serial routing => this is your I/O queue depth)\n", mpl);
        std::printf("layers with a miss  %.1f%%  max in one layer %llu\n",
                    100.0 * double(visits_with_miss) / double(layer_visits), (unsigned long long)max_misses);
    }
    return 0;
}
