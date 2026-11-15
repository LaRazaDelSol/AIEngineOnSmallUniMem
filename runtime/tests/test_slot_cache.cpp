// test_slot_cache — dependency-free unit tests for SlotCache (ctest).
#include "ps5lm/slot_cache.hpp"

#include <cstdio>
#include <cstdlib>

using namespace ps5lm;

static int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) { std::fprintf(stderr, "%s:%d CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

static SlotCache::Lookup get(SlotCache& c, uint32_t l, uint32_t e) {
    auto r = c.lookup_or_assign(l, e);
    if (r.slot != kNoSlot && !r.hit) c.mark_ready(r.slot);
    return r;
}

static void lru_matches_python_hand_checks() {
    // engine/tests: cap 2, routing [[0,1]],[[0,1]] -> 2 hits / 4 reads.
    SlotCache c(1, 8, 2, Policy::LRU);
    get(c, 0, 0); get(c, 0, 1); get(c, 0, 0); get(c, 0, 1);
    CHECK(c.stats().hits == 2 && c.stats().misses == 2);

    // Shared-budget regression: 2 layers x {0,1}, cap 2 -> LRU thrash, 0 hits.
    SlotCache s(2, 8, 2, Policy::LRU);
    for (int t = 0; t < 4; ++t)
        for (uint32_t l = 0; l < 2; ++l) { get(s, l, 0); get(s, l, 1); }
    CHECK(s.stats().hits == 0 && s.stats().misses == 16);
}

static void lru_evicts_least_recent() {
    SlotCache c(1, 8, 2, Policy::LRU);
    get(c, 0, 0); get(c, 0, 1); get(c, 0, 0);  // 1 is now LRU
    get(c, 0, 2);                              // evicts 1
    CHECK(c.find(0, 0) != kNoSlot);
    CHECK(c.find(0, 1) == kNoSlot);
    CHECK(c.find(0, 2) != kNoSlot);
    CHECK(c.stats().evictions == 1);
}

static void pinned_and_inflight_are_never_victims() {
    for (Policy p : {Policy::LRU, Policy::CLOCK}) {
        SlotCache c(1, 8, 2, p);
        auto a = get(c, 0, 0);
        c.pin(a.slot);
        auto b = c.lookup_or_assign(0, 1);  // left InFlight on purpose
        CHECK(!b.hit && c.state(b.slot) == SlotState::InFlight);
        auto d = c.lookup_or_assign(0, 2);
        CHECK(d.slot == kNoSlot);           // saturated: one pinned, one in flight
        CHECK(c.stats().saturated == 1);
        c.mark_ready(b.slot);
        d = c.lookup_or_assign(0, 2);       // now 1 is evictable
        CHECK(d.slot == b.slot && c.find(0, 1) == kNoSlot && c.find(0, 0) == a.slot);
        c.unpin(a.slot);
        CHECK(c.pins(a.slot) == 0);
    }
}

static void invalidate_forgets_and_recycles() {
    for (Policy p : {Policy::LRU, Policy::CLOCK}) {
        SlotCache c(1, 8, 2, p);
        auto a = c.lookup_or_assign(0, 0);
        c.invalidate(a.slot);               // e.g. short read
        CHECK(c.find(0, 0) == kNoSlot);
        auto again = c.lookup_or_assign(0, 0);
        CHECK(!again.hit);                  // re-requested as a miss
        CHECK(c.stats().misses == 2);
    }
}

static void clock_second_chance() {
    SlotCache c(1, 8, 2, Policy::CLOCK);
    get(c, 0, 0); get(c, 0, 1);
    get(c, 0, 2);  // both ref'd: sweep clears, evicts slot 0 (expert 0)
    CHECK(c.find(0, 0) == kNoSlot);
    CHECK(c.find(0, 1) != kNoSlot && c.find(0, 2) != kNoSlot);
}

static void key_isolation_across_layers() {
    SlotCache c(3, 4, 12, Policy::CLOCK);
    for (uint32_t l = 0; l < 3; ++l)
        for (uint32_t e = 0; e < 4; ++e) get(c, l, e);
    CHECK(c.stats().misses == 12);
    for (uint32_t l = 0; l < 3; ++l)
        for (uint32_t e = 0; e < 4; ++e) CHECK(get(c, l, e).hit);
}

static void cold_insert_is_next_victim() {
    // S14: a cold (prefetch) insert enters at the LRU tail: the next miss evicts it, not the hot entries.
    SlotCache c(1, 8, 3, Policy::LRU);
    get(c, 0, 0); get(c, 0, 1);
    auto p = c.lookup_or_assign(0, 2, /*cold=*/true);
    c.mark_ready(p.slot);
    get(c, 0, 3);                       // miss: must evict the cold expert 2
    CHECK(c.find(0, 2) == kNoSlot);
    CHECK(c.find(0, 0) != kNoSlot && c.find(0, 1) != kNoSlot);
    // A demand hit promotes a cold insert to MRU: it then survives the next miss.
    auto q = c.lookup_or_assign(0, 4, true);  // evicts LRU = expert 0
    c.mark_ready(q.slot);
    CHECK(get(c, 0, 4).hit);
    get(c, 0, 5);                       // evicts expert 1 (now LRU), not the promoted 4
    CHECK(c.find(0, 4) != kNoSlot && c.find(0, 1) == kNoSlot);
    auto keys = c.resident_keys_mru();
    CHECK(keys.size() == 3 && keys[0] == 5 && keys[1] == 4);
}

int main() {
    cold_insert_is_next_victim();
    lru_matches_python_hand_checks();
    lru_evicts_least_recent();
    pinned_and_inflight_are_never_victims();
    invalidate_forgets_and_recycles();
    clock_second_chance();
    key_isolation_across_layers();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("test_slot_cache: all passed");
    return 0;
}
