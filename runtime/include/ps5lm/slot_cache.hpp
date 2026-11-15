// slot_cache.hpp — portable expert slot cache (no OS dependencies).
//
// One pool of fixed-size slots shared by ALL layers, keyed by (layer, expert).
// This is the bookkeeping half of the streamer: it decides hit/miss, which
// slot a miss lands in, and which slot is evicted. The Linux io_uring half
// (expert_streamer.hpp) owns the bytes; this class only owns indices, so it
// builds and unit-tests on any platform (gate S1 runs it on a desktop).
//
// Policy::LRU reproduces engine/replay.py SlotCache read-for-read when no
// slot is ever pinned or in flight. Policy::CLOCK is the runtime default
// (O(1) hit path, no list surgery on the hot path).
//
// Thread model: confined to ONE thread (the streamer, CPU2). The only state
// read cross-thread is the per-slot ready flag, via ready().
#pragma once

#include <atomic>
#include <cmath>
#include <limits>
#include <queue>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <vector>

namespace ps5lm {

using SlotId = uint32_t;
inline constexpr SlotId kNoSlot = UINT32_MAX;

// S17 LFU: evict the slot whose expert has the lowest exponentially-decayed request frequency (half-life in tokens,
// engine calls tick() once per token). Per-expert history survives eviction; speculative (prefetch) lookups do not count.
// Measured on a 1,000-token EXL3 routing trace (tools/s17/policy_sim.py): hit 0.854 (LRU) -> 0.868 (half-life 32-40).
enum class Policy : uint8_t { LRU, CLOCK, LFU };
enum class SlotState : uint8_t { Free, InFlight, Ready };

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t saturated = 0;  // lookups refused: every slot pinned or in flight
    double hit_rate() const {
        uint64_t t = hits + misses;
        return t ? double(hits) / double(t) : 0.0;
    }
};

class SlotCache {
public:
    struct Lookup {
        SlotId slot = kNoSlot;  // kNoSlot => saturated, caller must back off
        bool hit = false;       // false => caller must issue the read into `slot`
    };

    SlotCache(uint32_t layers, uint32_t experts, uint32_t n_slots, Policy policy)
        : layers_(layers), experts_(experts), n_slots_(n_slots), policy_(policy),
          state_(new std::atomic<uint8_t>[n_slots]),
          key_(n_slots, kNoKey), pin_(n_slots, 0), ref_(n_slots, 0),
          prev_(n_slots, kNoSlot), next_(n_slots, kNoSlot),
          index_(size_t(layers) * experts, kNoSlot) {
        if (policy_ == Policy::LFU) {
            kscore_.assign(size_t(layers) * experts, -std::numeric_limits<double>::infinity());
            skey_.assign(n_slots, -std::numeric_limits<double>::infinity());
        }
        cend_[0] = n_slots;
        if (n_slots == 0) throw std::invalid_argument("SlotCache: zero slots");
        if (experts > 0xFFFF || layers > 0xFFFF) throw std::invalid_argument("SlotCache: key overflow");
        for (uint32_t s = 0; s < n_slots_; ++s) state_[s].store(uint8_t(SlotState::Free), std::memory_order_relaxed);
    }

    uint32_t n_slots() const { return n_slots_; }

    // S15b size classes (EXL3 packs mix K=3 and K=3.5 layers): slots [0, n0) belong to class 0, [n0, n) to class 1;
    // a layer's experts only ever occupy slots of its class, each class is its own LRU. LRU policy only.
    void set_classes(const std::vector<uint8_t>& layer_class, uint32_t n0) {
        if (policy_ == Policy::CLOCK || n0 == 0 || n0 >= n_slots_ || layer_class.size() != layers_)
            throw std::invalid_argument("SlotCache::set_classes");
        layer_cls_ = layer_class;
        ncls_ = 2;
        cbeg_[0] = 0; cend_[0] = n0; cbeg_[1] = n0; cend_[1] = n_slots_;
        fresh_[0] = 0; fresh_[1] = n0;
    }
    uint32_t class_slots(int c) const { return cend_[c] - cbeg_[c]; }
    Policy policy() const { return policy_; }
    const CacheStats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

    // Hit: returns the slot (Ready or InFlight) and refreshes recency.
    // Miss: picks a victim, remaps it to (layer, expert), marks it InFlight;
    //       the caller MUST then either complete the read (mark_ready) or
    //       invalidate() the slot.
    // cold=true (speculative prefetch, S14): a NEW assignment enters at the LRU tail instead of the MRU head,
    // so a wrong guess is the next victim instead of displacing a hot expert; a later demand hit promotes it.
    // S17: lfu_set_half_life / tick for Policy::LFU
    void lfu_set_half_life(double tokens) { lam_ = std::log(2.0) / tokens; }
    // S18: a use of an expert of `layer` counts w (MTP-layer misses stall the draft with nothing to overlap them)
    void lfu_set_layer_weight(uint32_t layer, double w) { if (lw_.empty()) lw_.assign(layers_, 1.0); if (layer < lw_.size()) lw_[layer] = w; }
    void tick() { now_ += 1.0; }
    double now() const { return now_; }
    // S22: stamp the next uses at time t instead of now (a batched prefill ticks a whole chunk first, then looks up
    // every token's experts: stamped at now_ they all counted as simultaneous, inflating LFU scores ~N/36-fold).
    // The key is log(sum_j w_j e^{lam t_j}), order-independent, so a past stamp equals token-serial feeding. NaN = now.
    void set_use_time(double t) { ut_ = t; }

    // spec = speculative (prefetch) lookup: LFU does not count it as a use (a wrong guess keeps a low score)
    Lookup lookup_or_assign(uint32_t layer, uint32_t expert, bool cold = false, bool spec = false) {
        const size_t ix = size_t(layer) * experts_ + expert;
        SlotId s = index_[ix];
        if (s != kNoSlot) {
            ++stats_.hits;
            if (policy_ == Policy::LFU) { if (!spec) { bump(ix); set_key(s, kscore_[ix]); } }
            else touch(s);
            return {s, true};
        }
        s = victim(ncls_ == 2 ? layer_cls_[layer] : 0);
        if (s == kNoSlot) { ++stats_.saturated; return {kNoSlot, false}; }
        last_evicted_ = key_[s];  // S31: what this miss pushed out (kNoKey if the slot was free)
        if (key_[s] != kNoKey) { index_[key_[s]] = kNoSlot; ++stats_.evictions; }
        ++stats_.misses;
        key_[s] = uint32_t(ix);
        index_[ix] = s;
        state_[s].store(uint8_t(SlotState::InFlight), std::memory_order_relaxed);
        if (policy_ == Policy::LFU) {
            if (!spec) bump(ix);
            // a speculative newcomer never seen before ranks like one old request (evicted before anything in use)
            set_key(s, std::isfinite(kscore_[ix]) ? kscore_[ix] : std::log(0.25) + lam_ * now_);
            return {s, false};
        }
        if (cold && policy_ == Policy::LRU) insert_lru_tail(s);
        else if (cold) ref_[s] = 0;
        else insert_mru(s);
        return {s, false};
    }

    // Peek without changing recency or stats (prefetch dedup).
    SlotId find(uint32_t layer, uint32_t expert) const { return index_[size_t(layer) * experts_ + expert]; }

    void mark_ready(SlotId s) { state_[s].store(uint8_t(SlotState::Ready), std::memory_order_release); }
    bool ready(SlotId s) const {
        return state_[s].load(std::memory_order_acquire) == uint8_t(SlotState::Ready);
    }
    SlotState state(SlotId s) const { return SlotState(state_[s].load(std::memory_order_acquire)); }

    // Failed/short read: forget the mapping; the slot becomes the next victim.
    void invalidate(SlotId s) {
        if (key_[s] != kNoKey) index_[key_[s]] = kNoSlot;
        key_[s] = kNoKey;
        ref_[s] = 0;
        state_[s].store(uint8_t(SlotState::Free), std::memory_order_relaxed);
        if (policy_ == Policy::LFU) { set_key(s, -1e300); return; }
        if (policy_ == Policy::LRU) { unlink(s); insert_lru_tail(s); }
    }

    // S17: a prefetched slot whose target layer bound WITHOUT using it -> next victim (LRU tail). Mapping kept: a later
    // hit still promotes it. No-op for CLOCK (clears the reference bit) and for unlinked slots.
    void demote(SlotId s) {
        if (policy_ == Policy::CLOCK) { ref_[s] = 0; return; }
        if (policy_ == Policy::LFU) { set_key(s, -1e300); return; }
        const int c = cls(s);
        if (prev_[s] == kNoSlot && next_[s] == kNoSlot && head_[c] != s) return;
        unlink(s);
        insert_lru_tail(s);
    }

    // GPU-lifetime pins. A pinned slot is never a victim. Pin before the
    // submit thread records it; unpin once the layer's timeline value passed.
    void pin(SlotId s) { ++pin_[s]; }
    void unpin(SlotId s) {
        if (pin_[s] == 0) throw std::logic_error("SlotCache: unpin underflow");
        --pin_[s];
    }
    uint32_t pins(SlotId s) const { return pin_[s]; }

    // S20 elastic KV: a retired slot holds no expert and is never assigned (its memory backs KV pages). Refused while
    // the slot is pinned or a read is in flight. unretire() makes it the next victim.
    bool retire(SlotId s) {
        if (pin_[s] || state(s) == SlotState::InFlight) return false;
        if (retired_.empty()) retired_.assign(n_slots_, 0);
        if (retired_[s]) return true;
        if (key_[s] != kNoKey) { index_[key_[s]] = kNoSlot; ++stats_.evictions; }
        key_[s] = kNoKey;
        ref_[s] = 0;
        state_[s].store(uint8_t(SlotState::Free), std::memory_order_relaxed);
        retired_[s] = 1;
        if (policy_ == Policy::LFU) set_key(s, std::numeric_limits<double>::infinity());
        else if (policy_ == Policy::LRU) unlink(s);
        return true;
    }
    void unretire(SlotId s) {
        if (retired_.empty() || !retired_[s]) return;
        retired_[s] = 0;
        if (policy_ == Policy::LFU) set_key(s, -1e300);
        else if (policy_ == Policy::LRU) insert_lru_tail(s);
    }
    bool retired(SlotId s) const { return !retired_.empty() && retired_[s]; }
    uint32_t class_begin(int c) const { return cbeg_[c]; }
    // S25: the size class of a layer, and how many of a class's slots a new assignment could take right now
    int layer_class(uint32_t layer) const { return ncls_ == 2 ? int(layer_cls_[layer]) : 0; }
    uint32_t evictable_in_class(int c) const {
        uint32_t n = 0;
        for (SlotId s = cbeg_[c]; s < cend_[c]; ++s) n += evictable(s);
        return n;
    }
    uint32_t class_end(int c) const { return cend_[c]; }
    uint32_t n_classes() const { return ncls_; }

    // S31 lowmem tier: the key (layer * experts + expert) a slot holds, kNoSlot if none
    uint32_t key_of(SlotId s) const { return key_[s]; }
    uint32_t last_evicted() const { return last_evicted_; }
    // S31: decayed use count of the expert in slot s right now (LFU; 1 = one use just now, a never-used prefetch 0.25)
    double score_now(SlotId s) const { return policy_ == Policy::LFU ? std::exp(skey_[s] - lam_ * now_) : 1.0; }
    // S35 shadow use: cache-aware routing displaced this true top-K expert, so it was NOT requested and earned no LFU
    // use, while the resident substitute did: the pool learned the biased routing. A shadow use of weight w adds to the
    // expert's decayed count without a slot or a read: if it is admitted later (prefetch, kept miss, tier) it ranks by its
    // true demand, and a resident expert (in flight / tier race) gets its heap key refreshed. LFU only.
    void shadow_use(uint32_t layer, uint32_t expert, double w) {
        if (policy_ != Policy::LFU || !(w > 0)) return;
        const size_t ix = size_t(layer) * experts_ + expert;
        const double t = std::isnan(ut_) ? now_ : ut_;
        const double sc = std::isfinite(kscore_[ix]) ? std::exp(kscore_[ix] - lam_ * t) + w : w;
        kscore_[ix] = std::log(sc) + lam_ * t;
        if (index_[ix] != kNoSlot) set_key(index_[ix], kscore_[ix]);
    }
    // S35: decayed use count of an expert right now, resident or not (0 = never used)
    double expert_score_now(uint32_t layer, uint32_t expert) const {
        if (policy_ != Policy::LFU) return 0;
        const double k = kscore_[size_t(layer) * experts_ + expert];
        return std::isfinite(k) ? std::exp(k - lam_ * now_) : 0.0;
    }
    // S31: class c has handed out every never-used slot (from now on each miss evicts)
    bool class_full(int c) const { return policy_ == Policy::CLOCK || fresh_[c] >= cend_[c]; }
    // S31: up to n Ready, evictable slots of class c that are next in line for eviction, first victim first.
    // LFU: lowest keys; LRU: from the tail; CLOCK: unsupported (empty). Never-used slots are skipped (nothing to keep).
    // Returns how many Ready evictable slots the class has (LRU: counted up to 4n), so the caller can leave headroom.
    uint32_t next_victims(int c, uint32_t n, std::vector<SlotId>& out) const {
        out.clear();
        if (policy_ == Policy::LRU) {
            uint32_t cnt = 0;
            for (SlotId s = tail_[c]; s != kNoSlot && cnt < 4 * n; s = prev_[s])
                if (key_[s] != kNoKey && evictable(s) && state(s) == SlotState::Ready) { if (out.size() < n) out.push_back(s); ++cnt; }
            return cnt;
        }
        if (policy_ != Policy::LFU) return 0;
        std::vector<std::pair<double, SlotId>> v;
        v.reserve(cend_[c] - cbeg_[c]);
        for (SlotId s = cbeg_[c]; s < cend_[c]; ++s)
            if (key_[s] != kNoKey && evictable(s) && state(s) == SlotState::Ready) v.push_back({skey_[s], s});
        const size_t k = std::min<size_t>(n, v.size());
        std::partial_sort(v.begin(), v.begin() + k, v.end());
        for (size_t i = 0; i < k; ++i) out.push_back(v[i].second);
        return uint32_t(v.size());
    }

    // Resident keys (layer*experts+expert), most recently used first (LRU); slot order (CLOCK).
    // Persisted by the server so a restart preloads the same warm pool.
    std::vector<uint32_t> resident_keys_mru() const {
        std::vector<uint32_t> out;
        if (policy_ == Policy::LFU) {  // highest decayed frequency first
            std::vector<std::pair<double, uint32_t>> v;
            for (SlotId s = 0; s < n_slots_; ++s)
                if (key_[s] != kNoKey && state(s) == SlotState::Ready) v.push_back({-skey_[s], key_[s]});
            std::sort(v.begin(), v.end());
            for (auto& x : v) out.push_back(x.second);
            return out;
        }
        if (policy_ == Policy::LRU) {
            for (uint32_t c = 0; c < ncls_; ++c)
                for (SlotId s = head_[c]; s != kNoSlot; s = next_[s])
                    if (key_[s] != kNoKey && state(s) == SlotState::Ready) out.push_back(key_[s]);
        } else {
            for (SlotId s = 0; s < n_slots_; ++s)
                if (key_[s] != kNoKey && state(s) == SlotState::Ready) out.push_back(key_[s]);
        }
        return out;
    }

private:
    static constexpr uint32_t kNoKey = UINT32_MAX;

    bool evictable(SlotId s) const {
        return pin_[s] == 0 && state(s) != SlotState::InFlight && (retired_.empty() || !retired_[s]);
    }

    void touch(SlotId s) {
        if (policy_ == Policy::CLOCK) { ref_[s] = 1; return; }
        unlink(s);
        insert_mru(s);
    }

    int cls(SlotId s) const { return ncls_ == 2 && s >= cbeg_[1] ? 1 : 0; }
    SlotId victim(int c) {
        if (policy_ == Policy::CLOCK) {
            for (uint32_t step = 0; step < 2 * n_slots_; ++step) {
                SlotId s = hand_;
                hand_ = (hand_ + 1 == n_slots_) ? 0 : hand_ + 1;
                if (state(s) == SlotState::Free && key_[s] == kNoKey && pin_[s] == 0 && !retired(s)) return s;
                if (!evictable(s)) continue;
                if (ref_[s]) { ref_[s] = 0; continue; }
                return s;
            }
            return kNoSlot;
        }
        // LRU: never-used slots first (in index order), then walk from the LRU end.
        while (fresh_[c] < cend_[c] && retired(fresh_[c])) ++fresh_[c];
        if (fresh_[c] < cend_[c]) return fresh_[c]++;
        if (policy_ == Policy::LFU) {  // lazy min-heap: skip stale entries, set aside non-evictable ones
            auto& h = heap_[c];
            SlotId out = kNoSlot;
            std::vector<std::pair<double, SlotId>> keep;
            while (!h.empty()) {
                const auto top = h.top();
                h.pop();
                if (top.first != skey_[top.second]) continue;  // stale
                if (!evictable(top.second)) { keep.push_back(top); continue; }
                out = top.second;
                break;
            }
            for (auto& x : keep) h.push(x);
            return out;
        }
        for (SlotId s = tail_[c]; s != kNoSlot; s = prev_[s])
            if (evictable(s)) { unlink(s); return s; }
        return kNoSlot;
    }

    // --- S17 LFU bookkeeping ---
    void bump(size_t ix) {  // decayed count + 1, stored time-invariant: log(score(t)) + lam * t
        const double t = std::isnan(ut_) ? now_ : ut_;
        const double inc = lw_.empty() ? 1.0 : lw_[ix / experts_];
        const double sc = std::isfinite(kscore_[ix]) ? std::exp(kscore_[ix] - lam_ * t) + inc : inc;
        kscore_[ix] = std::log(sc) + lam_ * t;
    }
    void set_key(SlotId s, double k) {
        skey_[s] = k;
        auto& h = heap_[cls(s)];
        h.push({k, s});
        if (h.size() > 8u * size_t(n_slots_) + 1024u) {  // drop stale entries
            std::priority_queue<std::pair<double, SlotId>, std::vector<std::pair<double, SlotId>>, std::greater<>> nh;
            for (SlotId x = cbeg_[cls(s)]; x < fresh_[cls(s)]; ++x) nh.push({skey_[x], x});  // assigned slots only
            h.swap(nh);
        }
    }

    // --- intrusive doubly linked list (LRU only): head_ = MRU, tail_ = LRU ---
    void unlink(SlotId s) {
        const int c = cls(s);
        if (prev_[s] == kNoSlot && next_[s] == kNoSlot && head_[c] != s) return;  // not linked
        if (prev_[s] != kNoSlot) next_[prev_[s]] = next_[s]; else head_[c] = next_[s];
        if (next_[s] != kNoSlot) prev_[next_[s]] = prev_[s]; else tail_[c] = prev_[s];
        prev_[s] = next_[s] = kNoSlot;
    }
    void insert_mru(SlotId s) {
        if (policy_ == Policy::CLOCK) { ref_[s] = 1; return; }
        const int c = cls(s);
        prev_[s] = kNoSlot;
        next_[s] = head_[c];
        if (head_[c] != kNoSlot) prev_[head_[c]] = s; else tail_[c] = s;
        head_[c] = s;
    }
    void insert_lru_tail(SlotId s) {
        const int c = cls(s);
        next_[s] = kNoSlot;
        prev_[s] = tail_[c];
        if (tail_[c] != kNoSlot) next_[tail_[c]] = s; else head_[c] = s;
        tail_[c] = s;
    }

    uint32_t layers_, experts_, n_slots_;
    Policy policy_;
    std::unique_ptr<std::atomic<uint8_t>[]> state_;
    std::vector<uint32_t> key_;   // slot -> layer*experts+expert
    std::vector<uint32_t> pin_;
    std::vector<uint8_t>  ref_;   // CLOCK reference bit
    std::vector<uint8_t>  retired_;  // S20 elastic KV (empty = none)
    std::vector<SlotId>   prev_, next_;
    std::vector<SlotId>   index_; // key -> slot
    SlotId head_[2] = {kNoSlot, kNoSlot}, tail_[2] = {kNoSlot, kNoSlot}, hand_ = 0, fresh_[2] = {0, 0};
    SlotId cbeg_[2] = {0, 0}, cend_[2] = {0, 0};
    uint32_t ncls_ = 1;
    std::vector<uint8_t> layer_cls_;
    std::vector<double> kscore_, skey_, lw_;  // S17 LFU: per-expert invariant score key, per-slot heap key
    std::priority_queue<std::pair<double, SlotId>, std::vector<std::pair<double, SlotId>>, std::greater<>> heap_[2];
    double now_ = 0, lam_ = 0.693147180559945 / 36.0, ut_ = std::numeric_limits<double>::quiet_NaN();
    CacheStats stats_;
    uint32_t last_evicted_ = kNoKey;
};

}  // namespace ps5lm
