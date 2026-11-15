// lowmem_tier.hpp — S31 second expert tier in CPU-only memory (the firmware-reserved low gigabyte, /dev/ps5lowmem).
//
// The GPU cannot read that range (S25 machine checks; S26-S31). The CPU can, write-back cached. So the tier is reached
// only by CPU memcpy, never by a GPU pointer:
//   admit: a pool slot that is next in line for eviction (still resident, pinned for the copy) -> tier entry
//   fill:  pool miss whose expert is in the tier -> memcpy tier entry -> the victim slot (instead of an SSD read)
// Expert bytes are immutable, so copying a resident slot while the GPU reads it is safe (read/read).
//
// Thread model: every public method runs on the streamer thread except the workers' copy loop. Jobs are split into
// parts so one 2 MB fill uses all workers (latency), each part done counts down an atomic.
#pragma once
#if !defined(__linux__)
#error "lowmem_tier.hpp is Linux-only"
#endif

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ps5lm {

struct TierStats {
    uint64_t lookups = 0;     // pool misses checked against the tier
    uint64_t hits = 0;        // ... served by a tier fill
    uint64_t fill_bytes = 0;
    uint64_t admits = 0;      // copies pool -> tier
    uint64_t admit_bytes = 0;
    uint64_t readmits = 0;    // candidate already in the tier: refreshed, no copy
    uint64_t admit_skips = 0; // no free tier entry / job (all pinned)
    double fill_s = 0;        // summed fill latency (request -> done seen by reap)
    uint64_t evicted = 0;     // pool evictions while the tier was on
    uint64_t evicted_kept = 0;// ... whose expert was in the tier (or being copied there) at that moment
};

class LowTier {
public:
    struct Job {
        uint8_t* dst = nullptr;
        const uint8_t* src = nullptr;
        uint32_t len = 0, parts = 0;
        std::atomic<uint32_t> left{0};
        uint32_t slot = 0, entry = 0;
        uint8_t kind = 0;  // 1 fill, 2 admit
        double t0 = 0;
    };

    // dev: char device to mmap; bytes: 0 = all it exposes. cls_bytes[c] = entry size of class c (the pool's slot stride
    // for that class), cls_weight[c] = how many pool slots class c has (tier entries are split in the same byte share).
    LowTier(const std::string& dev, uint64_t bytes, uint32_t layers, uint32_t experts, const std::vector<uint8_t>& layer_class,
            const uint32_t cls_bytes[2], const uint32_t cls_weight[2], uint32_t ncls, unsigned threads)
        : layers_(layers), experts_(experts), layer_cls_(layer_class), ncls_(ncls) {
        uint64_t avail = 0;
        const bool anon = dev == "anon";  // tests: ordinary anonymous memory of `bytes`
        if (anon) avail = bytes;
        else {
            std::ifstream f("/sys/module/ps5lowmem/parameters/tier_bytes");
            if (!(f >> avail) || !avail) throw std::runtime_error("lowmem tier: ps5lowmem module not loaded (no tier_bytes)");
        }
        if (!bytes || bytes > avail) bytes = avail;
        bytes &= ~uint64_t(4095);
        if (!anon) {
            fd_ = ::open(dev.c_str(), O_RDWR | O_CLOEXEC);
            if (fd_ < 0) throw std::runtime_error("lowmem tier: open " + dev + ": " + std::strerror(errno));
        }
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, anon ? MAP_PRIVATE | MAP_ANONYMOUS : MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("lowmem tier: mmap: " + std::string(std::strerror(errno)));
        base_ = static_cast<uint8_t*>(p);
        bytes_ = bytes;
        // entries: class c gets the byte share of its pool slots
        double wsum = 0;
        for (uint32_t c = 0; c < ncls_; ++c) wsum += double(cls_weight[c]) * cls_bytes[c];
        uint64_t off = 0;
        for (uint32_t c = 0; c < ncls_; ++c) {
            const uint64_t share = c + 1 == ncls_ ? bytes - off : uint64_t(double(bytes) * double(cls_weight[c]) * cls_bytes[c] / wsum);
            const uint64_t n = share / cls_bytes[c];
            cbeg_[c] = uint32_t(ent_off_.size());
            for (uint64_t i = 0; i < n; ++i) ent_off_.push_back(off + i * cls_bytes[c]);
            cend_[c] = uint32_t(ent_off_.size());
            ent_bytes_[c] = cls_bytes[c];
            off += n * cls_bytes[c];
        }
        const uint32_t ne = uint32_t(ent_off_.size());
        if (!ne) throw std::runtime_error("lowmem tier: no room for one entry");
        ent_key_.assign(ne, kNoKey);
        ent_pin_.assign(ne, 0);
        prev_.assign(ne, kNo);
        next_.assign(ne, kNo);
        index_.assign(size_t(layers) * experts, kNo);
        for (uint32_t c = 0; c < ncls_; ++c)
            for (uint32_t e = cbeg_[c]; e < cend_[c]; ++e) push_tail(c, e);  // free entries are the first victims
        jobs_ = std::vector<Job>(kJobs);
        for (uint32_t j = 0; j < kJobs; ++j) free_jobs_.push_back(j);
        nthreads_ = std::max(1u, threads);
        // own physical cores, away from 0-3 (SQPOLL on 0, the engine's main thread on 2, their SMT siblings 1 and 3):
        // threads inherit the creator's affinity, and the engine's main thread is pinned to CPU 2 (S31 gate 1: three
        // copy threads on that core made fills take 1.1 ms and decode 25% slower)
        const unsigned ncpu = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < nthreads_; ++i)
            th_.emplace_back([this, i, ncpu] {
                cpu_set_t set;
                CPU_ZERO(&set);
                const unsigned want = 4 + 2 * i;
                if (want < ncpu) CPU_SET(want, &set);
                else for (unsigned c = 4; c < ncpu; ++c) CPU_SET(c, &set);
                if (ncpu > 4) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
                work();
            });
        std::fprintf(stderr, "lowmem tier: %.0f MiB CPU-only (%s), %u entries", bytes / 1048576.0, dev.c_str(), ne);
        for (uint32_t c = 0; c < ncls_; ++c) std::fprintf(stderr, "%s%u x %.2f MB", c ? " + " : ": ", cend_[c] - cbeg_[c], ent_bytes_[c] / 1e6);
        std::fprintf(stderr, ", %u copy threads\n", nthreads_);
    }
    ~LowTier() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        if (base_) ::munmap(base_, bytes_);
        if (fd_ >= 0) ::close(fd_);
    }
    LowTier(const LowTier&) = delete;
    LowTier& operator=(const LowTier&) = delete;

    uint32_t entries() const { return uint32_t(ent_off_.size()); }
    const TierStats& stats() const { return st_; }
    // valid entry (complete copy) for (layer, expert)?
    bool has(uint32_t layer, uint32_t expert) const { return index_[size_t(layer) * experts_ + expert] != kNo; }
    uint32_t pending() const { return uint32_t(out_.size()); }
    void note_eviction(uint32_t key) {
        ++st_.evicted;
        if (index_[key] != kNo || std::find(admitting_.begin(), admitting_.end(), key) != admitting_.end()) ++st_.evicted_kept;
    }
    uint32_t pending_fills() const { return n_fills_; }

    // Fill `dst` (a pool slot that is InFlight for this expert) from the tier. False: not in the tier / no job free.
    bool fill(uint32_t layer, uint32_t expert, uint8_t* dst, uint32_t len, uint32_t slot, double now) {
        ++st_.lookups;
        const uint32_t e = index_[size_t(layer) * experts_ + expert];
        if (e == kNo || free_jobs_.empty()) return false;
        const uint32_t c = cls(layer);
        ++ent_pin_[e];
        // the expert is back in the pool: its tier copy is now a duplicate, so it becomes the next to overwrite (the
        // tier stays exclusive of the pool = pool + tier distinct experts). If it is re-admitted before that, no copy.
        unlink(c, e);
        push_tail(c, e);
        submit(1, dst, base_ + ent_off_[e], len, slot, e, now);
        ++st_.hits;
        st_.fill_bytes += len;
        ++n_fills_;
        return true;
    }

    // Copy a resident slot into the tier. The caller pinned the slot; done() reports it for unpinning.
    // Returns false (nothing queued, caller unpins) if already present (refreshed) or no entry/job is free.
    bool admit(uint32_t layer, uint32_t expert, const uint8_t* src, uint32_t len, uint32_t slot, double now) {
        const size_t ix = size_t(layer) * experts_ + expert;
        const uint32_t c = cls(layer);
        if (index_[ix] != kNo) { move_head(c, index_[ix]); ++st_.readmits; return false; }
        if (admitting_.size() && std::find(admitting_.begin(), admitting_.end(), uint32_t(ix)) != admitting_.end()) return false;
        if (free_jobs_.empty()) { ++st_.admit_skips; return false; }
        uint32_t e = tail_[c];
        while (e != kNo && ent_pin_[e]) e = prev_[e];
        if (e == kNo) { ++st_.admit_skips; return false; }
        if (ent_key_[e] != kNoKey) index_[ent_key_[e]] = kNo;  // invalid until the new copy completes
        ent_key_[e] = uint32_t(ix);
        ++ent_pin_[e];
        move_head(c, e);
        admitting_.push_back(uint32_t(ix));
        submit(2, base_ + ent_off_[e], src, len, slot, e, now);
        ++st_.admits;
        st_.admit_bytes += len;
        return true;
    }

    // Completed jobs since the last call: fills -> (slot, true), admits -> (slot, false). Streamer thread only.
    template <class F>
    unsigned reap(double now, F&& on_done) {
        unsigned n = 0;
        for (size_t i = 0; i < out_.size();) {
            Job& j = jobs_[out_[i]];
            if (j.left.load(std::memory_order_acquire)) { ++i; continue; }
            --ent_pin_[j.entry];
            if (j.kind == 1) {
                --n_fills_;
                st_.fill_s += now - j.t0;
            } else {
                const uint32_t ix = ent_key_[j.entry];
                index_[ix] = j.entry;
                admitting_.erase(std::find(admitting_.begin(), admitting_.end(), ix));
            }
            on_done(j.slot, j.kind == 1);
            free_jobs_.push_back(out_[i]);
            out_[i] = out_.back();
            out_.pop_back();
            ++n;
        }
        return n;
    }

private:
    static constexpr uint32_t kNo = UINT32_MAX, kNoKey = UINT32_MAX, kJobs = 192, kPart = 512u << 10;

    uint32_t cls(uint32_t layer) const { return ncls_ == 2 ? layer_cls_[layer] : 0; }
    int ecls(uint32_t e) const { return ncls_ == 2 && e >= cbeg_[1] ? 1 : 0; }

    void submit(uint8_t kind, uint8_t* dst, const uint8_t* src, uint32_t len, uint32_t slot, uint32_t entry, double now) {
        const uint32_t ji = free_jobs_.back();
        free_jobs_.pop_back();
        Job& j = jobs_[ji];
        j.dst = dst; j.src = src; j.len = len; j.slot = slot; j.entry = entry; j.kind = kind; j.t0 = now;
        j.parts = std::max(1u, std::min(nthreads_, (len + kPart - 1) / kPart));
        j.left.store(j.parts, std::memory_order_relaxed);
        out_.push_back(ji);
        {
            std::lock_guard<std::mutex> g(mu_);
            // fills first: a demand miss is waiting on them; admits are background
            for (uint32_t p = 0; p < j.parts; ++p) {
                if (kind == 1) q_.push_front({ji, p}); else q_.push_back({ji, p});
            }
        }
        if (j.parts == 1) cv_.notify_one(); else cv_.notify_all();
    }
    void work() {
        for (;;) {
            std::pair<uint32_t, uint32_t> w;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                w = q_.front();
                q_.pop_front();
            }
            Job& j = jobs_[w.first];
            const uint64_t per = ((uint64_t(j.len) + j.parts - 1) / j.parts + 4095) & ~uint64_t(4095);
            const uint64_t a = std::min<uint64_t>(j.len, per * w.second), b = std::min<uint64_t>(j.len, a + per);
            if (b > a) std::memcpy(j.dst + a, j.src + a, size_t(b - a));
            j.left.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    // per-class LRU of entries: head = most recent, tail = next to overwrite
    void unlink(int c, uint32_t e) {
        if (prev_[e] != kNo) next_[prev_[e]] = next_[e]; else head_[c] = next_[e];
        if (next_[e] != kNo) prev_[next_[e]] = prev_[e]; else tail_[c] = prev_[e];
        prev_[e] = next_[e] = kNo;
    }
    void push_tail(int c, uint32_t e) {
        next_[e] = kNo; prev_[e] = tail_[c];
        if (tail_[c] != kNo) next_[tail_[c]] = e; else head_[c] = e;
        tail_[c] = e;
    }
    void move_head(int c, uint32_t e) {
        if (head_[c] == e) return;
        unlink(c, e);
        prev_[e] = kNo; next_[e] = head_[c];
        if (head_[c] != kNo) prev_[head_[c]] = e; else tail_[c] = e;
        head_[c] = e;
    }

    uint32_t layers_, experts_;
    std::vector<uint8_t> layer_cls_;
    uint32_t ncls_ = 1;
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0;
    std::vector<uint64_t> ent_off_;
    std::vector<uint32_t> ent_key_, ent_pin_, prev_, next_, index_;
    uint32_t cbeg_[2] = {0, 0}, cend_[2] = {0, 0}, ent_bytes_[2] = {0, 0};
    uint32_t head_[2] = {kNo, kNo}, tail_[2] = {kNo, kNo};
    std::vector<uint32_t> admitting_;  // keys with an admit in flight (small)
    std::vector<Job> jobs_;
    std::vector<uint32_t> free_jobs_, out_;
    uint32_t n_fills_ = 0;
    unsigned nthreads_ = 1;
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::pair<uint32_t, uint32_t>> q_;
    bool stop_ = false;
    TierStats st_;
};

}  // namespace ps5lm
