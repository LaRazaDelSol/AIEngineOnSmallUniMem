// expert_streamer.hpp — Linux io_uring READ_FIXED streamer over a SlotCache.
//
// Bytes path (research/design-history/RUNTIME-SPEC.md §1.5): the arena's "experts" region (inside
// one boot-reserved 1 GiB / 2 MiB HugeTLB mapping) -> registered io_uring
// buffers -> NVMe DMA lands in unified GDDR6 -> the GPU reads the same bytes
// through the hipHostRegister'ed / host-pointer-imported mapping. No copies.
//
// Thread model: every method runs on the streamer thread (CPU 1); the SQPOLL
// kernel thread sits on CPU 0. cache().ready() may be polled from CPU 2-3.
//
// INVARIANT (DEVICE_LOST 10 s rule): a layer is handed to the GPU only after
// every one of its slots is ready(). The GPU never waits on I/O.
#pragma once
#if !defined(__linux__)
#error "expert_streamer.hpp is Linux-only (io_uring); slot_cache.hpp is the portable half"
#endif

#include "ps5lm/slot_cache.hpp"
#include "ps5lm/lowmem_tier.hpp"

#include <liburing.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ps5lm {

struct StreamerConfig {
    std::string pack_path;
    uint32_t layers = 48;
    uint32_t experts = 512;
    uint32_t slot_bytes = 1'019'904;  // 249 * 4096 (EXL3-1.66 geometry)
    uint64_t expert_base = 0;         // byte offset of the expert section in the pack
    Policy policy = Policy::CLOCK;
    unsigned ring_entries = 512;      // >= max in-flight; 512 x ~1 MiB = 0.5 GiB of reads queued
    int sqpoll_cpu = 0;               // SQPOLL kernel thread on CPU 0 (-1: no SQPOLL, syscall submit)
    unsigned sqpoll_idle_ms = 2000;   // poller never sleeps during a decode
    bool iopoll = true;               // polled NVMe completions: needs nvme.poll_queues>0
    unsigned split = 1;               // S14: issue each slot read as N page-aligned SQEs. A lone 1.5 MB miss is
                                      // ~qd1 (layer-serial routing); fio on the SN810: 448 us whole, 309 us as 8 parts
    // S15b (EXL3 packs, per-layer slot geometry): expert (l, e) at layer_base[l] + e * layer_bytes[l], read
    // layer_bytes[l] (<= slot_bytes, 4 KiB multiple). Empty = the fixed (layer * experts + e) * slot_bytes layout.
    std::vector<uint64_t> layer_base;
    std::vector<uint32_t> layer_bytes;
    // S15b size classes: explicit slot placement in the slab (empty = uniform slot_bytes). slot_off[s] = byte offset,
    // slot_chunk[s] = registered-buffer index; chunks = [start, len) pieces <= 1 GiB that no slot straddles.
    std::vector<uint64_t> slot_off;
    std::vector<uint16_t> slot_chunk;
    std::vector<std::pair<uint64_t, uint64_t>> chunks;
    std::vector<uint8_t> layer_class;  // with slot_off: class of each layer; class 0 = slots [0, n_class0)
    uint32_t n_class0 = 0;
    // S23 mirrored packs: byte-identical copies of pack_path on OTHER block devices. Every miss is read from the device
    // that would finish it first, (bytes in flight + this read) / bandwidth. Bandwidths are additive across devices on this
    // console (R6: SN810 6,574 + USB 10 Gbps 881 MB/s concurrently = the sum of each alone), and decode waits on bytes,
    // not on latency (R56). A mirror must have the primary's size; contents are the operator's responsibility
    // (scripts/ps5-mirror-pack.sh copies and compares).
    struct Mirror { std::string path; double mbps; };
    std::vector<Mirror> mirrors;
    double primary_mbps = 6400;       // SN810 at slot-size reads, qd >= 4 (R76)
};

struct StreamerStats {
    uint64_t reads_issued = 0;
    uint64_t reads_completed = 0;
    uint64_t bytes_read = 0;
    uint64_t io_errors = 0;
};

class ExpertStreamer {
public:
    // `slab` must be 4 KiB aligned, pinned and pre-faulted (arena "experts"
    // region); its size is rounded down to whole slots.
    ExpertStreamer(const StreamerConfig& c, std::byte* slab, uint64_t slab_bytes)
        : cfg_(c),
          cache_(c.layers, c.experts, c.slot_off.empty() ? uint32_t(slab_bytes / c.slot_bytes) : uint32_t(c.slot_off.size()), c.policy),
          slab_(slab), pending_(cache_.n_slots(), 0), failed_(cache_.n_slots(), 0), slot_dev_(cache_.n_slots(), 0) {
        if (c.slot_bytes % 4096) throw std::invalid_argument("slot_bytes must be 4 KiB aligned (O_DIRECT)");
        const uint32_t pages = c.slot_bytes / 4096, nsplit = std::max(1u, std::min(c.split, pages));
        part_bytes_ = ((pages + nsplit - 1) / nsplit) * 4096u;
        n_parts_ = (c.slot_bytes + part_bytes_ - 1) / part_bytes_;
        if (c.expert_base % 4096) throw std::invalid_argument("expert_base must be 4 KiB aligned (O_DIRECT)");
        if (reinterpret_cast<uintptr_t>(slab) % 4096) throw std::invalid_argument("slab must be 4 KiB aligned");
        n_slots_ = cache_.n_slots();
        slab_bytes_ = c.slot_off.empty() ? uint64_t(n_slots_) * c.slot_bytes : slab_bytes;
        if (!c.slot_off.empty()) cache_.set_classes(c.layer_class, c.n_class0);

        fd_ = ::open(c.pack_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_ < 0) fail("open(O_DIRECT) " + c.pack_path, errno);
        fds_.push_back(fd_);
        bw_.push_back(c.primary_mbps > 0 ? c.primary_mbps : 1.0);
        struct stat st0{};
        if (::fstat(fd_, &st0) != 0) fail("fstat " + c.pack_path, errno);
        for (const auto& m : c.mirrors) {
            if (fds_.size() >= 8) throw std::invalid_argument("at most 7 pack mirrors");
            const int f = ::open(m.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (f < 0) fail("open(O_DIRECT) mirror " + m.path, errno);
            fds_.push_back(f);
            struct stat st{};
            if (::fstat(f, &st) != 0) fail("fstat " + m.path, errno);
            if (S_ISREG(st.st_mode) && st.st_size != st0.st_size)
                throw std::invalid_argument("pack mirror " + m.path + ": size differs from the primary");
            bw_.push_back(m.mbps > 0 ? m.mbps : 1.0);
        }
        inflight_.assign(fds_.size(), 0);
        dev_reads_.assign(fds_.size(), 0);

        io_uring_params p{};
        if (c.sqpoll_cpu >= 0) {
            p.flags |= IORING_SETUP_SQPOLL | IORING_SETUP_SQ_AFF;
            p.sq_thread_cpu = uint32_t(c.sqpoll_cpu);
            p.sq_thread_idle = c.sqpoll_idle_ms;
        }
        if (c.iopoll) p.flags |= IORING_SETUP_IOPOLL;
        if (int r = io_uring_queue_init_params(c.ring_entries, &ring_, &p); r < 0) fail("io_uring_queue_init_params", -r);
        ring_up_ = true;

        if (int r = io_uring_register_files(&ring_, fds_.data(), unsigned(fds_.size())); r < 0) fail("io_uring_register_files", -r);

        // The kernel caps one registered buffer at 1 GiB: chunk the slab so no
        // slot straddles a chunk boundary; buf_index = slot / slots_per_chunk.
        slots_per_chunk_ = uint32_t((1ull << 30) / c.slot_bytes);
        if (c.slot_off.empty()) {
            for (uint32_t s = 0; s < n_slots_; s += slots_per_chunk_) {
                uint32_t n = std::min(slots_per_chunk_, n_slots_ - s);
                iov_.push_back({slab_ + uint64_t(s) * c.slot_bytes, size_t(n) * c.slot_bytes});
            }
        } else {
            for (auto& ch : c.chunks) iov_.push_back({slab_ + ch.first, size_t(ch.second)});
        }
        if (int r = io_uring_register_buffers(&ring_, iov_.data(), unsigned(iov_.size())); r < 0)
            fail("io_uring_register_buffers (raise RLIMIT_MEMLOCK: LimitMEMLOCK=infinity)", -r);
    }

    ~ExpertStreamer() {
        if (ring_up_) io_uring_queue_exit(&ring_);
        if (fds_.empty() && fd_ >= 0) ::close(fd_);
        for (int f : fds_) if (f >= 0) ::close(f);
    }
    ExpertStreamer(const ExpertStreamer&) = delete;
    ExpertStreamer& operator=(const ExpertStreamer&) = delete;

    // Base for vkImportMemoryHostPointerEXT; slot s lives at s * slot_bytes.
    std::span<std::byte> slab() const { return {slab_, size_t(slab_bytes_)}; }
    const std::byte* slot_data(SlotId s) const { return slab_ + slot_offset(s); }
    uint64_t slot_offset(SlotId s) const { return cfg_.slot_off.empty() ? uint64_t(s) * cfg_.slot_bytes : cfg_.slot_off[s]; }
    int slot_buf(SlotId s) const { return cfg_.slot_off.empty() ? int(s / slots_per_chunk_) : int(cfg_.slot_chunk[s]); }
    SlotCache& cache() { return cache_; }
    const StreamerConfig& config() const { return cfg_; }
    const StreamerStats& stats() const { return stats_; }
    uint32_t in_flight() const { return uint32_t(stats_.reads_issued - stats_.reads_completed) + (tier_ ? tier_->pending_fills() : 0u); }

    // S31 lowmem tier (lowmem_tier.hpp): CPU-only second tier in the reserved low gigabyte. Misses whose expert is
    // there are filled by memcpy instead of an SSD read; slots next in line for eviction are copied there first.
    // window = how many next victims (all classes) are kept admitted ahead of eviction.
    void enable_tier(const std::string& dev, uint64_t bytes, unsigned threads, uint32_t window, double min_score = 0) {
        tier_min_score_ = min_score;
        uint32_t cb[2] = {cfg_.slot_bytes, 0}, cw[2] = {n_slots_, 0}, nc = 1;
        std::vector<uint8_t> lc(cfg_.layers, 0);
        if (!cfg_.slot_off.empty() && !cfg_.layer_class.empty()) {
            nc = 2;
            lc = cfg_.layer_class;
            cb[0] = cb[1] = 0;
            for (uint32_t l = 0; l < cfg_.layers; ++l) cb[lc[l]] = std::max(cb[lc[l]], layer_len(l));
            cw[0] = cfg_.n_class0;
            cw[1] = n_slots_ - cfg_.n_class0;
            for (uint32_t c = 0; c < 2; ++c) cb[c] = (cb[c] + 4095u) & ~4095u;
        }
        tier_ = std::make_unique<LowTier>(dev, bytes, cfg_.layers, cfg_.experts, lc, cb, cw, nc, threads);
        tier_window_ = std::max(4u, window);
    }
    const LowTier* tier() const { return tier_.get(); }
    // S31: admission (pool -> tier copies) on/off; fills from what the tier holds are always on. The copies cost GPU
    // memory bandwidth (GDDR6 is shared), so the engine admits only where the SSD is the bound (long context, decode).
    void tier_admit_enable(bool on) { tier_admit_on_ = on; }
    bool tier_has(uint32_t layer, uint32_t expert) const { return tier_ && tier_->has(layer, expert); }
    uint32_t n_devices() const { return uint32_t(fds_.size()); }
    uint64_t device_reads(uint32_t d) const { return dev_reads_[d]; }  // S23: expert reads issued to device d (0 = primary)

    // Hit: returns the resident (or already in-flight) slot. Miss: queues a
    // READ_FIXED into the victim slot. kNoSlot: pool saturated (all pinned or
    // in flight): drop it if speculative, reap and retry if demand.
    SlotCache::Lookup request(uint32_t layer, uint32_t expert, bool cold = false, bool spec = false) {
        if (io_uring_sq_space_left(&ring_) < n_parts_) submit();
        if (io_uring_sq_space_left(&ring_) < n_parts_) return {kNoSlot, false};  // ring full even after submit
        auto r = cache_.lookup_or_assign(layer, expert, cold, spec);
        if (r.slot == kNoSlot || r.hit) return r;
        if (tier_ && cache_.last_evicted() != UINT32_MAX) tier_->note_eviction(cache_.last_evicted());
        const bool per_layer = !cfg_.layer_base.empty();
        const uint64_t off = per_layer ? cfg_.expert_base + cfg_.layer_base[layer] + uint64_t(expert) * cfg_.layer_bytes[layer]
                                       : cfg_.expert_base + (uint64_t(layer) * cfg_.experts + expert) * cfg_.slot_bytes;
        const uint32_t total = per_layer ? cfg_.layer_bytes[layer] : cfg_.slot_bytes;
        if (tier_ && tier_->fill(layer, expert, reinterpret_cast<uint8_t*>(slab_ + slot_offset(r.slot)), total, r.slot, now_s())) {
            pending_[r.slot] = 0;  // S31: filled from the tier; reap() marks it ready
            failed_[r.slot] = 0;
            return r;
        }
        const uint32_t parts = (total + part_bytes_ - 1) / part_bytes_;
        pending_[r.slot] = parts;
        failed_[r.slot] = 0;
        uint32_t dev = 0;  // S23: the device that finishes this read first
        if (fds_.size() > 1) {
            double best = (double(inflight_[0]) + total) / bw_[0];
            for (uint32_t d = 1; d < fds_.size(); ++d) {
                const double t = (double(inflight_[d]) + total) / bw_[d];
                if (t < best) { best = t; dev = d; }
            }
        }
        slot_dev_[r.slot] = uint8_t(dev);
        inflight_[dev] += total;
        ++dev_reads_[dev];
        for (uint32_t p = 0; p < parts; ++p) {
            const uint32_t pb = p * part_bytes_, len = std::min(part_bytes_, total - pb);
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            io_uring_prep_read_fixed(sqe, /*fixed file index*/ int(dev), slab_ + slot_offset(r.slot) + pb, len, off + pb, slot_buf(r.slot));
            sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, (uint64_t(len) << 32) | r.slot);
            ++unsubmitted_;
        }
        ++stats_.reads_issued;
        return r;
    }

    // With SQPOLL this mostly publishes the SQ tail (and wakes an idle poller).
    void submit() {
        if (!unsubmitted_) return;
        int r = io_uring_submit(&ring_);
        if (r < 0) fail("io_uring_submit", -r);
        unsubmitted_ = 0;
    }

    // Drain completions. wait=true blocks for at least one if any are in flight.
    // Returns how many slots became ready. Failed reads invalidate their slot
    // (the caller sees !ready() and re-requests).
    unsigned reap(bool wait = false) {
        submit();
        unsigned tier_ready = 0;
        if (tier_) {
            tier_ready = reap_tier();
            tier_admit();
            if (wait && !tier_ready && tier_->pending_fills()) {
                // a fill is in flight: wait for it OR a read completion, whichever comes first (no blocking wait_cqe)
                for (;;) {
                    if ((tier_ready = reap_tier())) break;
                    io_uring_cqe* c = nullptr;
                    if (io_uring_peek_cqe(&ring_, &c) == 0 && c) break;
                    if (!tier_->pending_fills()) break;
                    __builtin_ia32_pause();
                }
                wait = false;
            }
        }
        if (wait && uint32_t(stats_.reads_issued - stats_.reads_completed)) {
            io_uring_cqe* cqe = nullptr;
            int r = io_uring_wait_cqe(&ring_, &cqe);
            if (r < 0 && r != -EINTR) fail("io_uring_wait_cqe", -r);
        }
        io_uring_cqe* cqes[64];
        unsigned total_ready = 0, n;
        while ((n = io_uring_peek_batch_cqe(&ring_, cqes, 64)) > 0) {
            for (unsigned i = 0; i < n; ++i) {
                const uint64_t ud = io_uring_cqe_get_data64(cqes[i]);
                const SlotId s = SlotId(ud & 0xFFFFFFFFu);
                const int want = int(ud >> 32);
                inflight_[slot_dev_[s]] -= std::min<uint64_t>(inflight_[slot_dev_[s]], uint64_t(want));
                if (cqes[i]->res != want) {
                    ++stats_.io_errors;
                    if (stats_.io_errors <= 8)
                        std::fprintf(stderr, "expert read failed: slot %u res %d (%s)\n", s, cqes[i]->res,
                                     cqes[i]->res < 0 ? std::strerror(-cqes[i]->res) : "short read");
                    failed_[s] = 1;
                } else {
                    stats_.bytes_read += uint64_t(want);
                }
                if (--pending_[s]) continue;  // other parts of this slot still in flight
                ++stats_.reads_completed;
                if (failed_[s]) { cache_.invalidate(s); continue; }
                cache_.mark_ready(s);
                ++total_ready;
            }
            io_uring_cq_advance(&ring_, n);
        }
        return total_ready + tier_ready;
    }

private:
    static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    uint32_t layer_len(uint32_t l) const { return cfg_.layer_bytes.empty() ? cfg_.slot_bytes : cfg_.layer_bytes[l]; }
    unsigned reap_tier() {
        return tier_->reap(now_s(), [this](uint32_t s, bool fill) {
            if (fill) cache_.mark_ready(s);
            else cache_.unpin(s);
        });
    }
    // S31: keep the next `tier_window_` eviction victims copied into the tier before they are overwritten. Rescanned
    // after window/4 evictions or 2 ms. Leaves headroom: never pins when a class has few evictable slots (the wide
    // prefill pins two layers of experts per class; S24b size-class saturation).
    void tier_admit() {
        if (!tier_admit_on_) return;
        const uint64_t ev = cache_.stats().evictions;
        const double t = now_s();
        if (ev - tier_ev_ < tier_window_ / 4 && t - tier_t_ < 0.002) return;
        tier_ev_ = ev;
        tier_t_ = t;
        const uint32_t nc = cache_.n_classes();
        for (uint32_t c = 0; c < nc; ++c) {
            if (!cache_.class_full(int(c))) continue;
            const uint32_t share = nc == 1 ? tier_window_
                                           : std::max(4u, uint32_t(uint64_t(tier_window_) * cache_.class_slots(int(c)) / n_slots_));
            const uint32_t evictable = cache_.next_victims(int(c), share, cand_);
            if (evictable < 4 * share + 64) continue;
            for (SlotId s : cand_) {
                if (tier_->pending() >= kTierInflight) return;
                if (tier_min_score_ > 0 && cache_.score_now(s) < tier_min_score_) continue;  // rarely used: not worth a copy
                const uint32_t key = cache_.key_of(s), layer = key / cfg_.experts, expert = key % cfg_.experts;
                cache_.pin(s);
                if (!tier_->admit(layer, expert, reinterpret_cast<const uint8_t*>(slab_ + slot_offset(s)), layer_len(layer), s, t))
                    cache_.unpin(s);
            }
        }
    }
    static constexpr uint32_t kTierInflight = 64;

    [[noreturn]] static void fail(const std::string& what, int err) {
        throw std::runtime_error(what + ": " + std::strerror(err));
    }

    StreamerConfig cfg_;
    SlotCache cache_;
    io_uring ring_{};
    bool ring_up_ = false;
    int fd_ = -1;
    std::vector<int> fds_;             // S23: [0] = primary, then the mirrors (registered file indices)
    std::vector<double> bw_;           // MB/s per device
    std::vector<uint64_t> inflight_;   // bytes queued or in flight per device
    std::vector<uint64_t> dev_reads_;
    std::byte* slab_ = nullptr;
    uint64_t slab_bytes_ = 0;
    uint32_t n_slots_ = 0, slots_per_chunk_ = 0;
    unsigned unsubmitted_ = 0;
    std::vector<uint32_t> pending_;  // parts still in flight per slot
    std::vector<uint8_t> failed_;
    std::vector<uint8_t> slot_dev_;    // device of the read in flight for each slot
    uint32_t part_bytes_ = 0, n_parts_ = 1;
    std::vector<iovec> iov_;
    StreamerStats stats_;
    std::unique_ptr<LowTier> tier_;  // S31 (null = off)
    uint32_t tier_window_ = 64;
    double tier_min_score_ = 0;
    bool tier_admit_on_ = true;
    uint64_t tier_ev_ = 0;
    double tier_t_ = 0;
    std::vector<SlotId> cand_;
};

}  // namespace ps5lm
