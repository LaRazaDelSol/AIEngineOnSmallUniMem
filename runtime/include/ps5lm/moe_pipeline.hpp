// moe_pipeline.hpp — GPU-driven MoE decode pipeline (HIP, Linux).
//
// Submit side (CPU 2) enqueues a WHOLE token once; per MoE layer L:
//   pre_moe(L)                          dense ops (attention, norms) -> x_moe   [DenseOps hook]
//   quantize x_moe -> xq ; router(L) -> mailbox[L] (+ predictions L+1..L+K)
//   hipStreamWaitValue32(ready[L] >= seq)    <- released by the streamer
//   expert_ffn(L)  -> moe_out
//   hipStreamWriteValue32(done[L] = seq)     -> streamer unpins L's slots
//   post_moe(L)                         residual add + shared expert            [DenseOps hook]
//
// Streamer side (CPU 1, ExpertStreamer owner) — pump() loop:
//   mailbox[L].seq == seq -> demand ids -> request+pin; predictions -> speculative request
//   reap CQEs; when all of L's slots are ready -> write slot_ids[L], route_w[L] ->
//   release-store ready[L] = seq; done[L] == seq -> unpin.
//
// INVARIANT: ready[L] is released only when every slot of L is resident, so
// the GPU wait is bounded by I/O latency, never by an unissued read.
#pragma once
#if !defined(__linux__)
#error "moe_pipeline.hpp is Linux-only"
#endif

#include "ps5lm/expert_streamer.hpp"
#include "ps5lm/quant.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5lm {

// Must match moe_kernels.hip exactly.
struct RouterMailboxHost {
    int32_t ids[8][16];
    float weights[8][16];
    uint32_t seq;
    uint32_t done;
    uint32_t pad[14];
};

extern "C" {
hipError_t ps5lm_launch_quantize_act(const float* x, q::Q8Block* out, int n_blocks, hipStream_t st);
hipError_t ps5lm_launch_router(const float* x, const void* w_router_f16, int layer, int n_layers, int hidden,
                               int experts, int top_k, int lookahead, void* mb, uint32_t seq, hipStream_t st);
hipError_t ps5lm_launch_expert_ffn(int bits, const uint8_t* slab_dev, const uint32_t* slot_ids, const float* route_w,
                                   uint64_t slot_bytes, const q::Q8Block* xq, float* h, q::Q8Block* hq, int hidden,
                                   int ffn, int top_k, float* out, hipStream_t st);
}

inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// Dense (non-MoE) part of each layer; implemented by the model module.
struct DenseOps {
    virtual ~DenseOps() = default;
    // Must leave the MoE input (post-norm hidden, f32 [hidden]) in x_moe.
    virtual void pre_moe(int layer, float* x_moe, hipStream_t st) = 0;
    // Consumes moe_out (routed sum); adds shared expert + residual.
    virtual void post_moe(int layer, const float* moe_out, hipStream_t st) = 0;
};

struct PipelineConfig {
    int layers = 48, experts = 512, top_k = 10, hidden = 2560, ffn = 640;
    int bits = 2;           // PS5Q weight bits of the expert pack
    int lookahead = 4;      // GPU-predicted layers ahead (<= 7)
    int max_spec_per_layer = 10;  // speculative reads per predicted layer (top-m of the prediction)
};

class MoePipeline {
public:
    // `slab` is the arena "experts" region already owned by `st` (registered
    // with io_uring); here it is registered with the GPU (zero-copy).
    MoePipeline(const PipelineConfig& c, ExpertStreamer& st, const void* w_router_dev_f16)
        : c_(c), st_(st), w_router_(w_router_dev_f16) {
        if (c.top_k > 16 || c.lookahead > 7) throw std::invalid_argument("top_k<=16, lookahead<=7");
        auto slab = st.slab();
        hip_check(hipHostRegister(slab.data(), slab.size(), hipHostRegisterMapped | hipHostRegisterReadOnly),
                  "hipHostRegister(slab)");
        slab_registered_ = slab.data();
        hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&slab_dev_), slab.data(), 0),
                  "hipHostGetDevicePointer(slab)");

        // Coherent host-mapped control memory: written by one side, polled by the other.
        const unsigned fl = hipHostMallocMapped | hipHostMallocCoherent;
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&mb_), sizeof(RouterMailboxHost) * c.layers, fl), "mailbox");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&ready_), sizeof(uint32_t) * c.layers, fl), "ready");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&done_), sizeof(uint32_t) * c.layers, fl), "done");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&slot_ids_), sizeof(uint32_t) * 16 * c.layers, fl), "slot ids");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&route_w_), sizeof(float) * 16 * c.layers, fl), "route w");
        std::memset(mb_, 0, sizeof(RouterMailboxHost) * c.layers);
        std::memset(ready_, 0, sizeof(uint32_t) * c.layers);
        std::memset(done_, 0, sizeof(uint32_t) * c.layers);

        hip_check(hipMalloc(&x_moe_, sizeof(float) * c.hidden), "x_moe");
        hip_check(hipMalloc(&moe_out_, sizeof(float) * c.hidden), "moe_out");
        hip_check(hipMalloc(&xq_, sizeof(q::Q8Block) * (c.hidden / q::QK)), "xq");
        hip_check(hipMalloc(&h_, sizeof(float) * c.top_k * c.ffn), "h");
        hip_check(hipMalloc(&hq_, sizeof(q::Q8Block) * c.top_k * (c.ffn / q::QK)), "hq");
        hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "stream");
        pending_.resize(c.layers);
    }

    ~MoePipeline() {
        hipStreamSynchronize(stream_);
        hipStreamDestroy(stream_);
        for (void* p : {static_cast<void*>(x_moe_), static_cast<void*>(moe_out_), static_cast<void*>(xq_),
                        static_cast<void*>(h_), static_cast<void*>(hq_)})
            hipFree(p);
        for (void* p : {static_cast<void*>(mb_), static_cast<void*>(ready_), static_cast<void*>(done_),
                        static_cast<void*>(slot_ids_), static_cast<void*>(route_w_)})
            hipHostFree(p);
        if (slab_registered_) hipHostUnregister(slab_registered_);
    }
    MoePipeline(const MoePipeline&) = delete;
    MoePipeline& operator=(const MoePipeline&) = delete;

    static uint32_t seq_of(uint64_t token, int layers, int layer) {
        return uint32_t(token * uint64_t(layers) + uint64_t(layer) + 1);  // never 0 (0 = "nothing yet")
    }

    // ---------------- CPU 2: enqueue one token (non-blocking) ----------------
    void enqueue_token(uint64_t token, DenseOps& dense) {
        for (int L = 0; L < c_.layers; ++L) {
            const uint32_t seq = seq_of(token, c_.layers, L);
            dense.pre_moe(L, x_moe_, stream_);
            hip_check(ps5lm_launch_quantize_act(x_moe_, xq_, c_.hidden / q::QK, stream_), "quantize x");
            hip_check(ps5lm_launch_router(x_moe_, w_router_, L, c_.layers, c_.hidden, c_.experts, c_.top_k,
                                          c_.lookahead, &mb_[L], seq, stream_),
                      "router");
            hip_check(hipStreamWaitValue32(stream_, &ready_[L], seq, hipStreamWaitValueGte, 0xFFFFFFFFu),
                      "wait ready");
            hip_check(ps5lm_launch_expert_ffn(c_.bits, slab_dev_, &slot_ids_[16 * L], &route_w_[16 * L],
                                              st_.config().slot_bytes, xq_, h_, hq_, c_.hidden, c_.ffn, c_.top_k,
                                              moe_out_, stream_),
                      "expert ffn");
            hip_check(hipStreamWriteValue32(stream_, &done_[L], seq, 0), "write done");
            dense.post_moe(L, moe_out_, stream_);
        }
    }

    // ---------------- CPU 1: streamer pump (call in a tight loop) ----------------
    // Returns true when it made progress. Single-threaded with the ExpertStreamer.
    bool pump(uint64_t token) {
        bool progress = false;
        for (int L = 0; L < c_.layers; ++L) {
            Pending& p = pending_[L];
            const uint32_t seq = seq_of(token, c_.layers, L);
            // 0) GPU finished the layer we last released (possibly the previous
            //    token's) -> unpin. Runs FIRST so a new arrival never drops pins.
            if (p.released && !p.retired && load_acquire(&done_[L]) == p.seq) {
                for (int i = 0; i < p.n; ++i) st_.cache().unpin(p.slots[i]);
                p.retired = true;
                progress = true;
            }
            // 1) router output for L landed? (stream order guarantees done(prev)
            //    precedes router(next), so p is retired or never used here)
            if (p.seq != seq && load_acquire(&mb_[L].seq) == seq) {
                if (p.seq != 0 && !p.retired) throw std::logic_error("pipeline: layer reused before retire");
                p = {};
                p.seq = seq;
                for (int i = 0; i < c_.top_k; ++i) {
                    p.experts[i] = uint32_t(mb_[L].ids[0][i]);
                    p.weights[i] = mb_[L].weights[0][i];
                }
                p.n = c_.top_k;
                issue_demand(L, p);
                issue_predictions(L);
                progress = true;
            }
            // 2) retry demand that hit a saturated pool
            if (p.seq == seq && !p.released && p.unbound) { issue_demand(L, p); }
            // 3) all resident -> bind + release
            if (p.seq == seq && !p.released && !p.unbound && all_ready(p)) {
                for (int i = 0; i < p.n; ++i) {
                    slot_ids_[16 * L + i] = p.slots[i];
                    route_w_[16 * L + i] = p.weights[i];
                }
                store_release(&ready_[L], seq);
                p.released = true;
                progress = true;
            }
        }
        st_.submit();
        if (st_.reap()) progress = true;
        return progress;
    }

    void sync() { hip_check(hipStreamSynchronize(stream_), "sync"); }
    hipStream_t stream() const { return stream_; }

private:
    struct Pending {
        uint32_t seq = 0;
        int n = 0;
        uint32_t experts[16]{};
        float weights[16]{};
        SlotId slots[16]{};
        bool bound[16]{};
        bool unbound = false, released = false, retired = false;
    };

    static uint32_t load_acquire(const uint32_t* p) {
        return std::atomic_ref<uint32_t>(*const_cast<uint32_t*>(p)).load(std::memory_order_acquire);
    }
    static void store_release(uint32_t* p, uint32_t v) {
        std::atomic_ref<uint32_t>(*p).store(v, std::memory_order_release);
    }

    void issue_demand(int L, Pending& p) {
        p.unbound = false;
        for (int i = 0; i < p.n; ++i) {
            if (p.bound[i]) continue;
            auto r = st_.request(uint32_t(L), p.experts[i]);
            if (r.slot == kNoSlot) { p.unbound = true; continue; }  // saturated: retry next pump
            st_.cache().pin(r.slot);
            p.slots[i] = r.slot;
            p.bound[i] = true;
        }
    }

    void issue_predictions(int L) {
        for (int j = 1; j <= c_.lookahead && L + j < c_.layers; ++j)
            for (int i = 0; i < std::min(c_.top_k, c_.max_spec_per_layer); ++i) {
                const uint32_t e = uint32_t(mb_[L].ids[j][i]);
                if (st_.cache().find(uint32_t(L + j), e) != kNoSlot) continue;  // resident or in flight
                st_.request(uint32_t(L + j), e);  // speculative: dropped silently on saturation
            }
    }

    bool all_ready(const Pending& p) {
        for (int i = 0; i < p.n; ++i)
            if (!st_.cache().ready(p.slots[i])) {
                // A failed read invalidates its slot: re-request it.
                if (st_.cache().state(p.slots[i]) == SlotState::Free) {
                    st_.cache().unpin(p.slots[i]);
                    const_cast<Pending&>(p).bound[i] = false;
                    const_cast<Pending&>(p).unbound = true;
                }
                return false;
            }
        return true;
    }

    PipelineConfig c_;
    ExpertStreamer& st_;
    const void* w_router_;
    void* slab_registered_ = nullptr;
    uint8_t* slab_dev_ = nullptr;
    RouterMailboxHost* mb_ = nullptr;
    uint32_t *ready_ = nullptr, *done_ = nullptr, *slot_ids_ = nullptr;
    float* route_w_ = nullptr;
    float *x_moe_ = nullptr, *moe_out_ = nullptr, *h_ = nullptr;
    q::Q8Block *xq_ = nullptr, *hq_ = nullptr;
    hipStream_t stream_{};
    std::vector<Pending> pending_;
};

}  // namespace ps5lm
