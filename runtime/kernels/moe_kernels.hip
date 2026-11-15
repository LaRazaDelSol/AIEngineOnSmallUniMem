// moe_kernels.hip — PS5Q MoE decode kernels for RDNA2 (gfx1030), wave32, DP4A.
//
// Build: hipcc --offload-arch=gfx1030 -O3 -ffast-math -std=c++20 (wave32 is the
// gfx10 default; do NOT pass -mwavefrontsize64).
// PS5LM_SOFT_DOT4=1 swaps v_dot4_i32_i8 for the bit-identical software model
// (for ISAs without dot1-insts, e.g. gfx1010/gfx1013 — RUNTIME-SPEC gate C1).
//
// Per MoE layer, decode (batch 1), all on one stream:
//   quantize_act_q8 (x)      -> xq            [hidden/32 Q8 blocks]
//   router_topk               -> mailbox       ids/weights for L, predictions for L+1..L+K
//   ...host binds slots, waits for I/O, writes slot ids, bumps ready seq...
//   hipStreamWaitValue32(ready[L] >= seq)                       (GPU never runs on non-resident bytes)
//   gate_up_swiglu<B>         -> h             [top_k x ffn] f32
//   quantize_act_q8 (h)       -> hq            [top_k x ffn/32]
//   down_combine<B>           -> moe_out       [hidden] f32 = sum_e w_e * down_e(hq_e)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "ps5lm/quant.hpp"

namespace ps5lm::k {

using q::Q8Block;
using q::QK;

constexpr int kWave = 32;

__device__ __forceinline__ int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
#if PS5LM_SOFT_DOT4
    return q::sdot4(a, b, c);
#else
    return __builtin_amdgcn_sdot4(int(a), int(b), c, false);
#endif
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    return v;
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int o = kWave / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kWave));
    return v;
}
__device__ __forceinline__ int wave_isum(int v) {
#pragma unroll
    for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    return v;
}

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// Integer block dot: identical unpack order to q::dot_row_q8 (bit-exact s).
template <int B>
__device__ __forceinline__ int32_t block_dot(const uint8_t* __restrict__ qblk, const Q8Block& a) {
    const uint32_t* aw = reinterpret_cast<const uint32_t*>(a.qs);
    int32_t s = 0;
    if constexpr (B == 4) {
        const uint4 w = *reinterpret_cast<const uint4*>(qblk);  // one 16 B load per block
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s = dot4(wv[j] & 0x0F0F0F0Fu, aw[j], s);
            s = dot4((wv[j] >> 4) & 0x0F0F0F0Fu, aw[4 + j], s);
        }
    } else {
        const uint2 w = *reinterpret_cast<const uint2*>(qblk);  // one 8 B load per block
        const uint32_t wv[2] = {w.x, w.y};
#pragma unroll
        for (int sh = 0; sh < 4; ++sh)
#pragma unroll
            for (int j = 0; j < 2; ++j) s = dot4((wv[j] >> (2 * sh)) & 0x03030303u, aw[2 * sh + j], s);
    }
    return s;
}

// ---------------------------------------------------------------------------
// 1. Activation quantization: one wave32 per Q8 block, lane i owns element i.
__global__ void __launch_bounds__(256) quantize_act_q8(const float* __restrict__ x, Q8Block* __restrict__ out,
                                                         int n_blocks) {
    const int lane = threadIdx.x % kWave;
    const int blk = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
    if (blk >= n_blocks) return;
    const float v = x[blk * QK + lane];
    const float amax = wave_max(fabsf(v));
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    const int qv = max(-127, min(127, __float2int_rn(v * id)));
    out[blk].qs[lane] = int8_t(qv);
    const int s = wave_isum(qv);
    if (lane == 0) { out[blk].d = d; out[blk].sum = s; }
}

// ---------------------------------------------------------------------------
// 2. Router + GPU-side cross-layer predictor.
// grid.y = 1 + lookahead: y=0 routes layer L, y=j predicts layer L+j from the
// SAME hidden state x_L (residual stream drifts slowly between layers).
// One workgroup of 512 threads per y: thread e computes logit e.
// Mailbox is host-mapped memory; seq is written last with release semantics.
struct RouterMailbox {  // layout mirrored by ps5lm::RouterMailboxHost (moe_pipeline.hpp)
    int32_t ids[8][16];      // [0]=actual top-k for L, [j]=prediction for L+j
    float weights[8][16];
    uint32_t seq;            // = token*layers + L + 1, published after ALL blocks' stores land
    uint32_t done;           // last-block-done counter (device-internal)
    uint32_t pad[14];
};

constexpr int kMaxTopK = 16;

__global__ void __launch_bounds__(512) router_topk(const float* __restrict__ x,           // [hidden]
                                                    const __half* __restrict__ w_router,   // [layers][experts][hidden]
                                                    int layer, int n_layers, int hidden, int experts, int top_k,
                                                    RouterMailbox* __restrict__ mb, uint32_t seq) {
    extern __shared__ float smem[];       // hidden floats of x, then experts logits
    float* xs = smem;
    float* logits = smem + hidden;
    __shared__ float red_v[16];
    __shared__ int red_i[16];

    const int y = blockIdx.y;
    const int tgt = layer + y;
    if (tgt < n_layers) {  // out-of-range predictions still take part in the done count
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    // logit e: fp16 weights x f32 activations, 8 halves (16 B) per load
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        const __half* wr = w_router + (size_t(tgt) * experts + e) * hidden;
        float acc = 0.f;
        for (int k = 0; k < hidden; k += 8) {
            const uint4 raw = *reinterpret_cast<const uint4*>(wr + k);
            const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h2[j]);
                acc = fmaf(f.x, xs[k + 2 * j], acc);
                acc = fmaf(f.y, xs[k + 2 * j + 1], acc);
            }
        }
        logits[e] = acc;
    }
    __syncthreads();

    // top-k by k rounds of block argmax (k=10, experts=512: 10 x two-level reduce)
    const int lane = threadIdx.x % kWave, wv = threadIdx.x / kWave, nw = blockDim.x / kWave;
    float top_logit[kMaxTopK];
    int top_id[kMaxTopK];
    for (int r = 0; r < top_k; ++r) {
        float best = -INFINITY;
        int bi = -1;
        for (int e = threadIdx.x; e < experts; e += blockDim.x)
            if (logits[e] > best) { best = logits[e]; bi = e; }
#pragma unroll
        for (int o = kWave / 2; o > 0; o >>= 1) {
            const float ov = __shfl_xor(best, o, kWave);
            const int oi = __shfl_xor(bi, o, kWave);
            if (ov > best || (ov == best && oi < bi && oi >= 0)) { best = ov; bi = oi; }
        }
        if (lane == 0) { red_v[wv] = best; red_i[wv] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = red_v[0];
            int i = red_i[0];
            for (int w = 1; w < nw; ++w)
                if (red_v[w] > b || (red_v[w] == b && red_i[w] < i)) { b = red_v[w]; i = red_i[w]; }
            red_v[0] = b;
            red_i[0] = i;
            logits[i] = -INFINITY;
        }
        __syncthreads();
        top_logit[r] = red_v[0];
        top_id[r] = red_i[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // softmax restricted to the top-k (== full softmax + renormalize over top-k)
        float m = top_logit[0], z = 0.f;
        for (int r = 0; r < top_k; ++r) z += __expf(top_logit[r] - m);
        for (int r = 0; r < top_k; ++r) {
            mb->ids[y][r] = top_id[r];
            mb->weights[y][r] = __expf(top_logit[r] - m) / z;
        }
    }
    }  // tgt < n_layers
    // Publish: the LAST block to finish releases seq, so the host never sees a
    // new seq with stale ids/weights/predictions.
    if (threadIdx.x == 0) {
        __threadfence_system();
        const uint32_t prev = __hip_atomic_fetch_add(&mb->done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
        if (prev == gridDim.y - 1) {
            __hip_atomic_store(&mb->done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(&mb->seq, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Fused gate+up+SwiGLU. grid = (ffn / 8, top_k), block = 256 (8 waves).
// Wave w owns output row r = blockIdx.x*8 + w for expert blockIdx.y; lanes
// stride the 80 K-blocks; gate and up share every activation LDS read.
template <int B>
__global__ void __launch_bounds__(256) gate_up_swiglu(const uint8_t* __restrict__ slab,  // device ptr of mapped slab
                                                       const uint32_t* __restrict__ slot_ids,  // [top_k]
                                                       uint64_t slot_bytes, const Q8Block* __restrict__ xq,
                                                       int hidden, int ffn, float* __restrict__ h) {
    extern __shared__ Q8Block xs[];
    const int nb = hidden / QK;
    for (int i = threadIdx.x; i < nb * int(sizeof(Q8Block) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(xs)[i] = reinterpret_cast<const uint32_t*>(xq)[i];
    __syncthreads();

    const int e = blockIdx.y;
    const int r = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
    const int lane = threadIdx.x % kWave;
    if (r >= ffn) return;

    const q::ExpertLayout<B> L(q::ExpertShape{hidden, ffn});
    const size_t rb = q::row_bytes<B>(hidden), sb = q::scales_bytes<B>(hidden);
    const uint8_t* base = slab + uint64_t(slot_ids[e]) * slot_bytes;
    const uint8_t* rg = base + L.gate_off + size_t(r) * rb;
    const uint8_t* ru = base + L.up_off + size_t(r) * rb;
    const uint16_t* sg = reinterpret_cast<const uint16_t*>(rg);
    const uint16_t* su = reinterpret_cast<const uint16_t*>(ru);

    float g = 0.f, u = 0.f;
    for (int b = lane; b < nb; b += kWave) {
        const Q8Block& a = xs[b];
        const int32_t zs = q::Fmt<B>::zero * a.sum;
        g += h2f(sg[b]) * a.d * float(block_dot<B>(rg + sb + size_t(b) * q::Fmt<B>::quant_bytes, a) - zs);
        u += h2f(su[b]) * a.d * float(block_dot<B>(ru + sb + size_t(b) * q::Fmt<B>::quant_bytes, a) - zs);
    }
    g = wave_sum(g);
    u = wave_sum(u);
    if (lane == 0) h[size_t(e) * ffn + r] = g / (1.f + __expf(-g)) * u;
}

// ---------------------------------------------------------------------------
// 4. Down projection + routed-weight combine, deterministic (no atomics).
// grid = hidden / 8, block = 256. Each wave owns one output row and spreads
// its top_k x (ffn/32) = 10 x 20 = 200 (expert, block) items over 32 lanes.
template <int B>
__global__ void __launch_bounds__(256) down_combine(const uint8_t* __restrict__ slab,
                                                     const uint32_t* __restrict__ slot_ids,  // [top_k]
                                                     const float* __restrict__ route_w,      // [top_k]
                                                     uint64_t slot_bytes, const Q8Block* __restrict__ hq,  // [top_k][ffn/32]
                                                     int hidden, int ffn, int top_k, float* __restrict__ out) {
    extern __shared__ Q8Block hs[];
    const int nbf = ffn / QK, items = top_k * nbf;
    for (int i = threadIdx.x; i < items * int(sizeof(Q8Block) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(hs)[i] = reinterpret_cast<const uint32_t*>(hq)[i];
    __syncthreads();

    const int r = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
    const int lane = threadIdx.x % kWave;
    if (r >= hidden) return;

    const q::ExpertLayout<B> L(q::ExpertShape{hidden, ffn});
    const size_t rb = q::row_bytes<B>(ffn), sb = q::scales_bytes<B>(ffn);
    float acc = 0.f;
    for (int it = lane; it < items; it += kWave) {
        const int e = it / nbf, b = it % nbf;
        const uint8_t* row = slab + uint64_t(slot_ids[e]) * slot_bytes + L.down_off + size_t(r) * rb;
        const Q8Block& a = hs[it];
        const int32_t s = block_dot<B>(row + sb + size_t(b) * q::Fmt<B>::quant_bytes, a) - q::Fmt<B>::zero * a.sum;
        acc += route_w[e] * h2f(reinterpret_cast<const uint16_t*>(row)[b]) * a.d * float(s);
    }
    acc = wave_sum(acc);
    if (lane == 0) out[r] = acc;
}

// ---------------------------------------------------------------------------
static_assert(sizeof(RouterMailbox) == 8 * 16 * 4 * 2 + 16 * 4, "mailbox layout drift");

// Host launch wrappers (C ABI so the host side can be compiled by any C++20
// compiler and only this file needs hipcc).
extern "C" {

hipError_t ps5lm_launch_quantize_act(const float* x, Q8Block* out, int n_blocks, hipStream_t st) {
    const int per_wg = 256 / kWave;
    hipLaunchKernelGGL(quantize_act_q8, dim3((n_blocks + per_wg - 1) / per_wg), dim3(256), 0, st, x, out, n_blocks);
    return hipGetLastError();
}

hipError_t ps5lm_launch_router(const float* x, const void* w_router_f16, int layer, int n_layers, int hidden,
                               int experts, int top_k, int lookahead, void* mailbox, uint32_t seq,
                               hipStream_t st) {
    auto* mb = static_cast<RouterMailbox*>(mailbox);
    if (top_k > kMaxTopK || lookahead > 7) return hipErrorInvalidValue;
    const size_t lds = size_t(hidden + experts) * sizeof(float);
    hipLaunchKernelGGL(router_topk, dim3(1, 1 + lookahead), dim3(512), lds, st, x,
                       static_cast<const __half*>(w_router_f16), layer, n_layers, hidden, experts, top_k, mb, seq);
    return hipGetLastError();
}

hipError_t ps5lm_launch_expert_ffn(int bits, const uint8_t* slab_dev, const uint32_t* slot_ids, const float* route_w,
                                   uint64_t slot_bytes, const Q8Block* xq, float* h, Q8Block* hq, int hidden, int ffn,
                                   int top_k, float* out, hipStream_t st) {
    const size_t lds_x = size_t(hidden / QK) * sizeof(Q8Block);
    const size_t lds_h = size_t(top_k) * (ffn / QK) * sizeof(Q8Block);
    const dim3 blk(256);
    if (bits == 4) {
        hipLaunchKernelGGL(gate_up_swiglu<4>, dim3(ffn / 8, top_k), blk, lds_x, st, slab_dev, slot_ids, slot_bytes, xq,
                           hidden, ffn, h);
    } else if (bits == 2) {
        hipLaunchKernelGGL(gate_up_swiglu<2>, dim3(ffn / 8, top_k), blk, lds_x, st, slab_dev, slot_ids, slot_bytes, xq,
                           hidden, ffn, h);
    } else {
        return hipErrorInvalidValue;
    }
    if (hipError_t e = ps5lm_launch_quantize_act(h, hq, top_k * (ffn / QK), st); e != hipSuccess) return e;
    if (bits == 4)
        hipLaunchKernelGGL(down_combine<4>, dim3(hidden / 8), blk, lds_h, st, slab_dev, slot_ids, route_w, slot_bytes,
                           hq, hidden, ffn, top_k, out);
    else
        hipLaunchKernelGGL(down_combine<2>, dim3(hidden / 8), blk, lds_h, st, slab_dev, slot_ids, route_w, slot_bytes,
                           hq, hidden, ffn, top_k, out);
    return hipGetLastError();
}

}  // extern "C"

}  // namespace ps5lm::k
