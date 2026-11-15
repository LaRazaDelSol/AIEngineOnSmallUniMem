// test_kernels_hip — device kernels vs the CPU reference (run on the target GPU).
// Build with hipcc (CMake option PS5LM_HIP=ON). Exercises the zero-copy path:
// expert slots live in hipHostRegister'ed pinned host memory, as in production.
#include "ps5lm/moe_pipeline.hpp"
#include "ps5lm/quant.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace ps5lm;
using namespace ps5lm::q;

template <int B> static int run() {
    const ExpertShape s;
    const ExpertLayout<B> L(s);
    const int top_k = 10, n_slots = 16;
    uint8_t* slab = nullptr;
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&slab), size_t(n_slots) * L.bytes, hipHostMallocMapped), "slab");

    std::mt19937 g(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w(size_t(s.hidden));
    for (int sl = 0; sl < n_slots; ++sl) {
        uint8_t* base = slab + size_t(sl) * L.bytes;
        for (int r = 0; r < 2 * s.ffn; ++r) {
            for (auto& v : w) v = nd(g) / std::sqrt(float(s.hidden));
            quantize_row<B>(w.data(), s.hidden, base + L.gate_off + size_t(r) * row_bytes<B>(s.hidden));
        }
        for (int r = 0; r < s.hidden; ++r) {
            for (int k = 0; k < s.ffn; ++k) w[k] = nd(g) / std::sqrt(float(s.ffn));
            quantize_row<B>(w.data(), s.ffn, base + L.down_off + size_t(r) * row_bytes<B>(s.ffn));
        }
    }
    std::vector<float> x(s.hidden);
    for (auto& v : x) v = nd(g);
    uint32_t slot_ids[top_k];
    float route_w[top_k];
    for (int i = 0; i < top_k; ++i) { slot_ids[i] = uint32_t((i * 7) % n_slots); route_w[i] = 1.f / top_k; }

    // CPU reference
    std::vector<Q8Block> xq(s.hidden / QK), hq(s.ffn / QK);
    quantize_act(x.data(), s.hidden, xq.data());
    std::vector<float> ref(s.hidden, 0.f), h(s.ffn);
    for (int i = 0; i < top_k; ++i)
        expert_ffn_ref<B>(slab + size_t(slot_ids[i]) * L.bytes, xq.data(), route_w[i], ref.data(), s, h.data(), hq.data());

    // Device
    uint8_t* slab_dev = nullptr;
    hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&slab_dev), slab, 0), "dev ptr");
    float *dx, *dh, *dout, *dw;
    uint32_t* dids;
    Q8Block *dxq, *dhq;
    hip_check(hipMalloc(&dx, sizeof(float) * s.hidden), "x");
    hip_check(hipMalloc(&dout, sizeof(float) * s.hidden), "out");
    hip_check(hipMalloc(&dh, sizeof(float) * top_k * s.ffn), "h");
    hip_check(hipMalloc(&dxq, sizeof(Q8Block) * (s.hidden / QK)), "xq");
    hip_check(hipMalloc(&dhq, sizeof(Q8Block) * top_k * (s.ffn / QK)), "hq");
    hip_check(hipMalloc(&dids, sizeof(slot_ids)), "ids");
    hip_check(hipMalloc(&dw, sizeof(route_w)), "w");
    hip_check(hipMemcpy(dx, x.data(), sizeof(float) * s.hidden, hipMemcpyHostToDevice), "cp x");
    hip_check(hipMemcpy(dids, slot_ids, sizeof(slot_ids), hipMemcpyHostToDevice), "cp ids");
    hip_check(hipMemcpy(dw, route_w, sizeof(route_w), hipMemcpyHostToDevice), "cp w");
    hip_check(ps5lm_launch_quantize_act(dx, dxq, s.hidden / QK, nullptr), "q");
    hip_check(ps5lm_launch_expert_ffn(B, slab_dev, dids, dw, L.bytes, dxq, dh, dhq, s.hidden, s.ffn, top_k, dout,
                                      nullptr),
              "ffn");
    std::vector<float> got(s.hidden);
    hip_check(hipMemcpy(got.data(), dout, sizeof(float) * s.hidden, hipMemcpyDeviceToHost), "cp out");

    // Timing: 200 reps of the expert FFN (zero-copy weights in host-registered memory)
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0, nullptr);
    for (int r = 0; r < 200; ++r)
        ps5lm_launch_expert_ffn(B, slab_dev, dids, dw, L.bytes, dxq, dh, dhq, s.hidden, s.ffn, top_k, dout, nullptr);
    hipEventRecord(e1, nullptr);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    const double us = 1000.0 * ms / 200;
    const double gbs = double(top_k) * L.bytes / (us * 1e-6) / 1e9;

    double e2 = 0, r2 = 0;
    for (int i = 0; i < s.hidden; ++i) { e2 += double(got[i] - ref[i]) * (got[i] - ref[i]); r2 += double(ref[i]) * ref[i]; }
    const double rel = std::sqrt(e2 / r2);
    std::printf("Q%d: device vs CPU-ref rel-RMS %.2e | %.1f us/layer (top-%d) = %.1f GB/s weight stream | x48 layers = %.2f ms/token\n",
                B, rel, us, top_k, gbs, us * 48 / 1000);
    hipHostFree(slab);
    return rel < 1e-4 ? 0 : 1;  // integer path is exact; only f32 summation order differs
}

int main() {
    int rc = run<4>() | run<2>();
    std::puts(rc ? "test_kernels_hip: FAILED" : "test_kernels_hip: all passed");
    return rc;
}
