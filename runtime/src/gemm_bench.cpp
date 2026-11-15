// gemm_bench (S21) -- the wide-prefill dense GEMM (gemm_q4.comp) on a random PS5Q4 matrix: error vs a float64 CPU
// reference (and the production matvec's error for scale), GPU time, FMA throughput vs the fp32 peak.
//   gemm_bench --shaders DIR [--n 16480] [--k 2560] [--t 256] [--reps 20] [--spv gemm_q4] [--mode 0]
#include "ps5lm/quant.hpp"
#include "ps5lm/vk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ps5lm;
namespace qn = ps5lm::q;

struct alignas(8) PcGemm { uint64_t w, x, y; uint32_t K, N, T, mode; float alpha; uint32_t xs, ys, pad; uint64_t rmap; };
struct alignas(8) PcMatvecB { uint64_t w, x, y; uint32_t K, N, rpw, mode; float alpha; uint32_t xs, ys, nb; };

int main(int argc, char** argv) {
    std::string sd = "shaders", spv = "gemm_q4";
    uint32_t N = 16480, K = 2560, T = 256, mode = 0, ks = 1;
    bool bf16 = false;
    int reps = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--n") N = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--k") K = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--t") T = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
        else if (a == "--spv") spv = argv[i + 1];
        else if (a == "--mode") mode = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--ks") ks = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--bf16") bf16 = std::stoi(argv[i + 1]) != 0;
    }
    vk::Ctx ctx;
    if (bf16 && spv == "gemm_q4") spv = "gemm_bf16";
    const size_t rb = bf16 ? size_t(K) * 2 : qn::row_bytes<4>(int(K)), sb = bf16 ? 0 : qn::scales_bytes<4>(int(K)), wbytes = rb * N;
    std::mt19937 rng(77);
    std::vector<uint8_t> wh(wbytes);
    std::uniform_real_distribution<float> us(0.002f, 0.02f);
    for (uint32_t r = 0; r < N && bf16; ++r)
        for (uint32_t k = 0; k < K; ++k) {
            float f = us(rng) * (rng() & 1 ? 1.f : -1.f);
            uint32_t b; std::memcpy(&b, &f, 4);
            uint16_t h = uint16_t(b >> 16);
            std::memcpy(wh.data() + size_t(r) * rb + 2 * k, &h, 2);
        }
    for (uint32_t r = 0; r < N && !bf16; ++r) {
        uint8_t* row = wh.data() + size_t(r) * rb;
        std::memset(row, 0, sb);
        for (uint32_t b = 0; b < K / 32; ++b) { uint16_t h = qn::f32_to_f16(us(rng)); std::memcpy(row + 2 * b, &h, 2); }
        for (size_t i = sb; i < rb; ++i) row[i] = uint8_t(rng());
    }
    std::vector<float> xh(size_t(K) * T);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = nd(rng);
    // float64 reference (dequantised exactly: d * (q - 8))
    std::vector<double> ref(size_t(N) * T);
    {
        std::vector<double> wr(K);
        for (uint32_t r = 0; r < N; ++r) {
            const uint8_t* row = wh.data() + size_t(r) * rb;
            if (bf16) for (uint32_t k = 0; k < K; ++k) { uint16_t h; std::memcpy(&h, row + 2 * k, 2); uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); wr[k] = f; }
            for (uint32_t b = 0; b < K / 32 && !bf16; ++b) {
                uint16_t h; std::memcpy(&h, row + 2 * b, 2);
                const double d = qn::f16_to_f32(h);
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint8_t by = row[sb + 16 * b + i];
                    wr[32 * b + i] = d * (double(by & 15) - 8.0);
                    wr[32 * b + 16 + i] = d * (double(by >> 4) - 8.0);
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                double s = 0;
                const float* x = xh.data() + size_t(t) * K;
                for (uint32_t k = 0; k < K; ++k) s += wr[k] * x[k];
                double o = s;
                if (mode == 2) o = s / (1 + std::exp(-s));
                else if (mode == 3) o = 1 / (1 + std::exp(-s));
                ref[size_t(t) * N + r] = o;
            }
        }
    }
    auto wb = ctx.create(wbytes, true);
    std::memcpy(wb.map, wh.data(), wbytes);
    auto xb = ctx.create(xh.size() * 4, true);
    std::memcpy(xb.map, xh.data(), xh.size() * 4);
    auto yb = ctx.create(size_t(N) * T * 4, true);
    auto pb = ctx.create(size_t(N) * T * 4 * ks, true);
    auto pks = ctx.pipeline(sd + "/w_ksum.spv", 48);
    struct alignas(8) PcKsum { uint64_t part, y; uint32_t N, T, ks, mode; float alpha; uint32_t ps, ys, pad; };
    auto qp = ctx.timestamps(2);
    auto pp = ctx.pipeline(sd + "/" + spv + ".spv", sizeof(PcGemm));
    PcGemm p{wb.addr, xb.addr, ks > 1 ? pb.addr : yb.addr, K, N, T, mode, 1.f, K * 4, N * 4, ks, 0};
    PcKsum pk{pb.addr, yb.addr, N, T, ks, mode, 1.f, N * 4, N * 4, 0};
    const uint32_t gx = (N + 127) / 128, gy = (T + 63) / 64;
    auto run = [&](int n) {
        auto cb = ctx.cmd();
        vk::Ctx::begin(cb);
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        for (int r = 0; r < n; ++r) {
            vk::Ctx::dispatch(cb, pp, &p, gx, gy, ks);
            vk::Ctx::barrier(cb);
            if (ks > 1) { vk::Ctx::dispatch(cb, pks, &pk, (N + 255) / 256, T); vk::Ctx::barrier(cb); }
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vk::Ctx::end(cb);
        ctx.run(cb);
        auto ts = ctx.read_ts(qp, 2);
        vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
        return double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3 / n;
    };
    std::memset(yb.map, 0, size_t(N) * T * 4);
    run(1);
    const float* y = yb.as<float>();
    double maxrel = 0, sumsq = 0, refsq = 0;
    for (size_t i = 0; i < size_t(N) * T; ++i) {
        const double e = double(y[i]) - ref[i];
        sumsq += e * e; refsq += ref[i] * ref[i];
        maxrel = std::max(maxrel, std::fabs(e) / (std::fabs(ref[i]) + 1e-2));
    }
    std::vector<double> ts;
    for (int i = 0; i < 5; ++i) ts.push_back(run(reps));
    std::sort(ts.begin(), ts.end());
    const double t = ts[2], fma = double(N) * K * T;
    std::printf("%s ks %u N=%u K=%u T=%u mode %u: %8.1f us  %.2f us/token  %.2f TFMA/s (%.0f%% of 5.14)  max rel %.1e  rms rel %.1e\n", spv.c_str(), ks, N, K, T, mode,
                t, t / T, fma / t * 1e-6, fma / t * 1e-6 / 5.14 * 100, maxrel, std::sqrt(sumsq / refsq));
    return 0;
}
