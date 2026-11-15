// clk_bench (S15b) — effective GPU shader clock + streaming bandwidth on the console, measured, no SMU access.
//   clk_bench --shaders DIR [--seconds S] [--cus 36]
// Repeats a pure-FMA pass and a membw pass back to back for S seconds and prints per-second effective MHz and GB/s,
// so a clock ramp, a boost cap or thermal throttling shows up in the series.
#include "ps5lm/vk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace ps5lm;

int main(int argc, char** argv) {
    std::string sd = "shaders";
    double seconds = 10;
    uint32_t cus = 36;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--seconds") seconds = std::stod(argv[i + 1]);
        else if (a == "--cus") cus = uint32_t(std::stoul(argv[i + 1]));
    }
    vk::Ctx ctx;
    struct alignas(8) Pc { uint64_t out; uint32_t iters; float seed; };
    auto fp = ctx.pipeline(sd + "/fmaclk.spv", sizeof(Pc));
    auto out = ctx.create(64ull << 20, true);
    const uint32_t groups = cus * 64, iters = 4096;  // 64 WGs x 256 threads per CU: full occupancy
    const double flop = double(groups) * 256 * iters * 8 * 4 * 2;
    auto qp = ctx.timestamps(2);
    auto t0 = std::chrono::steady_clock::now();
    int n = 0;
    std::vector<double> mhz;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        auto cb = ctx.cmd();
        vk::Ctx::begin(cb);
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        Pc p{out.addr, iters, 0.5f + 1e-3f * float(n)};
        vk::Ctx::dispatch(cb, fp, &p, groups);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vk::Ctx::end(cb);
        ctx.run(cb);
        auto ts = ctx.read_ts(qp, 2);
        const double s = double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-9;
        const double tf = flop / s * 1e-12, m = flop / s / (double(cus) * 128) * 1e-6;
        mhz.push_back(m);
        if (n % 5 == 0) std::printf("pass %3d  %.3f ms  %.2f TFLOPS  effective clock %.0f MHz (at %u CUs)\n", n, s * 1e3, tf, m, cus);
        vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
        ++n;
    }
    std::sort(mhz.begin(), mhz.end());
    std::printf("passes %d: effective clock min %.0f / median %.0f / max %.0f MHz\n", n, mhz.front(), mhz[mhz.size() / 2], mhz.back());
    return 0;
}
