// mv_bench (S19) -- dense PS5Q4 matvec variants for the K=6144 matrices (out_proj / o_proj, N=2560), checked
// BIT-FOR-BIT against the production single-token kernel (matvec_lds6144), timed with GPU timestamps, in both the
// VRAM carve (device-local) and GTT (4 KiB host pages) placements, beside a plain streaming read of the same bytes.
//   mv_bench --shaders DIR [--reps 200] [--n 2560] [--k 6144]
#include "ps5lm/quant.hpp"
#include "ps5lm/vk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ps5lm;
namespace qn = ps5lm::q;

struct alignas(8) PcMatvecB { uint64_t w, x, y; uint32_t K, N, rpw, mode; float alpha; uint32_t xs, ys, nb; };
struct alignas(8) PcMembw { uint64_t src, outp; uint32_t n, magic; };

int main(int argc, char** argv) {
    std::string sd = "shaders", only;
    std::vector<std::string> extra;
    int reps = 200;
    uint32_t N = 2560, K = 6144;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
        else if (a == "--n") N = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--k") K = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--only") only = argv[i + 1];
        else if (a == "--var") extra.push_back(argv[i + 1]);  // spv:nb:rpw (rows per WG = rpw)
    }
    vk::Ctx ctx;
    for (int i = 1; i + 1 < argc; i += 2)
        if (std::string(argv[i]) == "--probe") { auto pp = ctx.pipeline(sd + "/" + argv[i + 1] + ".spv", 24); ctx.destroy(pp); return 0; }
    const size_t rb = qn::row_bytes<4>(int(K)), wbytes = rb * N;
    std::printf("mv_bench: N=%u K=%u row %zu B, matrix %.2f MB\n", N, K, rb, wbytes * 1e-6);
    std::mt19937 rng(77);
    std::vector<uint8_t> wh(wbytes);
    {
        std::uniform_real_distribution<float> us(0.002f, 0.02f);
        const size_t sb = qn::scales_bytes<4>(int(K));
        for (uint32_t r = 0; r < N; ++r) {
            uint8_t* row = wh.data() + size_t(r) * rb;
            std::memset(row, 0, sb);
            for (uint32_t b = 0; b < K / 32; ++b) {
                uint16_t h = qn::f32_to_f16(us(rng));
                std::memcpy(row + 2 * b, &h, 2);
            }
            for (size_t i = sb; i < rb; ++i) row[i] = uint8_t(rng());
        }
    }
    const uint32_t NBMAX = 4;
    std::vector<float> xh(size_t(K) * NBMAX);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xh) v = nd(rng);
    auto xb = ctx.create(xh.size() * 4, true);
    std::memcpy(xb.map, xh.data(), xh.size() * 4);
    auto yb = ctx.create(size_t(N) * 4 * NBMAX, true);
    auto yr = ctx.create(size_t(N) * 4 * NBMAX, true);  // reference outputs
    auto bout = ctx.create(64, true);
    auto qp = ctx.timestamps(2);

    auto time_it = [&](auto&& record) {
        std::vector<double> us;
        for (int rnd = 0; rnd < 5; ++rnd) {
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (int r = 0; r < reps; ++r) { record(cb); vk::Ctx::barrier(cb); }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            vk::Ctx::end(cb);
            ctx.run(cb);
            auto ts = ctx.read_ts(qp, 2);
            us.push_back(double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3 / reps);
            vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
        }
        std::sort(us.begin(), us.end());
        return us[2];
    };
    auto once = [&](auto&& record) {
        auto cb = ctx.cmd();
        vk::Ctx::begin(cb);
        record(cb);
        vk::Ctx::barrier(cb);
        vk::Ctx::end(cb);
        ctx.run(cb);
        vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
    };

    // reference = the production single-token kernel for this K (MV() in flash_engine.hpp)
    const bool k6 = K == 6144, kg = K != 6144 && K != 2560;   // other K: the generic matvec.comp (MV()'s fallback)
    const uint32_t ref_rpw = k6 ? 16 : kg ? (N >= 4096 ? 4 : 1) : (N >= 65536 ? 16 : N >= 8192 ? 4 : 1);
    const std::string ref_spv = k6 ? "matvec_lds6144" : kg ? "matvec" : "matvec_xr2560";
    auto pref = ctx.pipeline(sd + "/" + ref_spv + ".spv", sizeof(PcMatvecB));
    auto pmb = ctx.pipeline(sd + "/membw.spv", sizeof(PcMembw), 0);
    struct Var { std::string spv; uint32_t nb, rpw, rows_per_wg; bool exact_expected; };
    std::vector<Var> vars = {
        {"matvec_lds6144", 1, 16, 16, true},
        {"matvec_xr6144", 1, 4, 4, false},
        {"matvec_xr6144", 1, 8, 8, false},
        {"matvec_xr6_1", 1, 2, 2, true},
        {"matvec_xr6_1", 1, 4, 4, true},
        {"matvec_xr6_1", 1, 8, 8, true},
        {"matvec_xr6_1", 1, 16, 16, true},
        {"matvec_ldsb6144_2", 2, 16, 16, true},
        {"matvec_xr6_2", 2, 2, 2, true},
        {"matvec_xr6_2", 2, 4, 4, true},
        {"matvec_xr6_2", 2, 8, 8, true},
    };
    if (!extra.empty()) {
        vars.clear();
        vars.push_back({ref_spv, 1, ref_rpw, ref_rpw, true});
        for (const auto& e : extra) {
            const size_t a = e.find(':'), b = e.find(':', a + 1);
            const uint32_t nb = uint32_t(std::stoul(e.substr(a + 1, b - a - 1))), rpw = uint32_t(std::stoul(e.substr(b + 1)));
            vars.push_back({e.substr(0, a), nb, rpw, rpw, true});
        }
    }
    const int wmax = std::getenv("MVB_GTT") ? 2 : 1;
    for (int where = 0; where < wmax; ++where) {
        auto wb = where == 0 ? ctx.create(wbytes, false) : ctx.create(wbytes, true, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (wb.map) std::memcpy(wb.map, wh.data(), wbytes);
        else {  // device-local without a host mapping: stage through a host buffer
            auto st = ctx.create(wbytes, true, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            std::memcpy(st.map, wh.data(), wbytes);
            once([&](VkCommandBuffer cb) { VkBufferCopy c{0, 0, wbytes}; vkCmdCopyBuffer(cb, st.buf, wb.buf, 1, &c); });
            ctx.destroy(st);
        }
        std::printf("\n== placement: %s (memory type %u)\n", where == 0 ? "device-local (VRAM carve)" : "GTT (4 KiB host pages)", wb.mem_type);
        {
            PcMembw p{wb.addr, bout.addr, uint32_t(wbytes / 16), 0xDEADBEEFu};
            for (uint32_t g : {160u, 320u, 640u, 1440u}) {
                const double t = time_it([&](VkCommandBuffer cb) { vk::Ctx::dispatch(cb, pmb, &p, g); });
                std::printf("  stream read, %4u WGs: %6.2f us  %5.1f GB/s\n", g, t, wbytes / t * 1e-3);
            }
        }
        // reference outputs, per column, with the production kernel
        for (uint32_t c = 0; c < NBMAX; ++c) {
            PcMatvecB p{wb.addr, xb.addr + uint64_t(c) * K * 4, yr.addr + uint64_t(c) * N * 4, K, N, ref_rpw, 0, 1.f, 0, 0, 1};
            once([&](VkCommandBuffer cb) { vk::Ctx::dispatch(cb, pref, &p, (N + ref_rpw - 1) / ref_rpw); });
        }
        for (const Var& v : vars) {
            if (!only.empty() && v.spv.find(only) == std::string::npos) continue;
            vk::Pipeline pp;
            const bool ldsx = v.spv.find("ldsx") != std::string::npos;  // 72-byte push block, second matrix off
            try { pp = ctx.pipeline(sd + "/" + v.spv + ".spv", ldsx ? 72 : sizeof(PcMatvecB)); }
            catch (const std::exception& e) { std::printf("  %-20s skipped: %s\n", v.spv.c_str(), e.what()); continue; }
            struct alignas(8) { PcMatvecB b; uint8_t pad[32]; } pbuf{};
            pbuf.b = PcMatvecB{wb.addr, xb.addr, yb.addr, K, N, v.rpw, 0, 1.f, K * 4, N * 4, v.nb};
            if (ldsx) std::memset(reinterpret_cast<uint8_t*>(&pbuf) + 44, 0, sizeof pbuf - 44);  // N2, w2, y2, ... = 0
            const void* pptr = &pbuf;
            const uint32_t groups = (N + v.rows_per_wg - 1) / v.rows_per_wg;
            std::memset(yb.map, 0, size_t(N) * 4 * NBMAX);
            once([&](VkCommandBuffer cb) { vk::Ctx::dispatch(cb, pp, pptr, groups); });
            const float* y = yb.as<float>();
            const float* r = yr.as<float>();
            size_t diff = 0;
            double maxrel = 0;
            for (size_t i = 0; i < size_t(N) * v.nb; ++i) {
                if (std::memcmp(&y[i], &r[i], 4) != 0) ++diff;
                maxrel = std::max(maxrel, double(std::fabs(y[i] - r[i])) / (std::fabs(double(r[i])) + 1e-3));
            }
            const double t = time_it([&](VkCommandBuffer cb) { vk::Ctx::dispatch(cb, pp, pptr, groups); });
            std::printf("  %-20s NB=%u rpw %2u: %6.2f us  %5.1f GB/s  %s (%zu/%zu differ, max rel %.1e)\n", v.spv.c_str(), v.nb, v.rpw, t,
                        wbytes / t * 1e-3, diff == 0 ? "BIT-EXACT" : (v.exact_expected ? "MISMATCH" : "differs (expected)"), diff,
                        size_t(N) * v.nb, maxrel);
            ctx.destroy(pp);
        }
        ctx.destroy(wb);
    }
    return 0;
}
