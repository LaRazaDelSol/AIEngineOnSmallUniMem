// exl3_bench (S15b) -- cost of decoding EXL3 "mul1" K=3 trellis weights on GFX1013, gate_up-shaped.
//   exl3_bench --shaders DIR [--reps 200]
// 20 matrices of 2560x640 (= 10 experts x gate+up) per dispatch, like one layer's routed gate_up. Each variant is
// checked against a CPU reference that decodes every weight with exllamav3's single-weight dq() formula
// (independent of the shader's 8-at-a-time window math), then timed with GPU timestamps.
#include "ps5lm/vk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ps5lm;

static constexpr uint32_t KT = 160, NT = 40, NMAT = 20, HID = 2560, FFN = 640;

static uint32_t bytesum_mul1(uint32_t w) {
    uint32_t p = (w & 0xFFFFu) * 0x83DCD12Du;
    return (p & 0xFF) + ((p >> 8) & 0xFF) + ((p >> 16) & 0xFF) + (p >> 24);
}
// exllamav3 exl3_dq.cuh dq<bits=3>: 16-bit window of tile position pos (tail-biting, MSB-first funnel)
static uint32_t dq_idx(const uint32_t* tile, int pos) {
    const int K = 3, words = K * 256 / 32;
    int b0 = pos * K + K - 16 + 256 * K, b1 = b0 + 16, i0 = b0 / 32, i1 = (b1 - 1) / 32, s0 = (i1 + 1) * 32 - b1;
    uint32_t a = tile[i0 % words], b = tile[i1 % words];
    uint64_t merged = (uint64_t(a) << 32) | b;
    return uint32_t(merged >> (s0 & 31)) & 0xFFFF;  // __funnelshift_r clamps the shift to 0..31
}
// tensor-core permutation: tile position -> (row k, col n) inside the 16x16 tile
static void perm_rc(int pos, int& r, int& c) {
    int t = pos / 8, j = pos % 8, r0 = (t % 4) * 2;
    static const int dr[4] = {0, 1, 8, 9};
    r = r0 + dr[j % 4];
    c = t / 4 + (j >= 4 ? 8 : 0);
}

int main(int argc, char** argv) {
    std::string sd = "shaders";
    int reps = 200;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
    }
    vk::Ctx ctx;
    struct alignas(8) Pc { uint64_t trel, x, out; uint32_t c_lo = 0xDCD12D, c_hi = 0x83; };
    std::mt19937 rng(1234);
    // inputs: one pre-transformed x per matrix
    std::vector<float> xs(size_t(NMAT) * HID);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : xs) v = nd(rng);
    auto xb = ctx.create(xs.size() * 4 * 8, true);   // room for BATCH 8: x layout [m][b][2560]
    auto ob = ctx.create(size_t(NMAT) * FFN * 4 * 8, true);  // out layout [b][m][640]
    auto fill_x = [&](uint32_t B) {  // batch row b of matrix m = xs[m] * (1 + b)
        float* d = static_cast<float*>(xb.map);
        for (uint32_t m = 0; m < NMAT; ++m)
            for (uint32_t b = 0; b < B; ++b)
                for (uint32_t k = 0; k < HID; ++k) d[(size_t(m) * B + b) * HID + k] = xs[size_t(m) * HID + k] * float(1 + b);
    };
    auto qp = ctx.timestamps(2);

    struct Var { const char* spv; const char* name; uint32_t tw; bool nibble; bool off1024; uint32_t ntw = 1; uint32_t batch = 1; };
    const Var vars[] = {{"exl3_mv_v0.spv", "mul1 K=3, 32-bit mul", 24, false, false},
                        {"exl3_mv_v1.spv", "mul1 K=3, 24-bit split mul", 24, false, false},
                        {"exl3_mv_v3.spv", "mul1 K=3, packed pairs", 24, false, true},
                        {"exl3_mv_v2.spv", "4-bit nibble baseline", 32, true, false},
                        {"exl3_mv2_a.spv", "v4 sad+split, 2 tiles/wave", 24, false, true, 2},
                        {"exl3_mv2_b.spv", "v4 swar+split, 2 tiles/wave", 24, false, true, 2},
                        {"exl3_mv2_c.spv", "v4 sad+mul32, 2 tiles/wave", 24, false, true, 2},
                        {"exl3_mv2_d.spv", "v4 sad+split, 4 tiles/wave", 24, false, true, 4},
                        {"exl3_mv2_e.spv", "v4 sad+split, 1 tile/wave", 24, false, true, 1},
                        {"exl3_mv2_f.spv", "v4 sad+mul24(pc), 2 tiles/wave", 24, false, true, 2},
                        {"exl3_mv2_g.spv", "v4 sad+mul24(pc), 2 t/w, BATCH 4", 24, false, true, 2, 4},
                        {"exl3_mv2_h.spv", "v4 sad+mul24(pc), 1 t/w, BATCH 8", 24, false, true, 1, 8}};
    for (const Var& vr : vars) {
        const size_t words = size_t(NMAT) * KT * NT * vr.tw;
        std::vector<uint32_t> tw(words);
        for (auto& w : tw) w = rng();
        auto tb = ctx.create(words * 4, true);
        std::memcpy(tb.map, tw.data(), words * 4);
        auto p = ctx.pipeline(sd + "/" + vr.spv, sizeof(Pc), 32);
        fill_x(vr.batch);
        Pc pc{tb.addr, xb.addr, ob.addr};
        // correctness
        {
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            vk::Ctx::dispatch(cb, p, &pc, NT / vr.ntw, NMAT);
            vk::Ctx::barrier(cb);
            vk::Ctx::end(cb);
            ctx.run(cb);
            vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
            const float* o = static_cast<const float*>(ob.map);
            double maxrel = 0;
            for (uint32_t m : {0u, 7u, 19u}) {
                std::vector<double> ref(FFN, 0.0);
                for (uint32_t kt = 0; kt < KT; ++kt)
                    for (uint32_t nt = 0; nt < NT; ++nt) {
                        const uint32_t* tile = tw.data() + (size_t(m) * KT * NT + size_t(kt) * NT + nt) * vr.tw;
                        for (int pos = 0; pos < 256; ++pos) {
                            int r, c;
                            perm_rc(pos, r, c);
                            double d = vr.nibble ? double((tile[pos / 8] >> (4 * (pos % 8))) & 15u) : double(bytesum_mul1(dq_idx(tile, pos)));
                            ref[nt * 16 + c] += d * xs[size_t(m) * HID + kt * 16 + r];
                        }
                    }
                if (vr.off1024) {
                    double sx = 0;
                    for (uint32_t k = 0; k < HID; ++k) sx += xs[size_t(m) * HID + k];
                    for (auto& r : ref) r += 1024.0 * sx;
                }
                double num = 0, den = 0;
                const uint32_t bq = vr.batch - 1;  // check the LAST batch row (x scaled by B): exercises the batch indexing
                for (uint32_t n = 0; n < FFN; ++n) {
                    const double rr = ref[n] * double(vr.batch);
                    num += std::pow(o[(size_t(bq) * NMAT + m) * FFN + n] - rr, 2);
                    den += rr * rr;
                }
                maxrel = std::max(maxrel, std::sqrt(num / den));
            }
            std::printf("%-28s check vs CPU dq(): rel err %.2e %s\n", vr.name, maxrel, maxrel < 1e-5 ? "OK" : "MISMATCH");
        }
        // timing: reps dispatches back to back with a barrier between (the per-layer pattern)
        std::vector<double> us;
        for (int rnd = 0; rnd < 5; ++rnd) {
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (int r = 0; r < reps; ++r) { vk::Ctx::dispatch(cb, p, &pc, NT / vr.ntw, NMAT); vk::Ctx::barrier(cb); }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            vk::Ctx::end(cb);
            ctx.run(cb);
            auto ts = ctx.read_ts(qp, 2);
            us.push_back(double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3 / reps);
            vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
        }
        std::sort(us.begin(), us.end());
        const double t = us[2], wts = double(NMAT) * HID * FFN;
        std::printf("%-34s %.1f us/dispatch (20 x 2560x640, B=%u)  %.0f G weights/s  %.0f G weight-uses/s\n", vr.name, t,
                    vr.batch, wts / t * 1e-3, wts * vr.batch / t * 1e-3);
        ctx.destroy(p);
        ctx.destroy(tb);
    }
    return 0;
}
