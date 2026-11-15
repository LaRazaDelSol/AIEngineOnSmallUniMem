// exl3_moe_test (S15b) -- the EXL3 routed-MoE kernels on REAL experts from experts_exl3.pack vs an f64 CPU reference.
//   exl3_moe_test --model DIR --shaders DIR [--layer L] [--reps N]
// Loads 10 experts of one layer into GPU memory exactly as the streamer would (slot = 4 KiB-aligned span), runs
// had_in -> xmv(gate,up) -> act -> xmv(down) -> combine, and compares the routed output with a reference that decodes
// every weight with the per-position window formula (exl3_verify.py, validated against BF16 originals) in f64.
// Also times the 5-dispatch sequence (the per-layer GPU cost of the EXL3 routed MoE).
#include "ps5lm/vk.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ps5lm;
static constexpr uint32_t HID = 2560, FFN = 640, NE = 10;

static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (!m) f = s << 31; else { e = 127 - 15 + 1; while (!(m & 1024)) { m <<= 1; --e; } m &= 1023; f = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) f = (s << 31) | 0x7F800000 | (m << 13);
    else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}
// window of tile position p (16 bits ending at end(p) + tile bits), K2 = 2K
static uint32_t window(const uint32_t* tile, int p, int K2) {
    const int bits = K2 * 128, words = bits / 32;
    int end;
    if (K2 % 2 == 0) end = (K2 / 2) * (p + 1);
    else { const int KA = K2 / 2; end = (p % 2) ? (2 * KA + 1) * (p + 1) / 2 : (2 * KA + 1) * p / 2 + KA; }
    end += bits;
    const int hi = (end - 1) / 32, lo = (end - 16) / 32, s = (hi + 1) * 32 - end;
    const uint64_t merged = (uint64_t(tile[lo % words]) << 32) | tile[hi % words];
    return uint32_t(merged >> s) & 0xFFFF;
}
static double mul1(uint32_t w) {
    const uint32_t p = w * 0x83DCD12Du;
    const double s = (p & 0xFF) + ((p >> 8) & 0xFF) + ((p >> 16) & 0xFF) + (p >> 24);
    return (1024.0 + s) * double(h2f(0x1EEE)) + double(h2f(0xC931));
}
// y = H128(H128(x * suh) . W_inner) * svh for one EXL3 linear stored at `tr` (trellis [k/16][n/16][TW]), f64
static void ref_linear(const uint8_t* slot, uint32_t tr_off, uint32_t suh_off, uint32_t svh_off, uint32_t k, uint32_t n, int K2,
                       const std::vector<double>& x, std::vector<double>& y) {
    auto had = [](std::vector<double>& v) {
        for (size_t b = 0; b < v.size(); b += 128)
            for (int h = 1; h < 128; h <<= 1)
                for (int i = 0; i < 128; ++i) if (!(i & h)) { double a = v[b + i], c = v[b + i + h]; v[b + i] = a + c; v[b + i + h] = a - c; }
        for (auto& q : v) q /= std::sqrt(128.0);
    };
    const uint16_t* suh = reinterpret_cast<const uint16_t*>(slot + suh_off);
    const uint16_t* svh = reinterpret_cast<const uint16_t*>(slot + svh_off);
    std::vector<double> xt(k);
    for (uint32_t i = 0; i < k; ++i) xt[i] = x[i] * h2f(suh[i]);
    had(xt);
    y.assign(n, 0.0);
    const uint32_t TW = 4 * K2, KT = k / 16, NT = n / 16;  // words per tile = 256*K/32
    static const int dr[4] = {0, 1, 8, 9};
    for (uint32_t kt = 0; kt < KT; ++kt)
        for (uint32_t nt = 0; nt < NT; ++nt) {
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(slot + tr_off) + (size_t(kt) * NT + nt) * TW;
            for (int p = 0; p < 256; ++p) {
                const int t = p / 8, j = p % 8, r = (t % 4) * 2 + dr[j % 4], c = t / 4 + (j >= 4 ? 8 : 0);
                y[nt * 16 + c] += mul1(window(tile, p, K2)) * xt[kt * 16 + r];
            }
        }
    had(y);
    for (uint32_t i = 0; i < n; ++i) y[i] *= h2f(svh[i]);
}

int main(int argc, char** argv) {
    std::string model = ".", sd = "shaders", pack = "experts_exl3";
    int layer = 10, reps = 100;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--model") model = argv[i + 1];
        else if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--pack") pack = argv[i + 1];
        else if (a == "--layer") layer = std::stoi(argv[i + 1]);
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
    }
    // pack geometry
    std::ifstream mf(model + "/" + pack + ".txt");
    std::string line;
    int K2 = 0; uint64_t slot = 0, base = 0, span = 0;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int l, k2; uint64_t sb, b, sp;
        ss >> l >> k2 >> sb >> b >> sp;
        if (l == layer) { K2 = k2; slot = sb; base = b; span = sp; }
    }
    if (!K2) { std::fprintf(stderr, "layer %d not in experts_exl3.txt\n", layer); return 1; }
    const uint32_t TB = 102400u * uint32_t(K2);  // 640*2560*K/8
    const uint32_t OFF_D_SUH = 0, OFF_D_SVH = 1280, OFF_D_TR = 6404, OFF_G_SUH = 6404 + TB, OFF_G_SVH = OFF_G_SUH + 5120,
                   OFF_G_TR = OFF_G_SUH + 6404, OFF_U_SUH = OFF_G_TR + TB, OFF_U_SVH = OFF_U_SUH + 5120, OFF_U_TR = OFF_U_SUH + 6404;
    if (OFF_U_TR + TB != span) { std::fprintf(stderr, "layout mismatch: %u vs %llu\n", OFF_U_TR + TB, (unsigned long long)span); return 1; }

    vk::Ctx ctx;
    const uint32_t ex[NE] = {3, 77, 128, 200, 255, 301, 388, 402, 470, 511};
    auto sb = ctx.create(slot * NE, true);
    int fd = ::open((model + "/" + pack + ".pack").c_str(), O_RDONLY);
    for (uint32_t i = 0; i < NE; ++i)
        if (::pread(fd, static_cast<uint8_t*>(sb.map) + i * slot, slot, off_t(base + uint64_t(ex[i]) * slot)) != ssize_t(slot)) { std::perror("pread"); return 1; }
    ::close(fd);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(HID), w(16, 0.f);
    for (auto& v : x) v = nd(rng);
    for (uint32_t i = 0; i < NE; ++i) w[i] = 0.05f + 0.02f * float(i);
    auto xb = ctx.create(HID * 4, true), wb = ctx.create(16 * 4, true), tb = ctx.create(16 * 8, true);
    std::memcpy(xb.map, x.data(), HID * 4); std::memcpy(wb.map, w.data(), 64);
    auto* tab = static_cast<uint64_t*>(tb.map);
    for (uint32_t i = 0; i < 16; ++i) tab[i] = i < NE ? sb.addr + uint64_t(i) * slot : 0;
    auto xt = ctx.create(16 * 2 * HID * 4, true), pre = ctx.create(16 * 2 * FFN * 4, true), hd = ctx.create(16 * FFN * 4, true),
         yd = ctx.create(16 * HID * 4, true), out = ctx.create(HID * 4, true);
    const std::string tag = K2 == 6 ? "k3" : K2 == 5 ? "k25" : "k35";
    struct alignas(8) PcHad { uint64_t x, slots, xt; };
    struct alignas(8) PcMv { uint64_t slots, xin, out; uint32_t kt, nt, tr0, tr1, xs_e, xs_m, os_e, os_m; };
    struct alignas(8) PcAct { uint64_t pre, slots, hd; };
    struct alignas(8) PcCmb { uint64_t yd, slots, wt, out; uint32_t nexp; };
    auto pHad = ctx.pipeline(sd + "/exl3_had_in_" + tag + ".spv", sizeof(PcHad), 0);
    auto pMv = ctx.pipeline(sd + "/exl3_xmv_" + tag + (std::getenv("PS5LM_XMV_SUFFIX") ? std::getenv("PS5LM_XMV_SUFFIX") : "") + ".spv", sizeof(PcMv), 32);
    auto pAct = ctx.pipeline(sd + "/exl3_act_" + tag + ".spv", sizeof(PcAct), 0);
    auto pCmb = ctx.pipeline(sd + "/exl3_combine_" + tag + ".spv", sizeof(PcCmb), 0);
    PcHad ph{xb.addr, tb.addr, xt.addr};
    PcMv pgu{tb.addr, xt.addr, pre.addr, 160, 40, OFF_G_TR, OFF_U_TR, 2 * HID, HID, 2 * FFN, FFN};
    PcAct pa{pre.addr, tb.addr, hd.addr};
    PcMv pdn{tb.addr, hd.addr, yd.addr, 40, 160, OFF_D_TR, OFF_D_TR, FFN, 0, HID, 0};
    PcCmb pc{yd.addr, tb.addr, wb.addr, out.addr, NE};
    auto record = [&](VkCommandBuffer cb) {
        vk::Ctx::dispatch(cb, pHad, &ph, 20, 2, NE); vk::Ctx::barrier(cb);
        vk::Ctx::dispatch(cb, pMv, &pgu, 20, 2, NE); vk::Ctx::barrier(cb);
        vk::Ctx::dispatch(cb, pAct, &pa, 5, NE); vk::Ctx::barrier(cb);
        vk::Ctx::dispatch(cb, pMv, &pdn, 80, 1, NE); vk::Ctx::barrier(cb);
        vk::Ctx::dispatch(cb, pCmb, &pc, 20); vk::Ctx::barrier(cb);
    };
    {
        auto cb = ctx.cmd(); vk::Ctx::begin(cb); record(cb); vk::Ctx::end(cb); ctx.run(cb);
        vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
    }
    // reference
    std::vector<double> xd(x.begin(), x.end()), yref(HID, 0.0), g, u, dn;
    const uint8_t* sbase = static_cast<const uint8_t*>(sb.map);
    for (uint32_t i = 0; i < NE; ++i) {
        const uint8_t* s = sbase + i * slot;
        ref_linear(s, OFF_G_TR, OFF_G_SUH, OFF_G_SVH, HID, FFN, K2, xd, g);
        ref_linear(s, OFF_U_TR, OFF_U_SUH, OFF_U_SVH, HID, FFN, K2, xd, u);
        std::vector<double> h(FFN);
        for (uint32_t r = 0; r < FFN; ++r) h[r] = g[r] / (1.0 + std::exp(-g[r])) * u[r];
        ref_linear(s, OFF_D_TR, OFF_D_SUH, OFF_D_SVH, FFN, HID, K2, h, dn);
        for (uint32_t n = 0; n < HID; ++n) yref[n] += double(w[i]) * dn[n];
    }
    const float* o = static_cast<const float*>(out.map);
    double num = 0, den = 0;
    for (uint32_t n = 0; n < HID; ++n) { num += (o[n] - yref[n]) * (o[n] - yref[n]); den += yref[n] * yref[n]; }
    std::printf("layer %d (K=%g): routed MoE output vs f64 reference: rel err %.2e %s  (|y| %.3f)\n", layer, K2 / 2.0,
                std::sqrt(num / den), std::sqrt(num / den) < 1e-4 ? "OK" : "MISMATCH", std::sqrt(den));
    // timing
    auto qp = ctx.timestamps(2);
    auto cb = ctx.cmd(); vk::Ctx::begin(cb);
    vkCmdResetQueryPool(cb, qp, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    for (int r = 0; r < reps; ++r) record(cb);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    vk::Ctx::end(cb); ctx.run(cb);
    auto ts = ctx.read_ts(qp, 2);
    std::printf("routed MoE, 10 experts, 5 dispatches: %.1f us per layer\n", double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3 / reps);
    return 0;
}
