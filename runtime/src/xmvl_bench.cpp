// xmvl_bench (S22) -- the routed-expert trellis GEMMs of the wide prefill on REAL experts of one layer:
// exl3_xmvw (S21: one wave per entry of <= 16 pairs, tile decoded in registers) vs exl3_xmvl (S22: LDS-tiled, one
// workgroup per entry of <= PT pairs x 128 columns). Pair counts per expert come from a PS5LM_WDUMP line (a real
// 4096-token chunk) or are uniform (--pairs P). Inputs are random; both kernels' outputs are compared with each other
// and, for a few pairs, with an f64 reference that decodes every weight with the per-position window formula.
//   xmvl_bench --model DIR --shaders DIR [--pack experts_exl3_268] [--layer 10] [--counts FILE --line N | --pairs 64 --nexp 40]
//              [--split 64,32] (xmvl entry sizes: an expert's pairs go to PT-64 entries, then one PT-32 entry, the rest
//              <= 16 to exl3_xmvw)  [--reps 10]
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
static constexpr uint32_t HID = 2560, FFN = 640;

static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (!m) f = s << 31; else { e = 127 - 15 + 1; while (!(m & 1024)) { m <<= 1; --e; } m &= 1023; f = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) f = (s << 31) | 0x7F800000 | (m << 13);
    else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}
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
// inner product only (what the xmv kernels compute): y[n] = sum_k W_inner[k][n] x[k]
static void ref_inner(const uint8_t* slot, uint32_t tr_off, uint32_t k, uint32_t n, int K2, const float* x, std::vector<double>& y) {
    y.assign(n, 0.0);
    const uint32_t TW = 4 * K2, KT = k / 16, NT = n / 16;
    static const int dr[4] = {0, 1, 8, 9};
    for (uint32_t kt = 0; kt < KT; ++kt)
        for (uint32_t nt = 0; nt < NT; ++nt) {
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(slot + tr_off) + (size_t(kt) * NT + nt) * TW;
            for (int p = 0; p < 256; ++p) {
                const int t = p / 8, j = p % 8, r = (t % 4) * 2 + dr[j % 4], c = t / 4 + (j >= 4 ? 8 : 0);
                y[nt * 16 + c] += mul1(window(tile, p, K2)) * double(x[kt * 16 + r]);
            }
        }
}

struct alignas(8) PcXmvw { uint64_t ents, xin, out; uint32_t kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride; };
struct alignas(8) PcXmvl { uint64_t ents, xin, out, plist; uint32_t kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride; };
struct WEnt { uint64_t slot; uint32_t n, pad, p[16]; };
struct LEnt { uint64_t slot; uint32_t n, p0; };

int main(int argc, char** argv) {
    std::string model = ".", sd = "shaders", pack = "experts_exl3_268", counts_file, split = "64,32";
    int layer = 10, line_no = 0, reps = 10;
    uint32_t pairs = 64, nexp = 40;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--model") model = argv[i + 1];
        else if (a == "--shaders") sd = argv[i + 1];
        else if (a == "--pack") pack = argv[i + 1];
        else if (a == "--layer") layer = std::stoi(argv[i + 1]);
        else if (a == "--counts") counts_file = argv[i + 1];
        else if (a == "--line") line_no = std::stoi(argv[i + 1]);
        else if (a == "--pairs") pairs = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--nexp") nexp = uint32_t(std::stoul(argv[i + 1]));
        else if (a == "--split") split = argv[i + 1];
        else if (a == "--reps") reps = std::stoi(argv[i + 1]);
    }
    std::ifstream mf(model + "/" + pack + ".txt");
    std::string ln;
    int K2 = 0; uint64_t slot = 0, base = 0, span = 0;
    while (std::getline(mf, ln)) {
        if (ln.empty() || ln[0] == '#') continue;
        std::istringstream ss(ln);
        int l, k2; uint64_t sb, b, sp;
        ss >> l >> k2 >> sb >> b >> sp;
        if (l == layer) { K2 = k2; slot = sb; base = b; span = sp; }
    }
    if (!K2) { std::fprintf(stderr, "layer %d not in %s.txt\n", layer, pack.c_str()); return 1; }
    const uint32_t TB = 102400u * uint32_t(K2);
    const uint32_t OFF_D_TR = 6404, OFF_G_TR = 2 * 6404 + TB, OFF_U_TR = 3 * 6404 + 2 * TB;
    // pair counts per expert
    std::vector<uint32_t> cnt;
    if (!counts_file.empty()) {
        std::ifstream cf(counts_file);
        for (int i = 0; i <= line_no && std::getline(cf, ln); ++i) {}
        std::istringstream ss(ln);
        uint32_t l, N, c;
        ss >> l >> N;
        while (ss >> c) cnt.push_back(c);
        std::printf("counts: layer %u of a %u-token chunk (%zu experts)\n", l, N, cnt.size());
    } else cnt.assign(nexp, pairs);
    std::vector<uint32_t> ex;
    for (uint32_t e = 0; e < cnt.size(); ++e) if (cnt[e]) ex.push_back(e);
    const uint32_t NE = uint32_t(ex.size());
    uint32_t NP = 0, mx = 0;
    for (uint32_t e : ex) { NP += cnt[e]; mx = std::max(mx, cnt[e]); }
    std::printf("layer %d K=%g: %u experts used, %u pairs (%.1f per expert, max %u)\n", layer, K2 / 2.0, NE, NP, double(NP) / NE, mx);

    vk::Ctx ctx;
    auto sb = ctx.create(slot * NE, true);
    int fd = ::open((model + "/" + pack + ".pack").c_str(), O_RDONLY);
    for (uint32_t i = 0; i < NE; ++i)
        if (::pread(fd, static_cast<uint8_t*>(sb.map) + i * slot, slot, off_t(base + uint64_t(ex[i] % 512) * slot)) != ssize_t(slot)) { std::perror("pread"); return 1; }
    ::close(fd);
    // pairs grouped by expert: q = consecutive per expert
    std::vector<uint32_t> owner(NP);
    std::vector<uint32_t> first(NE + 1, 0);
    for (uint32_t i = 0, q = 0; i < NE; ++i) { first[i] = q; for (uint32_t j = 0; j < cnt[ex[i]]; ++j) owner[q++] = i; }
    first[NE] = NP;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto xg = ctx.create(uint64_t(NP) * 2 * HID * 4, true), xd = ctx.create(uint64_t(NP) * FFN * 4, true);
    for (uint64_t i = 0; i < uint64_t(NP) * 2 * HID; ++i) xg.as<float>()[i] = nd(rng);
    for (uint64_t i = 0; i < uint64_t(NP) * FFN; ++i) xd.as<float>()[i] = nd(rng);
    auto ogw = ctx.create(uint64_t(NP) * 2 * FFN * 4, true), ogl = ctx.create(uint64_t(NP) * 2 * FFN * 4, true);
    auto odw = ctx.create(uint64_t(NP) * HID * 4, true), odl = ctx.create(uint64_t(NP) * HID * 4, true);
    // xmvw entries (<= 16 pairs)
    std::vector<WEnt> we;
    for (uint32_t i = 0; i < NE; ++i)
        for (uint32_t q = first[i]; q < first[i + 1]; q += 16) {
            WEnt w{};
            w.slot = sb.addr + uint64_t(i) * slot;
            w.n = std::min(16u, first[i + 1] - q);
            for (uint32_t j = 0; j < w.n; ++j) w.p[j] = q + j;
            we.push_back(w);
        }
    // xmvl entries by size class + the <= 16 tails as xmvw entries
    std::vector<uint32_t> pts;
    std::vector<std::string> ptn;  // shader suffix: "64", "128", "64b16" (size = the leading number)
    { std::istringstream ss(split); std::string t; while (std::getline(ss, t, ',')) { ptn.push_back(t); pts.push_back(uint32_t(std::stoul(t))); } }
    std::vector<std::vector<LEnt>> le(pts.size());
    std::vector<WEnt> tail;
    std::vector<uint32_t> plist(NP);
    for (uint32_t q = 0; q < NP; ++q) plist[q] = q;
    for (uint32_t i = 0; i < NE; ++i) {
        uint32_t q = first[i], left = first[i + 1] - q;
        for (size_t c = 0; c < pts.size(); ++c) {
            const uint32_t lo = c + 1 < pts.size() ? pts[c + 1] : 16u;
            while (left > lo) {
                const uint32_t n = std::min(left, pts[c]);
                le[c].push_back({sb.addr + uint64_t(i) * slot, n, q});
                q += n; left -= n;
            }
        }
        for (; left; ) {
            WEnt w{};
            w.slot = sb.addr + uint64_t(i) * slot;
            w.n = std::min(16u, left);
            for (uint32_t j = 0; j < w.n; ++j) w.p[j] = q + j;
            tail.push_back(w);
            q += w.n; left -= w.n;
        }
    }
    auto bwe = ctx.create(std::max<size_t>(1, we.size()) * sizeof(WEnt), true), bte = ctx.create(std::max<size_t>(1, tail.size()) * sizeof(WEnt), true);
    std::memcpy(bwe.map, we.data(), we.size() * sizeof(WEnt));
    std::memcpy(bte.map, tail.data(), tail.size() * sizeof(WEnt));
    std::vector<vk::Buffer> ble;
    for (auto& v : le) { ble.push_back(ctx.create(std::max<size_t>(1, v.size()) * sizeof(LEnt), true)); std::memcpy(ble.back().map, v.data(), v.size() * sizeof(LEnt)); }
    auto bpl = ctx.create(uint64_t(NP) * 4, true);
    std::memcpy(bpl.map, plist.data(), uint64_t(NP) * 4);
    const std::string tag = K2 == 6 ? "k3" : K2 == 5 ? "k25" : "k35";
    auto pW = ctx.pipeline(sd + "/exl3_xmvw_" + tag + ".spv", sizeof(PcXmvw), 32);
    std::vector<vk::Pipeline> pL;
    for (auto& pt : ptn) pL.push_back(ctx.pipeline(sd + "/exl3_xmvl_" + tag + "_" + pt + ".spv", sizeof(PcXmvl), 32));
    auto qp = ctx.timestamps(2);
    auto timed = [&](auto&& rec, int n) {
        auto cb = ctx.cmd();
        vk::Ctx::begin(cb);
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        for (int r = 0; r < n; ++r) { rec(cb); vk::Ctx::barrier(cb); }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        vk::Ctx::end(cb);
        ctx.run(cb);
        auto ts = ctx.read_ts(qp, 2);
        vkFreeCommandBuffers(ctx.dev, ctx.pool, 1, &cb);
        return double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3 / n;
    };
    auto best = [&](auto&& rec) { rec(1); std::vector<double> t; for (int i = 0; i < 5; ++i) t.push_back(rec(reps)); std::sort(t.begin(), t.end()); return t[2]; };
    for (int mat = 0; mat < 2; ++mat) {  // 0 = gate/up, 1 = down
        const bool gu = mat == 0;
        const uint32_t kt = gu ? HID / 16 : FFN / 16, nt = gu ? FFN / 16 : HID / 16, tr0 = gu ? OFF_G_TR : OFF_D_TR, tr1 = gu ? OFF_U_TR : OFF_D_TR;
        const uint32_t xs_m = gu ? HID : 0, os_m = gu ? FFN : 0, xstride = gu ? 2 * HID * 4 : FFN * 4, ostride = gu ? 2 * FFN * 4 : HID * 4;
        const uint32_t nmat = gu ? 2 : 1, ncol = gu ? FFN : HID;
        const uint64_t xin = gu ? xg.addr : xd.addr;
        auto& ow = gu ? ogw : odw;
        auto& ol = gu ? ogl : odl;
        PcXmvw pw{bwe.addr, xin, ow.addr, kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride};
        PcXmvw ptl{bte.addr, xin, ol.addr, kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride};
        auto rw = [&](VkCommandBuffer cb) { vk::Ctx::dispatch(cb, pW, &pw, ncol / 32, nmat, uint32_t(we.size())); };
        auto rl = [&](VkCommandBuffer cb) {
            for (size_t c = 0; c < pts.size(); ++c)
                if (!le[c].empty()) {
                    PcXmvl p{ble[c].addr, xin, ol.addr, bpl.addr, kt, nt, tr0, tr1, xs_m, os_m, xstride, ostride};
                    vk::Ctx::dispatch(cb, pL[c], &p, ncol / 128, nmat, uint32_t(le[c].size()));
                }
            if (!tail.empty()) vk::Ctx::dispatch(cb, pW, &ptl, ncol / 32, nmat, uint32_t(tail.size()));
        };
        const double tw = best([&](int n) { return timed(rw, n); });
        const double tl = best([&](int n) { return timed(rl, n); });
        const uint64_t no = uint64_t(NP) * nmat * ncol;
        const float* a = ow.as<float>();
        const float* b = ol.as<float>();
        double num = 0, den = 0;
        for (uint64_t i = 0; i < no; ++i) { num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]); den += double(a[i]) * a[i]; }
        // f64 reference for the first pair of the first and last expert
        double rn = 0, rd = 0, rnw = 0;
        for (uint32_t pi : {0u, first[NE - 1]}) {
            const uint8_t* s = static_cast<const uint8_t*>(sb.map) + uint64_t(owner[pi]) * slot;
            for (uint32_t mm = 0; mm < nmat; ++mm) {
                std::vector<double> y;
                const float* x = gu ? xg.as<float>() + uint64_t(pi) * 2 * HID + mm * HID : xd.as<float>() + uint64_t(pi) * FFN;
                ref_inner(s, mm ? tr1 : tr0, gu ? HID : FFN, ncol, K2, x, y);
                for (uint32_t c = 0; c < ncol; ++c) {
                    const uint64_t o = uint64_t(pi) * (ostride / 4) + mm * os_m + c;
                    rn += (b[o] - y[c]) * (b[o] - y[c]); rnw += (a[o] - y[c]) * (a[o] - y[c]); rd += y[c] * y[c];
                }
            }
        }
        const double fma = double(no) * (gu ? HID : FFN);
        std::printf("%s: xmvw %8.1f us (%5.2f TFMA/s) | xmvl", gu ? "gate/up" : "down   ", tw, fma / tw * 1e-6);
        for (size_t c = 0; c < pts.size(); ++c) std::printf(" pt%s:%zu", ptn[c].c_str(), le[c].size());
        std::printf(" tail:%zu %8.1f us (%5.2f TFMA/s) = %.2fx | xmvl vs xmvw rel %.1e | vs f64: xmvl %.1e xmvw %.1e\n", tail.size(), tl,
                    fma / tl * 1e-6, tw / tl, std::sqrt(num / den), std::sqrt(rn / rd), std::sqrt(rnw / rd));
    }
    return 0;
}
