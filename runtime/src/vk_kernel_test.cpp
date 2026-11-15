// vk_kernel_test — gates K1 (expert kernels vs CPU double reference + timing) and K1b
// (memory-path bandwidth: driver host-visible BO vs device-local carve vs imported userptr).
//
//   vk_kernel_test --shaders DIR [--bits 4|2] [--iters N] [--rpw-gu R] [--rpw-dn R]
#include "ps5lm/quant.hpp"
#include "ps5lm/vk.hpp"

#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ps5lm;
namespace qn = ps5lm::q;

struct alignas(8) PcGateUp { uint64_t x, h, slots; uint32_t rpw, pad; };
struct alignas(8) PcDown { uint64_t h, slots, wt, resid, outp; uint32_t n_exp, rpw; };
struct alignas(8) PcRouter { uint64_t w, x, logits; uint32_t n_exp, pad; };
struct alignas(8) PcTopk { uint64_t logits, mail; uint32_t n_exp, k, seq, pad; };
struct alignas(8) PcMembw { uint64_t src, outp; uint32_t n, magic; };

static constexpr int H = 2560, F = 640, NE = 10;

template <int B> static void make_expert(uint64_t seed, uint8_t* slot, std::vector<float>& deq) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);
    const qn::ExpertLayout<B> L;
    const size_t rg = qn::row_bytes<B>(H), rd = qn::row_bytes<B>(F);
    deq.assign(size_t(3) * F * H, 0.f);
    std::vector<float> row(H);
    for (int m = 0; m < 2; ++m)
        for (int r = 0; r < F; ++r) {
            for (auto& v : row) v = nd(rng);
            uint8_t* dst = slot + (m ? L.up_off : L.gate_off) + size_t(r) * rg;
            qn::quantize_row<B>(row.data(), H, dst);
            qn::dequantize_row<B>(dst, H, deq.data() + (size_t(m) * F + r) * H);
        }
    for (int r = 0; r < H; ++r) {
        for (int k = 0; k < F; ++k) row[k] = nd(rng);
        uint8_t* dst = slot + L.down_off + size_t(r) * rd;
        qn::quantize_row<B>(row.data(), F, dst);
        qn::dequantize_row<B>(dst, F, deq.data() + size_t(2) * F * H + size_t(r) * F);
    }
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <int B> static int run(vk::Ctx& ctx, const std::string& sdir, int iters, uint32_t rpw_gu, uint32_t rpw_dn) {
    const size_t slot = qn::ExpertLayout<B>().bytes;
    const std::string tag = B == 4 ? "q4" : "q2";
    std::printf("== K1 PS5Q%d: slot %zu B, %d experts\n", B, slot, NE);

    // --- build experts (threaded quantisation) ---
    std::vector<uint8_t> host(slot * NE);
    std::vector<std::vector<float>> deq(NE);
    double t0 = now_ms();
    {
        std::vector<std::thread> th;
        for (int e = 0; e < NE; ++e) th.emplace_back([&, e] { make_expert<B>(1000 + e, host.data() + slot * e, deq[e]); });
        for (auto& t : th) t.join();
    }
    std::printf("quantised %d experts in %.1f s\n", NE, (now_ms() - t0) / 1000);

    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(H), resid(H), wt(NE);
    for (auto& v : x) v = nd(rng);
    for (auto& v : resid) v = nd(rng);
    float ws = 0;
    for (auto& v : wt) { v = std::exp(nd(rng)); ws += v; }
    for (auto& v : wt) v /= ws;

    // --- CPU double reference ---
    std::vector<double> yref(resid.begin(), resid.end());
    for (int e = 0; e < NE; ++e) {
        const float* D = deq[e].data();
        std::vector<double> h(F);
        for (int r = 0; r < F; ++r) {
            double g = 0, u = 0;
            for (int k = 0; k < H; ++k) { g += double(D[size_t(r) * H + k]) * x[k]; u += double(D[(size_t(F) + r) * H + k]) * x[k]; }
            h[r] = g / (1 + std::exp(-g)) * u;
        }
        for (int r = 0; r < H; ++r) {
            double a = 0;
            for (int k = 0; k < F; ++k) a += double(D[size_t(2) * F * H + size_t(r) * F + k]) * h[k];
            yref[r] += wt[e] * a;
        }
    }

    auto gu = ctx.pipeline(sdir + "/gate_up_" + tag + ".spv", sizeof(PcGateUp));
    auto dn = ctx.pipeline(sdir + "/down_combine_" + tag + ".spv", sizeof(PcDown));

    auto bx = ctx.create(H * 4), bh = ctx.create(NE * F * 4), bres = ctx.create(H * 4), bout = ctx.create(H * 4);
    auto bwt = ctx.create(NE * 4), bslots = ctx.create(NE * 8);
    std::memcpy(bx.map, x.data(), H * 4);
    std::memcpy(bres.map, resid.data(), H * 4);
    std::memcpy(bwt.map, wt.data(), NE * 4);

    // --- three memory paths for the slots ---
    struct Path { const char* name; vk::Buffer buf; void* raw = nullptr; size_t raw_len = 0; };
    std::vector<Path> paths;
    try { paths.push_back({"host-visible BO", ctx.create(slot * NE, true)}); } catch (const std::exception& ex) { std::printf("host-visible BO: %s\n", ex.what()); }
    try { paths.push_back({"device-local", ctx.create(slot * NE, false)}); } catch (const std::exception& ex) { std::printf("device-local: %s\n", ex.what()); }
    if (ctx.has_host_import) {
        size_t len = (slot * NE + (2u << 20) - 1) / (2u << 20) * (2u << 20);
        void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED) {
            try { paths.push_back({"imported userptr", ctx.import_host(p, len), p, len}); }
            catch (const std::exception& ex) { std::printf("import: %s\n", ex.what()); munmap(p, len); }
        }
    }

    int fails = 0;
    for (auto& P : paths) {
        // upload: mapped memcpy, or staging copy for device-local
        vk::Buffer stage{};
        if (P.buf.map) std::memcpy(P.buf.map, host.data(), host.size());
        else {
            stage = ctx.create(host.size(), true);
            std::memcpy(stage.map, host.data(), host.size());
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            VkBufferCopy rg{0, 0, host.size()};
            vkCmdCopyBuffer(cb, stage.buf, P.buf.buf, 1, &rg);
            vk::Ctx::end(cb);
            ctx.run(cb);
        }
        auto* sl = bslots.as<uint64_t>();
        for (int e = 0; e < NE; ++e) sl[e] = P.buf.addr + uint64_t(slot) * e;

        PcGateUp pg{bx.addr, bh.addr, bslots.addr, rpw_gu, 0};
        PcDown pd{bh.addr, bslots.addr, bwt.addr, bres.addr, bout.addr, uint32_t(NE), rpw_dn};
        const uint32_t gx = (F + rpw_gu - 1) / rpw_gu, dx = (H + rpw_dn - 1) / rpw_dn;

        // correctness (single run)
        {
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            vk::Ctx::dispatch(cb, gu, &pg, gx, NE);
            vk::Ctx::barrier(cb);
            vk::Ctx::dispatch(cb, dn, &pd, dx);
            vk::Ctx::barrier(cb);
            vk::Ctx::end(cb);
            ctx.run(cb);
        }
        const float* y = bout.as<float>();
        double num = 0, den = 0;
        for (int r = 0; r < H; ++r) { num += (y[r] - yref[r]) * (y[r] - yref[r]); den += (yref[r] - resid[r]) * (yref[r] - resid[r]); }
        const double rel = std::sqrt(num / den);
        const bool ok = rel < 1e-3 && std::isfinite(rel);
        fails += !ok;

        // timing: iters x (gate_up, down) with timestamps
        auto qp = ctx.timestamps(4 * iters);
        auto cb = ctx.cmd();
        vk::Ctx::begin(cb);
        vkCmdResetQueryPool(cb, qp, 0, 4 * iters);
        for (int i = 0; i < iters; ++i) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 4 * i);
            vk::Ctx::dispatch(cb, gu, &pg, gx, NE);
            vk::Ctx::barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 4 * i + 1);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 4 * i + 2);
            vk::Ctx::dispatch(cb, dn, &pd, dx);
            vk::Ctx::barrier(cb);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 4 * i + 3);
        }
        vk::Ctx::end(cb);
        ctx.run(cb);
        auto ts = ctx.read_ts(qp, 4 * iters);
        std::vector<double> tg, td;
        for (int i = 1; i < iters; ++i) {  // drop first (cold)
            tg.push_back(double(ts[4 * i + 1] - ts[4 * i]) * ctx.ts_period_ns * 1e-3);
            td.push_back(double(ts[4 * i + 3] - ts[4 * i + 2]) * ctx.ts_period_ns * 1e-3);
        }
        std::sort(tg.begin(), tg.end());
        std::sort(td.begin(), td.end());
        const double g_us = tg[tg.size() / 2], d_us = td[td.size() / 2];
        const qn::ExpertLayout<B> L;
        const double gu_bytes = double(L.down_off) * NE, dn_bytes = double(L.bytes - L.down_off) * NE;
        std::printf("[%s] rel_err %.2e %s | gate_up %.1f us (%.1f GB/s) | down %.1f us (%.1f GB/s) | layer %.1f us = %.1f GB/s"
                    " -> expert-compute ceiling %.0f tok/s (48 layers)\n",
                    P.name, rel, ok ? "PASS" : "FAIL", g_us, gu_bytes / g_us * 1e-3, d_us, dn_bytes / d_us * 1e-3, g_us + d_us,
                    double(L.bytes) * NE / (g_us + d_us) * 1e-3, 1e6 / (48 * (g_us + d_us)));
        vkDestroyQueryPool(ctx.dev, qp, nullptr);
        if (stage.buf) ctx.destroy(stage);
    }
    for (auto& P : paths) { ctx.destroy(P.buf); if (P.raw) munmap(P.raw, P.raw_len); }
    for (auto* b : {&bx, &bh, &bres, &bout, &bwt, &bslots}) ctx.destroy(*b);
    ctx.destroy(gu);
    ctx.destroy(dn);
    return fails;
}

static int run_router(vk::Ctx& ctx, const std::string& sdir) {
    const int E = 512, K = 10;
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto bw = ctx.create(size_t(E) * H * 4), bx = ctx.create(H * 4), bl = ctx.create(E * 4), bm = ctx.create(256);
    auto* w = bw.as<float>();
    auto* x = bx.as<float>();
    for (size_t i = 0; i < size_t(E) * H; ++i) w[i] = nd(rng) * 0.02f;
    for (int i = 0; i < H; ++i) x[i] = nd(rng);
    auto pr = ctx.pipeline(sdir + "/router_f32.spv", sizeof(PcRouter));
    auto pt = ctx.pipeline(sdir + "/topk.spv", sizeof(PcTopk));
    PcRouter a{bw.addr, bx.addr, bl.addr, uint32_t(E), 0};
    PcTopk b{bl.addr, bm.addr, uint32_t(E), uint32_t(K), 77u, 0};
    auto qp = ctx.timestamps(2);
    auto cb = ctx.cmd();
    vk::Ctx::begin(cb);
    vkCmdResetQueryPool(cb, qp, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vk::Ctx::dispatch(cb, pr, &a, E);
    vk::Ctx::barrier(cb);
    vk::Ctx::dispatch(cb, pt, &b, 1);
    vk::Ctx::barrier(cb);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    vk::Ctx::end(cb);
    ctx.run(cb);
    auto ts = ctx.read_ts(qp, 2);
    std::vector<std::pair<double, int>> ref(E);
    for (int r = 0; r < E; ++r) {
        double s = 0;
        for (int k = 0; k < H; ++k) s += double(w[size_t(r) * H + k]) * x[k];
        ref[r] = {s, r};
    }
    std::sort(ref.begin(), ref.end(), [](auto& p, auto& q) { return p.first > q.first; });
    const uint32_t* mail = bm.as<uint32_t>();
    int match = 0;
    for (int k = 0; k < K; ++k) match += int(mail[k]) == ref[k].second;
    const float* mw = reinterpret_cast<const float*>(mail + 16);
    double wsum = 0;
    for (int k = 0; k < K; ++k) wsum += mw[k];
    std::printf("== router+topk: %d/%d ids match, weight sum %.6f, seq %u, %.1f us -> %s\n", match, K, wsum, mail[32],
                double(ts[1] - ts[0]) * ctx.ts_period_ns * 1e-3, match == K && mail[32] == 77u ? "PASS" : "FAIL");
    for (auto* bb : {&bw, &bx, &bl, &bm}) ctx.destroy(*bb);
    ctx.destroy(pr);
    ctx.destroy(pt);
    vkDestroyQueryPool(ctx.dev, qp, nullptr);
    return match == K ? 0 : 1;
}

static void run_membw(vk::Ctx& ctx, const std::string& sdir) {
    const size_t bytes = 512ull << 20;
    auto pm = ctx.pipeline(sdir + "/membw.spv", sizeof(PcMembw), 0);
    auto bout = ctx.create(64);
    auto bench = [&](const char* name, vk::Buffer& b) {
        PcMembw p{b.addr, bout.addr, uint32_t(b.size / 16), 0xDEADBEEFu};
        auto qp = ctx.timestamps(2);
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            auto cb = ctx.cmd();
            vk::Ctx::begin(cb);
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            vk::Ctx::dispatch(cb, pm, &p, 36 * 40);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            vk::Ctx::end(cb);
            ctx.run(cb);
            auto ts = ctx.read_ts(qp, 2);
            best = std::min(best, double(ts[1] - ts[0]) * ctx.ts_period_ns);
        }
        std::printf("== K1b membw [%s] %.1f GB/s\n", name, double(b.size) / best);
        vkDestroyQueryPool(ctx.dev, qp, nullptr);
    };
    try { auto b = ctx.create(bytes, true); std::memset(b.map, 1, bytes); bench("host-visible BO", b); ctx.destroy(b); }
    catch (const std::exception& e) { std::printf("membw host-visible: %s\n", e.what()); }
    try { auto b = ctx.create(256ull << 20, false); bench("device-local (256 MiB)", b); ctx.destroy(b); }
    catch (const std::exception& e) { std::printf("membw device-local: %s\n", e.what()); }
    if (ctx.has_host_import) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        try { std::memset(p, 1, bytes); auto b = ctx.import_host(p, bytes); bench("imported userptr (4K pages)", b); ctx.destroy(b); }
        catch (const std::exception& e) { std::printf("membw import: %s\n", e.what()); }
        munmap(p, bytes);
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
            try { std::memset(p, 1, bytes); auto b = ctx.import_host(p, bytes); bench("imported userptr (hugetlb 2M)", b); ctx.destroy(b); }
            catch (const std::exception& e) { std::printf("membw import hugetlb: %s\n", e.what()); }
            munmap(p, bytes);
        } else std::printf("membw import hugetlb: no 2M hugepages reserved\n");
    }
    ctx.destroy(bout);
    ctx.destroy(pm);
}

int main(int argc, char** argv) {
    std::string sdir = "shaders";
    int bits = 0, iters = 20;
    uint32_t rpw_gu = 4, rpw_dn = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--shaders") sdir = nx();
        else if (a == "--bits") bits = std::stoi(nx());
        else if (a == "--iters") iters = std::stoi(nx());
        else if (a == "--rpw-gu") rpw_gu = uint32_t(std::stoul(nx()));
        else if (a == "--rpw-dn") rpw_dn = uint32_t(std::stoul(nx()));
    }
    vk::Ctx ctx;
    std::printf("device: %s  ts_period %.2f ns  host-import %d (align %llu)\n", ctx.props.deviceName, ctx.ts_period_ns,
                int(ctx.has_host_import), (unsigned long long)ctx.host_import_align);
    ctx.print_memory();
    int fails = 0;
    fails += run_router(ctx, sdir);
    run_membw(ctx, sdir);
    if (bits == 0 || bits == 4) fails += run<4>(ctx, sdir, iters, rpw_gu, rpw_dn);
    if (bits == 0 || bits == 2) fails += run<2>(ctx, sdir, iters, rpw_gu, rpw_dn);
    std::printf("K1 %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
