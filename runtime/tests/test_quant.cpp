// test_quant — PS5Q format invariants and CPU reference correctness (portable).
#include "ps5lm/quant.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace ps5lm::q;

static int failures = 0;
#define CHECK(cond)                                                                                     \
    do {                                                                                                \
        if (!(cond)) { std::fprintf(stderr, "%s:%d CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

static std::vector<float> randn(size_t n, unsigned seed, float sigma = 1.f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

static void fp16_roundtrip() {
    const float vals[] = {0.f, -0.f, 1.f, -2.5f, 65504.f, 6.1035156e-05f, 5.9604645e-08f, 0.1f, -1234.5f};
    for (float v : vals) {
        const float r = f16_to_f32(f32_to_f16(v));
        CHECK(std::fabs(r - v) <= std::fabs(v) * 1e-3f + 6e-8f);
    }
    CHECK(f32_to_f16(1e6f) == 0x7C00);  // overflow -> inf
    for (uint32_t h = 0; h < 0x7C00; ++h)  // every finite positive half round-trips exactly
        if (f32_to_f16(f16_to_f32(uint16_t(h))) != h) { ++failures; std::fprintf(stderr, "half %x\n", h); break; }
}

static void layouts() {
    CHECK(row_bytes<4>(2560) == 1440);
    CHECK(row_bytes<4>(640) == 368);
    CHECK(row_bytes<2>(2560) == 800);
    CHECK(row_bytes<2>(640) == 208);
    CHECK(scales_bytes<4>(640) % 16 == 0);
    CHECK(ExpertLayout<4>().bytes / 4096 == 680);
    CHECK(ExpertLayout<2>().bytes / 4096 == 380);
}

template <int B> static void roundtrip_error(float max_rel_rms) {
    const int K = 2560;
    auto x = randn(K, 7 + B);
    std::vector<uint8_t> row(row_bytes<B>(K));
    quantize_row<B>(x.data(), K, row.data());
    std::vector<float> y(K);
    dequantize_row<B>(row.data(), K, y.data());
    double e2 = 0, x2 = 0;
    for (int i = 0; i < K; ++i) { e2 += double(y[i] - x[i]) * (y[i] - x[i]); x2 += double(x[i]) * x[i]; }
    const double rel = std::sqrt(e2 / x2);
    std::printf("  Q%d dequant rel-RMS error %.4f\n", B, rel);
    CHECK(rel < max_rel_rms);
}

// The integer DP4A path equals the float dot of the dequantized operands
// (up to f32 summation order) — proves the unpack order is right.
template <int B> static void int_dot_matches_dequant() {
    for (int K : {640, 2560}) {
        auto w = randn(K, 100 + K + B), x = randn(K, 200 + K + B);
        std::vector<uint8_t> row(row_bytes<B>(K));
        quantize_row<B>(w.data(), K, row.data());
        std::vector<Q8Block> xq(K / QK);
        quantize_act(x.data(), K, xq.data());
        std::vector<float> wd(K);
        dequantize_row<B>(row.data(), K, wd.data());
        double ref = 0, mag = 0;
        for (int k = 0; k < K; ++k) {
            const float xa = xq[k / QK].d * float(xq[k / QK].qs[k % QK]);
            ref += double(wd[k]) * xa;
            mag += std::fabs(double(wd[k]) * xa);
        }
        const float got = dot_row_q8<B>(row.data(), K, xq.data());
        CHECK(std::fabs(got - ref) <= 1e-5 * mag + 1e-6);
    }
}

template <int B> static void expert_ffn_vs_float(double max_rel) {
    ExpertShape s;
    ExpertLayout<B> L(s);
    std::vector<uint8_t> slot(L.bytes);
    const float sig = 1.f / std::sqrt(float(s.hidden));
    auto gate = randn(size_t(s.ffn) * s.hidden, 11, sig), up = randn(size_t(s.ffn) * s.hidden, 12, sig);
    auto down = randn(size_t(s.hidden) * s.ffn, 13, 1.f / std::sqrt(float(s.ffn)));
    for (int r = 0; r < s.ffn; ++r) {
        quantize_row<B>(&gate[size_t(r) * s.hidden], s.hidden, &slot[L.gate_off + r * row_bytes<B>(s.hidden)]);
        quantize_row<B>(&up[size_t(r) * s.hidden], s.hidden, &slot[L.up_off + r * row_bytes<B>(s.hidden)]);
    }
    for (int r = 0; r < s.hidden; ++r)
        quantize_row<B>(&down[size_t(r) * s.ffn], s.ffn, &slot[L.down_off + r * row_bytes<B>(s.ffn)]);
    auto x = randn(s.hidden, 14);
    std::vector<Q8Block> xq(s.hidden / QK), hq(s.ffn / QK);
    quantize_act(x.data(), s.hidden, xq.data());
    std::vector<float> out(s.hidden, 0.f), h(s.ffn);
    expert_ffn_ref<B>(slot.data(), xq.data(), 0.5f, out.data(), s, h.data(), hq.data());

    // f32 reference on the dequantized weights (isolates activation-quant error)
    std::vector<float> wd(s.hidden), hf(s.ffn), ref(s.hidden);
    for (int r = 0; r < s.ffn; ++r) {
        double g = 0, u = 0;
        dequantize_row<B>(&slot[L.gate_off + r * row_bytes<B>(s.hidden)], s.hidden, wd.data());
        for (int k = 0; k < s.hidden; ++k) g += double(wd[k]) * x[k];
        dequantize_row<B>(&slot[L.up_off + r * row_bytes<B>(s.hidden)], s.hidden, wd.data());
        for (int k = 0; k < s.hidden; ++k) u += double(wd[k]) * x[k];
        hf[r] = silu(float(g)) * float(u);
    }
    double e2 = 0, r2 = 0;
    for (int r = 0; r < s.hidden; ++r) {
        dequantize_row<B>(&slot[L.down_off + r * row_bytes<B>(s.ffn)], s.ffn, wd.data());
        double acc = 0;
        for (int k = 0; k < s.ffn; ++k) acc += double(wd[k]) * hf[k];
        ref[r] = 0.5f * float(acc);
        e2 += double(out[r] - ref[r]) * (out[r] - ref[r]);
        r2 += double(ref[r]) * ref[r];
    }
    const double rel = std::sqrt(e2 / r2);
    std::printf("  Q%d expert FFN (Q8 act path) vs f32 rel-RMS %.4f\n", B, rel);
    CHECK(rel < max_rel);
}

int main() {
    fp16_roundtrip();
    layouts();
    roundtrip_error<4>(0.12f);
    roundtrip_error<2>(0.60f);
    int_dot_matches_dequant<4>();
    int_dot_matches_dequant<2>();
    expert_ffn_vs_float<4>(0.03);
    expert_ffn_vs_float<2>(0.06);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("test_quant: all passed");
    return 0;
}
