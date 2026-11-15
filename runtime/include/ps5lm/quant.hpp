// quant.hpp — PS5Q weight format (DP4A-native) + bit-exact CPU reference.
//
// Design goals, in order: (1) every expert slot is a whole number of 4 KiB
// pages (O_DIRECT, zero padding); (2) one wave32 reads a row with fully
// coalesced 16-byte-per-lane loads; (3) unpacking to v_dot4_i32_i8 operands
// is one AND (+ one shift) per 4 weights.
//
// Block: QK = 32 weights, one fp16 scale. Value = d * (q - 2^(B-1)), q in [0, 2^B).
//   Q4: 16 B quants + 2 B scale = 4.5 bpw.  byte i = elem i | elem (i+16) << 4
//   Q2:  8 B quants + 2 B scale = 2.5 bpw.  byte i = elem i | i+8 <<2 | i+16 <<4 | i+24 <<6
// Row layout (SoA, so lanes stride contiguous memory):
//   [ nb fp16 scales, padded to 16 B ][ nb * QK*B/8 quant bytes ]   (row stride 16 B aligned)
// Expert slot (Qwen3.8-Flash-Next: hidden 2560, expert ffn 640):
//   gate [640 x 2560] | up [640 x 2560] | down [2560 x 640]
//   Q4: 2,785,280 B = 680 x 4 KiB.   Q2: 1,556,480 B = 380 x 4 KiB.
//
// Activations: Q8 blocks of 32 (int8 quants, f32 scale, i32 sum of quants), so
//   dot(w_blk, a_blk) = dw * da * (sdot4(qw, qa) - 2^(B-1) * sum(qa)).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define PS5Q_HD __host__ __device__
#else
#define PS5Q_HD
#endif

namespace ps5lm::q {

inline constexpr int QK = 32;

template <int B> struct Fmt {
    static_assert(B == 2 || B == 4, "PS5Q supports 2- and 4-bit weights");
    static constexpr int bits = B;
    static constexpr int quant_bytes = QK * B / 8;  // per block
    static constexpr int zero = 1 << (B - 1);       // 8 or 2
    static constexpr int qmax = (1 << B) - 1;
};

PS5Q_HD constexpr size_t align16(size_t v) { return (v + 15) & ~size_t(15); }

template <int B> PS5Q_HD constexpr size_t scales_bytes(int K) { return align16(size_t(K / QK) * 2); }
template <int B> PS5Q_HD constexpr size_t row_bytes(int K) {
    return align16(scales_bytes<B>(K) + size_t(K / QK) * Fmt<B>::quant_bytes);
}

struct ExpertShape {
    int hidden = 2560;
    int ffn = 640;
};

template <int B> struct ExpertLayout {
    size_t gate_off, up_off, down_off, bytes;
    PS5Q_HD constexpr explicit ExpertLayout(ExpertShape s = {})
        : gate_off(0),
          up_off(size_t(s.ffn) * row_bytes<B>(s.hidden)),
          down_off(2 * size_t(s.ffn) * row_bytes<B>(s.hidden)),
          bytes(2 * size_t(s.ffn) * row_bytes<B>(s.hidden) + size_t(s.hidden) * row_bytes<B>(s.ffn)) {}
};
static_assert(ExpertLayout<4>().bytes == 2'785'280 && ExpertLayout<4>().bytes % 4096 == 0);
static_assert(ExpertLayout<2>().bytes == 1'556'480 && ExpertLayout<2>().bytes % 4096 == 0);

// ---- fp16 <-> f32 (portable, round-to-nearest-even) -------------------------
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 31) return uint16_t(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) ++half;
    return uint16_t(half);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FFu, x;
    if (exp == 0) {
        if (!mant) x = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            x = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) x = sign | 0x7F800000u | (mant << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---- weights ----------------------------------------------------------------
// steps: size of the offline scale-search grid (40 = the S12 reference; the HF converter uses ~12,
// ~3x faster, for a negligible rel-RMS change on real weights).
template <int B> inline void quantize_row(const float* x, int K, uint8_t* row, int steps = 40) {
    using F = Fmt<B>;
    const int nb = K / QK;
    auto* scales = reinterpret_cast<uint16_t*>(row);
    uint8_t* qs = row + scales_bytes<B>(K);
    std::memset(row, 0, row_bytes<B>(K));
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * QK;
        float amax = 0, m = 0;
        for (int i = 0; i < QK; ++i) if (std::fabs(xb[i]) > amax) { amax = std::fabs(xb[i]); m = xb[i]; }
        // Offline MSE-optimal scale: search d over a grid of candidates (both
        // signs: the asymmetric level set [-2^(B-1), 2^(B-1)-1] favours the
        // side mapped to the long end), then refit d by least squares for the
        // chosen integer codes. Free at runtime: the format is unchanged.
        uint8_t q[QK];
        auto encode = [&](float d, uint8_t* out) {
            const float id = d != 0.f ? 1.f / d : 0.f;
            float err = 0;
            for (int i = 0; i < QK; ++i) {
                const int v = std::clamp(int(std::lround(xb[i] * id)) + F::zero, 0, F::qmax);
                out[i] = uint8_t(v);
                const float e = xb[i] - d * float(v - F::zero);
                err += e * e;
            }
            return err;
        };
        float best_d = m / -float(F::zero), best_err = encode(best_d, q);
        if (amax > 0.f) {
            uint8_t tq[QK];
            for (int sgn = -1; sgn <= 1; sgn += 2)
                for (int step = 0; step <= steps; ++step) {
                    // |d| = amax/zero x [0.4, 2.4]: from clipping-heavy (small d) to
                    // no-clip on the short side of the level set (d >= amax/(zero-1)).
                    // Q2 lands at rel-RMS ~0.32 on Gaussian weights, about the
                    // 4-level scalar floor; below that needs vector/trellis codes.
                    const float d = float(sgn) * amax * (0.4f + 2.0f * float(step) / float(steps)) / float(F::zero);
                    // refit: d* = <x, c> / <c, c> for the codes c this d produces
                    encode(d, tq);
                    float num = 0, den = 0;
                    for (int i = 0; i < QK; ++i) { const float c = float(tq[i] - F::zero); num += xb[i] * c; den += c * c; }
                    const float dr = den > 0.f ? num / den : d;
                    const float er = encode(dr, tq);
                    if (er < best_err) { best_err = er; best_d = dr; }
                }
        }
        const uint16_t dh = f32_to_f16(best_d);
        scales[b] = dh;
        encode(f16_to_f32(dh), q);  // final codes against the fp16-rounded scale actually stored
        uint8_t* o = qs + size_t(b) * F::quant_bytes;
        if constexpr (B == 4) {
            for (int i = 0; i < 16; ++i) o[i] = uint8_t(q[i] | (q[i + 16] << 4));
        } else {
            for (int i = 0; i < 8; ++i) o[i] = uint8_t(q[i] | (q[i + 8] << 2) | (q[i + 16] << 4) | (q[i + 24] << 6));
        }
    }
}

// Converter path (same format, same objective, ~8x cheaper): score each grid candidate by its
// squared error WITHOUT a least-squares refit, keep the best two signs' winners, then refit once.
// Branch-free rounding instead of std::lround. Used for the 120B-parameter HF conversion.
template <int B> inline void quantize_row_fast(const float* x, int K, uint8_t* row, int steps = 16) {
    using F = Fmt<B>;
    const int nb = K / QK;
    auto* scales = reinterpret_cast<uint16_t*>(row);
    uint8_t* qs = row + scales_bytes<B>(K);
    std::memset(row, 0, row_bytes<B>(K));
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * QK;
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        uint8_t q[QK];
        auto encode = [&](float d, uint8_t* out) {
            const float id = d != 0.f ? 1.f / d : 0.f;
            float err = 0;
            for (int i = 0; i < QK; ++i) {
                const float t = xb[i] * id;
                int v = int(t + (t >= 0.f ? 0.5f : -0.5f)) + F::zero;
                v = v < 0 ? 0 : (v > F::qmax ? F::qmax : v);
                out[i] = uint8_t(v);
                const float e = xb[i] - d * float(v - F::zero);
                err += e * e;
            }
            return err;
        };
        float best_d = 0.f, best_err = 1e30f;
        if (amax > 0.f)
            for (int sgn = -1; sgn <= 1; sgn += 2)
                for (int step = 0; step <= steps; ++step) {
                    const float d = float(sgn) * amax * (0.4f + 2.0f * float(step) / float(steps)) / float(F::zero);
                    const float er = encode(d, q);
                    if (er < best_err) { best_err = er; best_d = d; }
                }
        if (amax > 0.f) {  // one least-squares refit of the winner
            encode(best_d, q);
            float num = 0, den = 0;
            for (int i = 0; i < QK; ++i) { const float c = float(q[i] - F::zero); num += xb[i] * c; den += c * c; }
            const float dr = den > 0.f ? num / den : best_d;
            if (encode(dr, q) < best_err) best_d = dr;
        }
        const uint16_t dh = f32_to_f16(best_d);
        scales[b] = dh;
        encode(f16_to_f32(dh), q);
        uint8_t* o = qs + size_t(b) * F::quant_bytes;
        if constexpr (B == 4) {
            for (int i = 0; i < 16; ++i) o[i] = uint8_t(q[i] | (q[i + 16] << 4));
        } else {
            for (int i = 0; i < 8; ++i) o[i] = uint8_t(q[i] | (q[i + 8] << 2) | (q[i + 16] << 4) | (q[i + 24] << 6));
        }
    }
}

template <int B> inline int weight_q(const uint8_t* row, int K, int k) {
    const uint8_t* o = row + scales_bytes<B>(K) + size_t(k / QK) * Fmt<B>::quant_bytes;
    const int i = k % QK;
    if constexpr (B == 4) return i < 16 ? (o[i] & 0xF) : (o[i - 16] >> 4);
    else return (o[i % 8] >> (2 * (i / 8))) & 3;
}

template <int B> inline void dequantize_row(const uint8_t* row, int K, float* y) {
    const auto* scales = reinterpret_cast<const uint16_t*>(row);
    for (int k = 0; k < K; ++k)
        y[k] = f16_to_f32(scales[k / QK]) * float(weight_q<B>(row, K, k) - Fmt<B>::zero);
}

// ---- activations (Q8) ---------------------------------------------------------
struct Q8Block {
    float d;
    int32_t sum;
    int8_t qs[QK];
};
static_assert(sizeof(Q8Block) == 40);

inline void quantize_act(const float* x, int K, Q8Block* out) {
    for (int b = 0; b < K / QK; ++b) {
        const float* xb = x + b * QK;
        float amax = 0;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        out[b].d = d;
        int32_t s = 0;
        for (int i = 0; i < QK; ++i) {
            const int v = int(std::lround(xb[i] * id));
            out[b].qs[i] = int8_t(std::clamp(v, -127, 127));
            s += out[b].qs[i];
        }
        out[b].sum = s;
    }
}

// Software model of v_dot4_i32_i8: four signed int8 lanes, accumulated into c.
PS5Q_HD inline int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    for (int i = 0; i < 4; ++i) c += int32_t(int8_t(a >> (8 * i))) * int32_t(int8_t(b >> (8 * i)));
    return c;
}

// Integer-domain row dot, using exactly the GPU kernel's unpack sequence (the
// kernel is tested against THIS, not against float math).
template <int B> inline float dot_row_q8(const uint8_t* row, int K, const Q8Block* a) {
    const auto* scales = reinterpret_cast<const uint16_t*>(row);
    const uint8_t* qs = row + scales_bytes<B>(K);
    float acc = 0;
    for (int b = 0; b < K / QK; ++b) {
        uint32_t aw[8];
        std::memcpy(aw, a[b].qs, 32);
        const uint8_t* o = qs + size_t(b) * Fmt<B>::quant_bytes;
        int32_t s = 0;
        if constexpr (B == 4) {
            uint32_t w[4];
            std::memcpy(w, o, 16);
            for (int j = 0; j < 4; ++j) {
                s = sdot4(w[j] & 0x0F0F0F0Fu, aw[j], s);
                s = sdot4((w[j] >> 4) & 0x0F0F0F0Fu, aw[4 + j], s);
            }
        } else {
            uint32_t w[2];
            std::memcpy(w, o, 8);
            for (int sh = 0; sh < 4; ++sh)
                for (int j = 0; j < 2; ++j) s = sdot4((w[j] >> (2 * sh)) & 0x03030303u, aw[2 * sh + j], s);
        }
        acc += f16_to_f32(scales[b]) * a[b].d * float(s - Fmt<B>::zero * a[b].sum);
    }
    return acc;
}

// ---- one routed expert, reference semantics of the fused GPU kernels ---------
inline float silu(float v) { return v / (1.f + std::exp(-v)); }

// out[hidden] += w_route * down( silu(gate x) * (up x) ), all through the Q8 int path.
template <int B>
inline void expert_ffn_ref(const uint8_t* slot, const Q8Block* xq, float w_route, float* out, ExpertShape s = {},
                           float* scratch_h = nullptr, Q8Block* scratch_hq = nullptr) {
    const ExpertLayout<B> L(s);
    const size_t rg = row_bytes<B>(s.hidden), rd = row_bytes<B>(s.ffn);
    for (int r = 0; r < s.ffn; ++r) {
        const float g = dot_row_q8<B>(slot + L.gate_off + r * rg, s.hidden, xq);
        const float u = dot_row_q8<B>(slot + L.up_off + r * rg, s.hidden, xq);
        scratch_h[r] = silu(g) * u;
    }
    quantize_act(scratch_h, s.ffn, scratch_hq);
    for (int r = 0; r < s.hidden; ++r) out[r] += w_route * dot_row_q8<B>(slot + L.down_off + r * rd, s.ffn, scratch_hq);
}

}  // namespace ps5lm::q
