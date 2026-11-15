// common.glsl -- shared declarations for PS5LM compute kernels (Vulkan, BDA, no descriptors).
// PS5Q layout (include/ps5lm/quant.hpp): row = [fp16 scales, padded to 16 B][quant bytes],
// blocks of 32 weights; value = d * (q - 2^(BITS-1)).
//   Q4: byte i = elem i | elem (i+16) << 4               (16 B/block, 4 words)
//   Q2: byte i = elem i | i+8 << 2 | i+16 << 4 | i+24 << 6  (8 B/block, 2 words)
// GFX1013 has no accelerated DP4A: dequantise in-register, fp32 FMA.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_control_flow_attributes : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer V4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer U64 { uint64_t v[]; };

const uint HIDDEN = 2560u;
const uint FFN = 640u;

#ifndef BITS
#define BITS 4
#endif
#if BITS == 4
const uint QWPB = 4u;   // quant words per block
const uint VPW = 8u;    // weights per quant word
const float ZERO = 8.0;
#else
const uint QWPB = 2u;
const uint VPW = 16u;
const float ZERO = 2.0;
#endif

uint align16w(uint bytes) { return ((bytes + 15u) & ~15u) >> 2; }  // bytes -> words, 16 B aligned
uint scale_words(uint K) { return align16w((K / 32u) * 2u); }
uint quant_words(uint K) { return (K / 32u) * QWPB; }
uint row_words(uint K) { return align16w(scale_words(K) * 4u + quant_words(K) * 4u); }

// Element index (within the row) of value v (0..VPW-1) of quant word w.
uint elem_of(uint w, uint v) {
#if BITS == 4
    uint b = w >> 2, j = w & 3u;   // v<4: low nibble of byte v; v>=4: high nibble of byte v-4
    return b * 32u + 4u * j + (v & 3u) + (v >> 2) * 16u;
#else
    uint b = w >> 1, j = w & 1u;   // v = 4*sh + k : byte k, shift sh
    return b * 32u + 4u * j + (v & 3u) + (v >> 2) * 8u;
#endif
}
// S14: extract the 4 codes of group g (= v >> 2) with ONE shift + mask into byte lanes, then convert the byte
// lane directly (ACO: v_cvt_f32_ubyte0..3). Same integer values and the same float ops/order as the S13
// per-value bitfield extract, so results are bit-identical; ~1.5 instead of 2 integer ops per weight.
float qval(uint q, uint v) {
#if BITS == 4
    const uint t = (q >> (4u * (v >> 2))) & 0x0F0F0F0Fu;
#else
    const uint t = (q >> (2u * (v >> 2))) & 0x03030303u;
#endif
    return float((t >> (8u * (v & 3u))) & 0xFFu);
}
float block_scale(U32 W, uint row_base, uint w) {
    uint b = w / QWPB;
    vec2 s = unpackHalf2x16(W.v[row_base + (b >> 1)]);
    return (b & 1u) != 0u ? s.y : s.x;
}
// Gather the VPW activations a lane needs for quant word w (groups of 4 are contiguous: vec4 loads).
void load_x(V4 X, uint w, out float xv[16], out float sx) {
    sx = 0.0;
    [[unroll]] for (uint g = 0u; g < 4u; ++g) {
        if (g < VPW / 4u) {
            vec4 t = X.v[elem_of(w, 4u * g) >> 2];
            xv[4u * g + 0u] = t.x; xv[4u * g + 1u] = t.y; xv[4u * g + 2u] = t.z; xv[4u * g + 3u] = t.w;
            sx += t.x + t.y + t.z + t.w;
        } else {
            xv[4u * g + 0u] = 0.0; xv[4u * g + 1u] = 0.0; xv[4u * g + 2u] = 0.0; xv[4u * g + 3u] = 0.0;
        }
    }
}
float word_dot(uint q, float xv[16], float sx) {
    float s = 0.0;
    [[unroll]] for (uint v = 0u; v < VPW; ++v) s += qval(q, v) * xv[v];
    return s - ZERO * sx;
}
