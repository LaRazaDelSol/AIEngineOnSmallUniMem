// exl3_common.glsl (S15b) -- EXL3 "mul1" expert slots streamed byte-for-byte from turboderp's shards.
// Slot = the expert's contiguous safetensors span, 4 KiB aligned by runtime/tools/s15/pack_exl3.py:
//   down.suh f16[640] | down.svh f16[2560] | down.mul1 u32 | down.trellis [40][160][16K u16]
//   gate.suh f16[2560] | gate.svh f16[640] | gate.mul1 | gate.trellis [160][40][16K u16] | up.(same as gate)
// Linear map (exllamav3 exl3.py get_weight_tensor): W = diag(suh) H128 W_inner H128 diag(svh)  (H128 = normalized
// Sylvester, block-diagonal), so y = H128(H128(x * suh) . W_inner) * svh.
// mul1 value = (1024 + bytesum(idx * 0x83DCD12D)) * K_INV + K_BIAS (exllamav3 codebook.cuh, fp16 constants 0x1EEE / 0xC931).
// KH = 2K: 6 (K=3), 5 (K=2.5) or 7 (K=3.5); odd KH = exllamav3 dq8_half.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer H16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer V2 { vec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer U64 { uint64_t v[]; };

const float K_INV = 0.0067672729;   // fp16 0x1EEE = 1.732421875 * 2^-8 (exact)
const float K_BIAS = -10.3828125;   // fp16 0xC931
const float RSQRT128 = 0.08838834764831845;

#ifndef KH
#define KH 6
#endif
const uint TBYTES = 102400u * KH;   // one 2560x640 trellis: 640*2560*K/8 = 102400 * 2K
// byte offsets inside a slot
const uint OFF_D_SUH = 0u, OFF_D_SVH = 1280u, OFF_D_TR = 6404u;
const uint OFF_G_SUH = 6404u + TBYTES, OFF_G_SVH = OFF_G_SUH + 5120u, OFF_G_TR = OFF_G_SUH + 6404u;
const uint OFF_U_SUH = OFF_G_TR + TBYTES, OFF_U_SVH = OFF_U_SUH + 5120u, OFF_U_TR = OFF_U_SUH + 6404u;
const uint TW = 4u * KH;            // uint32 words per 256-weight tile: 256*K/32 = 4 * 2K

// (1024 + s) of one 16-bit window as an fp16 bit pattern, consumed by a mixed-precision FMA (no convert)
float dec(uint w) {
    const uint p = w * 0x83DCD12Du;
    return float(unpackFloat2x16(dotPacked4x8AccSatEXT(p, 0x01010101u, 0x6400u)).x);
}

// Windows of tile positions 8t..8t+7 for lane t (exllamav3 dq8 / dq8_half): two 32-bit funnel results; positions
// 0..3 come from wlo, 4..7 from whi. Word indices / shifts depend only on the lane (precomputed by tile_geom).
struct Geom { uint ia0, ib0, s0, ia1, ib1, s1; };
Geom tile_geom(uint t) {
    Geom g;
#if KH == 6
    const int b1 = int(t * 8u + 257u) * 3, b0 = b1 - 16, b2 = b1 + 21;   // dq8<3,cb,4>
    g.ia0 = uint(b0 >> 5) % TW; g.ib0 = uint((b2 - 1) >> 5) % TW;
    g.s0 = uint(((b2 - 1) >> 5) + 1) * 32u - uint(b2);
    g.ia1 = g.ia0; g.ib1 = g.ib0; g.s1 = g.s0;                        // one 64-bit window covers all 8
#else
    // dq8_half<KA>, KA = (KH - 1) / 2 (K = 2.5 -> KH 5, K = 3.5 -> KH 7): bits per two positions = KH, four windows
    // span GSPAN = 18 + 3 KA bits (exllamav3 exl3_dq.cuh)
    const int GSPAN = 18 + 3 * ((KH - 1) / 2);
    const int e7 = int(t * 4u + 4u) * KH + 128 * KH, e3 = e7 - 2 * KH;
    g.ia1 = uint((e7 - GSPAN) >> 5) % TW; g.ib1 = uint((e7 - 1) >> 5) % TW; g.s1 = uint(((e7 - 1) >> 5) + 1) * 32u - uint(e7);
    g.ia0 = uint((e3 - GSPAN) >> 5) % TW; g.ib0 = uint((e3 - 1) >> 5) % TW; g.s0 = uint(((e3 - 1) >> 5) + 1) * 32u - uint(e3);
#endif
    return g;
}
// S17: the words one lane needs from one tile (loaded ahead of the decode: software pipelining)
struct TWords { uint a0, b0, a1, b1; };
TWords tile_load(uint64_t tile, Geom g) {
    TWords w;
    w.a0 = U32(tile).v[g.ia0]; w.b0 = U32(tile).v[g.ib0];
#if KH == 6
    w.a1 = 0u; w.b1 = 0u;
#else
    w.a1 = U32(tile).v[g.ia1]; w.b1 = U32(tile).v[g.ib1];
#endif
    return w;
}
// same arithmetic as tile_dec, from preloaded words
void tile_dec_w(TWords tw, Geom g, out float d[8]) {
#if KH == 6
    const uint64_t v = (uint64_t(tw.a0) << 32) | uint64_t(tw.b0);
    const uint w7 = uint(v >> g.s0), w3 = uint(v >> (g.s0 + 12u));
    d[0] = dec(bitfieldExtract(w3, 9, 16)); d[1] = dec(bitfieldExtract(w3, 6, 16));
    d[2] = dec(bitfieldExtract(w3, 3, 16)); d[3] = dec(w3 & 0xFFFFu);
    d[4] = dec(bitfieldExtract(w7, 9, 16)); d[5] = dec(bitfieldExtract(w7, 6, 16));
    d[6] = dec(bitfieldExtract(w7, 3, 16)); d[7] = dec(w7 & 0xFFFFu);
#else
    const uint w7 = uint(((uint64_t(tw.a1) << 32) | uint64_t(tw.b1)) >> g.s1);
    const uint w3 = uint(((uint64_t(tw.a0) << 32) | uint64_t(tw.b0)) >> g.s0);
    const int KA = (KH - 1) / 2, S6 = KA + 1, S5 = 2 * KA + 1, S4 = 3 * KA + 2;
    d[7] = dec(w7 & 0xFFFFu); d[6] = dec(bitfieldExtract(w7, S6, 16));
    d[5] = dec(bitfieldExtract(w7, S5, 16)); d[4] = dec(bitfieldExtract(w7, S4, 16));
    d[3] = dec(w3 & 0xFFFFu); d[2] = dec(bitfieldExtract(w3, S6, 16));
    d[1] = dec(bitfieldExtract(w3, S5, 16)); d[0] = dec(bitfieldExtract(w3, S4, 16));
#endif
}
// S19: the same words and windows with (a) 32-bit byte offsets, so a load is uniform tile base (SGPRs) + lane offset
// (global_load saddr+voffset: the U32(tile).v[index] form cost a 64-bit add pair per load) and (b) funnel shifts as
// 32-bit ops, (b >> s) | ((a << 1) << (31 - s)) == low 32 bits of (a:b) >> s for s in 0..31 (the uint64 shift
// form compiled to movs + 64-bit shifts). Bit-identical words -> bit-identical decode.
struct GeomS { uint oa0, ob0, oa1, ob1, s0, r0, s1, r1; };
GeomS tile_geom_s(uint t) {
    const Geom g = tile_geom(t);
    GeomS q;
    q.oa0 = g.ia0 * 4u; q.ob0 = g.ib0 * 4u; q.oa1 = g.ia1 * 4u; q.ob1 = g.ib1 * 4u;
    q.s0 = g.s0; q.r0 = 31u - g.s0; q.s1 = g.s1; q.r1 = 31u - g.s1;
    return q;
}
TWords tile_load_s(uint64_t tile, GeomS g) {
    TWords w;
    w.a0 = U32(tile + uint64_t(g.oa0)).v[0]; w.b0 = U32(tile + uint64_t(g.ob0)).v[0];
#if KH == 6
    w.a1 = 0u; w.b1 = 0u;
#else
    w.a1 = U32(tile + uint64_t(g.oa1)).v[0]; w.b1 = U32(tile + uint64_t(g.ob1)).v[0];
#endif
    return w;
}
uint fsh(uint a, uint b, uint s, uint r) { return (b >> s) | ((a << 1u) << r); }
void tile_dec_s(TWords tw, GeomS g, out float d[8]) {
#if KH == 6
    // w7 = (a:b) >> s0; w3 = (a:b) >> (s0 + 12) = (w7 >> 12) | ((a >> s0) << 20)  (s0 <= 31)
    const uint w7 = fsh(tw.a0, tw.b0, g.s0, g.r0);
    const uint w3 = (w7 >> 12u) | ((tw.a0 >> g.s0) << 20u);
    d[0] = dec(bitfieldExtract(w3, 9, 16)); d[1] = dec(bitfieldExtract(w3, 6, 16));
    d[2] = dec(bitfieldExtract(w3, 3, 16)); d[3] = dec(w3 & 0xFFFFu);
    d[4] = dec(bitfieldExtract(w7, 9, 16)); d[5] = dec(bitfieldExtract(w7, 6, 16));
    d[6] = dec(bitfieldExtract(w7, 3, 16)); d[7] = dec(w7 & 0xFFFFu);
#else
    const uint w7 = fsh(tw.a1, tw.b1, g.s1, g.r1);
    const uint w3 = fsh(tw.a0, tw.b0, g.s0, g.r0);
    const int KA = (KH - 1) / 2, S6 = KA + 1, S5 = 2 * KA + 1, S4 = 3 * KA + 2;
    d[7] = dec(w7 & 0xFFFFu); d[6] = dec(bitfieldExtract(w7, S6, 16));
    d[5] = dec(bitfieldExtract(w7, S5, 16)); d[4] = dec(bitfieldExtract(w7, S4, 16));
    d[3] = dec(w3 & 0xFFFFu); d[2] = dec(bitfieldExtract(w3, S6, 16));
    d[1] = dec(bitfieldExtract(w3, S5, 16)); d[0] = dec(bitfieldExtract(w3, S4, 16));
#endif
}
// decoded (1024 + s) of the lane's 8 positions in one tile at word address `tile` (byte address)
void tile_dec(uint64_t tile, Geom g, out float d[8]) {
#if KH == 6
    const uint a = U32(tile).v[g.ia0], b = U32(tile).v[g.ib0];
    const uint64_t v = (uint64_t(a) << 32) | uint64_t(b);
    const uint w7 = uint(v >> g.s0), w3 = uint(v >> (g.s0 + 12u));
    d[0] = dec(bitfieldExtract(w3, 9, 16)); d[1] = dec(bitfieldExtract(w3, 6, 16));
    d[2] = dec(bitfieldExtract(w3, 3, 16)); d[3] = dec(w3 & 0xFFFFu);
    d[4] = dec(bitfieldExtract(w7, 9, 16)); d[5] = dec(bitfieldExtract(w7, 6, 16));
    d[6] = dec(bitfieldExtract(w7, 3, 16)); d[7] = dec(w7 & 0xFFFFu);
#else
    const uint a1 = U32(tile).v[g.ia1], b1 = U32(tile).v[g.ib1];
    const uint w7 = uint(((uint64_t(a1) << 32) | uint64_t(b1)) >> g.s1);
    const uint a0 = U32(tile).v[g.ia0], b0 = U32(tile).v[g.ib0];
    const uint w3 = uint(((uint64_t(a0) << 32) | uint64_t(b0)) >> g.s0);
    // odd positions carry the extra bit: w6 = w7 >> (KA+1), w5 = w6 >> KA, w4 = w5 >> (KA+1)  (KA = 3: 4 / 7 / 11)
    const int KA = (KH - 1) / 2, S6 = KA + 1, S5 = 2 * KA + 1, S4 = 3 * KA + 2;
    d[7] = dec(w7 & 0xFFFFu); d[6] = dec(bitfieldExtract(w7, S6, 16));
    d[5] = dec(bitfieldExtract(w7, S5, 16)); d[4] = dec(bitfieldExtract(w7, S4, 16));
    d[3] = dec(w3 & 0xFFFFu); d[2] = dec(bitfieldExtract(w3, S6, 16));
    d[1] = dec(bitfieldExtract(w3, S5, 16)); d[0] = dec(bitfieldExtract(w3, S4, 16));
#endif
}
