#include <metal_stdlib>
using namespace metal;

// Helpers: arithmetic-only decomposition and packing
inline uint udiv(uint x, uint d) { return x / d; }
inline uint umod(uint x, uint d) { return x - d * (x / d); }

inline void decompose_u8(uint x, thread uint &b0, thread uint &b1, thread uint &b2, thread uint &b3) {
    uint q0 = udiv(x, 256u);
    b0 = x - q0 * 256u;
    uint q1 = udiv(q0, 256u);
    b1 = q0 - q1 * 256u;
    uint q2 = udiv(q1, 256u);
    b2 = q1 - q2 * 256u;
    b3 = q2;
}

inline uint pack_u8(uint r0, uint r1, uint r2, uint r3) {
    return r0 + 256u * r1 + 65536u * r2 + 16777216u * r3;
}

inline void decompose_u16(uint x, thread uint &w0, thread uint &w1) {
    uint q = udiv(x, 65536u);
    w0 = x - q * 65536u;
    w1 = q;
}

inline uint pack_u16(uint r0, uint r1) {
    return r0 + 65536u * r1;
}

kernel void avg_u8_trunc_packs(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint axb = a ^ b;
    uint hfx = (axb & 0xFEFEFEFEu) >> 1; // logical in Metal
    uint avgv = (a & b) + hfx;
    outC[gid] = avgv;
}

kernel void avg_u8_round_packs(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint axb = a ^ b;
    uint hfx = (axb & 0xFEFEFEFEu) >> 1; // logical in Metal
    uint avgv = (a & b) + hfx;
    uint rnd = axb & 0x01010101u; // add 1 when per-lane sum is odd
    outC[gid] = avgv + rnd;
}

// Arithmetic-only kernels (no &, ^, >>, <<)
kernel void avg_u8_trunc_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1, a2, a3; decompose_u8(a, a0, a1, a2, a3);
    uint b0, b1, b2, b3; decompose_u8(b, b0, b1, b2, b3);
    uint r0 = udiv(a0 + b0, 2u);
    uint r1 = udiv(a1 + b1, 2u);
    uint r2 = udiv(a2 + b2, 2u);
    uint r3 = udiv(a3 + b3, 2u);
    outC[gid] = pack_u8(r0, r1, r2, r3);
}

kernel void avg_u8_round_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1, a2, a3; decompose_u8(a, a0, a1, a2, a3);
    uint b0, b1, b2, b3; decompose_u8(b, b0, b1, b2, b3);
    uint r0 = udiv(a0 + b0 + 1u, 2u);
    uint r1 = udiv(a1 + b1 + 1u, 2u);
    uint r2 = udiv(a2 + b2 + 1u, 2u);
    uint r3 = udiv(a3 + b3 + 1u, 2u);
    outC[gid] = pack_u8(r0, r1, r2, r3);
}

kernel void avg_u16_trunc_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1; decompose_u16(a, a0, a1);
    uint b0, b1; decompose_u16(b, b0, b1);
    uint r0 = udiv(a0 + b0, 2u);
    uint r1 = udiv(a1 + b1, 2u);
    outC[gid] = pack_u16(r0, r1);
}

kernel void avg_u16_round_arith(
    device const uint *inA [[buffer(0)]],
    device const uint *inB [[buffer(1)]],
    device uint *outC [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint a = inA[gid];
    uint b = inB[gid];
    uint a0, a1; decompose_u16(a, a0, a1);
    uint b0, b1; decompose_u16(b, b0, b1);
    uint r0 = udiv(a0 + b0 + 1u, 2u);
    uint r1 = udiv(a1 + b1 + 1u, 2u);
    outC[gid] = pack_u16(r0, r1);
}