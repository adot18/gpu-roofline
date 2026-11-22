// ============================================================================
// gpu_roofline.cpp — Empirical Cross-Vendor LLM Inference Roofline Probe
//
// Compiles natively on NVIDIA (nvcc: sm_80/89/90/100) and AMD ROCm
// (hipcc: gfx90a MI250X, gfx942 MI300X/MI325X, gfx950 MI350X/MI355X).
//
// Usage:
//   ./gpu_roofline                           # Run probe + print ASCII sweep graph
//   ./gpu_roofline --json hw.json            # Export empirical ceilings for engine_roofline.py
//   ./gpu_roofline --csv sweep.csv           # Export 4 KiB..2 GiB sweep for plot_sweep.py
// ============================================================================

#include "gpu_compat.h"
#include <algorithm>
#include <chrono>
#include <cstring>

// 1A. Coalesced 128-bit (float4) global copy
__global__ void k_copy_vec4(const float4* __restrict__ src,
                            float4* __restrict__ dst,
                            size_t n_vec4) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = idx; i < n_vec4; i += stride) {
        dst[i] = src[i];
    }
}

// 1B. Strided uncoalesced copy (33-float jump wastes ~94% of each cache sector)
__global__ void k_copy_strided(const float* __restrict__ src,
                               float* __restrict__ dst,
                               size_t n_floats_pow2) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    size_t mask   = n_floats_pow2 - 1;
    for (size_t i = idx; i < n_floats_pow2; i += stride) {
        dst[i] = src[(i * 33ULL) & mask];
    }
}

// 1C. Pure Read-Only 128-bit (float4) HBM streaming (models weight-streaming decode)
__global__ void k_read_vec4(const float4* __restrict__ src,
                            float* __restrict__ out,
                            size_t n_vec4) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    float acc = 0.0f;
    for (size_t i = idx; i < n_vec4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == -999999.0f) out[idx] = acc;
}

// 3. On-chip Shared Memory (SRAM / LDS) read loop
__global__ void k_shared_bw(float4* out) {
    __shared__ float4 smem[1024]; // 16 KiB per block
    int t = threadIdx.x;
    for (int i = t; i < 1024; i += blockDim.x) {
        smem[i] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    }
    __syncthreads();

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll 16
    for (int r = 0; r < 1024; ++r) {
        float4 v = smem[(t + r) & 1023];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == -999999.0f) out[(size_t)blockIdx.x * blockDim.x + t] = acc;
}

// 4. FP32 Vector ALU Roofline (8 independent FMA chains hide ~4-cycle pipeline latency)
__global__ void k_fp32_valu(float* out, float a, float b) {
    float c0=0.f, c1=1.f, c2=2.f, c3=3.f, c4=4.f, c5=5.f, c6=6.f, c7=7.f;
    #pragma unroll 8
    for (int i = 0; i < 8192; ++i) {
        c0 = a * c0 + b; c1 = a * c1 + b; c2 = a * c2 + b; c3 = a * c3 + b;
        c4 = a * c4 + b; c5 = a * c5 + b; c6 = a * c6 + b; c7 = a * c7 + b;
    }
    float s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    if (s == -999999.0f) out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// 4B. FP32 SFU / Transcendental Roofline (exp2f + rsqrtf — realistic Softmax/RMSNorm ceiling)
__global__ void k_fp32_sfu(float* out, float a) {
    float c0=1.1f, c1=1.2f, c2=1.3f, c3=1.4f;
    #pragma unroll 4
    for (int i = 0; i < 2048; ++i) {
        c0 = exp2f(a * c0) + rsqrtf(c0 + 1.0f);
        c1 = exp2f(a * c1) + rsqrtf(c1 + 1.0f);
        c2 = exp2f(a * c2) + rsqrtf(c2 + 1.0f);
        c3 = exp2f(a * c3) + rsqrtf(c3 + 1.0f);
    }
    float s = c0 + c1 + c2 + c3;
    if (s == -999999.0f) out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// 5A. BF16 Matrix Core (AMD MFMA) / Tensor Core (NVIDIA WMMA) Roofline
//     Every architecture branch below executes exactly 268,435,456 FLOPs per wave/warp!
//     Uses non-constant, alternating bit patterns (~50% Hamming toggle) to trigger
//     realistic TDP power-limit clocks rather than zero/constant-input boost clocks.
__global__ void k_bf16_matrix(float* out, int* supported) {
#if defined(__gfx950__)
    // AMD CDNA4 (MI350X / MI355X): 16x16x32 BF16 MFMA = 16,384 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
    using fp32x4 = float  __attribute__((ext_vector_type(4)));
    float s0 = 0.73f + 0.001f * (float)(threadIdx.x & 31);
    float s1 = -0.61f - 0.001f * (float)((threadIdx.x + 7) & 31);
    bf16x8 a0 = {(__bf16)s0, (__bf16)-s1, (__bf16)s0, (__bf16)s1, (__bf16)-s0, (__bf16)s1, (__bf16)s0, (__bf16)-s1};
    bf16x8 a1 = {(__bf16)-s1, (__bf16)s0, (__bf16)-s0, (__bf16)s1, (__bf16)s1, (__bf16)-s0, (__bf16)-s1, (__bf16)s0};
    bf16x8 b0 = {(__bf16)s1, (__bf16)s0, (__bf16)-s1, (__bf16)s0, (__bf16)s0, (__bf16)-s1, (__bf16)s1, (__bf16)s0};
    bf16x8 b1 = {(__bf16)-s0, (__bf16)s1, (__bf16)s0, (__bf16)-s1, (__bf16)-s1, (__bf16)s0, (__bf16)s0, (__bf16)-s1};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 4096; ++i) {
        bf16x8 aa = (i & 1) ? a1 : a0;
        bf16x8 bb = (i & 1) ? b1 : b0;
        d0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(aa, bb, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(aa, bb, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(aa, bb, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(aa, bb, d3, 0, 0, 0);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__gfx942__) || defined(__gfx90a__)
    // AMD CDNA2/CDNA3 (MI250X / MI300X / MI325X): 16x16x16 BF16 MFMA = 8,192 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using bf16x4_raw = short __attribute__((ext_vector_type(4)));
    using fp32x4     = float __attribute__((ext_vector_type(4)));
    short p0 = (short)(0x3c5a ^ ((threadIdx.x & 15) * 0x0111));
    short p1 = (short)(0xbd3c ^ ((threadIdx.x & 15) * 0x0221));
    bf16x4_raw a0 = {p0, p1, (short)(p0 ^ 0x1234), (short)(p1 ^ 0x2412)};
    bf16x4_raw a1 = {(short)(~p0 & 0x7fff), p1, p0, (short)(~p1 & 0x7fff)};
    bf16x4_raw b0 = {p1, p0, (short)(p1 ^ 0x3121), (short)(p0 ^ 0x1543)};
    bf16x4_raw b1 = {p0, (short)(~p1 & 0x7fff), p1, (short)(~p0 & 0x7fff)};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 8192; ++i) {
        bf16x4_raw aa = (i & 1) ? a1 : a0;
        bf16x4_raw bb = (i & 1) ? b1 : b0;
        d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aa, bb, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aa, bb, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aa, bb, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(aa, bb, d3, 0, 0, 0);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
    // NVIDIA Ampere / Ada / Hopper / Blackwell: 16x16x16 BF16 WMMA = 8,192 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d0, d1, d2, d3;
    float v0 = 0.73f + 0.002f * (float)(threadIdx.x & 31);
    float v1 = -0.61f - 0.002f * (float)(threadIdx.x & 31);
    wmma::fill_fragment(a0, __float2bfloat16(v0));
    wmma::fill_fragment(a1, __float2bfloat16(v1));
    wmma::fill_fragment(b0, __float2bfloat16(v1));
    wmma::fill_fragment(b1, __float2bfloat16(v0));
    wmma::fill_fragment(d0, 0.0f); wmma::fill_fragment(d1, 0.0f);
    wmma::fill_fragment(d2, 0.0f); wmma::fill_fragment(d3, 0.0f);
    for (int i = 0; i < 8192; ++i) {
        if (i & 1) {
            wmma::mma_sync(d0, a1, b1, d0); wmma::mma_sync(d1, a1, b1, d1);
            wmma::mma_sync(d2, a1, b1, d2); wmma::mma_sync(d3, a1, b1, d3);
        } else {
            wmma::mma_sync(d0, a0, b0, d0); wmma::mma_sync(d1, a0, b0, d1);
            wmma::mma_sync(d2, a0, b0, d2); wmma::mma_sync(d3, a0, b0, d3);
        }
    }
    if (d0.x[0] + d1.x[0] + d2.x[0] + d3.x[0] == -999999.0f) out[threadIdx.x] = d0.x[0];
#else
    (void)out;
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 0;
#endif
}

// 5B. FP8 Matrix / Tensor Core Roofline (268,435,456 FLOPs per wave/warp)
__global__ void k_fp8_matrix(float* out, int* supported) {
#if defined(__gfx950__)
    // AMD CDNA4: 16x16x128 scaled f8f6f4 with cbsz=0, blgp=0 (FP8 E4M3) = 65,536 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using i8x8   = int   __attribute__((ext_vector_type(8)));
    using fp32x4 = float __attribute__((ext_vector_type(4)));
    int t = (threadIdx.x * 0x01010101) & 0x3f3f3f3f;
    int p0 = (int)(0x38b42c9e ^ t), p1 = (int)(0xb236a81c ^ t);
    i8x8 a0 = {p0, p1, p0 ^ 0x11223344, p1 ^ 0x44332211, p0, p1, p0 ^ 0x050a050a, p1 ^ 0x0a050a05};
    i8x8 a1 = {p1, p0, p1 ^ 0x22114433, p0 ^ 0x33441122, p1, p0, p1 ^ 0x030c030c, p0 ^ 0x0c030c03};
    i8x8 b0 = a1, b1 = a0;
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 1024; ++i) {
        i8x8 aa = (i & 1) ? a1 : a0;
        i8x8 bb = (i & 1) ? b1 : b0;
        d0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d0, 0, 0, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d1, 0, 0, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d2, 0, 0, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d3, 0, 0, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__gfx942__)
    // AMD CDNA3 (MI300X / MI325X): 16x16x32 FP8 MFMA = 16,384 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using fp32x4 = float __attribute__((ext_vector_type(4)));
    long t = (long)(threadIdx.x & 63) * 0x0101010101010101LL;
    long a0 = (0x38b42c9eb236a81cLL ^ t) & 0x7f7f7f7f7f7f7f7fLL;
    long a1 = (0xa428b61e349c2e14LL ^ t) & 0x7f7f7f7f7f7f7f7fLL;
    long b0 = a1, b1 = a0;
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 4096; ++i) {
        long aa = (i & 1) ? a1 : a0;
        long bb = (i & 1) ? b1 : b0;
        d0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(aa, bb, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(aa, bb, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(aa, bb, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(aa, bb, d3, 0, 0, 0);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__CUDACC__) && (__CUDA_ARCH__ >= 890)
    // NVIDIA Ada / Hopper / Blackwell: m16n8k32 FP8 e4m3 MMA = 8,192 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    unsigned t = (threadIdx.x & 31) * 0x01010101u;
    unsigned a0=0x38b42c9eu^t, a1=0xb236a81cu^t, a2=0x349c2e14u^t, a3=0xa428b61eu^t;
    unsigned b0=0x2c9e38b4u^t, b1=0xa81cb236u^t;
    float d0[4]={0,0,0,0}, d1[4]={0,0,0,0}, d2[4]={0,0,0,0}, d3[4]={0,0,0,0};
    for (int i = 0; i < 8192; ++i) {
        unsigned x0 = (i & 1) ? a1 : a0, x1 = (i & 1) ? a3 : a2;
        unsigned y0 = (i & 1) ? b1 : b0, y1 = (i & 1) ? b0 : b1;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d0[0]),"+f"(d0[1]),"+f"(d0[2]),"+f"(d0[3])
                     : "r"(x0),"r"(x1),"r"(a2),"r"(a3),"r"(y0),"r"(y1));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d1[0]),"+f"(d1[1]),"+f"(d1[2]),"+f"(d1[3])
                     : "r"(x0),"r"(x1),"r"(a2),"r"(a3),"r"(y0),"r"(y1));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d2[0]),"+f"(d2[1]),"+f"(d2[2]),"+f"(d2[3])
                     : "r"(x0),"r"(x1),"r"(a2),"r"(a3),"r"(y0),"r"(y1));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                     : "+f"(d3[0]),"+f"(d3[1]),"+f"(d3[2]),"+f"(d3[3])
                     : "r"(x0),"r"(x1),"r"(a2),"r"(a3),"r"(y0),"r"(y1));
    }
    if (d0[0]+d1[0]+d2[0]+d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#else
    (void)out;
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 0;
#endif
}

// 5C. FP4 / MXFP4 Matrix Core Roofline (268,435,456 FLOPs per wave)
__global__ void k_fp4_matrix(float* out, int* supported) {
#if defined(__gfx950__)
    // AMD CDNA4 (MI350X / MI355X): 16x16x128 scaled f8f6f4 with cbsz=4, blgp=4 (FP4 E2M1) = 65,536 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using i8x8   = int   __attribute__((ext_vector_type(8)));
    using fp32x4 = float __attribute__((ext_vector_type(4)));
    int t = (threadIdx.x * 0x11111111);
    i8x8 a0 = {(int)(0x2a3b4c5du^t), (int)(0x9e8f1234u^t), (int)(0x5678abcd^t), (int)(0xdcba8765^t), 0,0,0,0};
    i8x8 a1 = {(int)(0x9e8f1234u^t), (int)(0x2a3b4c5du^t), (int)(0xdcba8765^t), (int)(0x5678abcd^t), 0,0,0,0};
    i8x8 b0 = a1, b1 = a0;
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 1024; ++i) {
        i8x8 aa = (i & 1) ? a1 : a0;
        i8x8 bb = (i & 1) ? b1 : b0;
        d0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d0, 4, 4, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d1, 4, 4, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d2, 4, 4, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
        d3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(aa, bb, d3, 4, 4, 0, 0x7f7f7f7f, 0, 0x7f7f7f7f);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#else
    (void)out;
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 0;
#endif
}

// 6. Minimal kernel for measuring dispatch overhead
__global__ void k_noop(float* out) {
    if (threadIdx.x == 0) out[0] += 1.0f;
}

template <typename Fn>
float bench_gpu_ms(int reps, Fn fn) {
    fn(); // warmup
    GPU_CHECK(gpuDeviceSynchronize());
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuEventCreate(&t0));
    GPU_CHECK(gpuEventCreate(&t1));
    GPU_CHECK(gpuEventRecord(t0));
    for (int i = 0; i < reps; ++i) fn();
    GPU_CHECK(gpuEventRecord(t1));
    GPU_CHECK(gpuEventSynchronize(t1));
    float ms = 0.0f;
    GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
    GPU_CHECK(gpuEventDestroy(t0));
    GPU_CHECK(gpuEventDestroy(t1));
    return ms / reps;
}

static void format_bytes(size_t bytes, char* buf, size_t cap) {
    if (bytes >= (1ULL << 30))      snprintf(buf, cap, "%4zu GiB", bytes >> 30);
    else if (bytes >= (1ULL << 20)) snprintf(buf, cap, "%4zu MiB", bytes >> 20);
    else                            snprintf(buf, cap, "%4zu KiB", bytes >> 10);
}

int main(int argc, char** argv) {
    const char* csv_path  = nullptr;
    const char* json_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        }
    }

    int dev_count = 0;
    GPU_CHECK(gpuGetDeviceCount(&dev_count));
    GPU_CHECK(gpuSetDevice(0));

    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;
    double l2_mib = prop.l2CacheSize / (1024.0 * 1024.0);
    double vram_gib = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);

    printf("==========================================================================\n");
    printf("  GPU Roofline Probe (%s)\n", GPU_BACKEND_NAME);
    printf("  Device 0 : %s  (%d GPUs visible)\n", prop.name, dev_count);
    printf("  Arch     : %d SMs/CUs | %.1f GiB HBM/VRAM | %.1f MiB L2 Cache | Wave=%d\n",
           sms, vram_gib, l2_mib, GPU_WAVE_SIZE);
    printf("==========================================================================\n\n");

    const size_t HBM_BYTES = 1ULL * 1024 * 1024 * 1024; // 1 GiB (2 GiB R+W >> L2)
    const size_t STRIDE_B  = 256ULL * 1024 * 1024;      // 256 MiB power-of-2
    const size_t L2_BYTES  = std::max<size_t>(4ULL, (size_t)(l2_mib * 0.25)) * 1024 * 1024;

    float4 *d_x = nullptr, *d_y = nullptr;
    int* d_sup = nullptr;
    GPU_CHECK(gpuMalloc(&d_x, HBM_BYTES));
    GPU_CHECK(gpuMalloc(&d_y, HBM_BYTES));
    GPU_CHECK(gpuMalloc(&d_sup, sizeof(int)));
    GPU_CHECK(gpuMemset(d_x, 1, HBM_BYTES));
    GPU_CHECK(gpuMemset(d_y, 0, HBM_BYTES));
    GPU_CHECK(gpuMemset(d_sup, 0, sizeof(int)));

    int copy_blocks = sms * 16;

    // [1] HBM Bandwidth (Coalesced R+W, Read-Only Streaming, vs Strided)
    float hbm_ms  = bench_gpu_ms(20, [&]{ k_copy_vec4<<<copy_blocks, 256>>>(d_x, d_y, HBM_BYTES / 16); });
    float read_ms = bench_gpu_ms(20, [&]{ k_read_vec4<<<copy_blocks, 256>>>(d_x, (float*)d_y, HBM_BYTES / 16); });
    float bad_ms  = bench_gpu_ms(10, [&]{ k_copy_strided<<<copy_blocks, 256>>>((float*)d_x, (float*)d_y, STRIDE_B / 4); });
    double hbm_tbs      = (2.0 * HBM_BYTES) / (hbm_ms * 1e-3) / 1e12;
    double hbm_read_tbs = (1.0 * HBM_BYTES) / (read_ms * 1e-3) / 1e12;
    double bad_tbs      = (2.0 * STRIDE_B)  / (bad_ms * 1e-3) / 1e12;

    // [2] L2 Cache Bandwidth
    float l2_ms = bench_gpu_ms(200, [&]{ k_copy_vec4<<<copy_blocks, 256>>>(d_x, d_y, L2_BYTES / 16); });
    double l2_tbs = (2.0 * L2_BYTES) / (l2_ms * 1e-3) / 1e12;

    // [3] On-Chip Shared Memory / LDS
    float lds_ms = bench_gpu_ms(50, [&]{ k_shared_bw<<<sms * 4, 256>>>(d_y); });
    double lds_tbs = ((double)(sms * 4) * 256.0 * 1024.0 * 16.0) / (lds_ms * 1e-3) / 1e12;

    // [4] FP32 Vector ALU & SFU (Transcendental) Compute
    float valu_ms = bench_gpu_ms(10, [&]{ k_fp32_valu<<<sms * 8, 256>>>((float*)d_y, 1.00001f, 0.00001f); });
    double valu_tflops = ((double)(sms * 8) * 256.0 * 8192.0 * 16.0) / (valu_ms * 1e-3) / 1e12;
    float sfu_ms = bench_gpu_ms(10, [&]{ k_fp32_sfu<<<sms * 8, 256>>>((float*)d_y, 0.0001f); });
    double sfu_tops = ((double)(sms * 8) * 256.0 * 2048.0 * 8.0) / (sfu_ms * 1e-3) / 1e12;

    // [5] Matrix / Tensor Core Compute across Precisions (BF16, FP8, FP4)
    gpuStream_t tmp_s;
    GPU_CHECK(gpuStreamCreate(&tmp_s));
    double waves = (double)(sms * 8) * (256.0 / GPU_WAVE_SIZE);

    auto probe_matrix = [&](auto launch_fn, int& sup_out) -> double {
        GPU_CHECK(gpuMemset(d_sup, 0, sizeof(int)));
        // Warmup soak so power board settles at sustained TDP clock rather than boost
        for (int w = 0; w < 6; ++w) launch_fn();
        GPU_CHECK(gpuDeviceSynchronize());
        float ms = bench_gpu_ms(20, launch_fn);
        sup_out = 0;
        GPU_CHECK(gpuMemcpyAsync(&sup_out, d_sup, sizeof(int), gpuMemcpyDeviceToHost, tmp_s));
        GPU_CHECK(gpuStreamSynchronize(tmp_s));
        return sup_out ? ((waves * 268435456.0) / (ms * 1e-3)) / 1e12 : 0.0;
    };

    int host_sup = 0, fp8_sup = 0, fp4_sup = 0;
    double mat_tflops = probe_matrix([&]{ k_bf16_matrix<<<sms * 8, 256>>>((float*)d_y, d_sup); }, host_sup);
    double fp8_tflops = probe_matrix([&]{ k_fp8_matrix <<<sms * 8, 256>>>((float*)d_y, d_sup); }, fp8_sup);
    double fp4_tflops = probe_matrix([&]{ k_fp4_matrix <<<sms * 8, 256>>>((float*)d_y, d_sup); }, fp4_sup);

    // [6] Kernel Launch Latency: CPU Async Enqueue vs Eager GPU vs Graph Replay
    const int N_LAUNCH = 1000;
    k_copy_vec4<<<copy_blocks, 256, 0, tmp_s>>>(d_x, d_y, HBM_BYTES / 16);
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_LAUNCH; ++i) k_noop<<<1, 64, 0, tmp_s>>>((float*)d_y);
    auto c1 = std::chrono::high_resolution_clock::now();
    GPU_CHECK(gpuStreamSynchronize(tmp_s));
    double cpu_enq_us = std::chrono::duration<double, std::micro>(c1 - c0).count() / N_LAUNCH;

    float eager_total_ms = bench_gpu_ms(1, [&]{
        for (int i = 0; i < N_LAUNCH; ++i) k_noop<<<1, 64>>>((float*)d_y);
    });
    double eager_us = (eager_total_ms * 1000.0) / N_LAUNCH;

    gpuGraph_t graph;
    gpuGraphExec_t graph_exec;
    GPU_CHECK(gpuStreamBeginCapture(tmp_s, gpuStreamCaptureModeGlobal));
    for (int i = 0; i < N_LAUNCH; ++i) k_noop<<<1, 64, 0, tmp_s>>>((float*)d_y);
    GPU_CHECK(gpuStreamEndCapture(tmp_s, &graph));
    gpuGraphInit(&graph_exec, graph);
    GPU_CHECK(gpuGraphLaunch(graph_exec, tmp_s));
    GPU_CHECK(gpuStreamSynchronize(tmp_s));

    gpuEvent_t e0, e1;
    GPU_CHECK(gpuEventCreate(&e0)); GPU_CHECK(gpuEventCreate(&e1));
    GPU_CHECK(gpuEventRecord(e0, tmp_s));
    for (int r = 0; r < 10; ++r) GPU_CHECK(gpuGraphLaunch(graph_exec, tmp_s));
    GPU_CHECK(gpuEventRecord(e1, tmp_s));
    GPU_CHECK(gpuEventSynchronize(e1));
    float graph_ms = 0.0f;
    GPU_CHECK(gpuEventElapsedTime(&graph_ms, e0, e1));
    double graph_us = (graph_ms * 1000.0) / (10.0 * N_LAUNCH);

    // [7] Host Pinned PCIe Bandwidth (256 MiB H2D & D2H)
    const size_t PCIE_BYTES = 256ULL * 1024 * 1024;
    void* h_pinned = nullptr;
    gpuHostMalloc(&h_pinned, PCIE_BYTES);
    float h2d_ms = bench_gpu_ms(10, [&]{
        GPU_CHECK(gpuMemcpyAsync(d_y, h_pinned, PCIE_BYTES, gpuMemcpyHostToDevice, tmp_s));
        GPU_CHECK(gpuStreamSynchronize(tmp_s));
    });
    float d2h_ms = bench_gpu_ms(10, [&]{
        GPU_CHECK(gpuMemcpyAsync(h_pinned, d_x, PCIE_BYTES, gpuMemcpyDeviceToHost, tmp_s));
        GPU_CHECK(gpuStreamSynchronize(tmp_s));
    });
    double h2d_gbs = PCIE_BYTES / (h2d_ms * 1e-3) / 1e9;
    double d2h_gbs = PCIE_BYTES / (d2h_ms * 1e-3) / 1e9;
    gpuHostFree(h_pinned);

    // [8] Optional Peer-to-Peer (NVLink / xGMI): 1-to-1 AND All-GPU Concurrent Ring
    double p2p_gbs = 0.0;
    double p2p_all_gbs = 0.0;
    int can_peer = 0;
    if (dev_count >= 2) {
        GPU_CHECK(gpuDeviceCanAccessPeer(&can_peer, 0, 1));
        if (can_peer) {
            GPU_CHECK(gpuSetDevice(1));
            float4* d_peer = nullptr;
            GPU_CHECK(gpuMalloc(&d_peer, STRIDE_B));
            GPU_CHECK(gpuMemset(d_peer, 1, STRIDE_B));
            GPU_CHECK(gpuDeviceSynchronize());
            GPU_CHECK(gpuSetDevice(0));
            GPU_CHECK(gpuDeviceEnablePeerAccess(1, 0));
            float p2p_ms = bench_gpu_ms(20, [&]{
                k_copy_vec4<<<copy_blocks, 256>>>(d_peer, d_y, STRIDE_B / 16);
            });
            p2p_gbs = STRIDE_B / (p2p_ms * 1e-3) / 1e9;
            GPU_CHECK(gpuSetDevice(1));
            GPU_CHECK(gpuFree(d_peer));
            GPU_CHECK(gpuSetDevice(0));

            // [8b] Full-participation ring P2P across all visible GPUs (catches no-NVSwitch mesh drop)
            int n_g = std::min(dev_count, 16);
            float4* r_src[16] = {};
            float4* r_dst[16] = {};
            gpuStream_t r_stream[16] = {};
            bool ring_ok = true;
            for (int g = 0; g < n_g; ++g) {
                int peer = (g + 1) % n_g;
                int cp = 0;
                GPU_CHECK(gpuDeviceCanAccessPeer(&cp, g, peer));
                if (!cp) { ring_ok = false; break; }
            }
            if (ring_ok && n_g >= 2) {
                const size_t RING_B = 128ULL * 1024 * 1024;
                for (int g = 0; g < n_g; ++g) {
                    int peer = (g + 1) % n_g;
                    GPU_CHECK(gpuSetDevice(g));
                    GPU_CHECK(gpuStreamCreate(&r_stream[g]));
                    GPU_CHECK(gpuMalloc(&r_src[g], RING_B));
                    GPU_CHECK(gpuMalloc(&r_dst[g], RING_B));
                    GPU_CHECK(gpuMemset(r_src[g], 1, RING_B));
                    if (!(g == 0 && peer == 1)) {
                        // Ignore error if peer access was already enabled
                        (void)gpuDeviceEnablePeerAccess(peer, 0);
                    }
                    GPU_CHECK(gpuDeviceSynchronize());
                }
                for (int w = 0; w < 2; ++w) {
                    for (int g = 0; g < n_g; ++g) {
                        int peer = (g + 1) % n_g;
                        GPU_CHECK(gpuSetDevice(g));
                        k_copy_vec4<<<copy_blocks, 256, 0, r_stream[g]>>>(r_src[peer], r_dst[g], RING_B / 16);
                    }
                    for (int g = 0; g < n_g; ++g) {
                        GPU_CHECK(gpuSetDevice(g));
                        GPU_CHECK(gpuStreamSynchronize(r_stream[g]));
                    }
                }
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int r = 0; r < 10; ++r) {
                    for (int g = 0; g < n_g; ++g) {
                        int peer = (g + 1) % n_g;
                        GPU_CHECK(gpuSetDevice(g));
                        k_copy_vec4<<<copy_blocks, 256, 0, r_stream[g]>>>(r_src[peer], r_dst[g], RING_B / 16);
                    }
                }
                for (int g = 0; g < n_g; ++g) {
                    GPU_CHECK(gpuSetDevice(g));
                    GPU_CHECK(gpuStreamSynchronize(r_stream[g]));
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ring_s = std::chrono::duration<double>(t1 - t0).count() / 10.0;
                p2p_all_gbs = (double)RING_B / ring_s / 1e9; // per-GPU unidirectional bus BW under full participation
                for (int g = 0; g < n_g; ++g) {
                    GPU_CHECK(gpuSetDevice(g));
                    GPU_CHECK(gpuStreamDestroy(r_stream[g]));
                    GPU_CHECK(gpuFree(r_src[g]));
                    GPU_CHECK(gpuFree(r_dst[g]));
                }
            }
            GPU_CHECK(gpuSetDevice(0));
        }
    }

    printf("--- Memory Hierarchy -----------------------------------------------------\n");
    printf("[1a] HBM Bandwidth (Coalesced R+W)  : %7.2f TB/s  (2.0 GiB R+W)\n", hbm_tbs);
    printf("[1b] HBM Bandwidth (Read-Only)      : %7.2f TB/s  (1.0 GiB weight-stream upper bound)\n", hbm_read_tbs);
    printf("[1c] HBM Bandwidth (Uncoalesced)    : %7.2f TB/s  (%.1fx cache-line waste)\n", bad_tbs, hbm_tbs / bad_tbs);
    printf("[2]  L2 Cache Bandwidth             : %7.2f TB/s  (%zu MiB R+W in %.0f MiB L2)\n",
           l2_tbs, (2 * L2_BYTES) / (1024 * 1024), l2_mib);
    printf("[3]  Shared Memory / LDS Bandwidth  : %7.2f TB/s  (on-chip CU/SM scratchpad)\n\n", lds_tbs);

    printf("--- Compute Ceilings -----------------------------------------------------\n");
    printf("[4a] FP32 Vector ALU                : %7.1f TFLOP/s\n", valu_tflops);
    printf("[4b] FP32 SFU (exp2/rsqrt)          : %7.1f TOP/s   (Softmax / RMSNorm / SiLU ceiling)\n", sfu_tops);
    if (host_sup) {
        printf("[5a] BF16 Matrix / Tensor Cores     : %7.1f TFLOP/s  (%.1fx over Vector ALU)\n",
               mat_tflops, mat_tflops / valu_tflops);
    } else {
        printf("[5a] BF16 Matrix / Tensor Cores     : [compile with native arch flag]\n");
    }
    if (fp8_sup) {
        printf("[5b] FP8  Matrix / Tensor Cores     : %7.1f TFLOP/s  (%.2fx BF16)\n",
               fp8_tflops, host_sup ? (fp8_tflops / mat_tflops) : 0.0);
    }
    if (fp4_sup) {
        printf("[5c] FP4 / MXFP4 Matrix Cores       : %7.1f TFLOP/s  (%.2fx BF16)\n",
               fp4_tflops, host_sup ? (fp4_tflops / mat_tflops) : 0.0);
    }
    printf("\n");

    printf("--- Dispatch & Interconnects ---------------------------------------------\n");
    printf("[6a] CPU Async Enqueue Cost         : %7.2f us / launch\n", cpu_enq_us);
    printf("[6b] GPU Eager Launch Floor         : %7.2f us / launch\n", eager_us);
    printf("[6c] CUDA/HIP Graph Launch Floor    : %7.2f us / launch  (%.1fx lower gap)\n",
           graph_us, eager_us / std::max(graph_us, 0.01));
    printf("[7]  Pinned PCIe Host <-> GPU       : %7.1f GB/s H2D | %.1f GB/s D2H\n", h2d_gbs, d2h_gbs);
    if (can_peer) {
        printf("[8a] GPU0 <- GPU1 P2P (1-to-1)     : %7.1f GB/s unidirectional read\n", p2p_gbs);
        if (p2p_all_gbs > 0.0) {
            printf("[8b] All-GPU Ring P2P (%d GPUs)    : %7.1f GB/s per-GPU bus BW\n", std::min(dev_count, 16), p2p_all_gbs);
        }
    } else {
        printf("[8]  GPU0 <- GPU1 P2P (NVLink/xGMI) : [run with >=2 GPUs to measure]\n");
    }

    // Full Latency-Bound -> L2 Cache Peak -> HBM Bandwidth-Bound sweep (4 KiB .. 2 GiB R+W)
    const size_t SWEEP_RW_BYTES[] = {
        4ULL << 10,   16ULL << 10,  64ULL << 10,  256ULL << 10,
        1ULL << 20,   4ULL << 20,   16ULL << 20,  64ULL << 20,
        128ULL << 20, 256ULL << 20, 512ULL << 20, 1024ULL << 20, 2048ULL << 20
    };
    constexpr int N_SWEEP = sizeof(SWEEP_RW_BYTES) / sizeof(SWEEP_RW_BYTES[0]);
    double sweep_us[N_SWEEP];
    double sweep_tbs[N_SWEEP];
    double max_tbs = 0.01;

    for (int s = 0; s < N_SWEEP; ++s) {
        size_t rw_b  = SWEEP_RW_BYTES[s];
        size_t buf_b = rw_b / 2;
        size_t nv4   = std::max<size_t>(1, buf_b / 16);
        int blk      = (int)std::min<size_t>((nv4 + 255) / 256, (size_t)copy_blocks);
        int reps     = (rw_b <= (64ULL << 20)) ? 300 : 20;
        float ms     = bench_gpu_ms(reps, [&]{ k_copy_vec4<<<blk, 256>>>(d_x, d_y, nv4); });
        sweep_us[s]  = ms * 1000.0;
        sweep_tbs[s] = rw_b / (ms * 1e-3) / 1e12;
        max_tbs      = std::max(max_tbs, sweep_tbs[s]);
    }

    printf("\n--- Transfer Size Sweep: Latency-Bound -> L2 Cache Peak -> HBM Plateau ---\n");
    printf("  %-10s | %8s | %9s | %-28s | Regime\n", "R+W Size", "Latency", "Bandwidth", "Relative Throughput");
    printf("  -----------+----------+-----------+------------------------------+-----------------\n");
    for (int s = 0; s < N_SWEEP; ++s) {
        char sz[32];
        format_bytes(SWEEP_RW_BYTES[s], sz, sizeof(sz));
        int bars = (int)((sweep_tbs[s] / max_tbs) * 28.0 + 0.5);
        char bar_str[32];
        for (int b = 0; b < 28; ++b) bar_str[b] = (b < bars) ? '#' : '.';
        bar_str[28] = '\0';

        const char* regime = "HBM BW-Bound";
        if (sweep_us[s] < eager_us * 3.0) {
            regime = "Latency-Bound";
        } else if (SWEEP_RW_BYTES[s] <= (size_t)(l2_mib * 1024.0 * 1024.0 * 0.9)) {
            regime = "L2 Cache Peak";
        }
        printf("  %-10s | %6.1f us | %6.2f TB/s | [%s] | %s\n",
               sz, sweep_us[s], sweep_tbs[s], bar_str, regime);
    }

    if (csv_path != nullptr) {
        FILE* f = std::fopen(csv_path, "w");
        if (f) {
            std::fprintf(f, "device,rw_bytes,latency_us,bandwidth_tbs,l2_mib,hbm_tbs,bf16_tflops\n");
            for (int s = 0; s < N_SWEEP; ++s) {
                std::fprintf(f, "\"%s\",%zu,%.3f,%.5f,%.1f,%.4f,%.2f\n",
                             prop.name, SWEEP_RW_BYTES[s], sweep_us[s], sweep_tbs[s],
                             l2_mib, hbm_tbs, mat_tflops);
            }
            std::fclose(f);
            printf("\nSaved CSV sweep to: %s\n", csv_path);
        }
    }

    if (json_path != nullptr) {
        FILE* jf = std::fopen(json_path, "w");
        if (jf) {
            std::fprintf(jf,
                "{\n"
                "  \"device\": \"%s\",\n"
                "  \"backend\": \"%s\",\n"
                "  \"visible_gpus\": %d,\n"
                "  \"sms_or_cus\": %d,\n"
                "  \"vram_gib\": %.2f,\n"
                "  \"l2_mib\": %.2f,\n"
                "  \"wave_size\": %d,\n"
                "  \"hbm_tbs\": %.4f,\n"
                "  \"hbm_read_tbs\": %.4f,\n"
                "  \"hbm_uncoalesced_tbs\": %.4f,\n"
                "  \"l2_tbs\": %.4f,\n"
                "  \"shared_lds_tbs\": %.4f,\n"
                "  \"fp32_valu_tflops\": %.2f,\n"
                "  \"fp32_sfu_tops\": %.2f,\n"
                "  \"bf16_matrix_tflops\": %.2f,\n"
                "  \"fp8_matrix_tflops\": %.2f,\n"
                "  \"fp4_matrix_tflops\": %.2f,\n"
                "  \"cpu_enqueue_us\": %.3f,\n"
                "  \"eager_launch_us\": %.3f,\n"
                "  \"graph_launch_us\": %.3f,\n"
                "  \"pcie_h2d_gbs\": %.2f,\n"
                "  \"pcie_d2h_gbs\": %.2f,\n"
                "  \"p2p_read_gbs\": %.2f,\n"
                "  \"p2p_all_ring_gbs\": %.2f\n"
                "}\n",
                prop.name, GPU_BACKEND_NAME, dev_count, sms, vram_gib, l2_mib,
                GPU_WAVE_SIZE, hbm_tbs, hbm_read_tbs, bad_tbs, l2_tbs, lds_tbs,
                valu_tflops, sfu_tops, mat_tflops, fp8_tflops, fp4_tflops,
                cpu_enq_us, eager_us, graph_us, h2d_gbs, d2h_gbs, p2p_gbs, p2p_all_gbs);
            std::fclose(jf);
            printf("Saved empirical hardware JSON to: %s\n", json_path);
        }
    }

    if (host_sup) {
        double ridge = (mat_tflops * 1e12) / (hbm_read_tbs * 1e12);
        double bf16_batch_knee = ridge * 1.0;
        printf("\n--- LLM Inference Roofline Implications ----------------------------------\n");
        printf("  Empirical Ridge Points (vs Read-Only HBM %.2f TB/s):\n", hbm_read_tbs);
        printf("    * BF16 : %4.0f FLOPs/B  -> Linear Batch Knee M ~= %.0f tokens\n", ridge, bf16_batch_knee);
        if (fp8_sup) {
            double r8 = (fp8_tflops * 1e12) / (hbm_read_tbs * 1e12);
            printf("    * FP8  : %4.0f FLOPs/B  -> Linear Batch Knee M ~= %.0f tokens\n", r8, r8 * 0.5);
        }
        if (fp4_sup) {
            double r4 = (fp4_tflops * 1e12) / (hbm_read_tbs * 1e12);
            printf("    * FP4  : %4.0f FLOPs/B  -> Linear Batch Knee M ~= %.0f tokens\n", r4, r4 * 0.25);
        }
        printf("    * Decode  (M << knee) : HBM-Bandwidth-Bound -> speed scales with weight compression & SpecDec\n");
        printf("    * Prefill (M >= knee) : Matrix-Core-Bound   -> speed scales with TFLOP/s & Flash/Sparse Attn\n");
        printf("    * Tiny BS (bs = 1..4) : 400 eager launches cost ~%.2f ms vs ~%.2f ms with Graphs!\n",
               (400.0 * eager_us) / 1000.0, (400.0 * graph_us) / 1000.0);
    }
    printf("==========================================================================\n");

    GPU_CHECK(gpuEventDestroy(e0));
    GPU_CHECK(gpuEventDestroy(e1));
    GPU_CHECK(gpuGraphExecDestroy(graph_exec));
    GPU_CHECK(gpuGraphDestroy(graph));
    GPU_CHECK(gpuStreamDestroy(tmp_s));
    GPU_CHECK(gpuFree(d_sup));
    GPU_CHECK(gpuFree(d_x));
    GPU_CHECK(gpuFree(d_y));
    return 0;
}
