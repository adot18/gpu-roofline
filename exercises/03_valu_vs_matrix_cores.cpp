// Exercise 03: Compute Rooflines — Latency-Stalled vs ILP Vector ALU vs Matrix/Tensor Cores
// Shows:
//   1. Why 1 dependent FMA chain (`c = a * c + b`) stalls ~75% of cycles on ALU pipeline latency
//   2. How 8 independent accumulators (`c0..c7`) unlock ~100% of FP32 Vector ALU throughput
//   3. Why BF16 Matrix/Tensor Cores deliver ~8-16x higher TFLOP/s than Vector ALUs

#include "gpu_compat.h"

__global__ void valu_1chain_stalled(float* out, float a, float b) {
    float c0 = 1.f;
    for (int i = 0; i < 65536; ++i) {
        c0 = a * c0 + b; // Dependent chain: next FMA waits ~4 cycles on previous result!
    }
    if (c0 == -999999.f) out[threadIdx.x] = c0;
}

__global__ void valu_8chain_ilp(float* out, float a, float b) {
    float c0=0, c1=1, c2=2, c3=3, c4=4, c5=5, c6=6, c7=7;
    for (int i = 0; i < 8192; ++i) {
        c0 = a * c0 + b; c1 = a * c1 + b; c2 = a * c2 + b; c3 = a * c3 + b;
        c4 = a * c4 + b; c5 = a * c5 + b; c6 = a * c6 + b; c7 = a * c7 + b;
    }
    float s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    if (s == -999999.f) out[threadIdx.x] = s;
}

__global__ void bf16_matrix_core(float* out) {
#if defined(__gfx950__)
    using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
    using fp32x4 = float  __attribute__((ext_vector_type(4)));
    bf16x8 a={1,1,1,1,1,1,1,1}, b={1,1,1,1,1,1,1,1};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 4096; ++i) {
        d0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d3, 0, 0, 0);
    }
    if (d0[0]+d1[0]+d2[0]+d3[0] == -999999.f) out[threadIdx.x] = d0[0];
#elif defined(__gfx942__) || defined(__gfx90a__)
    using bf16x4_raw = short __attribute__((ext_vector_type(4)));
    using fp32x4     = float __attribute__((ext_vector_type(4)));
    bf16x4_raw a={0x3f80,0x3f80,0x3f80,0x3f80}, b={0x3f80,0x3f80,0x3f80,0x3f80};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 8192; ++i) {
        d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d3, 0, 0, 0);
    }
    if (d0[0]+d1[0]+d2[0]+d3[0] == -999999.f) out[threadIdx.x] = d0[0];
#elif defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d0, d1, d2, d3;
    wmma::fill_fragment(a, __float2bfloat16(1.f)); wmma::fill_fragment(b, __float2bfloat16(1.f));
    wmma::fill_fragment(d0, 0.f); wmma::fill_fragment(d1, 0.f);
    wmma::fill_fragment(d2, 0.f); wmma::fill_fragment(d3, 0.f);
    for (int i = 0; i < 8192; ++i) {
        wmma::mma_sync(d0, a, b, d0); wmma::mma_sync(d1, a, b, d1);
        wmma::mma_sync(d2, a, b, d2); wmma::mma_sync(d3, a, b, d3);
    }
    if (d0.x[0]+d1.x[0]+d2.x[0]+d3.x[0] == -999999.f) out[threadIdx.x] = d0.x[0];
#else
    (void)out;
#endif
}

template <typename Fn>
void bench_flops(const char* label, double flops_per_launch, Fn fn) {
    fn();
    GPU_CHECK(gpuDeviceSynchronize());
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuEventCreate(&t0)); GPU_CHECK(gpuEventCreate(&t1));
    GPU_CHECK(gpuEventRecord(t0));
    for (int i = 0; i < 10; ++i) fn();
    GPU_CHECK(gpuEventRecord(t1));
    GPU_CHECK(gpuEventSynchronize(t1));
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
    printf("%-42s : %7.1f TFLOP/s\n", label, (flops_per_launch * 10.0) / (ms * 1e-3) / 1e12);
    GPU_CHECK(gpuEventDestroy(t0)); GPU_CHECK(gpuEventDestroy(t1));
}

int main() {
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount * 8;
    float* out = nullptr;
    GPU_CHECK(gpuMalloc(&out, 4096));

    double valu_flops = (double)blocks * 256.0 * 65536.0 * 2.0;
    double mat_flops  = (double)blocks * (256.0 / GPU_WAVE_SIZE) * 268435456.0;

    bench_flops("[1] FP32 VALU (1 dependent FMA chain)",   valu_flops, [&]{ valu_1chain_stalled<<<blocks, 256>>>(out, 1.0001f, 0.0001f); });
    bench_flops("[2] FP32 VALU (8 independent ILP chains)", valu_flops, [&]{ valu_8chain_ilp<<<blocks, 256>>>(out, 1.0001f, 0.0001f); });
    bench_flops("[3] BF16 Matrix / Tensor Cores",           mat_flops,  [&]{ bf16_matrix_core<<<blocks, 256>>>(out); });

    GPU_CHECK(gpuFree(out));
    return 0;
}
