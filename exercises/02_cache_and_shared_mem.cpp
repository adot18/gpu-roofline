// Exercise 02: Automatic L2 Cache Cliff vs Manual On-Chip Shared Memory (SRAM/LDS)
// Shows:
//   1. How the exact same global copy kernel hits ~3x higher BW when R+W fits in L2
//   2. How `__shared__` memory reaches ~10-15x HBM BW inside each CU/SM
//   3. How shared-memory bank conflicts (`t * 8`) collapse on-chip SRAM throughput

#include "gpu_compat.h"

__global__ void copy_vec4(const float4* __restrict__ x, float4* __restrict__ y, size_t n) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = idx; i < n; i += stride) y[i] = x[i];
}

template <int STRIDE_MULT>
__global__ void read_shared(float4* out) {
    __shared__ float4 smem[1024];
    int t = threadIdx.x;
    for (int i = t; i < 1024; i += blockDim.x) smem[i] = make_float4(1.f, 2.f, 3.f, 4.f);
    __syncthreads();

    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll 16
    for (int r = 0; r < 1024; ++r) {
        float4 v = smem[(t * STRIDE_MULT + r) & 1023];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == -999999.f) out[(size_t)blockIdx.x * blockDim.x + t] = acc;
}

template <typename Fn>
void measure(const char* label, double bytes, int reps, Fn fn) {
    fn();
    GPU_CHECK(gpuDeviceSynchronize());
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuEventCreate(&t0)); GPU_CHECK(gpuEventCreate(&t1));
    GPU_CHECK(gpuEventRecord(t0));
    for (int i = 0; i < reps; ++i) fn();
    GPU_CHECK(gpuEventRecord(t1));
    GPU_CHECK(gpuEventSynchronize(t1));
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
    printf("%-44s : %6.2f TB/s\n", label, (bytes * reps) / (ms * 1e-3) / 1e12);
    GPU_CHECK(gpuEventDestroy(t0)); GPU_CHECK(gpuEventDestroy(t1));
}

int main() {
    const size_t MAX_B = 1ULL * 1024 * 1024 * 1024;
    float4 *x = nullptr, *y = nullptr;
    GPU_CHECK(gpuMalloc(&x, MAX_B)); GPU_CHECK(gpuMalloc(&y, MAX_B));
    GPU_CHECK(gpuMemset(x, 1, MAX_B)); GPU_CHECK(gpuMemset(y, 0, MAX_B));

    for (size_t mb : {8ULL, 32ULL, 128ULL, 256ULL, 1024ULL}) {
        size_t b = mb * 1024 * 1024;
        char name[64];
        snprintf(name, sizeof(name), "[L2->HBM] Copy %4zu MiB (R+W = %4zu MiB)", mb, 2 * mb);
        measure(name, 2.0 * b, (mb <= 64) ? 150 : 20, [&]{ copy_vec4<<<4096, 256>>>(x, y, b / 16); });
    }

    double sram_bytes = 1024.0 * 256.0 * 1024.0 * sizeof(float4);
    measure("[SRAM] __shared__ conflict-free (stride=1)", sram_bytes, 50, [&]{ read_shared<1><<<1024, 256>>>(y); });
    measure("[SRAM] __shared__ bank-conflict  (stride=8)", sram_bytes, 50, [&]{ read_shared<8><<<1024, 256>>>(y); });

    GPU_CHECK(gpuFree(x)); GPU_CHECK(gpuFree(y));
    return 0;
}
