// Exercise 01: HBM Bandwidth, Coalescing, and Little's Law Occupancy
// Compares:
//   1. Uncoalesced strided reads (adjacent threads jump 33 floats -> cache-line waste)
//   2. Starved occupancy (<<<16, 256>>> leaves >90% of CUs/SMs dark)
//   3. Vectorized 128-bit coalesced copy (<<<4096, 256>>> -> 100% HBM STREAM ceiling)

#include "gpu_compat.h"

__global__ void copy_vec4(const float4* __restrict__ x, float4* __restrict__ y, size_t n) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        y[i] = x[i];
    }
}

__global__ void copy_strided(const float* __restrict__ x, float* __restrict__ y, size_t n_pow2) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = idx; i < n_pow2; i += stride) {
        y[i] = x[(i * 33ULL) & (n_pow2 - 1)];
    }
}

template <typename Fn>
void run(const char* label, size_t bytes_moved, Fn fn) {
    fn();
    GPU_CHECK(gpuDeviceSynchronize());
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuEventCreate(&t0)); GPU_CHECK(gpuEventCreate(&t1));
    GPU_CHECK(gpuEventRecord(t0));
    for (int it = 0; it < 20; ++it) fn();
    GPU_CHECK(gpuEventRecord(t1));
    GPU_CHECK(gpuEventSynchronize(t1));
    float ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
    printf("%-40s : %6.3f ms | %6.2f TB/s\n",
           label, ms / 20.f, (bytes_moved * 20.0) / (ms * 1e-3) / 1e12);
    GPU_CHECK(gpuEventDestroy(t0)); GPU_CHECK(gpuEventDestroy(t1));
}

int main() {
    const size_t BYTES = 1ULL * 1024 * 1024 * 1024; // 1 GiB per buffer
    void *x = nullptr, *y = nullptr;
    GPU_CHECK(gpuMalloc(&x, BYTES));
    GPU_CHECK(gpuMalloc(&y, BYTES));
    GPU_CHECK(gpuMemset(x, 1, BYTES));
    GPU_CHECK(gpuMemset(y, 0, BYTES));

    run("[1] Uncoalesced strided <<<4096, 256>>>", 2 * BYTES, [&]{
        copy_strided<<<4096, 256>>>((float*)x, (float*)y, BYTES / sizeof(float));
    });
    run("[2] Starved occupancy   <<<  16, 256>>>", 2 * BYTES, [&]{
        copy_vec4<<<16, 256>>>((float4*)x, (float4*)y, BYTES / sizeof(float4));
    });
    run("[3] Coalesced float4    <<<4096, 256>>>", 2 * BYTES, [&]{
        copy_vec4<<<4096, 256>>>((float4*)x, (float4*)y, BYTES / sizeof(float4));
    });

    GPU_CHECK(gpuFree(x));
    GPU_CHECK(gpuFree(y));
    return 0;
}
