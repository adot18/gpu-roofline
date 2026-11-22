// Exercise 05: Interconnect Rooflines — Pageable vs Pinned PCIe & Multi-GPU P2P (NVLink / xGMI)
// Shows:
//   1. Why pageable `malloc()` stalls Host->Device KV/token transfers vs pinned `gpuHostMalloc`
//   2. Direct Peer-to-Peer `float4` reads across NVLink / xGMI (foundation of custom TP AllReduce)

#include "gpu_compat.h"
#include <cstdlib>
#include <cstring>

__global__ void p2p_read_vec4(const float4* __restrict__ remote_src,
                              float4* __restrict__ local_dst,
                              size_t n) {
    size_t idx    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (size_t i = idx; i < n; i += stride) local_dst[i] = remote_src[i];
}

int main() {
    const size_t BYTES = 256ULL * 1024 * 1024; // 256 MiB
    void* d0 = nullptr;
    GPU_CHECK(gpuSetDevice(0));
    GPU_CHECK(gpuMalloc(&d0, BYTES));

    void* h_pageable = std::malloc(BYTES);
    void* h_pinned   = nullptr;
    gpuHostMalloc(&h_pinned, BYTES);
    std::memset(h_pageable, 1, BYTES);
    std::memset(h_pinned,   1, BYTES);

    gpuStream_t s;
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuStreamCreate(&s));
    GPU_CHECK(gpuEventCreate(&t0)); GPU_CHECK(gpuEventCreate(&t1));

    auto bench_h2d = [&](const char* label, void* host_ptr) {
        GPU_CHECK(gpuMemcpyAsync(d0, host_ptr, BYTES, gpuMemcpyHostToDevice, s));
        GPU_CHECK(gpuStreamSynchronize(s));
        GPU_CHECK(gpuEventRecord(t0, s));
        for (int i = 0; i < 10; ++i)
            GPU_CHECK(gpuMemcpyAsync(d0, host_ptr, BYTES, gpuMemcpyHostToDevice, s));
        GPU_CHECK(gpuEventRecord(t1, s));
        GPU_CHECK(gpuEventSynchronize(t1));
        float ms = 0.f;
        GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
        printf("%-42s : %6.1f GB/s\n", label, (BYTES * 10.0) / (ms * 1e-3) / 1e9);
    };

    bench_h2d("[1] PCIe H2D from Pageable malloc()",  h_pageable);
    bench_h2d("[2] PCIe H2D from Pinned Host Memory", h_pinned);

    int devs = 0;
    GPU_CHECK(gpuGetDeviceCount(&devs));
    if (devs >= 2) {
        int can_peer = 0;
        GPU_CHECK(gpuDeviceCanAccessPeer(&can_peer, 0, 1));
        if (can_peer) {
            void* d1 = nullptr;
            GPU_CHECK(gpuSetDevice(1));
            GPU_CHECK(gpuMalloc(&d1, BYTES));
            GPU_CHECK(gpuMemset(d1, 1, BYTES));
            GPU_CHECK(gpuDeviceSynchronize());
            GPU_CHECK(gpuSetDevice(0));
            GPU_CHECK(gpuDeviceEnablePeerAccess(1, 0));

            p2p_read_vec4<<<2048, 256, 0, s>>>((float4*)d1, (float4*)d0, BYTES / 16);
            GPU_CHECK(gpuStreamSynchronize(s));
            GPU_CHECK(gpuEventRecord(t0, s));
            for (int i = 0; i < 20; ++i)
                p2p_read_vec4<<<2048, 256, 0, s>>>((float4*)d1, (float4*)d0, BYTES / 16);
            GPU_CHECK(gpuEventRecord(t1, s));
            GPU_CHECK(gpuEventSynchronize(t1));
            float ms = 0.f;
            GPU_CHECK(gpuEventElapsedTime(&ms, t0, t1));
            printf("%-42s : %6.1f GB/s\n",
                   "[3] GPU0 <- GPU1 Direct P2P (NVLink/xGMI)", (BYTES * 20.0) / (ms * 1e-3) / 1e9);
            GPU_CHECK(gpuSetDevice(1)); GPU_CHECK(gpuFree(d1)); GPU_CHECK(gpuSetDevice(0));
        }
    } else {
        printf("[3] Multi-GPU P2P : skipped (only %d GPU visible)\n", devs);
    }

    std::free(h_pageable);
    gpuHostFree(h_pinned);
    GPU_CHECK(gpuEventDestroy(t0)); GPU_CHECK(gpuEventDestroy(t1));
    GPU_CHECK(gpuStreamDestroy(s));
    GPU_CHECK(gpuFree(d0));
    return 0;
}
