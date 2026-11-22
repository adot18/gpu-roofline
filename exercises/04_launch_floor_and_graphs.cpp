// Exercise 04: Kernel Dispatch Overhead — Async CPU Enqueue vs Eager GPU vs CUDA/HIP Graphs
// Demonstrates why batch_size=1 LLM decode (~400 tiny kernels/token) is dominated by
// launch gaps unless captured into a CUDA/HIP Graph or fused into fewer kernels.

#include "gpu_compat.h"
#include <chrono>

__global__ void tiny_op(float* y) {
    if (threadIdx.x == 0) y[0] += 1.0f;
}

int main() {
    float* y = nullptr;
    GPU_CHECK(gpuMalloc(&y, sizeof(float)));
    GPU_CHECK(gpuMemset(y, 0, sizeof(float)));

    gpuStream_t s;
    GPU_CHECK(gpuStreamCreate(&s));
    const int N = 1000;

    // Warmup
    for (int i = 0; i < 100; ++i) tiny_op<<<1, 64, 0, s>>>(y);
    GPU_CHECK(gpuStreamSynchronize(s));

    // 1. Eager launch span measured on GPU events
    gpuEvent_t t0, t1;
    GPU_CHECK(gpuEventCreate(&t0)); GPU_CHECK(gpuEventCreate(&t1));
    auto c0 = std::chrono::high_resolution_clock::now();
    GPU_CHECK(gpuEventRecord(t0, s));
    for (int i = 0; i < N; ++i) tiny_op<<<1, 64, 0, s>>>(y);
    GPU_CHECK(gpuEventRecord(t1, s));
    auto c1 = std::chrono::high_resolution_clock::now();
    GPU_CHECK(gpuEventSynchronize(t1));

    float eager_ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&eager_ms, t0, t1));
    double cpu_us = std::chrono::duration<double, std::micro>(c1 - c0).count() / N;

    // 2. Capture all 1,000 launches into 1 reusable CUDA/HIP Graph
    gpuGraph_t graph;
    gpuGraphExec_t exec;
    GPU_CHECK(gpuStreamBeginCapture(s, gpuStreamCaptureModeGlobal));
    for (int i = 0; i < N; ++i) tiny_op<<<1, 64, 0, s>>>(y);
    GPU_CHECK(gpuStreamEndCapture(s, &graph));
    gpuGraphInit(&exec, graph);
    GPU_CHECK(gpuGraphLaunch(exec, s));
    GPU_CHECK(gpuStreamSynchronize(s));

    GPU_CHECK(gpuEventRecord(t0, s));
    for (int r = 0; r < 10; ++r) GPU_CHECK(gpuGraphLaunch(exec, s));
    GPU_CHECK(gpuEventRecord(t1, s));
    GPU_CHECK(gpuEventSynchronize(t1));
    float graph_ms = 0.f;
    GPU_CHECK(gpuEventElapsedTime(&graph_ms, t0, t1));

    printf("[1] CPU Enqueue Time (host loop) : %6.2f us / kernel\n", cpu_us);
    printf("[2] Eager GPU Execution Floor    : %6.2f us / kernel  (400-op step = %.2f ms)\n",
           (eager_ms * 1000.0) / N, (eager_ms * 0.4));
    printf("[3] Graph Replay Execution Floor : %6.2f us / kernel  (400-op step = %.2f ms)\n",
           (graph_ms * 1000.0) / (10.0 * N), (graph_ms * 0.04));

    GPU_CHECK(gpuGraphExecDestroy(exec));
    GPU_CHECK(gpuGraphDestroy(graph));
    GPU_CHECK(gpuEventDestroy(t0)); GPU_CHECK(gpuEventDestroy(t1));
    GPU_CHECK(gpuStreamDestroy(s));
    GPU_CHECK(gpuFree(y));
    return 0;
}
