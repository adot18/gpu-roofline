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

// 5. BF16 Matrix Core (AMD MFMA) / Tensor Core (NVIDIA WMMA) Roofline
//    Every architecture branch below executes exactly 268,435,456 FLOPs per wave/warp!
__global__ void k_bf16_matrix(float* out, int* supported) {
#if defined(__gfx950__)
    // AMD CDNA4 (MI350X / MI355X): 16x16x32 BF16 MFMA = 16,384 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
    using fp32x4 = float  __attribute__((ext_vector_type(4)));
    bf16x8 a = {1,1,1,1,1,1,1,1}, b = {1,1,1,1,1,1,1,1};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 4096; ++i) {
        d0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, d3, 0, 0, 0);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__gfx942__) || defined(__gfx90a__)
    // AMD CDNA2/CDNA3 (MI250X / MI300X / MI325X): 16x16x16 BF16 MFMA = 8,192 FLOPs/inst
    // Note: ROCm's 16x16x16bf16_1k builtin takes 4x16-bit `short` vector registers.
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using bf16x4_raw = short __attribute__((ext_vector_type(4)));
    using fp32x4     = float __attribute__((ext_vector_type(4)));
    bf16x4_raw a = {0x3f80, 0x3f80, 0x3f80, 0x3f80}; // 1.0 in BF16
    bf16x4_raw b = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    fp32x4 d0={0,0,0,0}, d1={0,0,0,0}, d2={0,0,0,0}, d3={0,0,0,0};
    for (int i = 0; i < 8192; ++i) {
        d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d0, 0, 0, 0);
        d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d1, 0, 0, 0);
        d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d2, 0, 0, 0);
        d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, d3, 0, 0, 0);
    }
    if (d0[0] + d1[0] + d2[0] + d3[0] == -999999.0f) out[threadIdx.x] = d0[0];
#elif defined(__CUDACC__) && (__CUDA_ARCH__ >= 800)
    // NVIDIA Ampere / Ada / Hopper / Blackwell: 16x16x16 BF16 WMMA = 8,192 FLOPs/inst
    if (blockIdx.x == 0 && threadIdx.x == 0) *supported = 1;
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d0, d1, d2, d3;
    wmma::fill_fragment(a, __float2bfloat16(1.0f));
    wmma::fill_fragment(b, __float2bfloat16(1.0f));
    wmma::fill_fragment(d0, 0.0f); wmma::fill_fragment(d1, 0.0f);
    wmma::fill_fragment(d2, 0.0f); wmma::fill_fragment(d3, 0.0f);
    for (int i = 0; i < 8192; ++i) {
        wmma::mma_sync(d0, a, b, d0); wmma::mma_sync(d1, a, b, d1);
        wmma::mma_sync(d2, a, b, d2); wmma::mma_sync(d3, a, b, d3);
    }
    if (d0.x[0] + d1.x[0] + d2.x[0] + d3.x[0] == -999999.0f) out[threadIdx.x] = d0.x[0];
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

    // [1] HBM Bandwidth (Coalesced vs Strided)
    float hbm_ms = bench_gpu_ms(20, [&]{ k_copy_vec4<<<copy_blocks, 256>>>(d_x, d_y, HBM_BYTES / 16); });
    float bad_ms = bench_gpu_ms(10, [&]{ k_copy_strided<<<copy_blocks, 256>>>((float*)d_x, (float*)d_y, STRIDE_B / 4); });
    double hbm_tbs = (2.0 * HBM_BYTES) / (hbm_ms * 1e-3) / 1e12;
    double bad_tbs = (2.0 * STRIDE_B)  / (bad_ms * 1e-3) / 1e12;

    // [2] L2 Cache Bandwidth
    float l2_ms = bench_gpu_ms(200, [&]{ k_copy_vec4<<<copy_blocks, 256>>>(d_x, d_y, L2_BYTES / 16); });
    double l2_tbs = (2.0 * L2_BYTES) / (l2_ms * 1e-3) / 1e12;

    // [3] On-Chip Shared Memory / LDS
    float lds_ms = bench_gpu_ms(50, [&]{ k_shared_bw<<<sms * 4, 256>>>(d_y); });
    double lds_tbs = ((double)(sms * 4) * 256.0 * 1024.0 * 16.0) / (lds_ms * 1e-3) / 1e12;

    // [4] FP32 Vector Compute
    float valu_ms = bench_gpu_ms(10, [&]{ k_fp32_valu<<<sms * 8, 256>>>((float*)d_y, 1.00001f, 0.00001f); });
    double valu_tflops = ((double)(sms * 8) * 256.0 * 8192.0 * 16.0) / (valu_ms * 1e-3) / 1e12;

    // [5] BF16 Matrix / Tensor Core Compute
    float mat_ms = bench_gpu_ms(10, [&]{ k_bf16_matrix<<<sms * 8, 256>>>((float*)d_y, d_sup); });
    int host_sup = 0;
    gpuStream_t tmp_s;
    GPU_CHECK(gpuStreamCreate(&tmp_s));
    GPU_CHECK(gpuMemcpyAsync(&host_sup, d_sup, sizeof(int), gpuMemcpyDeviceToHost, tmp_s));
    GPU_CHECK(gpuStreamSynchronize(tmp_s));
    double waves = (double)(sms * 8) * (256.0 / GPU_WAVE_SIZE);
    double mat_tflops = host_sup ? ((waves * 268435456.0) / (mat_ms * 1e-3)) / 1e12 : 0.0;

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

    // [8] Optional Peer-to-Peer (NVLink / xGMI) if >= 2 GPUs are visible
    double p2p_gbs = 0.0;
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
        }
    }

    printf("--- Memory Hierarchy -----------------------------------------------------\n");
    printf("[1a] HBM Bandwidth (Coalesced 128b) : %7.2f TB/s  (2.0 GiB R+W)\n", hbm_tbs);
    printf("[1b] HBM Bandwidth (Uncoalesced)    : %7.2f TB/s  (%.1fx cache-line waste)\n", bad_tbs, hbm_tbs / bad_tbs);
    printf("[2]  L2 Cache Bandwidth             : %7.2f TB/s  (%zu MiB R+W in %.0f MiB L2)\n",
           l2_tbs, (2 * L2_BYTES) / (1024 * 1024), l2_mib);
    printf("[3]  Shared Memory / LDS Bandwidth  : %7.2f TB/s  (on-chip CU/SM scratchpad)\n\n", lds_tbs);

    printf("--- Compute Ceilings -----------------------------------------------------\n");
    printf("[4]  FP32 Vector ALU                : %7.1f TFLOP/s\n", valu_tflops);
    if (host_sup) {
        printf("[5]  BF16 Matrix / Tensor Cores     : %7.1f TFLOP/s  (%.1fx over Vector ALU)\n\n",
               mat_tflops, mat_tflops / valu_tflops);
    } else {
        printf("[5]  BF16 Matrix / Tensor Cores     : [compile with native arch flag]\n\n");
    }

    printf("--- Dispatch & Interconnects ---------------------------------------------\n");
    printf("[6a] CPU Async Enqueue Cost         : %7.2f us / launch\n", cpu_enq_us);
    printf("[6b] GPU Eager Launch Floor         : %7.2f us / launch\n", eager_us);
    printf("[6c] CUDA/HIP Graph Launch Floor    : %7.2f us / launch  (%.1fx lower gap)\n",
           graph_us, eager_us / std::max(graph_us, 0.01));
    printf("[7]  Pinned PCIe Host <-> GPU       : %7.1f GB/s H2D | %.1f GB/s D2H\n", h2d_gbs, d2h_gbs);
    if (can_peer) {
        printf("[8]  GPU0 <- GPU1 P2P (NVLink/xGMI) : %7.1f GB/s unidirectional read\n", p2p_gbs);
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
                "  \"hbm_uncoalesced_tbs\": %.4f,\n"
                "  \"l2_tbs\": %.4f,\n"
                "  \"shared_lds_tbs\": %.4f,\n"
                "  \"fp32_valu_tflops\": %.2f,\n"
                "  \"bf16_matrix_tflops\": %.2f,\n"
                "  \"cpu_enqueue_us\": %.3f,\n"
                "  \"eager_launch_us\": %.3f,\n"
                "  \"graph_launch_us\": %.3f,\n"
                "  \"pcie_h2d_gbs\": %.2f,\n"
                "  \"pcie_d2h_gbs\": %.2f,\n"
                "  \"p2p_read_gbs\": %.2f\n"
                "}\n",
                prop.name, GPU_BACKEND_NAME, dev_count, sms, vram_gib, l2_mib,
                GPU_WAVE_SIZE, hbm_tbs, bad_tbs, l2_tbs, lds_tbs,
                valu_tflops, mat_tflops, cpu_enq_us, eager_us, graph_us,
                h2d_gbs, d2h_gbs, p2p_gbs);
            std::fclose(jf);
            printf("Saved empirical hardware JSON to: %s\n", json_path);
        }
    }

    if (host_sup) {
        double ridge = (mat_tflops * 1e12) / (hbm_tbs * 1e12);
        double bf16_batch_knee = ridge * 2.0;
        printf("\n--- LLM Inference Roofline Implications ----------------------------------\n");
        printf("  Empirical BF16 Ridge Point : %.0f FLOPs / Byte\n", ridge);
        printf("  BF16 Linear Layer Knee     : Batch tokens M ~= %.0f\n", bf16_batch_knee);
        printf("    * Decode  (M << %.0f) : HBM-Bandwidth-Bound -> speed scales with weight compression & SpecDec\n", bf16_batch_knee);
        printf("    * Prefill (M >= %.0f) : Matrix-Core-Bound   -> speed scales with TFLOP/s & Flash/Sparse Attn\n", bf16_batch_knee);
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
