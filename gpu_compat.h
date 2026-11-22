#pragma once
// Minimal cross-vendor header mapping NVIDIA CUDA (nvcc) and AMD ROCm (hipcc)
// onto a single thin C++ surface with zero external dependencies.

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>

  #define GPU_CHECK(cmd) do { \
      hipError_t err_ = (cmd); \
      if (err_ != hipSuccess) { \
          fprintf(stderr, "HIP error at %s:%d -> %s\n", \
                  __FILE__, __LINE__, hipGetErrorString(err_)); \
          std::exit(1); \
      } \
  } while (0)

  using gpuDeviceProp_t = hipDeviceProp_t;
  using gpuEvent_t      = hipEvent_t;
  using gpuStream_t     = hipStream_t;
  using gpuGraph_t      = hipGraph_t;
  using gpuGraphExec_t  = hipGraphExec_t;

  #define gpuGetDeviceCount          hipGetDeviceCount
  #define gpuSetDevice               hipSetDevice
  #define gpuGetDeviceProperties     hipGetDeviceProperties
  #define gpuDeviceCanAccessPeer     hipDeviceCanAccessPeer
  #define gpuDeviceEnablePeerAccess  hipDeviceEnablePeerAccess
  #define gpuMalloc                  hipMalloc
  #define gpuFree                    hipFree
  #define gpuMemset                  hipMemset
  #define gpuMemcpyAsync             hipMemcpyAsync
  #define gpuMemcpyHostToDevice      hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost      hipMemcpyDeviceToHost
  #define gpuDeviceSynchronize       hipDeviceSynchronize
  #define gpuEventCreate             hipEventCreate
  #define gpuEventDestroy            hipEventDestroy
  #define gpuEventRecord             hipEventRecord
  #define gpuEventSynchronize        hipEventSynchronize
  #define gpuEventElapsedTime        hipEventElapsedTime
  #define gpuStreamCreate            hipStreamCreate
  #define gpuStreamDestroy           hipStreamDestroy
  #define gpuStreamSynchronize       hipStreamSynchronize
  #define gpuStreamBeginCapture      hipStreamBeginCapture
  #define gpuStreamEndCapture        hipStreamEndCapture
  #define gpuStreamCaptureModeGlobal hipStreamCaptureModeGlobal
  #define gpuGraphLaunch             hipGraphLaunch
  #define gpuGraphDestroy            hipGraphDestroy
  #define gpuGraphExecDestroy        hipGraphExecDestroy

  inline void gpuHostMalloc(void** p, size_t bytes) { GPU_CHECK(hipHostMalloc(p, bytes, 0)); }
  inline void gpuHostFree(void* p)                  { GPU_CHECK(hipHostFree(p)); }
  inline void gpuGraphInit(gpuGraphExec_t* exec, gpuGraph_t g) {
      GPU_CHECK(hipGraphInstantiate(exec, g, nullptr, nullptr, 0));
  }

  constexpr int GPU_WAVE_SIZE = 64; // AMD CDNA wavefront = 64 lanes
  constexpr const char* GPU_BACKEND_NAME = "AMD ROCm / HIP";

#elif defined(__CUDACC__)
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <mma.h>

  #define GPU_CHECK(cmd) do { \
      cudaError_t err_ = (cmd); \
      if (err_ != cudaSuccess) { \
          fprintf(stderr, "CUDA error at %s:%d -> %s\n", \
                  __FILE__, __LINE__, cudaGetErrorString(err_)); \
          std::exit(1); \
      } \
  } while (0)

  using gpuDeviceProp_t = cudaDeviceProp;
  using gpuEvent_t      = cudaEvent_t;
  using gpuStream_t     = cudaStream_t;
  using gpuGraph_t      = cudaGraph_t;
  using gpuGraphExec_t  = cudaGraphExec_t;

  #define gpuGetDeviceCount          cudaGetDeviceCount
  #define gpuSetDevice               cudaSetDevice
  #define gpuGetDeviceProperties     cudaGetDeviceProperties
  #define gpuDeviceCanAccessPeer     cudaDeviceCanAccessPeer
  #define gpuDeviceEnablePeerAccess  cudaDeviceEnablePeerAccess
  #define gpuMalloc                  cudaMalloc
  #define gpuFree                    cudaFree
  #define gpuMemset                  cudaMemset
  #define gpuMemcpyAsync             cudaMemcpyAsync
  #define gpuMemcpyHostToDevice      cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost      cudaMemcpyDeviceToHost
  #define gpuDeviceSynchronize       cudaDeviceSynchronize
  #define gpuEventCreate             cudaEventCreate
  #define gpuEventDestroy            cudaEventDestroy
  #define gpuEventRecord             cudaEventRecord
  #define gpuEventSynchronize        cudaEventSynchronize
  #define gpuEventElapsedTime        cudaEventElapsedTime
  #define gpuStreamCreate            cudaStreamCreate
  #define gpuStreamDestroy           cudaStreamDestroy
  #define gpuStreamSynchronize       cudaStreamSynchronize
  #define gpuStreamBeginCapture      cudaStreamBeginCapture
  #define gpuStreamEndCapture        cudaStreamEndCapture
  #define gpuStreamCaptureModeGlobal cudaStreamCaptureModeGlobal
  #define gpuGraphLaunch             cudaGraphLaunch
  #define gpuGraphDestroy            cudaGraphDestroy
  #define gpuGraphExecDestroy        cudaGraphExecDestroy

  inline void gpuHostMalloc(void** p, size_t bytes) { GPU_CHECK(cudaMallocHost(p, bytes)); }
  inline void gpuHostFree(void* p)                  { GPU_CHECK(cudaFreeHost(p)); }
  inline void gpuGraphInit(gpuGraphExec_t* exec, gpuGraph_t g) {
      GPU_CHECK(cudaGraphInstantiate(exec, g, 0));
  }

  constexpr int GPU_WAVE_SIZE = 32; // NVIDIA warp = 32 lanes
  constexpr const char* GPU_BACKEND_NAME = "NVIDIA CUDA";

#else
  #error "Compile with `nvcc -x cu` (NVIDIA) or `hipcc -x hip` (AMD ROCm)."
#endif
