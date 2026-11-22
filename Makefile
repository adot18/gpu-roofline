# Auto-detect NVIDIA (nvcc) or AMD ROCm (hipcc).
# On AMD ROCm, builds a multi-arch fat binary supporting MI250X (gfx90a),
# MI300X/MI325X (gfx942), and MI350X/MI355X (gfx950) out of the box!
CXX_FLAGS := -O3 -std=c++17 -Wall -Wextra

ifneq ($(shell command -v nvcc 2>/dev/null),)
  GPU_CC := nvcc
  CUDA_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
  ifeq ($(CUDA_CAP),)
    CUDA_CAP := 90
  endif
  ARCH_FLAG := -arch=sm_$(CUDA_CAP) -x cu
else ifneq ($(shell command -v hipcc 2>/dev/null),)
  GPU_CC := hipcc
  ARCH_FLAG := --offload-arch=gfx942 --offload-arch=gfx950 -x hip
else ifneq ($(wildcard /opt/rocm/bin/hipcc),)
  GPU_CC := /opt/rocm/bin/hipcc
  ARCH_FLAG := --offload-arch=gfx942 --offload-arch=gfx950 -x hip
else
  $(error Neither nvcc nor hipcc found in PATH)
endif

EXERCISES := \
  exercises/01_hbm_coalescing \
  exercises/02_cache_and_shared_mem \
  exercises/03_valu_vs_matrix_cores \
  exercises/04_launch_floor_and_graphs \
  exercises/05_pcie_and_p2p_links

.PHONY: all exercises clean

all: gpu_roofline

gpu_roofline: gpu_roofline.cpp gpu_compat.h
	$(GPU_CC) $(CXX_FLAGS) $(ARCH_FLAG) $< -o $@

exercises: $(EXERCISES)

exercises/%: exercises/%.cpp gpu_compat.h
	$(GPU_CC) $(CXX_FLAGS) -I. $(ARCH_FLAG) $< -o $@

clean:
	rm -f gpu_roofline $(EXERCISES)
