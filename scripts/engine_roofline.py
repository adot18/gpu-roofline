#!/usr/bin/env python3
"""engine_roofline.py — Multi-Dimensional Hardware + Engine (SGLang / vLLM) Roofline Analyzer & Advisor.

Inspired by classic Roofline tools (e.g. KrxGu/RoofLine_Plot), extended to plot ALL FOUR
orthogonal GPU inference bottlenecks simultaneously across precisions (BF16, FP8, FP4/MXFP4,
FP32 VALU, FP32 SFU) against empirical hardware ceilings from `./gpu_roofline --json hw.json`:

  Panel 1: Compute vs. Memory Hierarchy Roofline  (TFLOP/s     vs. Operational Intensity FLOPs/Byte)
  Panel 2: All Data-Movement Bandwidths           (GB/s..TB/s  vs. Payload Size Bytes: L2, HBM, NVLink/xGMI, PCIe)
  Panel 3: Kernel Dispatch & Graph Efficiency     (GPU Busy %  vs. Mean Kernel Duration us)
  Panel 4: Matrix Tile & MoE Batch Efficiency     (Roofline %  vs. Tokens per GEMM / Expert M)

Supports three workload inputs (zero external Python dependencies required):
  1. Live SGLang or vLLM server (`--engine sglang|vllm --url http://127.0.0.1:30000 --trace-dir /tmp/traces`)
  2. Saved PyTorch Kineto profiler trace (`--trace trace.json[.gz]`) with stream-gap & fusion detection
  3. Model-on-Hardware architecture analyzer (`--model <hf_repo_or_config.json_or_preset>` or `--demo`)
     with automatic mixed-precision tracking (e.g. FP4 MoE experts + FP8/BF16 Attention/MLA)
"""

import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Hardware Ceiling Profiles (Sustained TDP Empirical + Datasheet Specs)
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    name: str
    sms_or_cus: int
    vram_gib: float
    l2_mib: float
    # Empirical achievable ceilings (from `./gpu_roofline --json` or TDP-sustained presets)
    hbm_tbs: float                   # Coalesced 50/50 R+W HBM
    hbm_read_tbs: float              # Coalesced Read-Only HBM (weight streaming)
    hbm_uncoalesced_tbs: float       # Strided uncoalesced HBM
    l2_tbs: float
    lds_tbs: float
    fp32_valu_tflops: float
    fp32_sfu_tops: float             # exp2/rsqrt transcendental ceiling
    bf16_matrix_tflops: float        # Sustained TDP-throttled BF16 Matrix/Tensor
    fp8_matrix_tflops: float         # Sustained TDP-throttled FP8 Matrix/Tensor (0 if unsupported)
    fp4_matrix_tflops: float         # Sustained TDP-throttled FP4/MXFP4 Matrix (0 -> upscale to FP8)
    eager_launch_us: float
    graph_launch_us: float
    pcie_gbs: float
    p2p_gbs: float                   # 1-to-1 unidirectional P2P
    p2p_all_gbs: float               # Full-node ring/collective bus BW
    # Datasheet theoretical dense boost peaks (visualized to show TDP/instruction tax!)
    spec_hbm_tbs: float
    spec_bf16_dense_tflops: float
    spec_fp8_dense_tflops: float
    spec_fp4_dense_tflops: float


# Realistic sustained TDP-limited presets (aligned with hipBLASLt / CUTLASS real-input clocks,
# NOT zero-input boost peaks). Pass `--hw hw.json` from `./gpu_roofline --json` for your exact node.
HW_PRESETS: Dict[str, HardwareProfile] = {
    "mi355x": HardwareProfile(
        name="AMD Instinct MI355X (gfx950)", sms_or_cus=256, vram_gib=288.0, l2_mib=256.0,
        hbm_tbs=5.85, hbm_read_tbs=6.35, hbm_uncoalesced_tbs=0.42, l2_tbs=18.5, lds_tbs=78.0,
        fp32_valu_tflops=145.0, fp32_sfu_tops=42.0,
        bf16_matrix_tflops=1680.0, fp8_matrix_tflops=3400.0, fp4_matrix_tflops=6800.0,
        eager_launch_us=2.6, graph_launch_us=0.42, pcie_gbs=54.0, p2p_gbs=410.0, p2p_all_gbs=360.0,
        spec_hbm_tbs=8.00, spec_bf16_dense_tflops=2500.0, spec_fp8_dense_tflops=5000.0, spec_fp4_dense_tflops=10000.0,
    ),
    "mi300x": HardwareProfile(
        name="AMD Instinct MI300X (gfx942)", sms_or_cus=304, vram_gib=192.0, l2_mib=256.0,
        hbm_tbs=4.25, hbm_read_tbs=4.45, hbm_uncoalesced_tbs=0.34, l2_tbs=15.5, lds_tbs=68.0,
        fp32_valu_tflops=126.0, fp32_sfu_tops=36.0,
        # Sustained 750W board limit on real-entropy inputs (per AMD hipBLASLt / cluster measurements):
        # ~616-708 TF BF16, ~1182-1270 TF FP8, 0 native FP4 (must upscale FP4 -> FP8).
        bf16_matrix_tflops=680.0, fp8_matrix_tflops=1220.0, fp4_matrix_tflops=0.0,
        eager_launch_us=2.9, graph_launch_us=0.48, pcie_gbs=52.0, p2p_gbs=360.0, p2p_all_gbs=317.0,
        spec_hbm_tbs=5.30, spec_bf16_dense_tflops=1307.0, spec_fp8_dense_tflops=2615.0, spec_fp4_dense_tflops=0.0,
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 SXM5 (sm_90)", sms_or_cus=132, vram_gib=80.0, l2_mib=50.0,
        hbm_tbs=2.95, hbm_read_tbs=3.10, hbm_uncoalesced_tbs=0.26, l2_tbs=11.2, lds_tbs=48.0,
        fp32_valu_tflops=60.0, fp32_sfu_tops=16.5,
        bf16_matrix_tflops=760.0, fp8_matrix_tflops=1520.0, fp4_matrix_tflops=0.0,
        eager_launch_us=2.2, graph_launch_us=0.35, pcie_gbs=55.0, p2p_gbs=400.0, p2p_all_gbs=385.0,
        spec_hbm_tbs=3.35, spec_bf16_dense_tflops=989.5, spec_fp8_dense_tflops=1979.0, spec_fp4_dense_tflops=0.0,
    ),
    "b200": HardwareProfile(
        name="NVIDIA B200 SXM (sm_100)", sms_or_cus=148, vram_gib=180.0, l2_mib=126.0,
        hbm_tbs=6.60, hbm_read_tbs=7.00, hbm_uncoalesced_tbs=0.50, l2_tbs=21.0, lds_tbs=90.0,
        fp32_valu_tflops=80.0, fp32_sfu_tops=28.0,
        # Sustained TDP clocks: ~1439 TF BF16, ~2788 TF FP8, ~5222-5627 TF MXFP4/NVFP4
        bf16_matrix_tflops=1440.0, fp8_matrix_tflops=2790.0, fp4_matrix_tflops=5400.0,
        eager_launch_us=2.0, graph_launch_us=0.30, pcie_gbs=58.0, p2p_gbs=840.0, p2p_all_gbs=820.0,
        spec_hbm_tbs=8.00, spec_bf16_dense_tflops=2250.0, spec_fp8_dense_tflops=4500.0, spec_fp4_dense_tflops=9000.0,
    ),
}


def load_hw_json(path: Path) -> HardwareProfile:
    d = json.loads(path.read_text())
    dev = d.get("device", path.stem)
    low = dev.lower()
    if "355" in low or "350" in low:
        shbm, sbf16, sfp8, sfp4 = 8.0, 2500.0, 5000.0, 10000.0
    elif "300" in low or "325" in low:
        shbm, sbf16, sfp8, sfp4 = 5.3, 1307.0, 2615.0, 0.0
    elif "h100" in low:
        shbm, sbf16, sfp8, sfp4 = 3.35, 989.5, 1979.0, 0.0
    elif "b200" in low or "b300" in low:
        shbm, sbf16, sfp8, sfp4 = 8.0, 2250.0, 4500.0, 9000.0
    else:
        shbm = float(d["hbm_tbs"]) * 1.24
        sbf16 = float(d["bf16_matrix_tflops"]) * 1.35
        sfp8 = float(d.get("fp8_matrix_tflops", 0.0)) * 1.35 or sbf16 * 2.0
        sfp4 = float(d.get("fp4_matrix_tflops", 0.0)) * 1.35

    bf16 = float(d["bf16_matrix_tflops"]) or (float(d["fp32_valu_tflops"]) * 10.0)
    fp8 = float(d.get("fp8_matrix_tflops", 0.0))
    if fp8 <= 0.0 and ("355" in low or "300" in low or "h100" in low or "b200" in low):
        fp8 = bf16 * 1.85
    fp4 = float(d.get("fp4_matrix_tflops", 0.0))
    hbm = float(d["hbm_tbs"])
    hbm_read = float(d.get("hbm_read_tbs", 0.0)) or (hbm * 1.07)
    sfu = float(d.get("fp32_sfu_tops", 0.0)) or (float(d["fp32_valu_tflops"]) * 0.28)
    p2p = float(d.get("p2p_read_gbs", 0.0))
    if p2p <= 1.0:
        p2p = 380.0
    p2p_all = float(d.get("p2p_all_ring_gbs", 0.0)) or (p2p * 0.88)

    return HardwareProfile(
        name=dev,
        sms_or_cus=int(d["sms_or_cus"]),
        vram_gib=float(d["vram_gib"]),
        l2_mib=float(d["l2_mib"]),
        hbm_tbs=hbm,
        hbm_read_tbs=hbm_read,
        hbm_uncoalesced_tbs=float(d.get("hbm_uncoalesced_tbs", hbm * 0.08)),
        l2_tbs=float(d["l2_tbs"]),
        lds_tbs=float(d["shared_lds_tbs"]),
        fp32_valu_tflops=float(d["fp32_valu_tflops"]),
        fp32_sfu_tops=sfu,
        bf16_matrix_tflops=bf16,
        fp8_matrix_tflops=fp8,
        fp4_matrix_tflops=fp4,
        eager_launch_us=float(d["eager_launch_us"]),
        graph_launch_us=float(d["graph_launch_us"]),
        pcie_gbs=float(d["pcie_h2d_gbs"]),
        p2p_gbs=p2p,
        p2p_all_gbs=p2p_all,
        spec_hbm_tbs=shbm,
        spec_bf16_dense_tflops=sbf16,
        spec_fp8_dense_tflops=sfp8,
        spec_fp4_dense_tflops=sfp4,
    )


# ---------------------------------------------------------------------------
# 2. Kernel / Operator Record & Precision Helpers
# ---------------------------------------------------------------------------

@dataclass
class OpPoint:
    name: str
    category: str          # "GEMM", "MoE", "Attention", "Norm/Elem", "Comm"
    calls: int
    total_us: float
    avg_us: float
    flops: float           # per call
    bytes_moved: float     # per call
    tokens_m: float        # effective batch/token dimension M (for Panel 4)
    pipe: str = "HBM"      # "HBM", "HBM_READ", "L2", "P2P", "PCIe"
    dtype: str = "bf16"    # "bf16", "fp8", "fp4", "fp32", "sfu"
    measured_gap_us: Optional[float] = None
    shape_source: str = "exact"  # "exact", "model", "graph_est"

    @property
    def oi(self) -> float:
        return self.flops / max(self.bytes_moved, 1.0)

    @property
    def achieved_tflops(self) -> float:
        return (self.flops / (max(self.avg_us, 0.01) * 1e-6)) / 1e12

    @property
    def achieved_tbs(self) -> float:
        return (self.bytes_moved / (max(self.avg_us, 0.01) * 1e-6)) / 1e12


@dataclass
class FusionOpportunity:
    pattern: str
    pair_desc: str
    occurrences: int
    total_dur_us: float
    saved_gap_us: float
    saved_bytes: float
    recommendation: str


CAT_COLOR = {
    "GEMM":      "#2563eb",  # blue
    "MoE":       "#7c3aed",  # purple
    "Attention": "#059669",  # emerald
    "Norm/Elem": "#d97706",  # amber
    "Comm":      "#dc2626",  # red
}

DTYPE_BADGE = {
    "fp4":  "FP4",
    "fp8":  "FP8",
    "bf16": "BF16",
    "fp32": "FP32",
    "sfu":  "SFU",
}


def matrix_ceiling_for_dtype(hw: HardwareProfile, dtype: str) -> Tuple[float, str]:
    """Return (sustained_tflops, label) accounting for FP4->FP8 fallback on non-FP4 silicon."""
    if dtype == "fp4":
        if hw.fp4_matrix_tflops > 0.0:
            return hw.fp4_matrix_tflops, "FP4 Matrix"
        if hw.fp8_matrix_tflops > 0.0:
            # No native FP4 tensor cores (e.g., MI300X / H100): must dequant/upscale FP4 -> FP8 (~10% tax)
            return hw.fp8_matrix_tflops * 0.90, "FP4->FP8 Upscale"
        return hw.bf16_matrix_tflops * 0.88, "FP4->BF16 Upscale"
    if dtype == "fp8":
        if hw.fp8_matrix_tflops > 0.0:
            return hw.fp8_matrix_tflops, "FP8 Matrix"
        return hw.bf16_matrix_tflops, "FP8->BF16 Upscale"
    if dtype == "sfu":
        return hw.fp32_sfu_tops, "FP32 SFU"
    if dtype == "fp32":
        return hw.fp32_valu_tflops, "FP32 VALU"
    return hw.bf16_matrix_tflops, "BF16 Matrix"


def bw_ceiling_for_pipe(hw: HardwareProfile, pipe: str) -> Tuple[float, str]:
    if pipe == "L2":
        return hw.l2_tbs, "L2 Cache BW"
    if pipe == "HBM_READ":
        return hw.hbm_read_tbs, "HBM Read BW"
    if pipe == "HBM_STRIDED":
        return hw.hbm_uncoalesced_tbs, "Uncoalesced HBM"
    return hw.hbm_tbs, "HBM R+W BW"


def ceiling_for_op(op: OpPoint, hw: HardwareProfile) -> Tuple[float, str]:
    """Return (empirical_ceiling_tflops_at_oi, limiting_regime_name)."""
    if op.pipe == "P2P":
        bw_gbs = hw.p2p_all_gbs if op.bytes_moved >= 128 * 1024 else hw.p2p_gbs
        p2p_ceil = (bw_gbs / 1000.0) * op.oi
        return p2p_ceil, "NVLink/xGMI Latency" if op.bytes_moved < 512 * 1024 else "NVLink/xGMI BusBW"
    bw_tbs, bw_label = bw_ceiling_for_pipe(hw, op.pipe)
    mem_ceil = bw_tbs * op.oi
    cmp_ceil, cmp_label = matrix_ceiling_for_dtype(hw, op.dtype)
    if mem_ceil < cmp_ceil:
        return mem_ceil, bw_label
    return cmp_ceil, cmp_label


# ---------------------------------------------------------------------------
# 3. Live SGLang / vLLM Profiler Trigger & Kineto Trace Parser
# ---------------------------------------------------------------------------

def http_post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}


def profile_live_engine(
    engine: str,
    base_url: str,
    trace_dir: Path,
    batch_size: int,
    prompt_len: int,
    output_len: int,
) -> Path:
    """Drive a live SGLang or vLLM OpenAI-compatible server with /start_profile -> workload -> /stop_profile."""
    base_url = base_url.rstrip("/")
    trace_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in trace_dir.glob("*.json*")}

    prompt = ("roofline " * max(8, prompt_len // 2)).strip()
    prompts = [prompt] * batch_size

    # Warmup (ensures CUDA/HIP graphs & JIT kernels are compiled before profiling)
    http_post_json(f"{base_url}/v1/completions", {
        "model": "default", "prompt": prompts, "max_tokens": max(2, min(output_len, 4)), "temperature": 0.0,
    })

    print(f"[{engine}] Starting Kineto profiler via {base_url}/start_profile ...")
    start_body = {"record_shapes": True, "with_flops": True} if engine == "sglang" else {}
    http_post_json(f"{base_url}/start_profile", start_body)

    http_post_json(f"{base_url}/v1/completions", {
        "model": "default", "prompt": prompts, "max_tokens": output_len, "temperature": 0.0,
    })

    http_post_json(f"{base_url}/stop_profile", {})
    time.sleep(1.5)

    candidates = [
        p for p in trace_dir.rglob("*.trace.json*")
        if p not in before or p.stat().st_mtime > before[p]
    ]
    if not candidates:
        candidates = sorted(trace_dir.rglob("*.json*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No trace files found in {trace_dir}. Start {engine} with "
            f"{'SGLANG_TORCH_PROFILER_DIR' if engine == 'sglang' else 'VLLM_TORCH_PROFILER_DIR'}={trace_dir}"
        )
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[{engine}] Captured trace: {newest}")
    return newest


DTYPE_BYTES = {
    "float": 4.0, "c10::half": 2.0, "c10::bfloat16": 2.0, "at::half": 2.0, "at::bfloat16": 2.0,
    "float8": 1.0, "c10::float8_e4m3fn": 1.0, "c10::float8_e4m3fnuz": 1.0, "c10::float8_e5m2": 1.0,
    "float4_e2m1fn_x2": 1.0, "c10::float4_e2m1fn_x2": 1.0,
    "unsigned char": 1.0, "uint8": 1.0, "char": 1.0, "int8": 1.0, "long": 8.0, "int": 4.0,
}


def classify_kernel(name: str, dtypes: Optional[list] = None) -> Tuple[str, str, str]:
    """Return (category, default_pipe, precision_dtype) from kernel name & tensor dtypes."""
    low = name.lower()
    dt_str = " ".join(str(x).lower() for x in (dtypes or []))

    # Precision inference
    if any(k in low for k in ("fp4", "mxfp4", "nvfp4", "f4_", "_f4", "f8f6f4", "e2m1")) or "float4" in dt_str:
        dtype = "fp4"
    elif any(k in low for k in ("fp8", "mxfp8", "f8b", "f8_", "_f8", "e4m3", "scaled_mm", "fp8_gemm", "w8a8")) or "float8" in dt_str:
        dtype = "fp8"
    elif any(k in low for k in ("softmax", "rmsnorm", "rms_norm", "layernorm", "silu", "swiglu", "act_and_mul", "rotary", "rope", "topk", "sinkhorn", "exp")):
        dtype = "sfu"
    elif "float" in dt_str and "half" not in dt_str and "bfloat16" not in dt_str:
        dtype = "fp32"
    else:
        dtype = "bf16"

    if any(k in low for k in ("nccl", "rccl", "allreduce", "all_reduce", "all_to_all", "all2all", "msccl", "deepep", "custom_ar", "quick_reduce")):
        return "Comm", "P2P", dtype
    if any(k in low for k in ("moe", "expert", "topk", "grouped_gemm", "fmoe")):
        return "MoE", "HBM_READ", dtype
    if any(k in low for k in ("flash", "fmha", "attn", "attention", "mla", "paged", "indexer")):
        return "Attention", "HBM_READ", dtype
    if any(k in low for k in ("gemm", "cijk_", "nvjet", "cutlass", "cublas", "hipblas", "scaled_mm", "wvsplitk", "matmul", "gemv", "woa", "wqkv")):
        return "GEMM", "HBM_READ", dtype
    if any(k in low for k in ("norm", "rope", "rotary", "silu", "swiglu", "act_and_mul", "elementwise", "quant", "cast", "kv_cache", "reshape_and_cache", "sinkhorn", "mhc", "engram")):
        return "Norm/Elem", "HBM", dtype
    return "Norm/Elem", "HBM", dtype


def estimate_from_dims(
    cat: str, dtype: str, dims: list, dtypes: list, explicit_flops: float, kernel_name: str, dur_us: float, gk_args: dict
) -> Tuple[float, float, float, str]:
    """Infer (flops, bytes_moved, tokens_m, shape_source) from Kineto CPU op Input Dims, or fallback."""
    valid_dims = [d for d in dims if isinstance(d, list) and len(d) >= 1 and all(isinstance(x, int) and x > 0 for x in d)]
    b_elem = 0.56 if dtype == "fp4" else (1.0 if dtype == "fp8" else 2.0)
    for dt in dtypes:
        if isinstance(dt, str) and dt.lower() in DTYPE_BYTES:
            b_elem = DTYPE_BYTES[dt.lower()]
            if "float4" in dt.lower():
                b_elem = 0.56
            break

    total_in_bytes = 0.0
    for idx, d in enumerate(valid_dims):
        eb = b_elem
        if idx < len(dtypes) and isinstance(dtypes[idx], str):
            eb = DTYPE_BYTES.get(dtypes[idx].lower(), b_elem)
            if "float4" in dtypes[idx].lower():
                eb = 0.56
        n_el = 1
        for x in d:
            n_el *= x
        total_in_bytes += n_el * eb

    tokens_m = float(valid_dims[0][0]) if valid_dims else 1.0

    if explicit_flops > 0:
        bytes_est = max(total_in_bytes * 1.15, 4096.0)
        return explicit_flops, bytes_est, tokens_m, "exact"

    if cat in ("GEMM", "MoE") and len(valid_dims) >= 2 and len(valid_dims[0]) >= 2 and len(valid_dims[1]) >= 2:
        m = float(math.prod(valid_dims[0][:-1]))
        k = float(valid_dims[0][-1])
        # Packed FP4 tensors often appear as uint8 [N, K//2] -> effective K is 2x
        w_last = valid_dims[1][-1]
        n = float(valid_dims[1][-1] if w_last == int(k) else valid_dims[1][0])
        b_w = 0.56 if dtype == "fp4" else (1.0 if dtype == "fp8" else b_elem)
        if cat == "MoE" and len(valid_dims[1]) == 3:
            e_cnt = valid_dims[1][0]
            n = float(valid_dims[1][1])
            active_e = min(float(e_cnt), max(1.0, m * 8.0))
            flops = 2.0 * m * 8.0 * k * n
            bytes_est = m * k * 2.0 + active_e * n * k * b_w
            return flops, bytes_est, max(1.0, (m * 8.0) / max(e_cnt, 1)), "exact"
        flops = 2.0 * m * n * k
        bytes_est = m * k * 2.0 + k * n * b_w + m * n * 2.0
        return flops, bytes_est, m, "exact"

    if cat == "Attention" and valid_dims:
        q_el = float(math.prod(valid_dims[0]))
        kv_bytes = max(total_in_bytes - q_el * 2.0, q_el * 16.0)
        flops = 2.0 * kv_bytes
        return flops, total_in_bytes + q_el * 2.0, tokens_m, "exact"

    if cat == "Comm" and valid_dims:
        bytes_est = max(total_in_bytes, 16384.0)
        return bytes_est * 0.25, bytes_est, tokens_m, "exact"

    if valid_dims:
        bytes_est = max(total_in_bytes * 2.0, 16384.0)
        flops = (bytes_est / max(b_elem, 1.0)) * 2.5
        return flops, bytes_est, tokens_m, "exact"

    # Fallback when CUDA/HIP Graph replay omits cpu_op Input Dims:
    # Try parsing M/N/K hints from kernel name or grid/block dims so points remain meaningful!
    m_match = re.search(r"_[Mm](\d+)[_x]", kernel_name)
    grid = gk_args.get("grid")
    grid_blocks = float(math.prod(grid)) if isinstance(grid, list) and grid else 256.0
    m_est = float(m_match.group(1)) if m_match else (4.0 if dur_us < 40.0 else 64.0)
    if cat in ("GEMM", "MoE"):
        # Approximate realistic bytes/FLOPs from grid & duration
        bytes_est = max(64 * 1024.0, grid_blocks * 128.0 * 1024.0)
        flops = bytes_est * (2.0 * m_est / (0.56 if dtype == "fp4" else (1.0 if dtype == "fp8" else 2.0)))
        return flops, bytes_est, m_est, "graph_est"
    if cat == "Attention":
        bytes_est = max(256 * 1024.0, grid_blocks * 64.0 * 1024.0)
        return bytes_est * 12.0, bytes_est, m_est, "graph_est"
    if cat == "Comm":
        bytes_est = 256 * 1024.0
        return bytes_est * 0.25, bytes_est, m_est, "graph_est"
    bytes_est = max(16384.0, grid_blocks * 4096.0)
    return bytes_est * 1.5, bytes_est, m_est, "graph_est"


def _fusion_rule(k1: str, c1: str, k2: str, c2: str) -> Optional[Tuple[str, str]]:
    """Identify known fusible adjacent GPU kernel pairs in SGLang / vLLM traces."""
    a, b = k1.lower(), k2.lower()
    if c1 == "Comm" and ("norm" in b):
        return "AllReduce + RMSNorm", "Fuse 1-shot P2P AllReduce with RMSNorm (eliminates HBM round-trip + sync gap)"
    if ("norm" in a) and ("quant" in b or "fp8" in b or "fp4" in b or "cast" in b) and c2 == "Norm/Elem":
        return "RMSNorm + Dynamic Quant", "Fuse RMSNorm directly with FP8/MXFP4 per-token/block quantization"
    if ("silu" in a or "swiglu" in a or "act_and_mul" in a) and ("quant" in b or "cast" in b):
        return "SwiGLU + Quant", "Fuse SiLU/SwiGLU activation with downstream FP8/FP4 quantization"
    if ("rope" in a or "rotary" in a) and ("kv" in b or "cache" in b or "attn" in b):
        return "RoPE + KV-Store/Attn", "Fuse rotary embedding directly into PagedKV write or Attention prologue"
    if ("topk" in a or "softmax" in a or "router" in a) and ("moe" in b or "permute" in b or "scatter" in b):
        return "Router TopK + MoE Permute", "Fuse MoE gating TopK selection with token-to-expert permutation"
    if c1 == "Norm/Elem" and c2 == "Norm/Elem" and ("copy" not in a or "copy" not in b):
        return "Adjacent Norm/Elementwise", "Merge back-to-back memory-bound elementwise ops via torch.compile / Triton"
    return None


def parse_kineto_trace(trace_path: Path, hw: Optional[HardwareProfile] = None) -> Tuple[List[OpPoint], List[FusionOpportunity]]:
    """Parse a PyTorch Kineto .json or .json.gz trace: correlate shapes, stream gaps, and fusion targets."""
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    events = data.get("traceEvents", data if isinstance(data, list) else [])
    cpu_ops = []
    corr_to_cpu = {}
    runtime_ev = []
    gpu_kernels = []

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if cat == "cpu_op":
            cpu_ops.append(ev)
        elif "runtime" in cat and "args" in ev and "correlation" in ev["args"]:
            runtime_ev.append(ev)
        elif "kernel" in cat or "gpu_memcpy" in cat:
            gpu_kernels.append(ev)

    by_tid: Dict[int, list] = {}
    for op in cpu_ops:
        args = op.get("args") or {}
        if "Input Dims" in args or "Flops" in args:
            by_tid.setdefault(op.get("tid", 0), []).append(op)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda e: e.get("ts", 0))

    for rt in runtime_ev:
        tid = rt.get("tid", 0)
        ts = rt.get("ts", 0)
        corr = rt["args"]["correlation"]
        best = None
        for op in by_tid.get(tid, []):
            op_ts = op.get("ts", 0)
            if op_ts <= ts <= op_ts + op.get("dur", 0):
                best = op
            elif op_ts > ts:
                break
        if best is not None:
            corr_to_cpu[corr] = best

    # Sort GPU kernels per (pid, tid/stream) to measure real inter-kernel gaps & fusible pairs
    l2_thresh_bytes = (hw.l2_mib * 0.5 * 1024**2) if hw else (64.0 * 1024**2)
    by_stream: Dict[Tuple[int, int], list] = {}
    for gk in gpu_kernels:
        dur = float(gk.get("dur", 0.0))
        if dur <= 0.0:
            continue
        key = (int(gk.get("pid", 0)), int(gk.get("tid", 0)))
        by_stream.setdefault(key, []).append(gk)

    grouped: Dict[str, dict] = {}
    fusion_agg: Dict[str, dict] = {}

    for stream_key, skernels in by_stream.items():
        skernels.sort(key=lambda e: float(e.get("ts", 0.0)))
        prev_rec = None
        for gk in skernels:
            ts = float(gk.get("ts", 0.0))
            dur = float(gk.get("dur", 0.0))
            raw_name = gk.get("name", "unknown_kernel")
            short = raw_name.split("(")[0].split("<")[0][-44:]
            gk_args = gk.get("args") or {}
            corr = gk_args.get("correlation")
            cpu_op = corr_to_cpu.get(corr)
            c_args = (cpu_op.get("args") if cpu_op else None) or {}
            in_dims = c_args.get("Input Dims", [])
            in_types = c_args.get("Input type", [])
            cat, pipe, dtype = classify_kernel(raw_name, in_types)
            flops, bytes_m, tok_m, src = estimate_from_dims(
                cat, dtype, in_dims, in_types, float(c_args.get("Flops", 0.0) or 0.0),
                raw_name, dur, gk_args,
            )

            # Small activation/norm ops stay in L2 rather than streaming from HBM
            if cat == "Norm/Elem" and bytes_m < l2_thresh_bytes:
                pipe = "L2"

            gap_us: Optional[float] = None
            if prev_rec is not None:
                raw_gap = ts - prev_rec["end_ts"]
                if 0.0 <= raw_gap <= 200.0:
                    gap_us = raw_gap
                    if prev_rec["dur"] < 45.0 or dur < 45.0:
                        rule = _fusion_rule(prev_rec["raw_name"], prev_rec["cat"], raw_name, cat)
                        if rule is not None:
                            pat, rec = rule
                            fa = fusion_agg.setdefault(pat, {
                                "pair": f"{prev_rec['short'][-22:]} -> {short[-22:]}",
                                "rec": rec, "count": 0, "dur": 0.0, "gap": 0.0, "bytes": 0.0,
                            })
                            fa["count"] += 1
                            fa["dur"] += min(prev_rec["dur"], dur)
                            fa["gap"] += raw_gap
                            fa["bytes"] += min(prev_rec["bytes_m"], bytes_m)

            g = grouped.setdefault(short, {
                "cat": cat, "pipe": pipe, "dtype": dtype, "calls": 0, "total_us": 0.0,
                "flops_sum": 0.0, "bytes_sum": 0.0, "m_sum": 0.0,
                "gap_sum": 0.0, "gap_cnt": 0, "src": src,
            })
            g["calls"] += 1
            g["total_us"] += dur
            g["flops_sum"] += flops
            g["bytes_sum"] += bytes_m
            g["m_sum"] += tok_m
            if gap_us is not None:
                g["gap_sum"] += gap_us
                g["gap_cnt"] += 1

            prev_rec = {
                "raw_name": raw_name, "short": short, "cat": cat,
                "dur": dur, "end_ts": ts + dur, "bytes_m": bytes_m,
            }

    pts = []
    for name, g in sorted(grouped.items(), key=lambda kv: kv[1]["total_us"], reverse=True)[:14]:
        c = g["calls"]
        avg_gap = (g["gap_sum"] / g["gap_cnt"]) if g["gap_cnt"] > 0 else None
        pts.append(OpPoint(
            name=name, category=g["cat"], calls=c, total_us=g["total_us"],
            avg_us=g["total_us"] / c, flops=g["flops_sum"] / c,
            bytes_moved=g["bytes_sum"] / c, tokens_m=g["m_sum"] / c,
            pipe=g["pipe"], dtype=g["dtype"], measured_gap_us=avg_gap, shape_source=g["src"],
        ))

    fusions = [
        FusionOpportunity(
            pattern=pat, pair_desc=v["pair"], occurrences=v["count"],
            total_dur_us=v["dur"], saved_gap_us=v["gap"],
            saved_bytes=v["bytes"], recommendation=v["rec"],
        )
        for pat, v in sorted(fusion_agg.items(), key=lambda kv: kv[1]["gap"] + kv[1]["dur"], reverse=True)
    ]
    return pts, fusions


# ---------------------------------------------------------------------------
# 4. Analytical Model-on-Hardware Synthesizer (`--model` or `--demo`)
# ---------------------------------------------------------------------------

MODEL_PRESETS: Dict[str, dict] = {
    "deepseek-v3": {
        "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
        "num_key_value_heads": 128, "kv_lora_rank": 512, "q_lora_rank": 1536,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"},
    },
    "deepseek-v4-fp4": {
        "num_hidden_layers": 64, "hidden_size": 7168, "num_attention_heads": 128,
        "num_key_value_heads": 128, "kv_lora_rank": 512, "q_lora_rank": 1536,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"targets": ["mlp.experts"], "weights": {"num_bits": 4, "type": "float"}},
                "group_1": {"targets": ["self_attn", "mlp.shared_experts"], "weights": {"num_bits": 8, "type": "float"}},
            },
            "kv_cache_scheme": {"num_bits": 8, "type": "float"},
        },
    },
    "qwen3-235b": {
        "num_hidden_layers": 94, "hidden_size": 4096, "num_attention_heads": 64,
        "num_key_value_heads": 4, "head_dim": 128,
        "n_routed_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 1536,
        "quantization_config": {"quant_method": "fp8"},
    },
    "llama3-70b": {
        "num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 28672,
    },
}


def load_hf_config(model_spec: str) -> dict:
    low = model_spec.lower()
    if low in MODEL_PRESETS:
        return dict(MODEL_PRESETS[low])
    p = Path(model_spec)
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text())
    if p.is_file():
        return json.loads(p.read_text())
    url = f"https://huggingface.co/{model_spec}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=15.0) as r:
        return json.loads(r.read().decode("utf-8"))


def detect_model_precisions(
    cfg: dict,
    quant: str = "auto",
    attn_quant: str = "auto",
    moe_quant: str = "auto",
    kv_quant: str = "auto",
) -> Dict[str, str]:
    """Infer per-component precisions (dense/attn, routed MoE experts, KV cache) from HF config.json."""
    qcfg = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    qmethod = str(qcfg.get("quant_method", "")).lower()
    qfmt = str(qcfg.get("fmt", "")).lower()
    exp_dt = str(qcfg.get("expert_dtype", "")).lower()
    ignored = " ".join(str(x).lower() for x in (qcfg.get("modules_to_not_convert") or qcfg.get("ignored_layers") or []))

    inferred_base = "bf16"
    inferred_moe = "bf16"
    inferred_attn = "bf16"
    inferred_kv = "bf16"

    if any(k in qmethod + " " + qfmt for k in ("fp4", "mxfp4", "nvfp4")):
        inferred_base = "fp4"
        inferred_moe = "fp4"
        inferred_attn = "bf16" if ("attn" in ignored or "q_proj" in ignored) else "fp8"
        inferred_kv = "fp8"
    elif "fp8" in qmethod or "e4m3" in qfmt:
        inferred_base = "fp8"
        inferred_moe = "fp8"
        inferred_attn = "bf16" if ("attn" in ignored) else "fp8"
        inferred_kv = "fp8"

    if any(k in exp_dt for k in ("fp4", "mxfp4", "nvfp4")):
        inferred_moe = "fp4"
    elif "fp8" in exp_dt:
        inferred_moe = "fp8"

    # Inspect compressed-tensors / modelopt config_groups for mixed-precision (e.g. FP4 experts + FP8 attn)
    groups = qcfg.get("config_groups") or {}
    if isinstance(groups, dict) and groups:
        for grp in groups.values():
            if not isinstance(grp, dict):
                continue
            targets = " ".join(str(t).lower() for t in (grp.get("targets") or []))
            bits = (grp.get("weights") or {}).get("num_bits")
            g_dt = "fp4" if bits == 4 else ("fp8" if bits == 8 else ("bf16" if bits == 16 else None))
            if g_dt is None:
                continue
            if "attn" in targets or "qkv" in targets or "shared_expert" in targets:
                inferred_attn = g_dt
                inferred_base = g_dt
            elif "expert" in targets or "moe" in targets:
                inferred_moe = g_dt
            elif "linear" in targets:
                inferred_attn = g_dt
                inferred_base = g_dt
            else:
                inferred_base = g_dt
                if inferred_moe == "bf16":
                    inferred_moe = g_dt
        kv_scheme = qcfg.get("kv_cache_scheme") or {}
        if isinstance(kv_scheme, dict) and kv_scheme.get("num_bits") == 8:
            inferred_kv = "fp8"

    # Fallback if top-level torch_dtype was bf16 with no quantization_config
    if quant != "auto":
        base = quant
        a_q = attn_quant if attn_quant != "auto" else quant
        m_q = moe_quant if moe_quant != "auto" else (inferred_moe if inferred_moe != "bf16" else quant)
        k_q = kv_quant if kv_quant != "auto" else ("fp8" if quant in ("fp8", "fp4") else "bf16")
    else:
        base = inferred_base
        a_q = attn_quant if attn_quant != "auto" else inferred_attn
        m_q = moe_quant if moe_quant != "auto" else inferred_moe
        k_q = kv_quant if kv_quant != "auto" else inferred_kv

    return {"base": base, "attn": a_q, "moe": m_q, "kv": k_q}


def _bytes_per_param(dtype: str) -> float:
    return {"bf16": 2.0, "fp8": 1.0, "fp4": 0.56}.get(dtype, 2.0)


def model_ops_on_hw(
    cfg: dict,
    hw: HardwareProfile,
    tp: int,
    quant: str,
    batch_decode: int,
    prompt_len: int,
    ctx_len: int,
    attn_quant: str = "auto",
    moe_quant: str = "auto",
    kv_quant: str = "auto",
    prefix_cache_hit: float = 0.0,
    spec_tokens: int = 4,
) -> Tuple[List[OpPoint], Dict[str, str]]:
    """Project an LLM architecture's Prefill + Decode operators onto `hw` with per-op precision tracking."""
    precs = detect_model_precisions(cfg, quant, attn_quant, moe_quant, kv_quant)
    tcfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    b_attn = _bytes_per_param(precs["attn"])
    b_moe = _bytes_per_param(precs["moe"])
    b_kv = _bytes_per_param(precs["kv"])

    layers = int(tcfg.get("num_hidden_layers", 48))
    d = int(tcfg.get("hidden_size", 5120))
    hq = int(tcfg.get("num_attention_heads", 40))
    hkv = int(tcfg.get("num_key_value_heads", max(1, hq // 8)))
    head_dim = int(tcfg.get("head_dim", d // hq))
    inter = int(tcfg.get("intermediate_size", d * 3))
    n_routed = int(tcfg.get("n_routed_experts") or tcfg.get("num_local_experts") or tcfg.get("num_experts") or 0)
    n_shared = int(tcfg.get("n_shared_experts") or 0)
    topk = int(tcfg.get("num_experts_per_tok", 8 if n_routed else 1))
    moe_inter = int(tcfg.get("moe_intermediate_size", inter // 4 if n_routed else inter))

    # MLA / MQA-latent & Sparse Indexer fields
    kv_lora = int(tcfg.get("kv_lora_rank") or (head_dim if (hkv == 1 and tcfg.get("q_lora_rank")) else 0))
    q_lora = int(tcfg.get("q_lora_rank") or 0)
    o_lora = int(tcfg.get("o_lora_rank") or 0)
    rope_dim = int(tcfg.get("qk_rope_head_dim") or 0)
    nope_dim = int(tcfg.get("qk_nope_head_dim") or head_dim)
    v_dim = int(tcfg.get("v_head_dim") or head_dim)
    is_mla = kv_lora > 0
    idx_topk = int(tcfg.get("index_topk") or 0)
    idx_heads = int(tcfg.get("index_n_heads") or 0)
    idx_dim = int(tcfg.get("index_head_dim") or 128)

    eff_prompt = max(1, int(round(prompt_len * max(0.02, 1.0 - prefix_cache_hit))))

    def sim_time_us(flops: float, bytes_m: float, cat: str, m_tok: float, dtype: str, pipe: str = "HBM_READ") -> float:
        if pipe == "P2P":
            bw_gbs = hw.p2p_all_gbs if bytes_m >= 128 * 1024 else hw.p2p_gbs
            return 6.0 + (bytes_m / (bw_gbs * 1e9)) * 1e6
        bw, _ = bw_ceiling_for_pipe(hw, pipe)
        if cat == "MoE" and m_tok < 16:
            # Tiny-tile scale-factor unpack + unaligned expert gather penalty
            bw_eff = bw * (0.46 + 0.34 * (m_tok / 16.0))
        else:
            bw_eff = bw * 0.86
        comp_peak, _ = matrix_ceiling_for_dtype(hw, dtype)
        comp_eff = comp_peak * min(0.86, 0.30 + 0.56 * min(1.0, m_tok / 128.0))
        t_mem_us = (bytes_m / (bw_eff * 1e12)) * 1e6
        t_cmp_us = (flops / (comp_eff * 1e12)) * 1e6
        return hw.graph_launch_us + max(t_mem_us, t_cmp_us)

    pts: List[OpPoint] = []

    # 1. Decode QKV+O Linear (MLA low-rank / compressed vs GQA)
    if is_mla:
        # wqkv_a [d, q_lora + kv_lora + rope_dim] + wq_b [q_lora, (hq/tp)*(nope+rope)] + wo(_a/_b)
        q_params = (d * q_lora + q_lora * (hq // tp) * (nope_dim + rope_dim)) if q_lora > 0 else (d * (hq // tp) * (nope_dim + rope_dim))
        kv_params = d * (kv_lora + rope_dim)
        o_params = ((hq // tp) * v_dim * o_lora + o_lora * d) if o_lora > 0 else ((hq // tp) * v_dim * d)
        attn_proj_params = float(q_params + kv_params + o_params)
        qkv_n = int(round(attn_proj_params / d))
    else:
        qkv_n = (hq + 2 * hkv) * head_dim // tp
    f_dec_qkv = 2.0 * batch_decode * d * qkv_n
    b_dec_qkv = d * qkv_n * b_attn + batch_decode * (d + qkv_n) * 2.0
    u = sim_time_us(f_dec_qkv, b_dec_qkv, "GEMM", batch_decode, precs["attn"], "HBM_READ")
    pts.append(OpPoint(f"Decode QKV+O ({precs['attn'].upper()},M={batch_decode})", "GEMM", layers, u * layers, u,
                       f_dec_qkv, b_dec_qkv, batch_decode, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))

    # 2. Decode FFN / MoE (Routed Experts + Optional Shared Expert)
    if n_routed > 0:
        active_e = min(n_routed, batch_decode * topk)
        m_per_e = max(1.0, (batch_decode * topk) / n_routed)
        w_per_e = 3.0 * d * (moe_inter / tp) * b_moe
        f_moe = 2.0 * batch_decode * topk * 3.0 * d * (moe_inter / tp)
        b_moe_bytes = active_e * w_per_e + batch_decode * d * 4.0
        u = sim_time_us(f_moe, b_moe_bytes, "MoE", m_per_e, precs["moe"], "HBM_READ")
        pts.append(OpPoint(f"Decode MoE ({precs['moe'].upper()},m/e={m_per_e:.1f})", "MoE", layers, u * layers, u,
                           f_moe, b_moe_bytes, m_per_e, pipe="HBM_READ", dtype=precs["moe"], shape_source="model"))
        if n_shared > 0:
            f_sh = 2.0 * batch_decode * n_shared * 3.0 * d * (moe_inter / tp)
            b_sh = n_shared * 3.0 * d * (moe_inter / tp) * b_attn + batch_decode * d * 4.0
            u_sh = sim_time_us(f_sh, b_sh, "GEMM", batch_decode, precs["attn"], "HBM_READ")
            pts.append(OpPoint(f"Decode Shared FFN ({precs['attn'].upper()})", "GEMM", layers, u_sh * layers, u_sh,
                               f_sh, b_sh, batch_decode, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))
    else:
        f_ffn = 2.0 * batch_decode * 3.0 * d * (inter / tp)
        b_ffn = 3.0 * d * (inter / tp) * b_attn + batch_decode * d * 4.0
        u = sim_time_us(f_ffn, b_ffn, "GEMM", batch_decode, precs["attn"], "HBM_READ")
        pts.append(OpPoint(f"Decode SwiGLU FFN ({precs['attn'].upper()})", "GEMM", layers, u * layers, u,
                           f_ffn, b_ffn, batch_decode, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))

    # 3. Speculative Verify GEMM + MoE (M = batch_decode * (1 + spec_tokens) verified in 1 weight read!)
    m_spec = batch_decode * max(1, 1 + spec_tokens)
    f_spec = 2.0 * m_spec * d * qkv_n
    b_spec = d * qkv_n * b_attn + m_spec * (d + qkv_n) * 2.0
    u = sim_time_us(f_spec, b_spec, "GEMM", m_spec, precs["attn"], "HBM_READ")
    pts.append(OpPoint(f"SpecDec Verify ({precs['attn'].upper()},M={m_spec})", "GEMM", layers, u * layers, u,
                       f_spec, b_spec, m_spec, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))
    if n_routed > 0 and spec_tokens > 0:
        active_sp_e = min(n_routed, m_spec * topk)
        m_sp_e = max(1.0, (m_spec * topk) / n_routed)
        w_per_e = 3.0 * d * (moe_inter / tp) * b_moe
        f_sp_moe = 2.0 * m_spec * topk * 3.0 * d * (moe_inter / tp)
        b_sp_moe = active_sp_e * w_per_e + m_spec * d * 4.0
        u_spm = sim_time_us(f_sp_moe, b_sp_moe, "MoE", m_sp_e, precs["moe"], "HBM_READ")
        pts.append(OpPoint(f"SpecVerify MoE ({precs['moe'].upper()},m/e={m_sp_e:.1f})", "MoE", layers, u_spm * layers, u_spm,
                           f_sp_moe, b_sp_moe, m_sp_e, pipe="HBM_READ", dtype=precs["moe"], shape_source="model"))

    # 4. Decode Paged KV Attention (MLA latent cache vs standard GQA cache, sparse topk if indexed)
    eff_ctx = min(ctx_len, idx_topk * 8) if idx_topk > 0 else ctx_len
    ctx_tag = f"{eff_ctx//1024}k" if eff_ctx >= 1024 else str(eff_ctx)
    if is_mla:
        kv_bytes = batch_decode * eff_ctx * (kv_lora + rope_dim) * b_kv
        f_dattn = 2.0 * batch_decode * eff_ctx * (hq / tp) * (kv_lora + rope_dim) * 2.0
        attn_lbl = f"Decode MLA Attn ({ctx_tag})"
    else:
        kv_bytes = batch_decode * eff_ctx * 2.0 * (hkv / tp) * head_dim * b_kv
        f_dattn = 4.0 * batch_decode * eff_ctx * (hq / tp) * head_dim
        attn_lbl = f"Decode KV Attn ({ctx_tag})"
    u = sim_time_us(f_dattn, kv_bytes, "Attention", batch_decode, precs["attn"], "HBM_READ")
    pts.append(OpPoint(attn_lbl, "Attention", layers, u * layers, u,
                       f_dattn, kv_bytes, batch_decode, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))

    # Optional Sparse Indexer (DeepSeek V3.2/V4 DSA lightning indexer: FP8 QK score + non-GEMM TopK select)
    if idx_topk > 0 and idx_heads > 0:
        b_idx = batch_decode * min(ctx_len, 65536) * idx_dim * 1.0
        f_idx = 2.0 * batch_decode * min(ctx_len, 65536) * idx_heads * idx_dim
        u_idx = hw.graph_launch_us + (b_idx / (hw.hbm_read_tbs * 1e12)) * 1e6 + 6.0  # non-GEMM top-k select overhead
        pts.append(OpPoint(f"Decode Sparse Indexer (top{idx_topk})", "Attention", layers, u_idx * layers, u_idx,
                           f_idx, b_idx, batch_decode, pipe="HBM_READ", dtype="fp8", shape_source="model"))

    # 5. Prefill Dense / MoE GEMM (scaled by uncached tokens `eff_prompt`)
    f_pre_gemm = 2.0 * eff_prompt * d * qkv_n
    b_pre_gemm = d * qkv_n * b_attn + eff_prompt * (d + qkv_n) * 2.0
    u = sim_time_us(f_pre_gemm, b_pre_gemm, "GEMM", eff_prompt, precs["attn"], "HBM_READ")
    pts.append(OpPoint(f"Prefill QKV ({precs['attn'].upper()},M={eff_prompt})", "GEMM", layers, u * layers, u,
                       f_pre_gemm, b_pre_gemm, eff_prompt, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))

    if n_routed > 0:
        m_pre_e = max(1.0, (eff_prompt * topk) / n_routed)
        active_pre_e = min(n_routed, eff_prompt * topk)
        f_pre_moe = 2.0 * eff_prompt * topk * 3.0 * d * (moe_inter / tp)
        b_pre_moe = active_pre_e * 3.0 * d * (moe_inter / tp) * b_moe + eff_prompt * d * 4.0
        u_pm = sim_time_us(f_pre_moe, b_pre_moe, "MoE", m_pre_e, precs["moe"], "HBM_READ")
        pts.append(OpPoint(f"Prefill MoE ({precs['moe'].upper()},m/e={m_pre_e:.0f})", "MoE", layers, u_pm * layers, u_pm,
                           f_pre_moe, b_pre_moe, m_pre_e, pipe="HBM_READ", dtype=precs["moe"], shape_source="model"))

    # 6. Prefill FlashAttention
    f_pattn = 2.0 * (eff_prompt ** 2) * (hq / tp) * (nope_dim + rope_dim if is_mla else head_dim)
    b_pattn = eff_prompt * ((hq + 2 * (1 if is_mla else hkv)) / tp) * head_dim * 2.0 * 2.0
    u = sim_time_us(f_pattn, b_pattn, "Attention", eff_prompt, precs["attn"], "HBM_READ")
    pts.append(OpPoint(f"Prefill FlashAttn (N={eff_prompt})", "Attention", layers, u * layers, u,
                       f_pattn, b_pattn, eff_prompt, pipe="HBM_READ", dtype=precs["attn"], shape_source="model"))

    # 7. Fused Add + RMSNorm + RoPE (Decode, L2 resident, SFU ceiling)
    b_norm = batch_decode * d * 2.0 * 3.0
    f_norm = batch_decode * d * 5.0
    u = hw.eager_launch_us * 0.25 + (b_norm / (hw.l2_tbs * 1e12)) * 1e6 + (f_norm / (hw.fp32_sfu_tops * 1e12)) * 1e6
    pts.append(OpPoint(f"Decode RMSNorm+RoPE (M={batch_decode})", "Norm/Elem", layers * 2, u * layers * 2, u,
                       f_norm, b_norm, batch_decode, pipe="L2", dtype="sfu", shape_source="model"))

    # 8. Multi-GPU TP AllReduce
    if tp >= 2:
        b_ar_dec = batch_decode * d * 2.0
        u_dec = sim_time_us(b_ar_dec * 0.5, b_ar_dec, "Comm", batch_decode, "bf16", pipe="P2P")
        pts.append(OpPoint(f"Decode TP{tp} AllReduce", "Comm", layers * 2, u_dec * layers * 2, u_dec,
                           b_ar_dec * 0.5, b_ar_dec, batch_decode, pipe="P2P", dtype="bf16", shape_source="model"))

        b_ar_pre = eff_prompt * d * 2.0
        u_pre = sim_time_us(b_ar_pre * 0.5, b_ar_pre, "Comm", eff_prompt, "bf16", pipe="P2P")
        pts.append(OpPoint(f"Prefill TP{tp} AllReduce", "Comm", layers * 2, u_pre * layers * 2, u_pre,
                           b_ar_pre * 0.5, b_ar_pre, eff_prompt, pipe="P2P", dtype="bf16", shape_source="model"))

    return pts, precs


# ---------------------------------------------------------------------------
# 5. Automated Roofline Bottleneck Analyzer & Optimization Advisor
# ---------------------------------------------------------------------------

def print_analysis_and_advice(
    hw: HardwareProfile,
    ops: List[OpPoint],
    fusions: Optional[List[FusionOpportunity]] = None,
    precs: Optional[Dict[str, str]] = None,
) -> List[str]:
    ridge_bf16 = hw.bf16_matrix_tflops / hw.hbm_read_tbs
    ridge_fp8 = hw.fp8_matrix_tflops / hw.hbm_read_tbs if hw.fp8_matrix_tflops > 0 else 0.0
    ridge_fp4 = hw.fp4_matrix_tflops / hw.hbm_read_tbs if hw.fp4_matrix_tflops > 0 else 0.0
    spec_gap_cmp = (1.0 - hw.bf16_matrix_tflops / max(hw.spec_bf16_dense_tflops, 1.0)) * 100.0
    spec_gap_hbm = (1.0 - hw.hbm_tbs / max(hw.spec_hbm_tbs, 0.1)) * 100.0

    print("\n===================================================================================================================")
    print(f"  Hardware Target : {hw.name}  ({hw.sms_or_cus} SMs/CUs | {hw.vram_gib:.0f} GiB HBM | {hw.l2_mib:.0f} MiB L2)")
    print(f"  HBM & Caches    : {hw.hbm_tbs:.2f} TB/s R+W / {hw.hbm_read_tbs:.2f} TB/s Read ({100-spec_gap_hbm:.0f}% of {hw.spec_hbm_tbs:.2f} spec) | "
          f"{hw.l2_tbs:.1f} TB/s L2 | {hw.lds_tbs:.0f} TB/s LDS")
    fp8_str = f" | FP8={hw.fp8_matrix_tflops:.0f} TF (ridge={ridge_fp8:.0f})" if hw.fp8_matrix_tflops > 0 else ""
    fp4_str = f" | FP4={hw.fp4_matrix_tflops:.0f} TF (ridge={ridge_fp4:.0f})" if hw.fp4_matrix_tflops > 0 else " | FP4=Emulated(upscale->FP8)"
    print(f"  Sustained Math  : BF16={hw.bf16_matrix_tflops:.0f} TF (ridge={ridge_bf16:.0f}){fp8_str}{fp4_str}")
    print(f"  VALU / SFU / P2P: VALU={hw.fp32_valu_tflops:.0f} TF | SFU={hw.fp32_sfu_tops:.0f} TOP/s | "
          f"Launch={hw.eager_launch_us:.2f}us/{hw.graph_launch_us:.2f}us | P2P={hw.p2p_gbs:.0f} (Bus={hw.p2p_all_gbs:.0f}) GB/s")
    if precs:
        print(f"  Model Precision : Attn/Dense={precs['attn'].upper()} | MoE Experts={precs['moe'].upper()} | KV Cache={precs['kv'].upper()}")
    print("===================================================================================================================\n")

    tot_us = sum(o.total_us for o in ops) or 1.0
    print(f"  {'Operator / Kernel':<34} | {'Dtype':>5} | {'Time%':>5} | {'Avg us':>7} | {'OI(F/B)':>7} | {'TFLOP/s':>7} | {'TB/s':>6} | {'%Roof':>6} | Limiting Regime")
    print("  " + "-" * 118)
    for o in ops:
        ceil_tf, regime = ceiling_for_op(o, hw)
        pct_roof = min(135.0, (o.achieved_tflops / max(ceil_tf, 1e-6)) * 100.0)
        eff_gap = o.measured_gap_us if o.measured_gap_us is not None else hw.eager_launch_us
        if o.avg_us < eff_gap * 2.5 and pct_roof < 45.0:
            regime = "Launch-Latency"
        dt_lbl = DTYPE_BADGE.get(o.dtype, o.dtype.upper())
        tag = "*" if o.shape_source == "graph_est" else " "
        print(f" {tag}{o.name:<34} | {dt_lbl:>5} | {100*o.total_us/tot_us:4.1f}% | {o.avg_us:7.1f} | {o.oi:7.1f} | "
              f"{o.achieved_tflops:7.1f} | {o.achieved_tbs:6.2f} | {pct_roof:5.1f}% | {regime}")

    if any(o.shape_source == "graph_est" for o in ops):
        print("  (* shape_source=graph_est: CUDA/HIP Graph replay omitted cpu_op Input Dims; OI/bytes estimated from grid/kernel name)")

    if fusions:
        print("\n--- Kernel Fusion & Optimization Opportunities (from Stream Timeline) -----------")
        for f in fusions[:8]:
            print(f"  * {f.pattern:<28} ({f.occurrences:>3}x, {f.pair_desc}) "
                  f"-> ~{f.saved_gap_us:.0f} us launch gaps + {f.saved_bytes/(1024**2):.1f} MiB traffic | {f.recommendation}")

    advice: List[str] = []
    advice.append(
        f"[Sustained Clocks vs. Datasheet] Real-entropy sustained BF16 ({hw.bf16_matrix_tflops:.0f} TF vs {hw.spec_bf16_dense_tflops:.0f} TF spec, "
        f"-{spec_gap_cmp:.0f}%) reflects board TDP throttling; HBM ({hw.hbm_tbs:.2f} R+W / {hw.hbm_read_tbs:.2f} Read TB/s) reflects tRFC & turnaround."
    )

    if hw.fp4_matrix_tflops <= 0.0 and any(o.dtype == "fp4" for o in ops):
        advice.append(
            f"[FP4 -> FP8 Upscale on {hw.name}] This GPU has no native FP4 tensor cores: FP4 experts save ~44% HBM weight bytes during "
            f"low-batch decode, but high-batch prefill upscales to FP8 (~{hw.fp8_matrix_tflops:.0f} TF ceiling) instead of native FP4."
        )

    low_oi_gemms = [o for o in ops if o.category in ("GEMM", "MoE") and o.oi < ridge_bf16 * 0.30]
    if low_oi_gemms:
        advice.append(
            f"[HBM Weight-Bound GEMM/MoE] {low_oi_gemms[0].name} operates at OI={low_oi_gemms[0].oi:.1f} FLOPs/B "
            f"(left of the {ridge_bf16:.0f} FLOPs/B BF16 Ridge Point). -> Use FP8/MXFP4 weight compression + Speculative Decoding (MTP/EAGLE3)."
        )

    tiny_moe = [o for o in ops if o.category == "MoE" and o.tokens_m < 16.0]
    if tiny_moe:
        advice.append(
            f"[MoE Tile-Padding Waste] {tiny_moe[0].name} averages m={tiny_moe[0].tokens_m:.1f} tokens/expert "
            f"(16x16 tiles pad unused rows). -> Use fused MoE kernels (AITER ck_moe / Triton / DeepGEMM masked grouped GEMM)."
        )

    kv_heavy = [o for o in ops if o.category == "Attention" and o.oi < 25.0 and o.bytes_moved > 8 * 1024**2]
    if kv_heavy:
        advice.append(
            f"[KV-Cache HBM Bottleneck] {kv_heavy[0].name} streams {kv_heavy[0].bytes_moved/(1024**2):.1f} MiB per layer at OI={kv_heavy[0].oi:.1f}. "
            f"-> Enable `--kv-cache-dtype fp8_e4m3`, Radix/Prefix Caching (skips prefill!), or MLA/Sparse Attention."
        )

    short_ops = [o for o in ops if o.avg_us < hw.eager_launch_us * 3.0]
    if short_ops:
        advice.append(
            f"[Kernel Dispatch Overhead] {len(short_ops)} kernel group(s) run in <{hw.eager_launch_us*3:.1f} us "
            f"(eager gap {hw.eager_launch_us:.1f} us vs {hw.graph_launch_us:.2f} us Graph). -> Ensure CUDA/HIP Graphs and fuse Norm+Quant/AllReduce."
        )

    small_comm = [o for o in ops if o.pipe == "P2P" and o.bytes_moved <= 512 * 1024]
    if small_comm:
        advice.append(
            f"[Latency-Bound Multi-GPU Collectives] {small_comm[0].name} moves {small_comm[0].bytes_moved/1024:.0f} KiB/call "
            f"(wire latency dominates {hw.p2p_all_gbs:.0f} GB/s bus BW). -> Enable custom 1-shot P2P AllReduce+RMSNorm fusion."
        )

    print("\n--- Automated Bottleneck Diagnosis & Optimization Playbook ---------------------")
    for idx, line in enumerate(advice, 1):
        print(f"  {idx}. {line}")
    print()
    return advice


# ---------------------------------------------------------------------------
# 6. Multi-Dimensional 4-Panel SVG Roofline Renderer (Zero External Deps!)
# ---------------------------------------------------------------------------

def render_4panel_svg(
    hw: HardwareProfile,
    ops: List[OpPoint],
    out_path: Path,
    subtitle: str,
    precision_view: str = "all",
):
    W, H = 1240, 900
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; background: #f8fafc;">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f8fafc"/>',
        f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="#0f172a">'
        f'Multi-Dimensional GPU Inference Roofline Atlas — {hw.name}</text>',
        f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12.5" fill="#475569">{subtitle}</text>',
    ]

    lx = 280
    for cat, col in CAT_COLOR.items():
        svg.append(f'<circle cx="{lx}" cy="70" r="5.5" fill="{col}"/>')
        svg.append(f'<text x="{lx + 10}" y="74" font-size="12" font-weight="600" fill="#1e293b">{cat}</text>')
        lx += 145

    def panel_box(px, py, pw, ph, title, x_lbl, y_lbl):
        svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>')
        svg.append(f'<text x="{px + 16}" y="{py + 24}" font-size="13.5" font-weight="700" fill="#0f172a">{title}</text>')
        ix, iy, iw, ih = px + 64, py + 42, pw - 86, ph - 82
        svg.append(f'<line x1="{ix}" y1="{iy}" x2="{ix}" y2="{iy+ih}" stroke="#334155" stroke-width="1.3"/>')
        svg.append(f'<line x1="{ix}" y1="{iy+ih}" x2="{ix+iw}" y2="{iy+ih}" stroke="#334155" stroke-width="1.3"/>')
        svg.append(f'<text x="{ix + iw/2}" y="{py + ph - 12}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">{x_lbl}</text>')
        svg.append(f'<text x="{px + 16}" y="{iy + ih/2}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155" '
                   f'transform="rotate(-90 {px + 16} {iy + ih/2})">{y_lbl}</text>')
        return ix, iy, iw, ih

    def draw_dot(placed: list, ix: float, iy: float, iw: float, ih: float, cx: float, cy: float, r: float, col: str, text: str):
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{col}" opacity="0.92" stroke="#ffffff" stroke-width="1.2"/>')
        right_side = cx > (ix + iw - 155)
        tx = cx - (r + 5) if right_side else cx + (r + 5)
        anchor = "end" if right_side else "start"
        ty = cy - 4.0 if cy > (iy + ih - 30) else cy + 3.0
        step = -12.0 if cy > (iy + ih * 0.6) else 12.0
        for _ in range(14):
            if any(abs(tx - px0) < 125 and abs(ty - py0) < 11.5 for px0, py0 in placed):
                ty += step
                if ty > iy + ih - 5.0 or ty < iy + 12.0:
                    step = -step
                    ty += 2.0 * step
            else:
                break
        ty = max(iy + 11.0, min(iy + ih - 5.0, ty))
        placed.append((tx, ty))
        svg.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="8.5" font-weight="500" fill="#1e293b">{text}</text>')

    # Determine which precision ceilings to draw in Panel 1
    active_dtypes = {o.dtype for o in ops}
    if precision_view == "all":
        show_fp4 = hw.fp4_matrix_tflops > 0 or ("fp4" in active_dtypes)
        show_fp8 = hw.fp8_matrix_tflops > 0 or ("fp8" in active_dtypes)
        show_bf16 = True
    elif precision_view == "auto":
        show_fp4 = ("fp4" in active_dtypes)
        show_fp8 = ("fp8" in active_dtypes) or not show_fp4
        show_bf16 = ("bf16" in active_dtypes) or True
    else:
        show_fp4 = (precision_view == "fp4")
        show_fp8 = (precision_view == "fp8")
        show_bf16 = (precision_view == "bf16")

    # =========================================================================
    # PANEL 1 (Top-Left): Multi-Precision Compute vs. Memory Roofline (Log-Log)
    # =========================================================================
    p1_title = "1. Multi-Precision Compute vs. Memory Roofline" if precision_view in ("all", "auto") else f"1. {precision_view.upper()} Compute vs. Memory Roofline"
    ix, iy, iw, ih = panel_box(
        22, 88, 586, 380,
        p1_title,
        "Operational Intensity (FLOPs / Byte, log10)",
        "Performance (TFLOP/s, log10)",
    )
    top_tf = max(
        4000.0,
        (hw.spec_fp4_dense_tflops if show_fp4 else 0.0) * 1.25,
        (hw.fp4_matrix_tflops if show_fp4 else 0.0) * 1.35,
        (hw.spec_fp8_dense_tflops if show_fp8 else 0.0) * 1.25,
        hw.spec_bf16_dense_tflops * 1.3,
    )
    oi_min, oi_max = 0.2, 12000.0
    tf_min, tf_max = 0.2, top_tf

    def p1x(oi):
        v = max(oi_min, min(oi_max, oi))
        return ix + (math.log10(v) - math.log10(oi_min)) / (math.log10(oi_max) - math.log10(oi_min)) * iw

    def p1y(tf):
        v = max(tf_min, min(tf_max, tf))
        return iy + ih - (math.log10(v) - math.log10(tf_min)) / (math.log10(tf_max) - math.log10(tf_min)) * ih

    for xv in [0.5, 1, 10, 100, 1000, 10000]:
        xx = p1x(xv)
        svg.append(f'<line x1="{xx:.1f}" y1="{iy}" x2="{xx:.1f}" y2="{iy+ih}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{iy+ih+15}" text-anchor="middle" font-size="10" fill="#64748b">{xv:g}</text>')
    for yv in [1, 10, 100, 1000, 10000]:
        if yv <= tf_max:
            yy = p1y(yv)
            svg.append(f'<line x1="{ix}" y1="{yy:.1f}" x2="{ix+iw}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
            svg.append(f'<text x="{ix-6}" y="{yy+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{yv:g}</text>')

    def draw_roof(bw_tbs, peak_tf, stroke, dash, label, lx_oi, dy=-5, width=2.0, anchor="start"):
        ois = [oi_min * (1.18 ** k) for k in range(78) if oi_min * (1.18 ** k) <= oi_max] + [oi_max]
        pts = " ".join(f"{'M' if i==0 else 'L'}{p1x(x):.1f},{p1y(min(peak_tf, bw_tbs * x)):.1f}" for i, x in enumerate(ois))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}"{da}/>')
        ly = p1y(min(peak_tf, bw_tbs * lx_oi)) + dy
        svg.append(f'<text x="{p1x(lx_oi):.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="8.7" font-weight="600" fill="{stroke}">{label}</text>')

    # Uncoalesced HBM reference slope
    draw_roof(hw.hbm_uncoalesced_tbs, hw.fp32_valu_tflops, "#fca5a5", "2,3",
              f"Uncoalesced ({hw.hbm_uncoalesced_tbs:.2f} TB/s)", 3.0, dy=11, width=1.4)

    # L2 Cache ceiling
    top_math = max(
        hw.fp4_matrix_tflops if (show_fp4 and hw.fp4_matrix_tflops > 0) else 0.0,
        hw.fp8_matrix_tflops if show_fp8 else 0.0,
        hw.bf16_matrix_tflops,
    )
    draw_roof(hw.l2_tbs, top_math, "#10b981", "5,3", f"L2 ({hw.l2_tbs:.1f} TB/s)", 1.3, dy=-6)

    # Datasheet boost spec ceiling for highest shown precision
    if show_fp4 and hw.spec_fp4_dense_tflops > 0:
        draw_roof(hw.spec_hbm_tbs, hw.spec_fp4_dense_tflops, "#94a3b8", "4,4",
                  f"FP4 Spec ({hw.spec_fp4_dense_tflops:.0f} TF)", 11000, dy=-5, width=1.6, anchor="end")
    elif show_fp8 and hw.spec_fp8_dense_tflops > 0:
        draw_roof(hw.spec_hbm_tbs, hw.spec_fp8_dense_tflops, "#94a3b8", "4,4",
                  f"FP8 Spec ({hw.spec_fp8_dense_tflops:.0f} TF)", 11000, dy=-5, width=1.6, anchor="end")
    else:
        draw_roof(hw.spec_hbm_tbs, hw.spec_bf16_dense_tflops, "#94a3b8", "4,4",
                  f"BF16 Spec ({hw.spec_bf16_dense_tflops:.0f} TF)", 11000, dy=-5, width=1.6, anchor="end")

    # Empirical sustained math roofs (labels anchored cleanly at right edge)
    if show_fp4:
        fp4_tf, _ = matrix_ceiling_for_dtype(hw, "fp4")
        lbl = f"FP4 ({fp4_tf:.0f} TF)" if hw.fp4_matrix_tflops > 0 else f"FP4->FP8 ({fp4_tf:.0f} TF)"
        draw_roof(hw.hbm_read_tbs, fp4_tf, "#7c3aed", "6,2" if hw.fp4_matrix_tflops <= 0 else "",
                  lbl, 11000, dy=-4, width=2.2, anchor="end")
    if show_fp8 and hw.fp8_matrix_tflops > 0:
        draw_roof(hw.hbm_read_tbs, hw.fp8_matrix_tflops, "#2563eb", "",
                  f"FP8 ({hw.fp8_matrix_tflops:.0f} TF)", 11000, dy=-4, width=2.2, anchor="end")
    if show_bf16:
        draw_roof(hw.hbm_read_tbs, hw.bf16_matrix_tflops, "#0f172a", "",
                  f"BF16 ({hw.hbm_read_tbs:.2f} TB/s, {hw.bf16_matrix_tflops:.0f} TF)", 11000, dy=11, width=2.2, anchor="end")

    draw_roof(hw.hbm_tbs, hw.fp32_valu_tflops, "#d97706", "3,3",
              f"FP32 VALU ({hw.fp32_valu_tflops:.0f} TF)", 11000, dy=-4, width=1.6, anchor="end")
    draw_roof(hw.hbm_tbs, hw.fp32_sfu_tops, "#b45309", "2,2",
              f"SFU ({hw.fp32_sfu_tops:.0f} TOP/s)", 11000, dy=11, width=1.4, anchor="end")

    # Ridge lines for active primary precision
    primary_dt = "fp4" if (show_fp4 and hw.fp4_matrix_tflops > 0) else ("fp8" if show_fp8 else "bf16")
    prim_tf, _ = matrix_ceiling_for_dtype(hw, primary_dt)
    ridge = prim_tf / hw.hbm_read_tbs
    rx = p1x(ridge)
    svg.append(f'<line x1="{rx:.1f}" y1="{p1y(prim_tf):.1f}" x2="{rx:.1f}" y2="{iy+ih}" stroke="#ef4444" stroke-dasharray="2,3"/>')
    svg.append(f'<text x="{rx-4:.1f}" y="{iy+ih-8}" text-anchor="end" font-size="9.2" fill="#dc2626">{primary_dt.upper()} Ridge={ridge:.0f}</text>')

    tot_us = sum(o.total_us for o in ops) or 1.0
    placed1: list = []
    for o in ops:
        if o.pipe == "P2P":
            continue
        ceil_tf, _ = ceiling_for_op(o, hw)
        cx, cy = p1x(o.oi), p1y(o.achieved_tflops)
        ceil_y = p1y(ceil_tf)
        svg.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{ceil_y:.1f}" stroke="#cbd5e1" stroke-dasharray="2,2"/>')
        r = 4.5 + min(5.5, math.sqrt(100.0 * o.total_us / tot_us))
        badge = DTYPE_BADGE.get(o.dtype, o.dtype.upper())
        lbl = o.name[:22] if badge in o.name.upper() else f"{o.name[:17]} [{badge}]"
        draw_dot(placed1, ix, iy, iw, ih, cx, cy, r, CAT_COLOR[o.category], lbl)

    # =========================================================================
    # PANEL 2 (Top-Right): All Memory & Interconnect Bandwidths vs Payload Size
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        632, 88, 586, 380,
        "2. All Data-Movement Pipes vs. Payload Size (L2 / HBM / P2P / PCIe)",
        "Payload Bytes Moved per Kernel or Collective (log2)",
        "Achieved Bandwidth (GB/s, log10)",
    )
    b_min, b_max = 4 * 1024.0, 2.0 * 1024**3
    gb_min, gb_max = 1.0, 30000.0

    def p2x(b):
        v = max(b_min, min(b_max, b))
        return ix + (math.log2(v) - math.log2(b_min)) / (math.log2(b_max) - math.log2(b_min)) * iw

    def p2y(gbs):
        v = max(gb_min, min(gb_max, gbs))
        return iy + ih - (math.log10(v) - math.log10(gb_min)) / (math.log10(gb_max) - math.log10(gb_min)) * ih

    for bv, bl in [(4096, "4K"), (65536, "64K"), (1048576, "1M"), (16777216, "16M"), (268435456, "256M"), (2147483648, "2G")]:
        xx = p2x(bv)
        svg.append(f'<line x1="{xx:.1f}" y1="{iy}" x2="{xx:.1f}" y2="{iy+ih}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{iy+ih+15}" text-anchor="middle" font-size="10" fill="#64748b">{bl}</text>')
    for yv, yl in [(10, "10 GB/s"), (100, "100 GB/s"), (1000, "1 TB/s"), (10000, "10 TB/s")]:
        yy = p2y(yv)
        svg.append(f'<line x1="{ix}" y1="{yy:.1f}" x2="{ix+iw}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ix-6}" y="{yy+3:.1f}" text-anchor="end" font-size="9.5" fill="#64748b">{yl}</text>')

    bs_grid = [b_min * (2 ** (k * 0.5)) for k in range(39)]
    l2_cap = hw.l2_mib * 0.85 * 1024**2
    pts_mem = []
    for b in bs_grid:
        hit = 1.0 / (1.0 + (b / l2_cap) ** 4)
        bw_gbs = (hit * hw.l2_tbs + (1.0 - hit) * hw.hbm_read_tbs) * 1000.0
        t_s = hw.graph_launch_us * 1e-6 + b / (bw_gbs * 1e9)
        pts_mem.append((b, (b / t_s) / 1e9))
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_mem)) +
               '" fill="none" stroke="#0f172a" stroke-width="2.2"/>')
    svg.append(f'<text x="{p2x(24*1024**2):.1f}" y="{p2y(hw.l2_tbs*950)-6:.1f}" font-size="9.2" font-weight="600" fill="#0f172a">On-Chip L2 -> HBM Read Cliff</text>')

    pts_p2p = [(b, (b / (6e-6 + b / (hw.p2p_all_gbs * 1e9))) / 1e9) for b in bs_grid]
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_p2p)) +
               '" fill="none" stroke="#dc2626" stroke-width="2"/>')
    svg.append(f'<text x="{p2x(160*1024**2):.1f}" y="{p2y(hw.p2p_all_gbs)-6:.1f}" font-size="9.2" font-weight="600" fill="#dc2626">All-GPU P2P Bus ({hw.p2p_all_gbs:.0f} GB/s)</text>')

    pts_pcie = [(b, (b / (10e-6 + b / (hw.pcie_gbs * 1e9))) / 1e9) for b in bs_grid]
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_pcie)) +
               '" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3"/>')
    svg.append(f'<text x="{p2x(160*1024**2):.1f}" y="{p2y(hw.pcie_gbs)+13:.1f}" font-size="9.2" font-weight="600" fill="#475569">Host PCIe ({hw.pcie_gbs:.0f} GB/s)</text>')

    placed2: list = []
    for o in ops:
        cx, cy = p2x(o.bytes_moved), p2y(o.achieved_tbs * 1000.0)
        draw_dot(placed2, ix, iy, iw, ih, cx, cy, 5.0, CAT_COLOR[o.category], o.name[:22])

    # =========================================================================
    # PANEL 3 (Bottom-Left): Dispatch / Launch Overhead vs Kernel Duration (us)
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        22, 494, 586, 380,
        "3. Kernel Dispatch Efficiency vs. Kernel Duration (Eager vs. Graphs)",
        "Mean GPU Kernel Duration (microseconds, log10)",
        "Effective GPU Busy % (t_kern / (t_kern + t_gap))",
    )
    u_min, u_max = 0.5, 2000.0

    def p3x(u):
        v = max(u_min, min(u_max, u))
        return ix + (math.log10(v) - math.log10(u_min)) / (math.log10(u_max) - math.log10(u_min)) * iw

    def p3y(pct):
        return iy + ih - (max(0.0, min(100.0, pct)) / 100.0) * ih

    for uv in [1, 5, 10, 50, 100, 500, 1000]:
        xx = p3x(uv)
        svg.append(f'<line x1="{xx:.1f}" y1="{iy}" x2="{xx:.1f}" y2="{iy+ih}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{iy+ih+15}" text-anchor="middle" font-size="10" fill="#64748b">{uv} us</text>')
    for yv in [0, 25, 50, 75, 100]:
        yy = p3y(yv)
        svg.append(f'<line x1="{ix}" y1="{yy:.1f}" x2="{ix+iw}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ix-6}" y="{yy+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{yv}%</text>')

    us_grid = [u_min * (1.2 ** k) for k in range(48) if u_min * (1.2 ** k) <= u_max]
    eager_curve = " ".join(f"{'M' if i==0 else 'L'}{p3x(u):.1f},{p3y(100*u/(u+hw.eager_launch_us)):.1f}" for i, u in enumerate(us_grid))
    graph_curve = " ".join(f"{'M' if i==0 else 'L'}{p3x(u):.1f},{p3y(100*u/(u+hw.graph_launch_us)):.1f}" for i, u in enumerate(us_grid))
    svg.append(f'<path d="{eager_curve}" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="5,3"/>')
    svg.append(f'<path d="{graph_curve}" fill="none" stroke="#059669" stroke-width="2.2"/>')
    svg.append(f'<text x="{p3x(2.2):.1f}" y="{p3y(92):.1f}" font-size="9.2" font-weight="600" fill="#059669">CUDA/HIP Graph ({hw.graph_launch_us:.2f} us gap)</text>')
    svg.append(f'<text x="{p3x(12):.1f}" y="{p3y(68):.1f}" font-size="9.2" font-weight="600" fill="#dc2626">Eager Launch ({hw.eager_launch_us:.1f} us gap)</text>')

    placed3: list = []
    for o in ops:
        gap = o.measured_gap_us if o.measured_gap_us is not None else hw.eager_launch_us
        busy = 100.0 * o.avg_us / max(o.avg_us + gap, 0.01)
        cx, cy = p3x(o.avg_us), p3y(busy)
        draw_dot(placed3, ix, iy, iw, ih, cx, cy, 5.0, CAT_COLOR[o.category], o.name[:22])

    # =========================================================================
    # PANEL 4 (Bottom-Right): Multi-Precision Matrix Utilization vs Batch M
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        632, 494, 586, 380,
        "4. Precision-Matched Matrix Utilization vs. Tokens M (Batch Knee)",
        "Tokens per GEMM or Routed MoE Expert M (log2)",
        "% of Op's Native Sustained Roof (BF16 / FP8 / FP4)",
    )
    m_min, m_max = 1.0, 8192.0

    def p4x(m):
        v = max(m_min, min(m_max, m))
        return ix + (math.log2(v) - math.log2(m_min)) / (math.log2(m_max) - math.log2(m_min)) * iw

    def p4y(pct):
        return iy + ih - (max(0.0, min(105.0, pct)) / 105.0) * ih

    for mv in [1, 4, 16, 64, 256, 1024, 4096]:
        xx = p4x(mv)
        svg.append(f'<line x1="{xx:.1f}" y1="{iy}" x2="{xx:.1f}" y2="{iy+ih}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{iy+ih+15}" text-anchor="middle" font-size="10" fill="#64748b">M={mv}</text>')
    for yv in [0, 25, 50, 75, 100]:
        yy = p4y(yv)
        svg.append(f'<line x1="{ix}" y1="{yy:.1f}" x2="{ix+iw}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ix-6}" y="{yy+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{yv}%</text>')

    ms_grid = [2 ** (k * 0.25) for k in range(53)]
    for dt_k, b_w, col, lbl, ly_pct in [
        ("bf16", 2.0, "#0f172a", "BF16 Weights", 52),
        ("fp8",  1.0, "#2563eb", "FP8 Weights",  68),
        ("fp4",  0.56, "#7c3aed", "FP4 Weights", 83),
    ]:
        ceil_k, _ = matrix_ceiling_for_dtype(hw, dt_k)
        pts_m = []
        for m in ms_grid:
            oi_m = (2.0 * m) / b_w
            util = min(86.0, 100.0 * (hw.hbm_read_tbs * oi_m) / max(ceil_k, 1.0))
            pts_m.append((m, util))
        d_str = " ".join(f"{'M' if i==0 else 'L'}{p4x(x):.1f},{p4y(y):.1f}" for i, (x, y) in enumerate(pts_m))
        da = ' stroke-dasharray="4,3"' if (dt_k == "fp4" and hw.fp4_matrix_tflops <= 0) else ""
        svg.append(f'<path d="{d_str}" fill="none" stroke="{col}" stroke-width="2"{da}/>')
        knee_m = (ceil_k / hw.hbm_read_tbs) * (b_w / 2.0) * 0.65
        suffix = " (FP8 upscale)" if (dt_k == "fp4" and hw.fp4_matrix_tflops <= 0) else ""
        svg.append(f'<text x="{p4x(knee_m):.1f}" y="{p4y(ly_pct):.1f}" font-size="8.8" font-weight="600" fill="{col}">{lbl}{suffix}</text>')

    placed4: list = []
    for o in ops:
        if o.category not in ("GEMM", "MoE", "Attention") or o.dtype == "sfu":
            continue
        ceil_k, _ = matrix_ceiling_for_dtype(hw, o.dtype)
        util_pct = min(102.0, 100.0 * o.achieved_tflops / max(ceil_k, 1.0))
        cx, cy = p4x(o.tokens_m), p4y(util_pct)
        badge = DTYPE_BADGE.get(o.dtype, o.dtype.upper())
        lbl4 = o.name[:22] if badge in o.name.upper() else f"{o.name[:17]} [{badge}]"
        draw_dot(placed4, ix, iy, iw, ih, cx, cy, 5.2, CAT_COLOR[o.category], lbl4)

    svg.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg) + "\n")
    print(f"Wrote 4-panel Roofline Atlas SVG -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Multi-dimensional SGLang / vLLM / Model Roofline Analyzer")
    ap.add_argument("--hw", type=Path, help="Empirical hardware JSON from `./gpu_roofline --json hw.json`")
    ap.add_argument("--gpu", choices=sorted(HW_PRESETS.keys()), default="mi355x", help="Hardware preset when --hw not given")
    ap.add_argument("--engine", choices=["sglang", "vllm"], help="Profile a live SGLang or vLLM server")
    ap.add_argument("--url", default="http://127.0.0.1:30000", help="OpenAI-compatible server URL")
    ap.add_argument("--launch-cmd", help="Optional shell command to launch SGLang/vLLM before profiling")
    ap.add_argument("--trace-dir", type=Path, default=Path("/tmp/roofline_traces"), help="Profiler trace output directory")
    ap.add_argument("--trace", type=Path, help="Existing PyTorch Kineto .trace.json or .trace.json.gz")
    ap.add_argument("--model", help="HF repo ID, path to config.json, or preset (deepseek-v3, deepseek-v4-fp4, qwen3-235b, llama3-70b)")
    ap.add_argument("--demo", action="store_true", help="Run mixed-precision MoE LLM analysis + generate 4-panel SVG")
    ap.add_argument("--tp", type=int, default=2, help="Tensor Parallel degree (for --model / --demo)")
    ap.add_argument("--quant", choices=["auto", "bf16", "fp8", "fp4"], default="auto", help="Global weight precision (auto reads config.json)")
    ap.add_argument("--attn-quant", choices=["auto", "bf16", "fp8", "fp4"], default="auto", help="Override Attention/Dense precision")
    ap.add_argument("--moe-quant", choices=["auto", "bf16", "fp8", "fp4"], default="auto", help="Override routed MoE expert precision")
    ap.add_argument("--kv-quant", choices=["auto", "bf16", "fp8", "fp4"], default="auto", help="Override KV cache precision")
    ap.add_argument("--precision-view", choices=["all", "auto", "bf16", "fp8", "fp4"], default="all", help="Which compute ceilings to render in Panel 1")
    ap.add_argument("--prefix-cache-hit", type=float, default=0.0, help="Prefix cache hit ratio 0.0..0.98 (scales uncached prefill tokens)")
    ap.add_argument("--spec-tokens", type=int, default=4, help="Speculative draft tokens verified per step (M_verify = BS * (1 + spec_tokens))")
    ap.add_argument("--batch-size", type=int, default=4, help="Concurrent decode batch size")
    ap.add_argument("--prompt-len", type=int, default=4096, help="Prefill prompt tokens")
    ap.add_argument("--output-len", type=int, default=16, help="Decode tokens to profile on live server")
    ap.add_argument("--ctx-len", type=int, default=32768, help="Decode KV cache context length (for --model / --demo)")
    ap.add_argument("-o", "--output", type=Path, default=Path("assets/multi_bottleneck_roofline.svg"))
    args = ap.parse_args()

    hw = load_hw_json(args.hw) if args.hw else HW_PRESETS[args.gpu]

    proc = None
    fusions: List[FusionOpportunity] = []
    precs: Optional[Dict[str, str]] = None
    try:
        if args.launch_cmd:
            print(f"Launching server: {args.launch_cmd}")
            proc = subprocess.Popen(args.launch_cmd, shell=True)
            for _ in range(180):
                try:
                    urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=2.0)
                    break
                except Exception:
                    time.sleep(2.0)

        if args.engine:
            trace_file = profile_live_engine(
                args.engine, args.url, args.trace_dir, args.batch_size, args.prompt_len, args.output_len
            )
            ops, fusions = parse_kineto_trace(trace_file, hw)
            subtitle = f"Live {args.engine.upper()} Kineto Profile ({trace_file.name}: bs={args.batch_size}, prompt={args.prompt_len})"
        elif args.trace:
            ops, fusions = parse_kineto_trace(args.trace, hw)
            subtitle = f"PyTorch Kineto Trace ({args.trace.name})"
        elif args.model:
            cfg = load_hf_config(args.model)
            ops, precs = model_ops_on_hw(
                cfg, hw, args.tp, args.quant, args.batch_size, args.prompt_len, args.ctx_len,
                args.attn_quant, args.moe_quant, args.kv_quant, args.prefix_cache_hit, args.spec_tokens,
            )
            hit_str = f", Hit={args.prefix_cache_hit*100:.0f}%" if args.prefix_cache_hit > 0 else ""
            mod_lbl = Path(args.model).name if Path(args.model).exists() else args.model
            subtitle = (f"{mod_lbl} (Attn={precs['attn'].upper()}, MoE={precs['moe'].upper()}, KV={precs['kv'].upper()}, "
                        f"TP={args.tp}, BS={args.batch_size}, SpecK={args.spec_tokens} @ {args.ctx_len//1024}k{hit_str})")
        else:
            demo_cfg = MODEL_PRESETS["deepseek-v4-fp4"]
            ops, precs = model_ops_on_hw(
                demo_cfg, hw, args.tp, args.quant, args.batch_size, args.prompt_len, args.ctx_len,
                args.attn_quant, args.moe_quant, args.kv_quant, args.prefix_cache_hit, args.spec_tokens,
            )
            subtitle = (f"Mixed-Precision MoE Reference (Attn={precs['attn'].upper()}, MoE={precs['moe'].upper()}, "
                        f"KV={precs['kv'].upper()}, TP={args.tp}, BS={args.batch_size} @ {args.ctx_len//1024}k ctx)")
    finally:
        if proc is not None:
            proc.terminate()

    print_analysis_and_advice(hw, ops, fusions=fusions, precs=precs)
    render_4panel_svg(hw, ops, args.output, subtitle, precision_view=args.precision_view)


if __name__ == "__main__":
    main()
