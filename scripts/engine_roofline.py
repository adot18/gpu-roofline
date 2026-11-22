#!/usr/bin/env python3
"""engine_roofline.py — Multi-Dimensional Hardware + Engine (SGLang / vLLM) Roofline Analyzer & Advisor.

Inspired by classic Roofline tools (e.g. KrxGu/RoofLine_Plot), extended to plot ALL FOUR
orthogonal GPU inference bottlenecks simultaneously against real hardware ceilings measured by
`./gpu_roofline --json hw.json`:

  Panel 1: Compute vs. Memory Hierarchy Roofline  (TFLOP/s     vs. Operational Intensity FLOPs/Byte)
  Panel 2: All Data-Movement Bandwidths           (GB/s..TB/s  vs. Payload Size Bytes: L2, HBM, NVLink/xGMI, PCIe)
  Panel 3: Kernel Dispatch & Graph Efficiency     (GPU Busy %  vs. Mean Kernel Duration us)
  Panel 4: Matrix Tile & MoE Batch Efficiency     (Roofline %  vs. Tokens per GEMM / Expert M)

Supports three workload inputs (zero external Python dependencies required):
  1. Live SGLang or vLLM server (`--engine sglang|vllm --url http://127.0.0.1:30000 --trace-dir /tmp/traces`)
  2. Saved PyTorch Kineto profiler trace (`--trace trace.json[.gz]`)
  3. Model-on-Hardware architecture analyzer (`--model <hf_repo_or_config.json>` or `--demo`)
"""

import argparse
import gzip
import json
import math
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Hardware Ceiling Profiles (Empirical + Datasheet Spec Comparison)
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    name: str
    sms_or_cus: int
    vram_gib: float
    l2_mib: float
    # Empirical achievable ceilings (from `./gpu_roofline --json` or calibrated presets)
    hbm_tbs: float
    hbm_uncoalesced_tbs: float
    l2_tbs: float
    lds_tbs: float
    fp32_valu_tflops: float
    bf16_matrix_tflops: float
    eager_launch_us: float
    graph_launch_us: float
    pcie_gbs: float
    p2p_gbs: float
    # Datasheet theoretical dense peaks (to visualize why 100% spec is unreachable!)
    spec_hbm_tbs: float
    spec_bf16_dense_tflops: float


HW_PRESETS: Dict[str, HardwareProfile] = {
    "mi355x": HardwareProfile(
        name="AMD Instinct MI355X (gfx950)", sms_or_cus=256, vram_gib=288.0, l2_mib=256.0,
        hbm_tbs=5.85, hbm_uncoalesced_tbs=0.42, l2_tbs=18.5, lds_tbs=78.0,
        fp32_valu_tflops=145.0, bf16_matrix_tflops=1920.0,
        eager_launch_us=2.6, graph_launch_us=0.42, pcie_gbs=54.0, p2p_gbs=410.0,
        spec_hbm_tbs=8.00, spec_bf16_dense_tflops=2500.0,
    ),
    "mi300x": HardwareProfile(
        name="AMD Instinct MI300X (gfx942)", sms_or_cus=304, vram_gib=192.0, l2_mib=256.0,
        hbm_tbs=4.25, hbm_uncoalesced_tbs=0.34, l2_tbs=15.5, lds_tbs=68.0,
        fp32_valu_tflops=138.0, bf16_matrix_tflops=980.0,
        eager_launch_us=2.9, graph_launch_us=0.48, pcie_gbs=52.0, p2p_gbs=360.0,
        spec_hbm_tbs=5.30, spec_bf16_dense_tflops=1307.0,
    ),
    "h100": HardwareProfile(
        name="NVIDIA H100 SXM5 (sm_90)", sms_or_cus=132, vram_gib=80.0, l2_mib=50.0,
        hbm_tbs=2.95, hbm_uncoalesced_tbs=0.26, l2_tbs=11.2, lds_tbs=48.0,
        fp32_valu_tflops=60.0, bf16_matrix_tflops=790.0,
        eager_launch_us=2.2, graph_launch_us=0.35, pcie_gbs=55.0, p2p_gbs=380.0,
        spec_hbm_tbs=3.35, spec_bf16_dense_tflops=989.5,
    ),
    "b200": HardwareProfile(
        name="NVIDIA B200 SXM (sm_100)", sms_or_cus=148, vram_gib=192.0, l2_mib=126.0,
        hbm_tbs=6.60, hbm_uncoalesced_tbs=0.50, l2_tbs=21.0, lds_tbs=90.0,
        fp32_valu_tflops=80.0, bf16_matrix_tflops=1850.0,
        eager_launch_us=2.0, graph_launch_us=0.30, pcie_gbs=58.0, p2p_gbs=760.0,
        spec_hbm_tbs=8.00, spec_bf16_dense_tflops=2250.0,
    ),
}


def load_hw_json(path: Path) -> HardwareProfile:
    d = json.loads(path.read_text())
    dev = d.get("device", path.stem)
    # Match datasheet spec if recognizable, otherwise estimate ~1.25x empirical
    low = dev.lower()
    if "355" in low or "350" in low:
        shbm, sbf16 = 8.0, 2500.0
    elif "300" in low:
        shbm, sbf16 = 5.3, 1307.0
    elif "h100" in low:
        shbm, sbf16 = 3.35, 989.5
    elif "b200" in low:
        shbm, sbf16 = 8.0, 2250.0
    else:
        shbm = d["hbm_tbs"] * 1.24
        sbf16 = d["bf16_matrix_tflops"] * 1.28

    p2p = d.get("p2p_read_gbs", 0.0)
    if p2p <= 1.0:
        p2p = 380.0
    return HardwareProfile(
        name=dev,
        sms_or_cus=int(d["sms_or_cus"]),
        vram_gib=float(d["vram_gib"]),
        l2_mib=float(d["l2_mib"]),
        hbm_tbs=float(d["hbm_tbs"]),
        hbm_uncoalesced_tbs=float(d.get("hbm_uncoalesced_tbs", d["hbm_tbs"] * 0.08)),
        l2_tbs=float(d["l2_tbs"]),
        lds_tbs=float(d["shared_lds_tbs"]),
        fp32_valu_tflops=float(d["fp32_valu_tflops"]),
        bf16_matrix_tflops=float(d["bf16_matrix_tflops"]) or (float(d["fp32_valu_tflops"]) * 10.0),
        eager_launch_us=float(d["eager_launch_us"]),
        graph_launch_us=float(d["graph_launch_us"]),
        pcie_gbs=float(d["pcie_h2d_gbs"]),
        p2p_gbs=float(p2p),
        spec_hbm_tbs=shbm,
        spec_bf16_dense_tflops=sbf16,
    )


# ---------------------------------------------------------------------------
# 2. Kernel / Operator Record
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
    pipe: str = "HBM"      # "HBM", "L2", "P2P", "PCIe"

    @property
    def oi(self) -> float:
        return self.flops / max(self.bytes_moved, 1.0)

    @property
    def achieved_tflops(self) -> float:
        return (self.flops / (max(self.avg_us, 0.01) * 1e-6)) / 1e12

    @property
    def achieved_tbs(self) -> float:
        return (self.bytes_moved / (max(self.avg_us, 0.01) * 1e-6)) / 1e12


CAT_COLOR = {
    "GEMM":      "#2563eb",  # blue
    "MoE":       "#7c3aed",  # purple
    "Attention": "#059669",  # emerald
    "Norm/Elem": "#d97706",  # amber
    "Comm":      "#dc2626",  # red
}


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

    # Synthetic prompt of ~prompt_len tokens
    prompt = ("roofline " * max(8, prompt_len // 2)).strip()
    prompts = [prompt] * batch_size

    # Warmup (ensures CUDA/HIP graphs & JIT kernels are already compiled)
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
    time.sleep(1.5)  # allow async trace flush

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
    "float8": 1.0, "c10::float8_e4m3fn": 1.0, "c10::float8_e4m3fnuz": 1.0,
    "unsigned char": 1.0, "uint8": 1.0, "char": 1.0, "int8": 1.0, "long": 8.0, "int": 4.0,
}


def classify_kernel(name: str) -> Tuple[str, str]:
    low = name.lower()
    if any(k in low for k in ("nccl", "rccl", "allreduce", "all_reduce", "all_to_all", "all2all", "msccl", "deepep")):
        return "Comm", "P2P"
    if any(k in low for k in ("moe", "expert", "topk")):
        return "MoE", "HBM"
    if any(k in low for k in ("flash", "fmha", "attn", "attention", "mla", "paged")):
        return "Attention", "HBM"
    if any(k in low for k in ("gemm", "cijk_", "nvjet", "cutlass", "cublas", "hipblas", "scaled_mm", "wvsplitk", "matmul")):
        return "GEMM", "HBM"
    if any(k in low for k in ("norm", "rope", "rotary", "silu", "swiglu", "act_and_mul", "elementwise", "quant")):
        return "Norm/Elem", "HBM"
    return "Norm/Elem", "HBM"


def estimate_from_dims(cat: str, dims: list, dtypes: list, explicit_flops: float) -> Tuple[float, float, float]:
    """Infer (flops, bytes_moved, tokens_m) from Kineto CPU op Input Dims + Input type."""
    valid_dims = [d for d in dims if isinstance(d, list) and len(d) >= 1 and all(isinstance(x, int) and x > 0 for x in d)]
    b_elem = 2.0
    for dt in dtypes:
        if isinstance(dt, str) and dt.lower() in DTYPE_BYTES:
            b_elem = DTYPE_BYTES[dt.lower()]
            break

    total_in_bytes = 0.0
    for idx, d in enumerate(valid_dims):
        eb = b_elem
        if idx < len(dtypes) and isinstance(dtypes[idx], str):
            eb = DTYPE_BYTES.get(dtypes[idx].lower(), b_elem)
        n_el = 1
        for x in d:
            n_el *= x
        total_in_bytes += n_el * eb

    tokens_m = float(valid_dims[0][0]) if valid_dims else 1.0

    if explicit_flops > 0:
        bytes_est = max(total_in_bytes * 1.15, 4096.0)
        return explicit_flops, bytes_est, tokens_m

    if cat in ("GEMM", "MoE") and len(valid_dims) >= 2 and len(valid_dims[0]) >= 2 and len(valid_dims[1]) >= 2:
        m = float(math.prod(valid_dims[0][:-1]))
        k = float(valid_dims[0][-1])
        n = float(valid_dims[1][-1] if valid_dims[1][0] == int(k) else valid_dims[1][0])
        if cat == "MoE" and len(valid_dims[1]) == 3:
            # [E, N, K] fused MoE weights: assume topk=8 active experts if m is small
            e_cnt = valid_dims[1][0]
            n = float(valid_dims[1][1])
            active_e = min(float(e_cnt), max(1.0, m * 8.0))
            flops = 2.0 * m * 8.0 * k * n
            bytes_est = m * k * 2.0 + active_e * n * k * b_elem
            return flops, bytes_est, max(1.0, (m * 8.0) / max(e_cnt, 1))
        flops = 2.0 * m * n * k
        bytes_est = m * k * 2.0 + k * n * b_elem + m * n * 2.0
        return flops, bytes_est, m

    if cat == "Attention" and valid_dims:
        # Approximate Q [B_or_M, H, D] + KV cache
        q_el = float(math.prod(valid_dims[0]))
        kv_bytes = max(total_in_bytes - q_el * 2.0, q_el * 16.0)
        flops = 2.0 * kv_bytes  # ~2 FLOPs per KV byte for decode/GQA
        return flops, total_in_bytes + q_el * 2.0, tokens_m

    if cat == "Comm":
        bytes_est = max(total_in_bytes, 16384.0)
        return bytes_est * 0.25, bytes_est, tokens_m

    # Norm / Elementwise: ~4 FLOPs per element, read + write
    bytes_est = max(total_in_bytes * 2.0, 16384.0)
    flops = (bytes_est / max(b_elem, 1.0)) * 2.5
    return flops, bytes_est, tokens_m


def parse_kineto_trace(trace_path: Path) -> List[OpPoint]:
    """Parse a PyTorch Kineto .json or .json.gz trace and correlate CPU shapes -> GPU kernels."""
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

    # Sort CPU ops per thread to match runtime launches -> nearest enclosing CPU op with dims
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

    grouped: Dict[str, dict] = {}
    for gk in gpu_kernels:
        dur = float(gk.get("dur", 0.0))
        if dur <= 0.0:
            continue
        raw_name = gk.get("name", "unknown_kernel")
        short = raw_name.split("(")[0].split("<")[0][-44:]
        cat, pipe = classify_kernel(raw_name)
        corr = (gk.get("args") or {}).get("correlation")
        cpu_op = corr_to_cpu.get(corr)
        c_args = (cpu_op.get("args") if cpu_op else None) or {}
        flops, bytes_m, tok_m = estimate_from_dims(
            cat,
            c_args.get("Input Dims", []),
            c_args.get("Input type", []),
            float(c_args.get("Flops", 0.0) or 0.0),
        )
        g = grouped.setdefault(short, {
            "cat": cat, "pipe": pipe, "calls": 0, "total_us": 0.0,
            "flops_sum": 0.0, "bytes_sum": 0.0, "m_sum": 0.0,
        })
        g["calls"] += 1
        g["total_us"] += dur
        g["flops_sum"] += flops
        g["bytes_sum"] += bytes_m
        g["m_sum"] += tok_m

    pts = []
    for name, g in sorted(grouped.items(), key=lambda kv: kv[1]["total_us"], reverse=True)[:12]:
        c = g["calls"]
        pts.append(OpPoint(
            name=name, category=g["cat"], calls=c, total_us=g["total_us"],
            avg_us=g["total_us"] / c, flops=g["flops_sum"] / c,
            bytes_moved=g["bytes_sum"] / c, tokens_m=g["m_sum"] / c, pipe=g["pipe"],
        ))
    return pts


# ---------------------------------------------------------------------------
# 4. Analytical Model-on-Hardware Synthesizer (`--model` or `--demo`)
# ---------------------------------------------------------------------------

def load_hf_config(model_spec: str) -> dict:
    p = Path(model_spec)
    if p.is_dir() and (p / "config.json").exists():
        return json.loads((p / "config.json").read_text())
    if p.is_file():
        return json.loads(p.read_text())
    url = f"https://huggingface.co/{model_spec}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=15.0) as r:
        return json.loads(r.read().decode("utf-8"))


def model_ops_on_hw(
    cfg: dict, hw: HardwareProfile, tp: int, quant: str, batch_decode: int, prompt_len: int, ctx_len: int
) -> List[OpPoint]:
    """Project an LLM architecture's Prefill + Decode operators onto `hw` to visualize bottlenecks."""
    b_w = {"bf16": 2.0, "fp8": 1.0, "fp4": 0.56}[quant]
    layers = int(cfg.get("num_hidden_layers", 48))
    d = int(cfg.get("hidden_size", 5120))
    hq = int(cfg.get("num_attention_heads", 40))
    hkv = int(cfg.get("num_key_value_heads", max(1, hq // 8)))
    head_dim = int(cfg.get("head_dim", d // hq))
    inter = int(cfg.get("intermediate_size", d * 3))
    n_routed = int(cfg.get("n_routed_experts") or cfg.get("num_local_experts") or cfg.get("num_experts") or 0)
    topk = int(cfg.get("num_experts_per_tok", 8 if n_routed else 1))
    moe_inter = int(cfg.get("moe_intermediate_size", inter // 4 if n_routed else inter))

    def sim_time_us(flops: float, bytes_m: float, cat: str, m_tok: float, pipe: str = "HBM") -> float:
        if pipe == "P2P":
            # Collective wire latency (~6 us) + fabric transfer time
            return 6.0 + (bytes_m / (hw.p2p_gbs * 1e9)) * 1e6
        # Note: across 40-60 layers, per-layer weights & paged KV caches exceed total L2 and stream from HBM;
        # only immediate activations (`pipe == "L2"`) stay resident in L2!
        bw = hw.l2_tbs if pipe == "L2" else hw.hbm_tbs
        if cat == "MoE" and m_tok < 16:
            bw_eff = bw * (0.48 + 0.30 * (m_tok / 16.0))  # tiny-tile scale unpack + unaligned expert gather penalty
        else:
            bw_eff = bw * 0.84
        comp_peak = hw.fp32_valu_tflops if cat == "Norm/Elem" else hw.bf16_matrix_tflops
        comp_eff = comp_peak * min(0.85, 0.30 + 0.55 * min(1.0, m_tok / 128.0))
        t_mem_us = (bytes_m / (bw_eff * 1e12)) * 1e6
        t_cmp_us = (flops / (comp_eff * 1e12)) * 1e6
        return hw.graph_launch_us + max(t_mem_us, t_cmp_us)

    pts: List[OpPoint] = []

    # 1. Decode QKV+O Linear (M = batch_decode)
    qkv_n = (hq + 2 * hkv) * head_dim // tp
    f_dec_qkv = 2.0 * batch_decode * d * qkv_n
    b_dec_qkv = d * qkv_n * b_w + batch_decode * (d + qkv_n) * 2.0
    u = sim_time_us(f_dec_qkv, b_dec_qkv, "GEMM", batch_decode)
    pts.append(OpPoint(f"Decode QKV Linear (M={batch_decode})", "GEMM", layers, u * layers, u, f_dec_qkv, b_dec_qkv, batch_decode))

    # 2. Decode FFN / MoE
    if n_routed > 0:
        active_e = min(n_routed, batch_decode * topk)
        m_per_e = max(1.0, (batch_decode * topk) / n_routed)
        w_per_e = 3.0 * d * (moe_inter / tp) * b_w
        f_moe = 2.0 * batch_decode * topk * 3.0 * d * (moe_inter / tp)
        b_moe = active_e * w_per_e + batch_decode * d * 4.0
        u = sim_time_us(f_moe, b_moe, "MoE", m_per_e)
        pts.append(OpPoint(f"Decode Fused MoE (m/exp={m_per_e:.1f})", "MoE", layers, u * layers, u, f_moe, b_moe, m_per_e))
    else:
        f_ffn = 2.0 * batch_decode * 3.0 * d * (inter / tp)
        b_ffn = 3.0 * d * (inter / tp) * b_w + batch_decode * d * 4.0
        u = sim_time_us(f_ffn, b_ffn, "GEMM", batch_decode)
        pts.append(OpPoint(f"Decode SwiGLU FFN (M={batch_decode})", "GEMM", layers, u * layers, u, f_ffn, b_ffn, batch_decode))

    # 3. Speculative Verify GEMM (M = batch_decode * 5 tokens verified in 1 weight read!)
    m_spec = batch_decode * 5
    f_spec = 2.0 * m_spec * d * qkv_n
    b_spec = d * qkv_n * b_w + m_spec * (d + qkv_n) * 2.0
    u = sim_time_us(f_spec, b_spec, "GEMM", m_spec)
    pts.append(OpPoint(f"SpecDec Verify GEMM (M={m_spec})", "GEMM", layers, u * layers, u, f_spec, b_spec, m_spec))

    # 4. Decode Paged KV Attention
    kv_bytes = batch_decode * ctx_len * 2.0 * (hkv / tp) * head_dim * (1.0 if quant != "bf16" else 2.0)
    f_dattn = 4.0 * batch_decode * ctx_len * (hq / tp) * head_dim
    u = sim_time_us(f_dattn, kv_bytes, "Attention", batch_decode)
    pts.append(OpPoint(f"Decode Paged KV Attn ({ctx_len//1024}k)", "Attention", layers, u * layers, u, f_dattn, kv_bytes, batch_decode))

    # 5. Prefill Dense GEMM (M = prompt_len)
    f_pre_gemm = 2.0 * prompt_len * d * qkv_n
    b_pre_gemm = d * qkv_n * b_w + prompt_len * (d + qkv_n) * 2.0
    u = sim_time_us(f_pre_gemm, b_pre_gemm, "GEMM", prompt_len)
    pts.append(OpPoint(f"Prefill QKV GEMM (M={prompt_len})", "GEMM", layers, u * layers, u, f_pre_gemm, b_pre_gemm, prompt_len))

    # 6. Prefill FlashAttention (causal)
    f_pattn = 2.0 * (prompt_len ** 2) * (hq / tp) * head_dim
    b_pattn = prompt_len * ((hq + 2 * hkv) / tp) * head_dim * 2.0 * 2.0
    u = sim_time_us(f_pattn, b_pattn, "Attention", prompt_len)
    pts.append(OpPoint(f"Prefill FlashAttn (N={prompt_len})", "Attention", layers, u * layers, u, f_pattn, b_pattn, prompt_len))

    # 7. Fused Add + RMSNorm (Decode)
    b_norm = batch_decode * d * 2.0 * 3.0
    f_norm = batch_decode * d * 5.0
    u = hw.eager_launch_us * 0.25 + (b_norm / (hw.l2_tbs * 1e12)) * 1e6
    pts.append(OpPoint(f"Decode RMSNorm+RoPE (M={batch_decode})", "Norm/Elem", layers * 2, u * layers * 2, u, f_norm, b_norm, batch_decode, pipe="L2"))

    # 8. Multi-GPU TP AllReduce (Decode vs Prefill)
    if tp >= 2:
        b_ar_dec = batch_decode * d * 2.0
        u_dec = sim_time_us(b_ar_dec * 0.5, b_ar_dec, "Comm", batch_decode, pipe="P2P")
        pts.append(OpPoint(f"Decode TP{tp} AllReduce", "Comm", layers * 2, u_dec * layers * 2, u_dec, b_ar_dec * 0.5, b_ar_dec, batch_decode, pipe="P2P"))

        b_ar_pre = prompt_len * d * 2.0
        u_pre = sim_time_us(b_ar_pre * 0.5, b_ar_pre, "Comm", prompt_len, pipe="P2P")
        pts.append(OpPoint(f"Prefill TP{tp} AllReduce", "Comm", layers * 2, u_pre * layers * 2, u_pre, b_ar_pre * 0.5, b_ar_pre, prompt_len, pipe="P2P"))

    return pts


# ---------------------------------------------------------------------------
# 5. Automated Roofline Bottleneck Analyzer & Optimization Advisor
# ---------------------------------------------------------------------------

def ceiling_for_op(op: OpPoint, hw: HardwareProfile) -> Tuple[float, str]:
    """Return (empirical_ceiling_tflops_at_oi, limiting_regime_name)."""
    if op.pipe == "P2P":
        p2p_ceil = (hw.p2p_gbs / 1000.0) * op.oi
        return p2p_ceil, "NVLink/xGMI Latency" if op.bytes_moved < 512 * 1024 else "NVLink/xGMI BW"
    bw = hw.l2_tbs if op.pipe == "L2" else hw.hbm_tbs
    mem_ceil = bw * op.oi
    cmp_ceil = hw.fp32_valu_tflops if op.category == "Norm/Elem" else hw.bf16_matrix_tflops
    if mem_ceil < cmp_ceil:
        return mem_ceil, ("L2 Cache BW" if op.pipe == "L2" else "HBM BW-Bound")
    return cmp_ceil, ("FP32 VALU Ceiling" if op.category == "Norm/Elem" else "Matrix Core Ceiling")


def print_analysis_and_advice(hw: HardwareProfile, ops: List[OpPoint]) -> List[str]:
    ridge = hw.bf16_matrix_tflops / hw.hbm_tbs
    spec_gap_cmp = (1.0 - hw.bf16_matrix_tflops / hw.spec_bf16_dense_tflops) * 100.0
    spec_gap_hbm = (1.0 - hw.hbm_tbs / hw.spec_hbm_tbs) * 100.0

    print("\n====================================================================================================")
    print(f"  Hardware Target : {hw.name}  ({hw.sms_or_cus} SMs/CUs | {hw.vram_gib:.0f} GiB HBM | {hw.l2_mib:.0f} MiB L2)")
    print(f"  Empirical Roofs : {hw.hbm_tbs:.2f} TB/s HBM ({100-spec_gap_hbm:.0f}% of {hw.spec_hbm_tbs:.2f} TB/s spec) | "
          f"{hw.bf16_matrix_tflops:.0f} TFLOP/s BF16 ({100-spec_gap_cmp:.0f}% of {hw.spec_bf16_dense_tflops:.0f} spec)")
    print(f"  Secondary Roofs : {hw.l2_tbs:.1f} TB/s L2 | {hw.lds_tbs:.0f} TB/s LDS | {hw.fp32_valu_tflops:.0f} TFLOP/s VALU | "
          f"Ridge = {ridge:.0f} FLOPs/B")
    print(f"  Dispatch & P2P  : {hw.eager_launch_us:.2f} us Eager vs {hw.graph_launch_us:.2f} us Graph | "
          f"{hw.p2p_gbs:.0f} GB/s P2P | {hw.pcie_gbs:.0f} GB/s PCIe")
    print("====================================================================================================\n")

    tot_us = sum(o.total_us for o in ops) or 1.0
    print(f"  {'Operator / Kernel':<34} | {'Time%':>5} | {'Avg us':>7} | {'OI(F/B)':>7} | {'TFLOP/s':>7} | {'TB/s':>6} | {'%Roof':>6} | Limiting Regime")
    print("  " + "-" * 108)
    for o in ops:
        ceil_tf, regime = ceiling_for_op(o, hw)
        pct_roof = min(135.0, (o.achieved_tflops / max(ceil_tf, 1e-6)) * 100.0)
        if o.avg_us < hw.eager_launch_us * 2.5 and pct_roof < 45.0:
            regime = "Launch-Latency"
        print(f"  {o.name:<34} | {100*o.total_us/tot_us:4.1f}% | {o.avg_us:7.1f} | {o.oi:7.1f} | "
              f"{o.achieved_tflops:7.1f} | {o.achieved_tbs:6.2f} | {pct_roof:5.1f}% | {regime}")

    # Generate targeted, actionable optimization prescriptions
    advice: List[str] = []
    advice.append(
        f"[Hardware Spec vs. Reality] Datasheet peak ({hw.spec_bf16_dense_tflops:.0f} TFLOP/s, {hw.spec_hbm_tbs:.2f} TB/s) "
        f"drops by {spec_gap_cmp:.0f}% compute (all-SM TDP power throttling + non-MMA address/scale instructions) and "
        f"{spec_gap_hbm:.0f}% HBM (DRAM tRFC refresh cycles + R/W bus turnaround). Always benchmark against empirical ceilings!"
    )

    low_oi_gemms = [o for o in ops if o.category in ("GEMM", "MoE") and o.oi < ridge * 0.25]
    if low_oi_gemms:
        advice.append(
            f"[HBM Weight-Bound GEMM/MoE] {low_oi_gemms[0].name} operates at OI={low_oi_gemms[0].oi:.1f} FLOPs/B "
            f"(far left of the {ridge:.0f} FLOPs/B Ridge Point). Matrix Cores sit >95% idle waiting on HBM. "
            f"-> Enable FP8 / MXFP4 weight quantization and Speculative Decoding (MTP / EAGLE3) to verify 4-6 tokens per weight read."
        )

    tiny_moe = [o for o in ops if o.category == "MoE" and o.tokens_m < 16.0]
    if tiny_moe:
        advice.append(
            f"[MoE Tile-Padding Waste] {tiny_moe[0].name} averages only m={tiny_moe[0].tokens_m:.1f} tokens/expert "
            f"(16x16 MFMA/WGMMA tiles pad unused rows with zeros). -> Use fused MoE kernels (AITER ck_moe / Triton / DeepGEMM masked grouped GEMM) "
            f"and avoid wide EP until concurrency raises m/expert >= 16."
        )

    kv_heavy = [o for o in ops if o.category == "Attention" and o.oi < 20.0 and o.bytes_moved > 8 * 1024**2]
    if kv_heavy:
        advice.append(
            f"[KV-Cache HBM Bottleneck] {kv_heavy[0].name} streams {kv_heavy[0].bytes_moved/(1024**2):.1f} MiB of KV cache per layer "
            f"at OI={kv_heavy[0].oi:.1f}. -> Enable `--kv-cache-dtype fp8_e4m3`, Prefix Caching (RadixAttention), or MLA/Sparse Attention."
        )

    short_ops = [o for o in ops if o.avg_us < hw.eager_launch_us * 3.0]
    if short_ops:
        advice.append(
            f"[Kernel Dispatch Overhead] {len(short_ops)} op group(s) run in <{hw.eager_launch_us*3:.1f} us "
            f"(eager launch gap is {hw.eager_launch_us:.1f} us vs {hw.graph_launch_us:.2f} us with Graphs!). "
            f"-> Ensure CUDA/HIP Graphs (`--cuda-graph-max-bs`) are active and fuse Add+RMSNorm+Quant."
        )

    small_comm = [o for o in ops if o.pipe == "P2P" and o.bytes_moved <= 512 * 1024]
    if small_comm:
        advice.append(
            f"[Latency-Bound Multi-GPU Collectives] {small_comm[0].name} moves only {small_comm[0].bytes_moved/1024:.0f} KiB per call "
            f"(wire latency dominates {hw.p2p_gbs:.0f} GB/s P2P bandwidth). -> Enable custom 1-shot P2P AllReduce fused with RMSNorm."
        )

    print("\n--- Automated Bottleneck Diagnosis & Optimization Playbook ---------------------")
    for idx, line in enumerate(advice, 1):
        print(f"  {idx}. {line}")
    print()
    return advice


# ---------------------------------------------------------------------------
# 6. Multi-Dimensional 4-Panel SVG Roofline Renderer (Zero External Deps!)
# ---------------------------------------------------------------------------

def render_4panel_svg(hw: HardwareProfile, ops: List[OpPoint], out_path: Path, subtitle: str):
    W, H = 1220, 880
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; background: #f8fafc;">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#f8fafc"/>',
        f'<text x="{W/2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="#0f172a">'
        f'Multi-Dimensional GPU Inference Roofline Atlas — {hw.name}</text>',
        f'<text x="{W/2}" y="50" text-anchor="middle" font-size="12.5" fill="#475569">{subtitle}</text>',
    ]

    # Shared category legend bar at top
    lx = 270
    for cat, col in CAT_COLOR.items():
        svg.append(f'<circle cx="{lx}" cy="70" r="5.5" fill="{col}"/>')
        svg.append(f'<text x="{lx + 10}" y="74" font-size="12" font-weight="600" fill="#1e293b">{cat}</text>')
        lx += 145

    def panel_box(px, py, pw, ph, title, x_lbl, y_lbl):
        svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>')
        svg.append(f'<text x="{px + 16}" y="{py + 24}" font-size="13.5" font-weight="700" fill="#0f172a">{title}</text>')
        ix, iy, iw, ih = px + 62, py + 42, pw - 84, ph - 82
        svg.append(f'<line x1="{ix}" y1="{iy}" x2="{ix}" y2="{iy+ih}" stroke="#334155" stroke-width="1.3"/>')
        svg.append(f'<line x1="{ix}" y1="{iy+ih}" x2="{ix+iw}" y2="{iy+ih}" stroke="#334155" stroke-width="1.3"/>')
        svg.append(f'<text x="{ix + iw/2}" y="{py + ph - 12}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">{x_lbl}</text>')
        svg.append(f'<text x="{px + 16}" y="{iy + ih/2}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155" '
                   f'transform="rotate(-90 {px + 16} {iy + ih/2})">{y_lbl}</text>')
        return ix, iy, iw, ih

    def draw_dot(placed: list, ix: float, iy: float, iw: float, ih: float, cx: float, cy: float, r: float, col: str, text: str):
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{col}" opacity="0.9" stroke="#ffffff" stroke-width="1.2"/>')
        right_side = cx > (ix + iw - 135)
        tx = cx - (r + 5) if right_side else cx + (r + 5)
        anchor = "end" if right_side else "start"
        ty = cy + 3.0
        for _ in range(10):
            if any(abs(tx - px0) < 115 and abs(ty - py0) < 11.5 for px0, py0 in placed):
                ty += 12.0 if ty < (iy + ih - 14) else -12.0
            else:
                break
        ty = max(iy + 10.0, min(iy + ih - 4.0, ty))
        placed.append((tx, ty))
        svg.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="8.8" font-weight="500" fill="#1e293b">{text}</text>')

    # =========================================================================
    # PANEL 1 (Top-Left): Hierarchical Compute vs. Memory Roofline (Log-Log)
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        22, 88, 578, 372,
        "1. Compute vs. Memory Roofline (Datasheet vs. Empirical)",
        "Operational Intensity (FLOPs / Byte, log10)",
        "Performance (TFLOP/s, log10)",
    )
    oi_min, oi_max = 0.2, 8000.0
    tf_min, tf_max = 0.2, 4000.0

    def p1x(oi):
        v = max(oi_min, min(oi_max, oi))
        return ix + (math.log10(v) - math.log10(oi_min)) / (math.log10(oi_max) - math.log10(oi_min)) * iw

    def p1y(tf):
        v = max(tf_min, min(tf_max, tf))
        return iy + ih - (math.log10(v) - math.log10(tf_min)) / (math.log10(tf_max) - math.log10(tf_min)) * ih

    for xv in [0.5, 1, 10, 100, 1000, 5000]:
        xx = p1x(xv)
        svg.append(f'<line x1="{xx:.1f}" y1="{iy}" x2="{xx:.1f}" y2="{iy+ih}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{xx:.1f}" y="{iy+ih+15}" text-anchor="middle" font-size="10" fill="#64748b">{xv:g}</text>')
    for yv in [1, 10, 100, 1000]:
        yy = p1y(yv)
        svg.append(f'<line x1="{ix}" y1="{yy:.1f}" x2="{ix+iw}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ix-6}" y="{yy+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{yv:g}</text>')

    def draw_roof(bw_tbs, peak_tf, stroke, dash, label, lx_oi, dy=-5):
        ois = [oi_min * (1.18 ** k) for k in range(70) if oi_min * (1.18 ** k) <= oi_max] + [oi_max]
        pts = " ".join(f"{'M' if i==0 else 'L'}{p1x(x):.1f},{p1y(min(peak_tf, bw_tbs * x)):.1f}" for i, x in enumerate(ois))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{pts}" fill="none" stroke="{stroke}" stroke-width="2"{da}/>')
        ly = p1y(min(peak_tf, bw_tbs * lx_oi)) + dy
        svg.append(f'<text x="{p1x(lx_oi):.1f}" y="{ly:.1f}" font-size="9.2" font-weight="600" fill="{stroke}">{label}</text>')

    draw_roof(hw.spec_hbm_tbs, hw.spec_bf16_dense_tflops, "#94a3b8", "4,4",
              f"Datasheet ({hw.spec_bf16_dense_tflops:.0f} TF/s, {hw.spec_hbm_tbs:.1f} TB/s)", 360, dy=-5)
    draw_roof(hw.l2_tbs, hw.bf16_matrix_tflops, "#10b981", "5,3", f"L2 ({hw.l2_tbs:.1f} TB/s)", 1.4, dy=-6)
    draw_roof(hw.hbm_tbs, hw.bf16_matrix_tflops, "#0f172a", "",
              f"Empirical ({hw.hbm_tbs:.2f} TB/s, {hw.bf16_matrix_tflops:.0f} TF/s)", 360, dy=11)
    draw_roof(hw.hbm_tbs, hw.fp32_valu_tflops, "#d97706", "3,3", f"FP32 VALU ({hw.fp32_valu_tflops:.0f} TF/s)", 360, dy=-5)

    # Ridge line
    ridge = hw.bf16_matrix_tflops / hw.hbm_tbs
    rx = p1x(ridge)
    svg.append(f'<line x1="{rx:.1f}" y1="{p1y(hw.bf16_matrix_tflops):.1f}" x2="{rx:.1f}" y2="{iy+ih}" stroke="#ef4444" stroke-dasharray="2,3"/>')
    svg.append(f'<text x="{rx-4:.1f}" y="{iy+ih-8}" text-anchor="end" font-size="9.5" fill="#dc2626">Ridge={ridge:.0f} F/B</text>')

    tot_us = sum(o.total_us for o in ops) or 1.0
    placed1: list = []
    for o in ops:
        if o.pipe == "P2P":
            continue
        cx, cy = p1x(o.oi), p1y(o.achieved_tflops)
        ceil_y = p1y(min(hw.fp32_valu_tflops if o.category == "Norm/Elem" else hw.bf16_matrix_tflops,
                         (hw.l2_tbs if o.pipe == "L2" else hw.hbm_tbs) * o.oi))
        svg.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{ceil_y:.1f}" stroke="#cbd5e1" stroke-dasharray="2,2"/>')
        r = 4.5 + min(5.5, math.sqrt(100.0 * o.total_us / tot_us))
        draw_dot(placed1, ix, iy, iw, ih, cx, cy, r, CAT_COLOR[o.category], o.name[:23])

    # =========================================================================
    # PANEL 2 (Top-Right): All Memory & Interconnect Bandwidths vs Payload Size
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        620, 88, 578, 372,
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
    # Curve A: L2 -> HBM
    l2_cap = hw.l2_mib * 0.85 * 1024**2
    pts_mem = []
    for b in bs_grid:
        hit = 1.0 / (1.0 + (b / l2_cap) ** 4)
        bw_gbs = (hit * hw.l2_tbs + (1.0 - hit) * hw.hbm_tbs) * 1000.0
        t_s = hw.graph_launch_us * 1e-6 + b / (bw_gbs * 1e9)
        pts_mem.append((b, (b / t_s) / 1e9))
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_mem)) +
               '" fill="none" stroke="#0f172a" stroke-width="2.2"/>')
    svg.append(f'<text x="{p2x(24*1024**2):.1f}" y="{p2y(hw.l2_tbs*950)-6:.1f}" font-size="9.5" font-weight="600" fill="#0f172a">On-Chip L2 -> HBM Cliff</text>')

    # Curve B: NVLink / xGMI P2P
    pts_p2p = [(b, (b / (6e-6 + b / (hw.p2p_gbs * 1e9))) / 1e9) for b in bs_grid]
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_p2p)) +
               '" fill="none" stroke="#dc2626" stroke-width="2"/>')
    svg.append(f'<text x="{p2x(200*1024**2):.1f}" y="{p2y(hw.p2p_gbs)-6:.1f}" font-size="9.5" font-weight="600" fill="#dc2626">NVLink/xGMI ({hw.p2p_gbs:.0f} GB/s)</text>')

    # Curve C: Host PCIe DMA
    pts_pcie = [(b, (b / (10e-6 + b / (hw.pcie_gbs * 1e9))) / 1e9) for b in bs_grid]
    svg.append('<path d="' + " ".join(f"{'M' if i==0 else 'L'}{p2x(x):.1f},{p2y(y):.1f}" for i, (x, y) in enumerate(pts_pcie)) +
               '" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3"/>')
    svg.append(f'<text x="{p2x(200*1024**2):.1f}" y="{p2y(hw.pcie_gbs)+13:.1f}" font-size="9.5" font-weight="600" fill="#475569">Host PCIe ({hw.pcie_gbs:.0f} GB/s)</text>')

    placed2: list = []
    for o in ops:
        cx, cy = p2x(o.bytes_moved), p2y(o.achieved_tbs * 1000.0)
        draw_dot(placed2, ix, iy, iw, ih, cx, cy, 5.0, CAT_COLOR[o.category], o.name[:22])

    # =========================================================================
    # PANEL 3 (Bottom-Left): Dispatch / Launch Overhead vs Kernel Duration (us)
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        22, 482, 578, 372,
        "3. Kernel Dispatch Efficiency vs. Kernel Duration (Eager vs. Graphs)",
        "Mean GPU Kernel Duration (microseconds, log10)",
        "Effective GPU Busy % (t_kern / (t_kern + t_launch))",
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
    svg.append(f'<text x="{p3x(2.2):.1f}" y="{p3y(92):.1f}" font-size="9.5" font-weight="600" fill="#059669">CUDA/HIP Graph ({hw.graph_launch_us:.2f} us gap)</text>')
    svg.append(f'<text x="{p3x(12):.1f}" y="{p3y(68):.1f}" font-size="9.5" font-weight="600" fill="#dc2626">Eager Launch ({hw.eager_launch_us:.1f} us gap)</text>')

    placed3: list = []
    for o in ops:
        busy_eager = 100.0 * o.avg_us / (o.avg_us + hw.eager_launch_us)
        cx, cy = p3x(o.avg_us), p3y(busy_eager)
        draw_dot(placed3, ix, iy, iw, ih, cx, cy, 5.0, CAT_COLOR[o.category], o.name[:22])

    # =========================================================================
    # PANEL 4 (Bottom-Right): Matrix Core Utilization vs Batch / Expert Tokens M
    # =========================================================================
    ix, iy, iw, ih = panel_box(
        620, 482, 578, 372,
        "4. Matrix Core Utilization vs. Batch / Expert Tokens M (Batch Knee)",
        "Tokens per GEMM or Routed MoE Expert M (log2)",
        "% of Empirical BF16 Matrix Ceiling",
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
    for b_w, col, lbl in [(2.0, "#64748b", "BF16 Weights"), (1.0, "#2563eb", "FP8 Weights"), (0.5, "#7c3aed", "MXFP4/NVFP4 Weights")]:
        pts_m = []
        for m in ms_grid:
            oi_m = (2.0 * m) / b_w
            util = min(88.0, 100.0 * (hw.hbm_tbs * oi_m) / hw.bf16_matrix_tflops)
            pts_m.append((m, util))
        d_str = " ".join(f"{'M' if i==0 else 'L'}{p4x(x):.1f},{p4y(y):.1f}" for i, (x, y) in enumerate(pts_m))
        svg.append(f'<path d="{d_str}" fill="none" stroke="{col}" stroke-width="2"/>')
        knee_m = (hw.bf16_matrix_tflops / hw.hbm_tbs) * (b_w / 2.0) * 0.7
        svg.append(f'<text x="{p4x(knee_m):.1f}" y="{p4y(70 if b_w==1.0 else (82 if b_w==0.5 else 56)):.1f}" font-size="9.2" font-weight="600" fill="{col}">{lbl}</text>')

    placed4: list = []
    for o in ops:
        if o.category not in ("GEMM", "MoE", "Attention"):
            continue
        util_pct = min(102.0, 100.0 * o.achieved_tflops / hw.bf16_matrix_tflops)
        cx, cy = p4x(o.tokens_m), p4y(util_pct)
        draw_dot(placed4, ix, iy, iw, ih, cx, cy, 5.2, CAT_COLOR[o.category], o.name[:22])

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
    ap.add_argument("--model", help="HuggingFace repo ID or path to config.json for analytical model-on-HW projection")
    ap.add_argument("--demo", action="store_true", help="Run reference MoE LLM analysis + generate 4-panel SVG")
    ap.add_argument("--tp", type=int, default=2, help="Tensor Parallel degree (for --model / --demo)")
    ap.add_argument("--quant", choices=["bf16", "fp8", "fp4"], default="fp8", help="Weight precision (for --model / --demo)")
    ap.add_argument("--batch-size", type=int, default=4, help="Concurrent decode batch size")
    ap.add_argument("--prompt-len", type=int, default=4096, help="Prefill prompt tokens")
    ap.add_argument("--output-len", type=int, default=16, help="Decode tokens to profile on live server")
    ap.add_argument("--ctx-len", type=int, default=32768, help="Decode KV cache context length (for --model / --demo)")
    ap.add_argument("-o", "--output", type=Path, default=Path("assets/multi_bottleneck_roofline.svg"))
    args = ap.parse_args()

    hw = load_hw_json(args.hw) if args.hw else HW_PRESETS[args.gpu]

    proc = None
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
            ops = parse_kineto_trace(trace_file)
            subtitle = f"Live {args.engine.upper()} Kineto Profile ({trace_file.name}: bs={args.batch_size}, prompt={args.prompt_len})"
        elif args.trace:
            ops = parse_kineto_trace(args.trace)
            subtitle = f"PyTorch Kineto Trace ({args.trace.name})"
        elif args.model:
            cfg = load_hf_config(args.model)
            ops = model_ops_on_hw(cfg, hw, args.tp, args.quant, args.batch_size, args.prompt_len, args.ctx_len)
            subtitle = f"Model Projection: {args.model} ({args.quant.upper()}, TP={args.tp}, Decode BS={args.batch_size} @ {args.ctx_len//1024}k ctx, Prefill={args.prompt_len})"
        else:
            demo_cfg = {
                "num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128,
                "num_key_value_heads": 8, "head_dim": 128, "n_routed_experts": 256,
                "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            }
            ops = model_ops_on_hw(demo_cfg, hw, args.tp, args.quant, args.batch_size, args.prompt_len, args.ctx_len)
            subtitle = f"Sparse MoE Reference ({args.quant.upper()}, TP={args.tp}, Decode BS={args.batch_size} @ {args.ctx_len//1024}k ctx, Prefill={args.prompt_len})"
    finally:
        if proc is not None:
            proc.terminate()

    print_analysis_and_advice(hw, ops)
    render_4panel_svg(hw, ops, args.output, subtitle)


if __name__ == "__main__":
    main()
