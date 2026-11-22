#!/usr/bin/env python3
"""Zero-dependency unit & integration tests for engine_roofline.py and plot_sweep.py."""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import engine_roofline as er  # noqa: E402


class TestEngineRoofline(unittest.TestCase):
    def test_mixed_precision_deepseek_detection(self):
        cfg = er.MODEL_PRESETS["deepseek-v4-fp4"]
        precs = er.detect_model_precisions(cfg)
        self.assertEqual(precs["moe"], "fp4")
        self.assertEqual(precs["attn"], "fp8")
        self.assertEqual(precs["kv"], "fp8")

        # VLM nested text_config + expert_dtype="fp4" schema
        vlm_cfg = {
            "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4", "scale_fmt": "ue8m0"},
            "text_config": {
                "num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 64,
                "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
                "q_lora_rank": 1280, "o_lora_rank": 1024,
                "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
                "moe_intermediate_size": 2304, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
            },
        }
        vlm_precs = er.detect_model_precisions(vlm_cfg)
        self.assertEqual(vlm_precs["moe"], "fp4")
        self.assertEqual(vlm_precs["attn"], "fp8")
        ops, _ = er.model_ops_on_hw(vlm_cfg, er.HW_PRESETS["mi355x"], tp=2, quant="auto", batch_decode=24, prompt_len=16384, ctx_len=65536, spec_tokens=5)
        self.assertTrue(any("SpecVerify MoE (FP4" in o.name for o in ops))

        # Override via flags
        precs_ov = er.detect_model_precisions(cfg, quant="fp8", moe_quant="fp4", attn_quant="bf16")
        self.assertEqual(precs_ov["moe"], "fp4")
        self.assertEqual(precs_ov["attn"], "bf16")

    def test_fp4_upscale_fallback_on_mi300x_vs_mi355x(self):
        mi355x = er.HW_PRESETS["mi355x"]
        mi300x = er.HW_PRESETS["mi300x"]
        cfg = er.MODEL_PRESETS["deepseek-v4-fp4"]

        ops_355, _ = er.model_ops_on_hw(cfg, mi355x, tp=2, quant="auto", batch_decode=4, prompt_len=16384, ctx_len=32768)
        ops_300, _ = er.model_ops_on_hw(cfg, mi300x, tp=2, quant="auto", batch_decode=4, prompt_len=16384, ctx_len=32768)

        pre_moe_355 = [o for o in ops_355 if "Prefill MoE" in o.name][0]
        pre_moe_300 = [o for o in ops_300 if "Prefill MoE" in o.name][0]
        self.assertEqual(pre_moe_355.dtype, "fp4")
        self.assertEqual(pre_moe_300.dtype, "fp4")

        c_355, reg_355 = er.ceiling_for_op(pre_moe_355, mi355x)
        c_300, reg_300 = er.ceiling_for_op(pre_moe_300, mi300x)
        self.assertIn("FP4", reg_355)
        self.assertIn("Upscale", reg_300)
        self.assertGreater(c_355, c_300 * 3.0)

    def test_prefix_cache_hit_shrinks_prefill(self):
        hw = er.HW_PRESETS["mi300x"]
        cfg = er.MODEL_PRESETS["deepseek-v4-fp4"]
        ops_0, _ = er.model_ops_on_hw(cfg, hw, tp=4, quant="auto", batch_decode=4, prompt_len=8192, ctx_len=32768, prefix_cache_hit=0.0)
        ops_90, _ = er.model_ops_on_hw(cfg, hw, tp=4, quant="auto", batch_decode=4, prompt_len=8192, ctx_len=32768, prefix_cache_hit=0.90)

        p0 = [o for o in ops_0 if "Prefill QKV" in o.name][0]
        p90 = [o for o in ops_90 if "Prefill QKV" in o.name][0]
        self.assertGreater(p0.tokens_m, p90.tokens_m * 8.0)

    def test_kineto_trace_parser_and_fusion_detector(self):
        trace = {
            "traceEvents": [
                # 1. AllReduce -> RMSNorm -> DynamicFP8Quant -> FP8 GEMM -> FP4 MoE
                {"ph": "X", "cat": "cpu_op", "name": "nccl::all_reduce", "tid": 1, "ts": 100, "dur": 20,
                 "args": {"Input Dims": [[4, 7168]], "Input type": ["c10::BFloat16"]}},
                {"ph": "X", "cat": "hip_runtime", "name": "hipLaunchKernel", "tid": 1, "ts": 105, "dur": 2, "args": {"correlation": 1}},
                {"ph": "X", "cat": "kernel", "name": "rcclGenericRingKernel", "pid": 0, "tid": 7, "ts": 120, "dur": 12.0,
                 "args": {"correlation": 1}},

                {"ph": "X", "cat": "cpu_op", "name": "aiter::rmsnorm", "tid": 1, "ts": 130, "dur": 10,
                 "args": {"Input Dims": [[4, 7168], [7168]], "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
                {"ph": "X", "cat": "hip_runtime", "name": "hipLaunchKernel", "tid": 1, "ts": 133, "dur": 2, "args": {"correlation": 2}},
                {"ph": "X", "cat": "kernel", "name": "fused_rmsnorm_kernel", "pid": 0, "tid": 7, "ts": 135.5, "dur": 3.0,
                 "args": {"correlation": 2}},

                {"ph": "X", "cat": "cpu_op", "name": "sglang::scaled_fp8_quant", "tid": 1, "ts": 145, "dur": 8,
                 "args": {"Input Dims": [[4, 7168]], "Input type": ["c10::BFloat16"]}},
                {"ph": "X", "cat": "hip_runtime", "name": "hipLaunchKernel", "tid": 1, "ts": 147, "dur": 2, "args": {"correlation": 3}},
                {"ph": "X", "cat": "kernel", "name": "dynamic_fp8_quant_kernel", "pid": 0, "tid": 7, "ts": 141.5, "dur": 2.5,
                 "args": {"correlation": 3}},

                {"ph": "X", "cat": "cpu_op", "name": "aten::_scaled_mm", "tid": 1, "ts": 160, "dur": 40,
                 "args": {"Input Dims": [[1024, 7168], [7168, 8192]], "Input type": ["c10::Float8_e4m3fnuz", "c10::Float8_e4m3fnuz"]}},
                {"ph": "X", "cat": "hip_runtime", "name": "hipLaunchKernel", "tid": 1, "ts": 165, "dur": 3, "args": {"correlation": 4}},
                {"ph": "X", "cat": "kernel", "name": "Cijk_Alik_Bljk_F8B_gemm_kernel", "pid": 0, "tid": 7, "ts": 170.0, "dur": 62.0,
                 "args": {"correlation": 4}},

                # 2. Graph-replayed FP4 MoE kernel with NO cpu_op shape correlation
                {"ph": "X", "cat": "kernel", "name": "aiter_f8f6f4_moe_mxfp4_kernel", "pid": 0, "tid": 7, "ts": 234.0, "dur": 45.0,
                 "args": {"correlation": 999, "grid": [512, 1, 1]}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "test.trace.json.gz"
            with gzip.open(tp, "wt", encoding="utf-8") as f:
                json.dump(trace, f)
            hw = er.HW_PRESETS["mi355x"]
            ops, fusions = er.parse_kineto_trace(tp, hw)

        by_name = {o.name: o for o in ops}
        self.assertEqual(by_name["Cijk_Alik_Bljk_F8B_gemm_kernel"].dtype, "fp8")
        self.assertEqual(by_name["fused_rmsnorm_kernel"].pipe, "L2")
        self.assertEqual(by_name["fused_rmsnorm_kernel"].dtype, "sfu")
        self.assertAlmostEqual(by_name["fused_rmsnorm_kernel"].measured_gap_us, 3.5, places=2)
        self.assertEqual(by_name["aiter_f8f6f4_moe_mxfp4_kernel"].dtype, "fp4")
        self.assertEqual(by_name["aiter_f8f6f4_moe_mxfp4_kernel"].shape_source, "graph_est")

        patterns = {f.pattern for f in fusions}
        self.assertIn("AllReduce + RMSNorm", patterns)
        self.assertIn("RMSNorm + Dynamic Quant", patterns)

    def test_hw_json_load_and_svg_views(self):
        hw_data = {
            "device": "AMD Instinct MI355X",
            "sms_or_cus": 256,
            "vram_gib": 288.0,
            "l2_mib": 256.0,
            "hbm_tbs": 5.80,
            "hbm_read_tbs": 6.30,
            "l2_tbs": 18.2,
            "shared_lds_tbs": 77.0,
            "fp32_valu_tflops": 144.0,
            "fp32_sfu_tops": 41.0,
            "bf16_matrix_tflops": 1650.0,
            "fp8_matrix_tflops": 3350.0,
            "fp4_matrix_tflops": 6650.0,
            "eager_launch_us": 2.5,
            "graph_launch_us": 0.4,
            "pcie_h2d_gbs": 54.0,
            "p2p_read_gbs": 405.0,
            "p2p_all_ring_gbs": 355.0,
        }
        with tempfile.TemporaryDirectory() as td:
            hw_p = Path(td) / "hw.json"
            hw_p.write_text(json.dumps(hw_data))
            hw = er.load_hw_json(hw_p)
            self.assertAlmostEqual(hw.fp4_matrix_tflops, 6650.0)
            self.assertAlmostEqual(hw.hbm_read_tbs, 6.30)

            ops, _ = er.model_ops_on_hw(er.MODEL_PRESETS["deepseek-v4-fp4"], hw, tp=2, quant="auto", batch_decode=4, prompt_len=4096, ctx_len=32768)
            for view in ("all", "fp4", "fp8", "bf16"):
                out_svg = Path(td) / f"out_{view}.svg"
                er.render_4panel_svg(hw, ops, out_svg, f"test {view}", precision_view=view)
                text = out_svg.read_text()
                self.assertTrue(text.startswith("<svg "))
                self.assertIn("</svg>", text)

    def test_demangle_and_model_bound_trace(self):
        self.assertEqual(
            er.shorten_kernel_name("_ZN5aiter26cross_device_reduce_1stageINS_8RankDataEEE"),
            "aiter::cross_device_reduce_1stage",
        )
        self.assertEqual(
            er.shorten_kernel_name("mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1"),
            "mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x2",
        )
        cat, pipe, dt = er.classify_kernel("gemm2_a4w4_port_b64_n128_k256_w1x4")
        self.assertEqual((cat, pipe, dt), ("MoE", "HBM_READ", "fp4"))
        cat2, pipe2, dt2 = er.classify_kernel("_ZN5aiter26cross_device_reduce_1stageE")
        self.assertEqual((cat2, pipe2), ("Comm", "P2P"))

        hint, _ = er.build_model_hint(er.MODEL_PRESETS["deepseek-v4-fp4"], tp=2, batch_decode=24, spec_tokens=0, ctx_len=32768)
        trace = {
            "traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256",
                 "pid": 0, "tid": 7, "ts": 100.0, "dur": 220.0, "args": {"correlation": 10}},
                {"ph": "X", "cat": "kernel", "name": "gemm2_a4w4_port_b64_n128_k256",
                 "pid": 0, "tid": 7, "ts": 325.0, "dur": 110.0, "args": {"correlation": 11}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "graph.trace.json"
            tp.write_text(json.dumps(trace))
            ops, _ = er.parse_kineto_trace(tp, er.HW_PRESETS["mi355x"], model_hint=hint)
        self.assertEqual(len(ops), 2)
        self.assertTrue(all(o.shape_source == "model_bound" for o in ops))
        self.assertAlmostEqual(ops[0].bytes_moved / ops[1].bytes_moved, 2.0, delta=0.05)

    def test_cli_demo_and_plot_sweep(self):
        with tempfile.TemporaryDirectory() as td:
            svg1 = Path(td) / "atlas.svg"
            svg2 = Path(td) / "sweep.svg"
            r1 = subprocess.run(
                [sys.executable, str(ROOT / "scripts/engine_roofline.py"), "--demo", "--gpu", "mi355x", "-o", str(svg1)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("Model Precision", r1.stdout)
            self.assertTrue(svg1.exists())

            subprocess.run(
                [sys.executable, str(ROOT / "scripts/plot_sweep.py"), "--demo", "-o", str(svg2)],
                capture_output=True, text=True, check=True,
            )
            self.assertTrue(svg2.exists())


if __name__ == "__main__":
    unittest.main()
