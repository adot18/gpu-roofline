#!/usr/bin/env python3
"""Plot `gpu_roofline --csv <file.csv>` sweeps across one or more accelerators,
or run `--demo` to render the reference 3-regime SVG diagram for README.md.

Zero external dependencies required for SVG output (pure Python stdlib)!

Usage:
    # Overlay real CSV sweeps from MI300X, MI355X, H100, B200:
    python3 scripts/plot_sweep.py mi300x.csv mi355x.csv h100.csv -o comparison.svg

    # Generate the reference README diagram illustrating the 3 regimes:
    python3 scripts/plot_sweep.py --demo -o assets/latency_vs_bandwidth_regimes.svg
"""

import argparse
import csv
import math
from pathlib import Path

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def load_csv(path: Path):
    rows = list(csv.DictReader(path.open()))
    label = rows[0]["device"] if rows else path.stem
    pts = [(float(r["rw_bytes"]), float(r["bandwidth_tbs"]), float(r["latency_us"])) for r in rows]
    return label, pts


def demo_curves():
    """Analytical + empirical reference curves showing how transfer size shifts a GPU
    from Latency-Bound (t ~= t_launch) -> L2 Cache Peak -> HBM Bandwidth Plateau."""
    sizes = [
        4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024,
        1024**2, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2,
        128 * 1024**2, 256 * 1024**2, 512 * 1024**2, 1024**3, 2 * 1024**3,
    ]

    def synth(t_launch_s, l2_bw_tbs, hbm_bw_tbs, l2_cap_bytes):
        pts = []
        for b in sizes:
            # Smooth L2 residency transition around l2_cap_bytes
            hit_frac = 1.0 / (1.0 + (b / l2_cap_bytes) ** 4)
            eff_bw = hit_frac * l2_bw_tbs + (1.0 - hit_frac) * hbm_bw_tbs
            t_s = t_launch_s + b / (eff_bw * 1e12)
            achieved_tbs = (b / t_s) / 1e12
            pts.append((float(b), achieved_tbs, t_s * 1e6))
        return pts

    return [
        ("AMD MI355X (256 MiB L2, ~5.8 TB/s HBM3e)", synth(2.2e-6, 18.5, 5.80, 220 * 1024**2)),
        ("AMD MI300X (256 MiB L2, ~4.2 TB/s HBM3)",  synth(2.4e-6, 15.5, 4.20, 220 * 1024**2)),
        ("NVIDIA H100 (50 MiB L2, ~3.0 TB/s HBM3)",  synth(2.0e-6, 11.5, 3.00, 44 * 1024**2)),
    ]


def render_svg(series, out_path: Path, is_demo: bool = False):
    W, H = 920, 500
    ml, mr, mt, mb = 78, 36, 68, 72
    pw, ph = W - ml - mr, H - mt - mb

    all_x = [p[0] for _, pts in series for p in pts]
    all_y = [p[1] for _, pts in series for p in pts]
    log_min, log_max = math.log2(min(all_x)), math.log2(max(all_x))
    y_max = max(20.0, math.ceil(max(all_y) / 2.0) * 2.0)

    def sx(b):
        return ml + (math.log2(b) - log_min) / (log_max - log_min) * pw

    def sy(tbs):
        return mt + ph - (tbs / y_max) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; background: #ffffff;">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff" rx="8"/>',
    ]

    # Regime background bands
    x_lat_end = sx(4 * 1024**2)
    x_l2_end  = sx(256 * 1024**2)
    lines += [
        f'<rect x="{ml:.1f}" y="{mt}" width="{x_lat_end - ml:.1f}" height="{ph}" fill="#eff6ff" opacity="0.65"/>',
        f'<rect x="{x_lat_end:.1f}" y="{mt}" width="{x_l2_end - x_lat_end:.1f}" height="{ph}" fill="#ecfdf5" opacity="0.65"/>',
        f'<rect x="{x_l2_end:.1f}" y="{mt}" width="{ml + pw - x_l2_end:.1f}" height="{ph}" fill="#fef2f2" opacity="0.55"/>',
        f'<text x="{(ml + x_lat_end)/2:.1f}" y="{mt + 20}" text-anchor="middle" font-size="12" font-weight="600" fill="#1d4ed8">1. Latency-Bound (t ~= t_launch)</text>',
        f'<text x="{(x_lat_end + x_l2_end)/2:.1f}" y="{mt + 20}" text-anchor="middle" font-size="12" font-weight="600" fill="#047857">2. L2 Cache Peak (On-Chip SRAM)</text>',
        f'<text x="{(x_l2_end + ml + pw)/2:.1f}" y="{mt + 20}" text-anchor="middle" font-size="12" font-weight="600" fill="#b91c1c">3. HBM Bandwidth-Bound</text>',
    ]

    # Title
    subtitle = "Illustrative reference profiles (replace with `./gpu_roofline --csv out.csv`)" if is_demo else "Measured via `./gpu_roofline --csv`"
    lines += [
        f'<text x="{W/2}" y="28" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">'
        f'Effective GPU Memory Bandwidth vs. Transfer Size (Latency-Bound -> L2 Cliff -> HBM)</text>',
        f'<text x="{W/2}" y="48" text-anchor="middle" font-size="12" fill="#64748b">{subtitle}</text>',
    ]

    # Horizontal gridlines
    step_y = 2 if y_max <= 22 else 5
    for yv in range(0, int(y_max) + 1, step_y):
        yy = sy(yv)
        lines.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml+pw}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 10}" y="{yy + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv} TB/s</text>')

    # X ticks
    xticks = [
        (4 * 1024, "4 KiB"), (64 * 1024, "64 KiB"), (1024**2, "1 MiB"),
        (16 * 1024**2, "16 MiB"), (64 * 1024**2, "64 MiB"),
        (256 * 1024**2, "256 MiB"), (1024**3, "1 GiB"), (2 * 1024**3, "2 GiB"),
    ]
    for xv, lbl in xticks:
        if min(all_x) <= xv <= max(all_x):
            xx = sx(xv)
            lines.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#e2e8f0" stroke-dasharray="3,3"/>')
            lines.append(f'<text x="{xx:.1f}" y="{mt + ph + 20}" text-anchor="middle" font-size="11" fill="#475569">{lbl}</text>')

    # Axes
    lines += [
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#334155" stroke-width="1.5"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#334155" stroke-width="1.5"/>',
        f'<text x="{W/2}" y="{H - 18}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">'
        f'Total Bytes Transferred per Kernel (Read + Write, log scale)</text>',
        f'<text x="20" y="{mt + ph/2}" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b" '
        f'transform="rotate(-90 20 {mt + ph/2})">Effective Bandwidth (TB/s)</text>',
    ]

    # Curves + Legend
    for idx, (label, pts) in enumerate(series):
        col = PALETTE[idx % len(PALETTE)]
        path_d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(pts))
        lines.append(f'<path d="{path_d}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for p in pts:
            lines.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="3.5" fill="{col}"/>')

        ly = mt + 44 + idx * 20
        lines.append(f'<line x1="{ml + 14}" y1="{ly - 4}" x2="{ml + 36}" y2="{ly - 4}" stroke="{col}" stroke-width="2.8"/>')
        lines.append(f'<circle cx="{ml + 25}" cy="{ly - 4}" r="3.5" fill="{col}"/>')
        lines.append(f'<text x="{ml + 44}" y="{ly}" font-size="11.5" font-weight="600" fill="#1e293b">{label}</text>')

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", type=Path, help="CSV files produced by `./gpu_roofline --csv`")
    ap.add_argument("--demo", action="store_true", help="Render illustrative 3-regime diagram")
    ap.add_argument("-o", "--output", type=Path, default=Path("assets/latency_vs_bandwidth_regimes.svg"))
    args = ap.parse_args()

    if args.demo or not args.csvs:
        render_svg(demo_curves(), args.output, is_demo=True)
    else:
        render_svg([load_csv(p) for p in args.csvs], args.output, is_demo=False)


if __name__ == "__main__":
    main()
