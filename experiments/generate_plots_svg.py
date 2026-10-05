#!/usr/bin/env python3
"""
Generate publication-quality SVG vector charts and raster PNG charts
for B0 (Sequential Fine-Tuning) Continual Relation Extraction benchmark.
Fixes all text clipping, spacing, dark mode backgrounds, and visual clarity.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "fewrel" / "5shot" / "B0_sequential_ft" / "seed_2021"
PLOTS_DIR = RESULTS_DIR / "plots"


def generate_svg_and_html(results_dir: Path | str | None = None) -> dict[str, Path]:
    target_results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    target_plots_dir = target_results_dir / "plots"
    target_plots_dir.mkdir(parents=True, exist_ok=True)

    matrix_file = target_results_dir / "performance_matrix.json"
    summary_file = target_results_dir / "summary.json"

    with matrix_file.open("r", encoding="utf-8") as f:
        m_data = json.load(f)
    with summary_file.open("r", encoding="utf-8") as f:
        s_data = json.load(f)

    matrix = m_data["matrix"]
    num_tasks = len(matrix)
    task_labels = [f"T{i + 1}" for i in range(num_tasks)]

    colors = [
        "#2563eb",  # T1 - Royal Blue
        "#ea580c",  # T2 - Vibrant Orange
        "#16a34a",  # T3 - Forest Green
        "#dc2626",  # T4 - Crimson Red
        "#9333ea",  # T5 - Deep Purple
        "#b45309",  # T6 - Warm Brown
        "#db2777",  # T7 - Magenta/Pink
        "#4b5563",  # T8 - Slate Gray
    ]

    # Enhanced Dimensions to give plenty of breathing room
    width, height = 1000, 600
    margin_left, margin_right = 85, 220
    margin_top, margin_bottom = 85, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def map_x(t: int) -> float:
        return margin_left + (t / max(1, num_tasks - 1)) * plot_w

    def map_y(val: float) -> float:
        return margin_top + plot_h - (val * plot_h)

    # -------------------------------------------------------------
    # 1. SVG: Forgetting Curves (Individual Task Degradation)
    # -------------------------------------------------------------
    svg1 = []
    svg1.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">')
    
    # Solid background for crisp viewing in dark mode / image viewers
    svg1.append(f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>')

    # Header Title & Subtitle
    svg1.append(f'<text x="{margin_left + plot_w/2}" y="36" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">B0 Sequential Fine-Tuning: Catastrophic Forgetting Curves</text>')
    svg1.append(f'<text x="{margin_left + plot_w/2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">FewRel Track A (8 tasks × 10 relations, 5-shot, seed 2021) — M=0, No Replay</text>')

    # Chart Canvas Frame
    svg1.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>')

    # Horizontal Gridlines & Y-ticks
    for y_pct in range(0, 101, 20):
        y_val = y_pct / 100.0
        y_pos = map_y(y_val)
        svg1.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_w}" y2="{y_pos}" stroke="#e2e8f0" stroke-dasharray="4,4" stroke-width="1"/>')
        svg1.append(f'<text x="{margin_left - 14}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#475569" font-weight="600">{y_pct}%</text>')

    # Vertical Gridlines & X-ticks
    for t_idx, label in enumerate(task_labels):
        x_pos = map_x(t_idx)
        svg1.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg1.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#1e293b" font-weight="700">Stage {t_idx}</text>')
        svg1.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 40}" text-anchor="middle" font-size="11" fill="#64748b">(after {label})</text>')

    # Axis Labels
    svg1.append(f'<text x="{margin_left + plot_w/2}" y="{height - 18}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Continual Training Stage</text>')
    svg1.append(f'<text transform="rotate(-90)" x="-{margin_top + plot_h/2}" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Task Accuracy (%)</text>')

    # Render Task Degradation Lines
    for j in range(num_tasks - 1):
        color = colors[j]
        points = []
        for t in range(j, num_tasks):
            v = matrix[t][j]
            if v is not None:
                points.append((t, v, map_x(t), map_y(v)))

        # Polyline with clean smooth stroke
        poly_pts = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in points])
        svg1.append(f'<polyline points="{poly_pts}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/>')

        # Markers & Data Badges
        for idx, (t, v, px, py) in enumerate(points):
            svg1.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.8"/>')
            
            # Start score label (Top of first point)
            if idx == 0:
                svg1.append(f'<rect x="{px - 22:.1f}" y="{py - 24:.1f}" width="44" height="18" fill="#ffffff" stroke="{color}" stroke-width="1" rx="4" opacity="0.95"/>')
                svg1.append(f'<text x="{px:.1f}" y="{py - 11:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{color}">{v*100:.1f}%</text>')
            
            # End score label (Clean badge at Stage 7, offset leftwards to prevent edge clipping)
            elif idx == len(points) - 1:
                svg1.append(f'<rect x="{px - 48:.1f}" y="{py - 9:.1f}" width="44" height="18" fill="#ffffff" stroke="{color}" stroke-width="1" rx="4" opacity="0.95"/>')
                svg1.append(f'<text x="{px - 26:.1f}" y="{py + 4:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{color}">{v*100:.1f}%</text>')

    # Legend Box on the Right
    leg_x = width - margin_right + 25
    leg_y = margin_top + 10
    leg_w = margin_right - 35
    leg_h = (num_tasks - 1) * 32 + 48
    svg1.append(f'<rect x="{leg_x}" y="{leg_y}" width="{leg_w}" height="{leg_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="8" />')
    svg1.append(f'<text x="{leg_x + 16}" y="{leg_y + 24}" font-size="13" font-weight="700" fill="#0f172a">Task Degradation</text>')

    for j in range(num_tasks - 1):
        color = colors[j]
        item_y = leg_y + 50 + j * 32
        drop = (matrix[j][j] - matrix[num_tasks - 1][j]) * 100
        # Line swatch
        svg1.append(f'<line x1="{leg_x + 16}" y1="{item_y}" x2="{leg_x + 40}" y2="{item_y}" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>')
        svg1.append(f'<circle cx="{leg_x + 28}" cy="{item_y}" r="4" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        # Label with colored task tag and red drop badge
        svg1.append(f'<text x="{leg_x + 48}" y="{item_y + 4}" font-size="12" font-weight="700" fill="#1e293b">T{j+1}: <tspan font-weight="bold" fill="#dc2626">-{drop:.1f}%</tspan></text>')

    svg1.append('</svg>')
    svg1_content = "\n".join(svg1)

    svg1_path = target_plots_dir / "forgetting_curve.svg"
    with svg1_path.open("w", encoding="utf-8") as f:
        f.write(svg1_content)

    # -------------------------------------------------------------
    # 2. SVG: Accuracy Evolution (Avg Acc vs New vs Old)
    # -------------------------------------------------------------
    svg2 = []
    svg2.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">')
    svg2.append(f'<rect width="{width}" height="{height}" fill="#ffffff" rx="8"/>')

    # Title & Subtitle
    svg2.append(f'<text x="{margin_left + plot_w/2}" y="36" text-anchor="middle" font-size="20" font-weight="700" fill="#0f172a">B0 Sequential Fine-Tuning: Accuracy Evolution Over Tasks</text>')
    svg2.append(f'<text x="{margin_left + plot_w/2}" y="58" text-anchor="middle" font-size="13" fill="#64748b">Divergence Between New Task Acquisition (High) and Old Knowledge Retention (Severe Drop)</text>')

    # Canvas
    svg2.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>')

    for y_pct in range(0, 101, 20):
        y_val = y_pct / 100.0
        y_pos = map_y(y_val)
        svg2.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_w}" y2="{y_pos}" stroke="#e2e8f0" stroke-dasharray="4,4" stroke-width="1"/>')
        svg2.append(f'<text x="{margin_left - 14}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#475569" font-weight="600">{y_pct}%</text>')

    for t_idx, label in enumerate(task_labels):
        x_pos = map_x(t_idx)
        svg2.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg2.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#1e293b" font-weight="700">Stage {t_idx}</text>')
        svg2.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 40}" text-anchor="middle" font-size="11" fill="#64748b">(after {label})</text>')

    svg2.append(f'<text x="{margin_left + plot_w/2}" y="{height - 18}" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Continual Training Stage</text>')
    svg2.append(f'<text transform="rotate(-90)" x="-{margin_top + plot_h/2}" y="28" text-anchor="middle" font-size="13" font-weight="700" fill="#334155">Accuracy (%)</text>')

    # Series Data
    aa_pts = [(t, s_data["average_accuracy_by_stage"][t], map_x(t), map_y(s_data["average_accuracy_by_stage"][t])) for t in range(num_tasks)]
    new_pts = [(t, matrix[t][t], map_x(t), map_y(matrix[t][t])) for t in range(num_tasks)]
    old_pts = []
    for t in range(1, num_tasks):
        old_val = s_data["old_vs_new_by_stage"][f"stage_{t}"]["old_acc"]
        old_pts.append((t, old_val, map_x(t), map_y(old_val)))

    # 1. New Task Line (Green dashed)
    p_new = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in new_pts])
    svg2.append(f'<polyline points="{p_new}" fill="none" stroke="#16a34a" stroke-dasharray="6,4" stroke-width="2.5" stroke-linecap="round"/>')
    for _, v, px, py in new_pts:
        svg2.append(f'<rect x="{px - 4.5:.1f}" y="{py - 4.5:.1f}" width="9" height="9" fill="#16a34a" stroke="#fff" stroke-width="1.8"/>')

    # 2. Old Tasks Line (Red dotted)
    p_old = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in old_pts])
    svg2.append(f'<polyline points="{p_old}" fill="none" stroke="#dc2626" stroke-dasharray="3,3" stroke-width="2.5" stroke-linecap="round"/>')
    for _, v, px, py in old_pts:
        svg2.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#dc2626" stroke="#fff" stroke-width="1.8"/>')

    # 3. Average Accuracy Line (Royal Blue solid thick)
    p_aa = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in aa_pts])
    svg2.append(f'<polyline points="{p_aa}" fill="none" stroke="#2563eb" stroke-width="3.5" stroke-linecap="round"/>')
    for t, v, px, py in aa_pts:
        svg2.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="#2563eb" stroke="#fff" stroke-width="2"/>')
        svg2.append(f'<rect x="{px - 22:.1f}" y="{py - 24:.1f}" width="44" height="17" fill="#ffffff" stroke="#2563eb" stroke-width="1" rx="4" opacity="0.95"/>')
        svg2.append(f'<text x="{px:.1f}" y="{py - 11:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#1e40af">{v*100:.1f}%</text>')

    # Legend Box on Right
    leg2_x = width - margin_right + 25
    leg2_y = margin_top + 10
    leg2_w = margin_right - 35
    svg2.append(f'<rect x="{leg2_x}" y="{leg2_y}" width="{leg2_w}" height="150" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" rx="8"/>')
    svg2.append(f'<text x="{leg2_x + 16}" y="{leg2_y + 24}" font-size="13" font-weight="700" fill="#0f172a">Metrics</text>')

    # Item 1: Average Accuracy
    svg2.append(f'<line x1="{leg2_x + 16}" y1="{leg2_y + 50}" x2="{leg2_x + 40}" y2="{leg2_y + 50}" stroke="#2563eb" stroke-width="3.5" stroke-linecap="round"/>')
    svg2.append(f'<circle cx="{leg2_x + 28}" cy="{leg2_y + 50}" r="4" fill="#2563eb" stroke="#fff" stroke-width="1.5"/>')
    svg2.append(f'<text x="{leg2_x + 48}" y="{leg2_y + 54}" font-size="12" font-weight="700" fill="#1e293b">Avg Accuracy</text>')

    # Item 2: New Task
    svg2.append(f'<line x1="{leg2_x + 16}" y1="{leg2_y + 82}" x2="{leg2_x + 40}" y2="{leg2_y + 82}" stroke="#16a34a" stroke-dasharray="5,3" stroke-width="3" stroke-linecap="round"/>')
    svg2.append(f'<rect x="{leg2_x + 24}" y="{leg2_y + 78}" width="8" height="8" fill="#16a34a" stroke="#fff" stroke-width="1.5"/>')
    svg2.append(f'<text x="{leg2_x + 48}" y="{leg2_y + 86}" font-size="12" font-weight="700" fill="#1e293b">New Task Acc</text>')

    # Item 3: Old Tasks
    svg2.append(f'<line x1="{leg2_x + 16}" y1="{leg2_y + 114}" x2="{leg2_x + 40}" y2="{leg2_y + 114}" stroke="#dc2626" stroke-dasharray="3,3" stroke-width="3" stroke-linecap="round"/>')
    svg2.append(f'<circle cx="{leg2_x + 28}" cy="{leg2_y + 114}" r="4" fill="#dc2626" stroke="#fff" stroke-width="1.5"/>')
    svg2.append(f'<text x="{leg2_x + 48}" y="{leg2_y + 118}" font-size="12" font-weight="700" fill="#1e293b">Old Tasks Acc</text>')

    svg2.append('</svg>')
    svg2_content = "\n".join(svg2)

    svg2_path = target_plots_dir / "accuracy_over_tasks.svg"
    with svg2_path.open("w", encoding="utf-8") as f:
        f.write(svg2_content)

    # -------------------------------------------------------------
    # Convert SVGs to High-Resolution PNGs via macOS sips
    # -------------------------------------------------------------
    png_forg_path = target_plots_dir / "forgetting_curve.png"
    png_acc_path = target_plots_dir / "accuracy_over_tasks.png"

    try:
        subprocess.run(["sips", "-s", "format", "png", str(svg1_path), "--out", str(png_forg_path)], check=True, capture_output=True)
        subprocess.run(["sips", "-s", "format", "png", str(svg2_path), "--out", str(png_acc_path)], check=True, capture_output=True)
    except Exception as e:
        print(f"Warning: Could not rasterize SVG with sips: {e}")

    # -------------------------------------------------------------
    # 3. Interactive HTML Dashboard with High Contrast & Heatmap
    # -------------------------------------------------------------
    table_rows = []
    for t in range(num_tasks):
        stage_name = f"Stage {t} (after T{t+1})"
        cells = [f'<td class="stage-cell font-semibold text-slate-800">{stage_name}</td>']
        for j in range(num_tasks):
            val = matrix[t][j]
            if val is None:
                cells.append('<td class="empty-cell text-slate-400">-</td>')
            else:
                pct = val * 100
                if pct >= 80:
                    bg = "rgba(34, 197, 94, 0.18)"
                    fg = "#15803d"
                elif pct >= 65:
                    bg = "rgba(59, 130, 246, 0.18)"
                    fg = "#1d4ed8"
                elif pct >= 50:
                    bg = "rgba(245, 158, 11, 0.22)"
                    fg = "#b45309"
                else:
                    bg = "rgba(239, 68, 68, 0.25)"
                    fg = "#b91c1c"
                cells.append(f'<td style="background:{bg}; color:{fg}; font-weight:700;">{pct:.2f}%</td>')
        stage_aa = s_data["average_accuracy_by_stage"][t] * 100
        cells.append(f'<td class="font-extrabold text-blue-700 bg-blue-50">{stage_aa:.2f}%</td>')
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    degradation_rows = []
    for j in range(num_tasks - 1):
        init_score = matrix[j][j] * 100
        final_score = matrix[num_tasks - 1][j] * 100
        drop = init_score - final_score
        pct_relative = (drop / init_score) * 100
        degradation_rows.append(f"""
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="font-bold text-slate-800">Task {j+1} (T{j+1})</td>
            <td class="text-emerald-700 font-semibold">{init_score:.2f}%</td>
            <td class="text-rose-700 font-bold">{final_score:.2f}%</td>
            <td class="text-rose-600 font-black">-{drop:.2f}%</td>
            <td class="text-slate-500 font-medium">-{pct_relative:.1f}% relative drop</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>B0 Baseline: Performance Matrix & Catastrophic Forgetting Plots</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        table {{ border-collapse: collapse; }}
        th, td {{ padding: 10px 14px; text-align: center; border: 1px solid #e2e8f0; font-size: 13.5px; }}
        .stage-cell {{ text-align: left; background: #f8fafc; }}
        .empty-cell {{ background: #f8fafc; color: #cbd5e1; }}
    </style>
</head>
<body class="bg-slate-100 text-slate-800 p-6 md:p-10 font-sans">
    <div class="max-w-6xl mx-auto space-y-8">
        
        <!-- Header -->
        <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
            <div>
                <span class="px-3.5 py-1 bg-amber-100 text-amber-900 text-xs font-black rounded-full uppercase tracking-wider">Experiment E001</span>
                <h1 class="text-2xl md:text-3xl font-black text-slate-900 mt-2">B0: Sequential Fine-Tuning Baseline</h1>
                <p class="text-sm text-slate-600 mt-1 font-medium">FewRel Track A (8 tasks × 10 relations, 5-shot, seed 2021) — M=0, No Replay Lower Bound</p>
            </div>
            <div class="flex flex-wrap gap-4">
                <div class="bg-blue-50 border border-blue-100 px-5 py-3 rounded-xl text-center min-w-[100px]">
                    <div class="text-xs text-blue-700 uppercase font-black">Final AA</div>
                    <div class="text-2xl font-black text-blue-700">{s_data['final_average_accuracy']*100:.2f}%</div>
                </div>
                <div class="bg-indigo-50 border border-indigo-100 px-5 py-3 rounded-xl text-center min-w-[100px]">
                    <div class="text-xs text-indigo-700 uppercase font-black">AIA</div>
                    <div class="text-2xl font-black text-indigo-700">{s_data['average_incremental_accuracy']*100:.2f}%</div>
                </div>
                <div class="bg-rose-50 border border-rose-100 px-5 py-3 rounded-xl text-center min-w-[100px]">
                    <div class="text-xs text-rose-700 uppercase font-black">BWT</div>
                    <div class="text-2xl font-black text-rose-700">{s_data['backward_transfer']*100:.2f}%</div>
                </div>
            </div>
        </div>

        <!-- Plots Section -->
        <div class="space-y-8">
            <!-- Chart 1: Forgetting Curves -->
            <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                <div class="mb-4">
                    <h2 class="text-xl font-black text-slate-900">1. Catastrophic Forgetting Curves ($A_{{t, j}}$)</h2>
                    <p class="text-sm text-slate-500 mt-0.5">Theo dõi trực tiếp sự sụt giảm hiệu năng dốc đứng của từng task cũ khi mô hình học tiếp các task sau.</p>
                </div>
                <div class="w-full overflow-hidden rounded-xl border border-slate-200 bg-white">
                    {svg1_content}
                </div>
            </div>

            <!-- Chart 2: Accuracy Over Tasks -->
            <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
                <div class="mb-4">
                    <h2 class="text-xl font-black text-slate-900">2. Accuracy Evolution ($AA_t$ vs $New_t$ vs $Old_t$)</h2>
                    <p class="text-sm text-slate-500 mt-0.5">Sự phân kỳ rõ rệt: mô hình liên tục học tốt task mới (~88%) nhưng lập tức quên sạch các task cũ (~55%).</p>
                </div>
                <div class="w-full overflow-hidden rounded-xl border border-slate-200 bg-white">
                    {svg2_content}
                </div>
            </div>
        </div>

        <!-- Performance Matrix Heatmap Table -->
        <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm">
            <div class="mb-4">
                <h2 class="text-xl font-black text-slate-900">3. Ma trận hiệu năng 8×8 (Performance Matrix $A_{{t, j}}$)</h2>
                <p class="text-sm text-slate-500 mt-0.5">Hàng $t$: Stage sau khi học xong Task $t+1$. Cột $j$: Độ chính xác trên tập kiểm tra của Task $j+1$.</p>
            </div>
            <div class="overflow-x-auto rounded-xl border border-slate-200">
                <table class="w-full">
                    <thead>
                        <tr class="bg-slate-100 font-bold text-slate-700">
                            <th class="stage-cell">Training Stage</th>
                            <th>T1 Eval</th>
                            <th>T2 Eval</th>
                            <th>T3 Eval</th>
                            <th>T4 Eval</th>
                            <th>T5 Eval</th>
                            <th>T6 Eval</th>
                            <th>T7 Eval</th>
                            <th>T8 Eval</th>
                            <th class="bg-blue-100 text-blue-900">Average ($AA_t$)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Task Degradation Breakdown Table -->
        <div class="bg-white p-6 md:p-8 rounded-2xl border border-slate-200 shadow-sm">
            <div class="mb-4">
                <h2 class="text-xl font-black text-slate-900">4. Bảng phân tích chi tiết độ quên từng Task (Catastrophic Forgetting Breakdown)</h2>
                <p class="text-sm text-slate-500 mt-0.5">Mức sụt giảm tuyệt đối ($A_{{j, j}} - A_{{7, j}}$) từ khi vừa học xong cho đến khi kết thúc task 8.</p>
            </div>
            <div class="overflow-x-auto rounded-xl border border-slate-200">
                <table class="w-full text-left">
                    <thead>
                        <tr class="bg-slate-100 font-bold text-slate-700">
                            <th>Task</th>
                            <th>Hiệu năng ban đầu ($A_{{j, j}}$)</th>
                            <th>Hiệu năng cuối cùng ($A_{{7, j}}$)</th>
                            <th>Độ sụt giảm tuyệt đối ($F_j$)</th>
                            <th>Độ suy giảm tương đối</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(degradation_rows)}
                    </tbody>
                </table>
            </div>
        </div>

    </div>
</body>
</html>
"""

    html_path = target_results_dir / "dashboard.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "forgetting_svg": svg1_path,
        "accuracy_svg": svg2_path,
        "forgetting_png": png_forg_path,
        "accuracy_png": png_acc_path,
        "dashboard_html": html_path,
    }


if __name__ == "__main__":
    generate_svg_and_html()
