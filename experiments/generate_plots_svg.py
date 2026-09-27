#!/usr/bin/env python3
"""
Generate publication-quality SVG vector charts and an interactive HTML report
for B0 (Sequential Fine-Tuning) Continual Relation Extraction benchmark.
Includes exact labels, axes, percentages, and data markers.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "fewrel" / "5shot" / "B0_sequential_ft" / "seed_2021"
PLOTS_DIR = RESULTS_DIR / "plots"


def generate_svg_and_html():
    matrix_file = RESULTS_DIR / "performance_matrix.json"
    summary_file = RESULTS_DIR / "summary.json"

    with matrix_file.open("r", encoding="utf-8") as f:
        m_data = json.load(f)
    with summary_file.open("r", encoding="utf-8") as f:
        s_data = json.load(f)

    matrix = m_data["matrix"]
    num_tasks = len(matrix)
    task_labels = [f"T{i + 1}" for i in range(num_tasks)]

    colors = [
        "#1f77b4",  # T1 - Blue
        "#ff7f0e",  # T2 - Orange
        "#2ca02c",  # T3 - Green
        "#d62728",  # T4 - Red
        "#9467bd",  # T5 - Purple
        "#8c564b",  # T6 - Brown
        "#e377c2",  # T7 - Pink
        "#7f7f7f",  # T8 - Gray
    ]

    # Dimensions
    width, height = 900, 520
    margin_left, margin_right = 80, 180
    margin_top, margin_bottom = 70, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def map_x(t: int) -> float:
        return margin_left + (t / (num_tasks - 1)) * plot_w

    def map_y(val: float) -> float:
        # val in 0.0 .. 1.0 -> map to plot_h
        return margin_top + plot_h - (val * plot_h)

    # -------------------------------------------------------------
    # 1. SVG: Forgetting Curves (Individual Task Degradation)
    # -------------------------------------------------------------
    svg1 = []
    svg1.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">')
    
    # Title & Subtitle
    svg1.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">B0 Sequential Fine-Tuning: Catastrophic Forgetting Curves</text>')
    svg1.append(f'<text x="{width/2}" y="52" text-anchor="middle" font-size="13" fill="#64748b">FewRel Track A (8 tasks × 10 relations, 5-shot, seed 2021) — M=0, No Replay</text>')

    # Plot Background & Grid
    svg1.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="4"/>')

    # Horizontal Gridlines & Y-ticks
    for y_pct in range(0, 101, 20):
        y_val = y_pct / 100.0
        y_pos = map_y(y_val)
        svg1.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_w}" y2="{y_pos}" stroke="#e2e8f0" stroke-dasharray="4,4" stroke-width="1"/>')
        svg1.append(f'<text x="{margin_left - 12}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#64748b" font-weight="500">{y_pct}%</text>')

    # Vertical Gridlines & X-ticks
    for t_idx, label in enumerate(task_labels):
        x_pos = map_x(t_idx)
        svg1.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg1.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#334155" font-weight="bold">Stage {t_idx}</text>')
        svg1.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 40}" text-anchor="middle" font-size="11" fill="#64748b">(after {label})</text>')

    # Axes Labels
    svg1.append(f'<text x="{margin_left + plot_w/2}" y="{height - 12}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Continual Training Stage</text>')
    svg1.append(f'<text transform="rotate(-90)" x="-{margin_top + plot_h/2}" y="24" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Task Accuracy (%)</text>')

    # Render Task Degradation Lines
    for j in range(num_tasks - 1):
        color = colors[j]
        points = []
        for t in range(j, num_tasks):
            v = matrix[t][j]
            if v is not None:
                points.append((t, v, map_x(t), map_y(v)))

        # Polyline
        poly_pts = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in points])
        svg1.append(f'<polyline points="{poly_pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')

        # Markers & Values
        for idx, (t, v, px, py) in enumerate(points):
            svg1.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
            # Label start and end values
            if idx == 0:
                svg1.append(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" font-size="10" font-weight="bold" fill="{color}">{v*100:.1f}%</text>')
            elif idx == len(points) - 1:
                svg1.append(f'<text x="{px + 6:.1f}" y="{py + 12:.1f}" font-size="10" font-weight="bold" fill="{color}">{v*100:.1f}%</text>')

    # Legend on Right
    leg_x = width - margin_right + 25
    leg_y = margin_top + 15
    svg1.append(f'<rect x="{leg_x - 10}" y="{leg_y - 12}" width="{margin_right - 25}" height="{num_tasks * 28 + 15}" fill="#f8fafc" stroke="#e2e8f0" rx="6" />')
    svg1.append(f'<text x="{leg_x}" y="{leg_y + 4}" font-size="12" font-weight="bold" fill="#1e293b">Task Traces</text>')

    for j in range(num_tasks - 1):
        color = colors[j]
        item_y = leg_y + 24 + j * 26
        drop = (matrix[j][j] - matrix[num_tasks - 1][j]) * 100
        svg1.append(f'<line x1="{leg_x}" y1="{item_y}" x2="{leg_x + 20}" y2="{item_y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        svg1.append(f'<circle cx="{leg_x + 10}" cy="{item_y}" r="3.5" fill="{color}" stroke="#fff" stroke-width="1"/>')
        svg1.append(f'<text x="{leg_x + 28}" y="{item_y + 4}" font-size="11" font-weight="600" fill="#334155">T{j+1} <tspan font-weight="normal" fill="#ef4444">(-{drop:.1f}%)</tspan></text>')

    svg1.append('</svg>')
    svg1_content = "\n".join(svg1)

    svg1_path = PLOTS_DIR / "forgetting_curve.svg"
    with svg1_path.open("w", encoding="utf-8") as f:
        f.write(svg1_content)

    # -------------------------------------------------------------
    # 2. SVG: Accuracy Evolution (Avg Acc vs New vs Old)
    # -------------------------------------------------------------
    svg2 = []
    svg2.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">')
    svg2.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">B0 Sequential Fine-Tuning: Accuracy Evolution Over Tasks</text>')
    svg2.append(f'<text x="{width/2}" y="52" text-anchor="middle" font-size="13" fill="#64748b">Comparison between Average Accuracy, Newly Learned Task, and Previously Learned Tasks</text>')

    svg2.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" rx="4"/>')

    for y_pct in range(0, 101, 20):
        y_val = y_pct / 100.0
        y_pos = map_y(y_val)
        svg2.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_w}" y2="{y_pos}" stroke="#e2e8f0" stroke-dasharray="4,4" stroke-width="1"/>')
        svg2.append(f'<text x="{margin_left - 12}" y="{y_pos + 4}" text-anchor="end" font-size="12" fill="#64748b" font-weight="500">{y_pct}%</text>')

    for t_idx, label in enumerate(task_labels):
        x_pos = map_x(t_idx)
        svg2.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
        svg2.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#334155" font-weight="bold">Stage {t_idx}</text>')
        svg2.append(f'<text x="{x_pos}" y="{margin_top + plot_h + 40}" text-anchor="middle" font-size="11" fill="#64748b">(after {label})</text>')

    svg2.append(f'<text x="{margin_left + plot_w/2}" y="{height - 12}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Continual Training Stage</text>')
    svg2.append(f'<text transform="rotate(-90)" x="-{margin_top + plot_h/2}" y="24" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Accuracy (%)</text>')

    # Data series
    # Series 1: Average Accuracy
    aa_pts = [(t, s_data["average_accuracy_by_stage"][t], map_x(t), map_y(s_data["average_accuracy_by_stage"][t])) for t in range(num_tasks)]
    # Series 2: New Task Accuracy
    new_pts = [(t, matrix[t][t], map_x(t), map_y(matrix[t][t])) for t in range(num_tasks)]
    # Series 3: Old Tasks Accuracy
    old_pts = []
    for t in range(1, num_tasks):
        old_val = s_data["old_vs_new_by_stage"][f"stage_{t}"]["old_acc"]
        old_pts.append((t, old_val, map_x(t), map_y(old_val)))

    # Render Lines
    # New Task (Green dashed)
    p_new = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in new_pts])
    svg2.append(f'<polyline points="{p_new}" fill="none" stroke="#16a34a" stroke-dasharray="6,4" stroke-width="2.2" stroke-linecap="round"/>')
    for _, v, px, py in new_pts:
        svg2.append(f'<rect x="{px - 4:.1f}" y="{py - 4:.1f}" width="8" height="8" fill="#16a34a" stroke="#fff" stroke-width="1.5"/>')

    # Old Tasks (Red dotted)
    p_old = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in old_pts])
    svg2.append(f'<polyline points="{p_old}" fill="none" stroke="#dc2626" stroke-dasharray="3,3" stroke-width="2.2" stroke-linecap="round"/>')
    for _, v, px, py in old_pts:
        svg2.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#dc2626" stroke="#fff" stroke-width="1.5"/>')

    # Average Accuracy (Blue solid thick)
    p_aa = " ".join([f"{px:.1f},{py:.1f}" for _, _, px, py in aa_pts])
    svg2.append(f'<polyline points="{p_aa}" fill="none" stroke="#2563eb" stroke-width="3.2" stroke-linecap="round"/>')
    for t, v, px, py in aa_pts:
        svg2.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#2563eb" stroke="#fff" stroke-width="2"/>')
        svg2.append(f'<text x="{px:.1f}" y="{py - 9:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#1e40af">{v*100:.1f}%</text>')

    # Legend
    leg2_x = width - margin_right + 20
    leg2_y = margin_top + 20
    svg2.append(f'<rect x="{leg2_x - 10}" y="{leg2_y - 12}" width="{margin_right - 15}" height="120" fill="#f8fafc" stroke="#e2e8f0" rx="6"/>')
    svg2.append(f'<text x="{leg2_x}" y="{leg2_y + 4}" font-size="12" font-weight="bold" fill="#1e293b">Metrics</text>')

    # Item 1: AA
    svg2.append(f'<line x1="{leg2_x}" y1="{leg2_y + 24}" x2="{leg2_x + 22}" y2="{leg2_y + 24}" stroke="#2563eb" stroke-width="3"/>')
    svg2.append(f'<circle cx="{leg2_x + 11}" cy="{leg2_y + 24}" r="4" fill="#2563eb" stroke="#fff" stroke-width="1"/>')
    svg2.append(f'<text x="{leg2_x + 28}" y="{leg2_y + 28}" font-size="11" font-weight="600" fill="#1e293b">Average Acc</text>')

    # Item 2: New Task
    svg2.append(f'<line x1="{leg2_x}" y1="{leg2_y + 54}" x2="{leg2_x + 22}" y2="{leg2_y + 54}" stroke="#16a34a" stroke-dasharray="4,3" stroke-width="2.5"/>')
    svg2.append(f'<rect x="{leg2_x + 7}" y="{leg2_y + 50}" width="8" height="8" fill="#16a34a" stroke="#fff" stroke-width="1"/>')
    svg2.append(f'<text x="{leg2_x + 28}" y="{leg2_y + 58}" font-size="11" font-weight="600" fill="#1e293b">New Task Acc</text>')

    # Item 3: Old Tasks
    svg2.append(f'<line x1="{leg2_x}" y1="{leg2_y + 84}" x2="{leg2_x + 22}" y2="{leg2_y + 84}" stroke="#dc2626" stroke-dasharray="3,3" stroke-width="2.5"/>')
    svg2.append(f'<circle cx="{leg2_x + 11}" cy="{leg2_y + 84}" r="3.5" fill="#dc2626" stroke="#fff" stroke-width="1"/>')
    svg2.append(f'<text x="{leg2_x + 28}" y="{leg2_y + 88}" font-size="11" font-weight="600" fill="#1e293b">Old Tasks Acc</text>')

    svg2.append('</svg>')
    svg2_content = "\n".join(svg2)

    svg2_path = PLOTS_DIR / "accuracy_over_tasks.svg"
    with svg2_path.open("w", encoding="utf-8") as f:
        f.write(svg2_content)

    # -------------------------------------------------------------
    # 3. Interactive HTML Dashboard with Table, Heatmap & Vector SVGs
    # -------------------------------------------------------------
    # Generate Heatmap Table HTML
    table_rows = []
    for t in range(num_tasks):
        stage_name = f"Stage {t} (after T{t+1})"
        cells = [f'<td class="stage-cell font-semibold">{stage_name}</td>']
        for j in range(num_tasks):
            val = matrix[t][j]
            if val is None:
                cells.append('<td class="empty-cell">-</td>')
            else:
                pct = val * 100
                # Color intensity based on performance: green (high) -> yellow -> red (low)
                # 80-90%: green, 60-80%: blue/cyan, 40-60%: orange, <40%: red
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
                cells.append(f'<td style="background:{bg}; color:{fg}; font-weight:600;">{pct:.2f}%</td>')
        # Average acc for this stage
        stage_aa = s_data["average_accuracy_by_stage"][t] * 100
        cells.append(f'<td class="font-bold text-blue-700 bg-blue-50">{stage_aa:.2f}%</td>')
        table_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Degradation Summary Table
    degradation_rows = []
    for j in range(num_tasks - 1):
        init_score = matrix[j][j] * 100
        final_score = matrix[num_tasks - 1][j] * 100
        drop = init_score - final_score
        pct_relative = (drop / init_score) * 100
        degradation_rows.append(f"""
        <tr>
            <td class="font-semibold">Task {j+1} (T{j+1})</td>
            <td class="text-green-700 font-medium">{init_score:.2f}%</td>
            <td class="text-red-700 font-bold">{final_score:.2f}%</td>
            <td class="text-rose-600 font-bold">-{drop:.2f}%</td>
            <td class="text-slate-600">-{pct_relative:.1f}% relative drop</td>
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
        th, td {{ padding: 8px 12px; text-align: center; border: 1px solid #e2e8f0; font-size: 13px; }}
        .stage-cell {{ text-align: left; background: #f8fafc; }}
        .empty-cell {{ background: #f1f5f9; color: #94a3b8; }}
    </style>
</head>
<body class="bg-slate-50 text-slate-800 p-6 md:p-10 font-sans">
    <div class="max-w-6xl mx-auto space-y-10">
        
        <!-- Header -->
        <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <span class="px-3 py-1 bg-amber-100 text-amber-800 text-xs font-bold rounded-full uppercase tracking-wider">Experiment E001</span>
                <h1 class="text-2xl font-black text-slate-900 mt-2">B0: Sequential Fine-Tuning Baseline</h1>
                <p class="text-sm text-slate-600 mt-1">FewRel Track A (8 tasks × 10 relations, 5-shot, seed 2021) — Pure Lower Bound (M=0, No Replay)</p>
            </div>
            <div class="flex flex-wrap gap-3">
                <div class="bg-slate-100 px-4 py-2 rounded-lg text-center">
                    <div class="text-xs text-slate-500 uppercase font-bold">Final AA</div>
                    <div class="text-xl font-black text-blue-600">{s_data['final_average_accuracy']*100:.2f}%</div>
                </div>
                <div class="bg-slate-100 px-4 py-2 rounded-lg text-center">
                    <div class="text-xs text-slate-500 uppercase font-bold">AIA</div>
                    <div class="text-xl font-black text-indigo-600">{s_data['average_incremental_accuracy']*100:.2f}%</div>
                </div>
                <div class="bg-slate-100 px-4 py-2 rounded-lg text-center">
                    <div class="text-xs text-slate-500 uppercase font-bold">BWT</div>
                    <div class="text-xl font-black text-rose-600">{s_data['backward_transfer']*100:.2f}%</div>
                </div>
            </div>
        </div>

        <!-- Plots Section -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Chart 1: Forgetting Curves -->
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col">
                <h2 class="text-lg font-bold text-slate-900 mb-2">1. Catastrophic Forgetting Curves ($A_{{t, j}}$)</h2>
                <p class="text-xs text-slate-500 mb-4">Theo dõi trực tiếp sự sụt giảm hiệu năng của từng task cũ khi mô hình học tiếp các task sau.</p>
                <div class="w-full flex-grow flex items-center justify-center">
                    {svg1_content}
                </div>
            </div>

            <!-- Chart 2: Accuracy Over Tasks -->
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col">
                <h2 class="text-lg font-bold text-slate-900 mb-2">2. Accuracy Evolution ($AA_t$ vs $New_t$ vs $Old_t$)</h2>
                <p class="text-xs text-slate-500 mb-4">Sự phân kỳ rõ rệt giữa khả năng tiếp thu task mới ($New_t \\approx 88\%$) và việc xóa sổ kiến thức cũ ($Old_t \\to 55\%$).</p>
                <div class="w-full flex-grow flex items-center justify-center">
                    {svg2_content}
                </div>
            </div>
        </div>

        <!-- Performance Matrix Heatmap Table -->
        <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h2 class="text-lg font-bold text-slate-900 mb-2">3. Ma trận hiệu năng 8×8 (Performance Matrix $A_{{t, j}}$)</h2>
            <p class="text-xs text-slate-500 mb-4">Hàng $t$: Stage sau khi học xong Task $t+1$. Cột $j$: Độ chính xác trên tập kiểm tra của Task $j+1$.</p>
            <div class="overflow-x-auto">
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
        <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <h2 class="text-lg font-bold text-slate-900 mb-2">4. Bảng phân tích chi tiết độ quên từng Task (Catastrophic Forgetting Breakdown)</h2>
            <p class="text-xs text-slate-500 mb-4">Mức sụt giảm tuyệt đối ($A_{{j, j}} - A_{{7, j}}$) từ khi vừa học xong cho đến khi hoàn thành task 8.</p>
            <div class="overflow-x-auto">
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

    html_path = RESULTS_DIR / "dashboard.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated Vector SVG 1: {svg1_path}")
    print(f"Generated Vector SVG 2: {svg2_path}")
    print(f"Generated Interactive Dashboard HTML: {html_path}")


if __name__ == "__main__":
    generate_svg_and_html()
