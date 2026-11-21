#!/usr/bin/env python3
"""Generate publication-grade 4-panel SVG/PNG dashboard comparing P0, P2b, P3, P2a, and P1.

Visualizes:
1. Average Incremental Accuracy Evolution (Stage 1..8) for P0, P2b, P3, P2a, P1.
2. Router Performance & Coverage Evolution (Stage 1..8): P2b vs P2a vs P3 (Top-1 vs Top-2).
3. P3 Task Router Confusion Matrix (8x8 Heatmap, Top-1 Acc = 64.69%, Top-2 Coverage = 81.41%).
4. Final Per-Task Accuracy (T1..T8) 5-way Comparison Bar Chart.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
P0_METRICS = REPO_ROOT / "runs" / "p0_shared_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P1_METRICS = REPO_ROOT / "runs" / "p1_oracle_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P2A_METRICS = REPO_ROOT / "runs" / "p2a_centroid_router" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P2B_METRICS = REPO_ROOT / "runs" / "p2b_learned_router" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P3_METRICS = REPO_ROOT / "runs" / "p3_soft_router" / "fewrel_5shot" / "seed_2021" / "metrics.json"

VAULT_ATTACHMENTS = Path("/Users/vinhlq2512/Obisidian/Knowledge Hub/06 - Attachments")


def build_p0_to_p3_svg(p0: dict, p1: dict, p2a: dict, p2b: dict, p3: dict) -> str:
    p0_m = p0["proto_matrix_A"]
    p1_m = p1["proto_matrix_A"]
    p2a_m = p2a["routed_proto_matrix_A"]
    p2b_m = p2b["proto_matrix_A"]
    p3_m = p3["performance_matrix"]["matrix"]
    p3_cov_m = p3["router_topk_matrix"]["matrix"]
    p3_r1_m = p3["router_top1_matrix"]["matrix"]
    conf_m = p3["confusion_matrix"]

    # Stage averages (1..8)
    p0_stage = [sum(p0_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p1_stage = [sum(p1_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p2a_stage = [sum(p2a_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p2b_stage = [sum(p2b_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p3_stage = [p3["summary"]["average_accuracy_by_stage"][s] * 100 for s in range(8)]

    # Router Coverage / Accuracy over stages
    p2b_r_stage = [sum(p2b["router_acc_matrix_A"][s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p2a_r_stage = [sum(p2a["router_acc_matrix_A"][s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p3_r1_stage = [sum(p3_r1_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p3_cov_stage = [sum(p3_cov_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]

    # Final per-task accuracies (T1..T8 at stage 7)
    p0_final = [p0_m[7][t] * 100 for t in range(8)]
    p1_final = [p1_m[7][t] * 100 for t in range(8)]
    p2a_final = [p2a_m[7][t] * 100 for t in range(8)]
    p2b_final = [p2b_m[7][t] * 100 for t in range(8)]
    p3_final = [p3_m[7][t] * 100 for t in range(8)]

    W, H = 1320, 1050
    svg: list[str] = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" '
        'style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">'
    )

    svg.append(
        """<defs>
      <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="100%" stop-color="#f8fafc"/>
      </linearGradient>
      <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.08"/>
      </filter>
    </defs>"""
    )
    svg.append(f'<rect width="{W}" height="{H}" fill="url(#bgGrad)"/>')

    # Header
    svg.append(f'<rect x="30" y="20" width="{W - 60}" height="84" rx="10" fill="#0f172a" filter="url(#shadow)"/>')
    svg.append(
        '<text x="50" y="53" font-size="19" font-weight="800" fill="#ffffff">'
        "ĐỐI SÁNH THỰC NGHIỆM 5 MÔ HÌNH: P0 (SÀN) — P2b — P3 (SOFT ROUTER) — P2a — P1 (TRẦN ORACLE)"
        "</text>"
    )
    svg.append(
        '<text x="50" y="79" font-size="12.5" font-weight="500" fill="#94a3b8">'
        "Protocol v1 Track A (FewRel 8×10, strict 5-shot, Rehearsal-Free M=0, Seed 2021) | Google Colab GPU (CUDA)"
        "</text>"
    )
    svg.append(f'<rect x="{W - 340}" y="32" width="290" height="58" rx="6" fill="#1e293b"/>')
    svg.append(
        f'<text x="{W - 195}" y="53" text-anchor="middle" font-size="11" font-weight="600" fill="#38bdf8">'
        "P3 SOFT TOP-2 ROUTER COVERAGE"
        "</text>"
    )
    svg.append(
        f'<text x="{W - 195}" y="76" text-anchor="middle" font-size="14.5" font-weight="800" fill="#4ade80">'
        "Top-2 Cov: 81.41% | AIA: 90.58%"
        "</text>"
    )

    # -------------------------------------------------------------------------
    # PANEL 1: Average Incremental Accuracy Evolution (Top Left)
    # -------------------------------------------------------------------------
    p1_x, p1_y, p1_w, p1_h = 30, 125, 435, 370
    svg.append(
        f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p1_x + 18}" y="{p1_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "1. Diễn tiến Độ chính xác Tích lũy (AIA)"
        "</text>"
    )
    svg.append(
        f'<text x="{p1_x + 18}" y="{p1_y + 48}" font-size="11" fill="#64748b">'
        "Độ chính xác trung bình trên các tác vụ đã học sau mỗi Stage 1..8"
        "</text>"
    )

    # Grid
    gx, gy, gw, gh = p1_x + 48, p1_y + 68, p1_w - 68, 230
    for tick in [40, 50, 60, 70, 80, 90, 100]:
        ty = gy + gh - ((tick - 40) / 60) * gh
        svg.append(f'<line x1="{gx}" y1="{ty:.1f}" x2="{gx + gw}" y2="{ty:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{tick}%</text>')

    def pt(stage_idx: int, val: float) -> tuple[float, float]:
        cx = gx + (stage_idx / 7) * gw
        cy = gy + gh - ((val - 40) / 60) * gh
        return cx, cy

    def polyline(vals: list[float], stroke: str, stroke_w: float = 2.5, dash: str = "") -> str:
        coords = [f"{pt(i, v)[0]:.1f},{pt(i, v)[1]:.1f}" for i, v in enumerate(vals)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(coords)}" fill="none" stroke="{stroke}" stroke-width="{stroke_w}"{dash_attr}/>'

    svg.append(polyline(p1_stage, "#10b981", 3.0))        # P1 (Green)
    svg.append(polyline(p2a_stage, "#0284c7", 2.8))       # P2a (Blue)
    svg.append(polyline(p3_stage, "#8b5cf6", 2.8))        # P3 (Purple)
    svg.append(polyline(p2b_stage, "#f59e0b", 2.2, "5,3"))# P2b (Amber)
    svg.append(polyline(p0_stage, "#ef4444", 2.2, "3,3")) # P0 (Red)

    for i, v in enumerate(p1_stage):
        x, y = pt(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#10b981"/>')
    for i, v in enumerate(p2a_stage):
        x, y = pt(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#0284c7"/>')
    for i, v in enumerate(p3_stage):
        x, y = pt(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#8b5cf6"/>')
    for i, v in enumerate(p0_stage):
        x, y = pt(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#ef4444"/>')

    for s in range(8):
        sx = gx + (s / 7) * gw
        svg.append(f'<text x="{sx:.1f}" y="{gy + gh + 18}" text-anchor="middle" font-size="10.5" fill="#475569">S{s + 1}</text>')

    # Legend
    leg_y = p1_y + p1_h - 22
    items = [
        ("#10b981", "P1 Oracle: 70.1%"),
        ("#0284c7", "P2a Centroid: 59.4%"),
        ("#8b5cf6", "P3 Soft: 55.5%"),
        ("#f59e0b", "P2b Learned: 56.0%"),
        ("#ef4444", "P0 Sàn: 52.8%"),
    ]
    cur_lx = p1_x + 18
    for color, label in items:
        svg.append(f'<rect x="{cur_lx}" y="{leg_y - 8}" width="10" height="10" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{cur_lx + 14}" y="{leg_y}" font-size="9.5" font-weight="600" fill="#334155">{label}</text>')
        cur_lx += 82

    # -------------------------------------------------------------------------
    # PANEL 2: Router Performance & Top-2 Coverage Evolution (Top Right)
    # -------------------------------------------------------------------------
    p2_x, p2_y, p2_w, p2_h = 485, 125, 415, 370
    svg.append(
        f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p2_x + 18}" y="{p2_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "2. Năng lực Định tuyến: Top-1 vs Top-2 Coverage"
        "</text>"
    )
    svg.append(
        f'<text x="{p2_x + 18}" y="{p2_y + 48}" font-size="11" fill="#64748b">'
        "Tỷ lệ câu kiểm thử có tác vụ thật nằm trong ứng viên định tuyến"
        "</text>"
    )

    gx2, gy2, gw2, gh2 = p2_x + 48, p2_y + 68, p2_w - 68, 230
    for tick in [40, 50, 60, 70, 80, 90, 100]:
        ty = gy2 + gh2 - ((tick - 40) / 60) * gh2
        svg.append(f'<line x1="{gx2}" y1="{ty:.1f}" x2="{gx2 + gw2}" y2="{ty:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx2 - 8}" y="{ty + 4:.1f}" text-anchor="end" font-size="10" fill="#94a3b8">{tick}%</text>')

    def pt2(stage_idx: int, val: float) -> tuple[float, float]:
        cx = gx2 + (stage_idx / 7) * gw2
        cy = gy2 + gh2 - ((val - 40) / 60) * gh2
        return cx, cy

    def polyline2(vals: list[float], stroke: str, stroke_w: float = 2.5, dash: str = "") -> str:
        coords = [f"{pt2(i, v)[0]:.1f},{pt2(i, v)[1]:.1f}" for i, v in enumerate(vals)]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(coords)}" fill="none" stroke="{stroke}" stroke-width="{stroke_w}"{dash_attr}/>'

    svg.append(polyline2(p3_cov_stage, "#059669", 3.2))      # P3 Top-2 Coverage (Bold Emerald)
    svg.append(polyline2(p2a_r_stage, "#0284c7", 2.6))       # P2a Top-1 Centroid (Blue)
    svg.append(polyline2(p3_r1_stage, "#8b5cf6", 2.2, "4,3"))# P3 Top-1 Router (Purple dash)
    svg.append(polyline2(p2b_r_stage, "#f59e0b", 2.2, "3,3"))# P2b Learned Router (Amber dash)

    for i, v in enumerate(p3_cov_stage):
        x, y = pt2(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#059669"/>')
    for i, v in enumerate(p2a_r_stage):
        x, y = pt2(i, v)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#0284c7"/>')

    for s in range(8):
        sx = gx2 + (s / 7) * gw2
        svg.append(f'<text x="{sx:.1f}" y="{gy2 + gh2 + 18}" text-anchor="middle" font-size="10.5" fill="#475569">S{s + 1}</text>')

    # Legend 2
    leg_y2 = p2_y + p2_h - 22
    items2 = [
        ("#059669", "P3 Top-2 Cov: 81.4% (AIA: 90.6%)"),
        ("#0284c7", "P2a Top-1: 64.7% (AIA: 78.2%)"),
        ("#f59e0b", "P2b Learned Top-1: 50.9%"),
    ]
    cur_lx2 = p2_x + 18
    for color, label in items2:
        svg.append(f'<rect x="{cur_lx2}" y="{leg_y2 - 8}" width="10" height="10" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{cur_lx2 + 14}" y="{leg_y2}" font-size="9.5" font-weight="600" fill="#334155">{label}</text>')
        cur_lx2 += 130

    # -------------------------------------------------------------------------
    # PANEL 3: Task Confusion Matrix Heatmap (Top Far-Right)
    # -------------------------------------------------------------------------
    p3_x, p3_y, p3_w, p3_h = 920, 125, 370, 370
    svg.append(
        f'<rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="{p3_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p3_x + 16}" y="{p3_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "3. Ma trận Nhầm lẫn Task Router"
        "</text>"
    )
    svg.append(
        f'<text x="{p3_x + 16}" y="{p3_y + 48}" font-size="11" fill="#64748b">'
        "True Task (Hàng) vs Predicted Task (Cột) ở Stage 8"
        "</text>"
    )

    hm_x, hm_y = p3_x + 40, p3_y + 68
    cell_s = 34
    for r in range(8):
        svg.append(f'<text x="{hm_x - 8}" y="{hm_y + r * cell_s + 22}" text-anchor="end" font-size="10.5" font-weight="700" fill="#475569">T{r + 1}</text>')
        for c in range(8):
            val = conf_m[r][c]
            if r == c:
                # Diagonal: Emerald / Blue
                intensity = int(min(255, 255 - (val - 0.5) * 350))
                color = f"rgb(16, {int(150 + val * 105)}, {int(100 + val * 80)})"
                text_col = "#ffffff" if val > 0.6 else "#0f172a"
            else:
                red_int = int(min(255, val * 1200))
                color = f"rgba(239, 68, 68, {min(0.85, val * 4.5):.2f})" if val > 0.05 else "#f8fafc"
                text_col = "#7f1d1d" if val > 0.08 else "#94a3b8"

            svg.append(
                f'<rect x="{hm_x + c * cell_s}" y="{hm_y + r * cell_s}" width="{cell_s - 2}" height="{cell_s - 2}" '
                f'rx="4" fill="{color}"/>'
            )
            pct_str = f"{val * 100:.0f}"
            svg.append(
                f'<text x="{hm_x + c * cell_s + cell_s / 2 - 1}" y="{hm_y + r * cell_s + cell_s / 2 + 4}" '
                f'text-anchor="middle" font-size="10" font-weight="{700 if r == c else 500}" fill="{text_col}">'
                f"{pct_str}</text>"
            )

    for c in range(8):
        svg.append(f'<text x="{hm_x + c * cell_s + cell_s / 2 - 1}" y="{hm_y + 8 * cell_s + 15}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#475569">T{c + 1}</text>')

    # -------------------------------------------------------------------------
    # PANEL 4: Final Per-Task Accuracy (T1..T8) 5-way Bar Comparison (Bottom Full Width)
    # -------------------------------------------------------------------------
    p4_x, p4_y, p4_w, p4_h = 30, 520, W - 60, 490
    svg.append(
        f'<rect x="{p4_x}" y="{p4_y}" width="{p4_w}" height="{p4_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p4_x + 24}" y="{p4_y + 34}" font-size="16" font-weight="700" fill="#0f172a">'
        "4. Đối sánh Độ chính xác Chi tiết theo từng Tác vụ (T1..T8) sau khi Học xong Tác vụ 8"
        "</text>"
    )
    svg.append(
        f'<text x="{p4_x + 24}" y="{p4_y + 54}" font-size="12" fill="#64748b">'
        "So sánh 5 mô hình trên tập kiểm thử riêng biệt của từng tác vụ khi kiểm tra CIL toàn cục"
        "</text>"

    )

    # Legend for Panel 4
    leg_x = p4_x + p4_w - 630
    items4 = [
        ("#ef4444", "P0 Sàn (52.8%)"),
        ("#f59e0b", "P2b Learned (56.0%)"),
        ("#8b5cf6", "P3 Soft Top-2 (55.5%)"),
        ("#0284c7", "P2a Centroid (59.4%)"),
        ("#10b981", "P1 Oracle (70.1%)"),
    ]
    for color, label in items4:
        svg.append(f'<rect x="{leg_x}" y="{p4_y + 24}" width="12" height="12" rx="3" fill="{color}"/>')
        svg.append(f'<text x="{leg_x + 18}" y="{p4_y + 35}" font-size="11.5" font-weight="600" fill="#334155">{label}</text>')
        leg_x += 125

    gx4, gy4, gw4, gh4 = p4_x + 55, p4_y + 85, p4_w - 85, 330
    for tick in [0, 20, 40, 60, 80, 100]:
        ty = gy4 + gh4 - (tick / 100) * gh4
        svg.append(f'<line x1="{gx4}" y1="{ty:.1f}" x2="{gx4 + gw4}" y2="{ty:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx4 - 12}" y="{ty + 4:.1f}" text-anchor="end" font-size="11" fill="#94a3b8">{tick}%</text>')

    group_w = gw4 / 8
    bar_w = 19
    gap = 4

    for t in range(8):
        cx = gx4 + t * group_w + group_w / 2
        start_bx = cx - (5 * bar_w + 4 * gap) / 2

        # 5 bars: P0, P2b, P3, P2a, P1
        vals = [
            (p0_final[t], "#ef4444"),
            (p2b_final[t], "#f59e0b"),
            (p3_final[t], "#8b5cf6"),
            (p2a_final[t], "#0284c7"),
            (p1_final[t], "#10b981"),
        ]

        for b_idx, (val, color) in enumerate(vals):
            bx = start_bx + b_idx * (bar_w + gap)
            bh = (val / 100) * gh4
            by = gy4 + gh4 - bh
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="3" fill="{color}"/>')
            if val > 25:
                svg.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9" font-weight="700" fill="{color}">'
                    f"{val:.0f}</text>"
                )

        svg.append(
            f'<text x="{cx:.1f}" y="{gy4 + gh4 + 24}" text-anchor="middle" font-size="13" font-weight="800" fill="#1e293b">'
            f"Task {t + 1}</text>"
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{gy4 + gh4 + 40}" text-anchor="middle" font-size="10" fill="#64748b">'
            f"P3 Top-2: {p3_cov_m[7][t] * 100:.1f}%</text>"
        )

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> int:
    p0 = json.loads(P0_METRICS.read_text(encoding="utf-8"))
    p1 = json.loads(P1_METRICS.read_text(encoding="utf-8"))
    p2a = json.loads(P2A_METRICS.read_text(encoding="utf-8"))
    p2b = json.loads(P2B_METRICS.read_text(encoding="utf-8"))
    p3 = json.loads(P3_METRICS.read_text(encoding="utf-8"))

    svg_content = build_p0_to_p3_svg(p0, p1, p2a, p2b, p3)

    out_dir = REPO_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "p0_to_p3_comparison.svg"
    out_svg.write_text(svg_content, encoding="utf-8")
    print(f"Saved SVG to: {out_svg}")

    # Render PNG using qlmanage
    subprocess.run(["qlmanage", "-t", "-s", "1320", "-o", "/tmp", str(out_svg)], check=True)
    tmp_png = Path("/tmp") / f"{out_svg.name}.png"
    out_png = out_dir / "p0_to_p3_comparison.png"
    if tmp_png.exists():
        tmp_png.replace(out_png)
        print(f"Rendered PNG to: {out_png}")

    VAULT_ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    vault_svg = VAULT_ATTACHMENTS / "p0-to-p3-comparison.svg"
    vault_svg.write_text(svg_content, encoding="utf-8")
    if out_png.exists():
        vault_png = VAULT_ATTACHMENTS / "p0-to-p3-comparison.png"
        vault_png.write_bytes(out_png.read_bytes())
        print(f"Copied PNG to Vault: {vault_png}")

    return 0


if __name__ == "__main__":
    main()
