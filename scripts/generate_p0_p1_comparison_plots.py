#!/usr/bin/env python3
"""Generate publication-grade SVG comparison visualizer for P0 vs P1.

Visualizes:
1. Average Incremental Accuracy Evolution (Stage 1..8) with Shaded Value Gap.
2. Catastrophic Forgetting Retention Curve for Task 1 (Stage 1..8).
3. Final Per-Task Accuracy (T1..T8) Bar Comparison with Delta Gains.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
P0_METRICS = REPO_ROOT / "runs" / "p0_shared_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P1_METRICS = REPO_ROOT / "runs" / "p1_oracle_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"

VAULT_ATTACHMENTS = Path("/Users/vinhlq2512/Obisidian/Knowledge Hub/06 - Attachments")


def build_comparison_svg(p0_data: dict, p1_data: dict) -> str:
    p0_m = p0_data["proto_matrix_A"]
    p1_m = p1_data["proto_matrix_A"]

    # 1. Stage Average Accuracies
    p0_stage_avg = [sum(p0_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p1_stage_avg = [sum(p1_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]

    # 2. Task 1 Retention Curve
    p0_t1_decay = [p0_m[s][0] * 100 for s in range(8)]
    p1_t1_decay = [p1_m[s][0] * 100 for s in range(8)]

    # 3. Final Per-Task Accuracy
    p0_final = [p0_m[7][t] * 100 for t in range(8)]
    p1_final = [p1_m[7][t] * 100 for t in range(8)]

    W, H = 1200, 920
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">')

    # Background gradient
    svg.append('''<defs>
      <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="100%" stop-color="#f8fafc"/>
      </linearGradient>
      <linearGradient id="p1Grad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#10b981" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#10b981" stop-opacity="0.02"/>
      </linearGradient>
      <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
        <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.08"/>
      </filter>
    </defs>''')

    svg.append(f'<rect width="{W}" height="{H}" fill="url(#bgGrad)"/>')

    # Header
    svg.append(f'<rect x="30" y="25" width="{W - 60}" height="75" rx="10" fill="#0f172a" filter="url(#shadow)"/>')
    svg.append(f'<text x="50" y="58" font-size="20" font-weight="800" fill="#ffffff">ĐỐI SÁNH THỰC NGHIỆM P0 (SHARED PROMPT) VÀ P1 (ORACLE-TASK PROMPT)</text>')
    svg.append(f'<text x="50" y="82" font-size="13" font-weight="500" fill="#94a3b8">Protocol v1 Track A (FewRel 8 tasks × 10 relations, strict 5-shot) | Backbone: 100% Frozen BERT-base | M=0 (Rehearsal-Free)</text>')
    svg.append(f'<rect x="{W - 270}" y="42" width="220" height="42" rx="6" fill="#1e293b"/>')
    svg.append(f'<text x="{W - 160}" y="60" text-anchor="middle" font-size="11" font-weight="600" fill="#38bdf8">TASK-AWARENESS VALUE GAP</text>')
    svg.append(f'<text x="{W - 160}" y="76" text-anchor="middle" font-size="15" font-weight="800" fill="#4ade80">Δ = +17.31% (Prototype Acc)</text>')

    # -------------------------------------------------------------
    # PANEL 1: Average Incremental Accuracy Evolution (Left Top)
    # -------------------------------------------------------------
    p1_x, p1_y, p1_w, p1_h = 45, 125, 535, 340
    svg.append(f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.append(f'<text x="{p1_x + 20}" y="{p1_y + 32}" font-size="15" font-weight="700" fill="#0f172a">1. Diễn Tiến Độ Chính Xác Trung Bình Qua 8 Stage</text>')
    svg.append(f'<text x="{p1_x + 20}" y="{p1_y + 50}" font-size="11" fill="#64748b">So sánh mức suy giảm hiệu năng toàn cục tích lũy (Average Incremental Accuracy)</text>')

    gw1, gh1 = p1_w - 90, p1_h - 110
    gx1, gy1 = p1_x + 55, p1_y + 65

    # Grid lines 0%..100%
    for pct in [20, 40, 60, 80, 100]:
        py = gy1 + gh1 - (pct / 100.0) * gh1
        svg.append(f'<line x1="{gx1}" y1="{py}" x2="{gx1 + gw1}" y2="{py}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx1 - 10}" y="{py + 4}" text-anchor="end" font-size="10" font-weight="600" fill="#94a3b8">{pct}%</text>')

    for s in range(8):
        px = gx1 + (s / 7.0) * gw1
        svg.append(f'<line x1="{px}" y1="{gy1}" x2="{px}" y2="{gy1 + gh1}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{px}" y="{gy1 + gh1 + 20}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">S{s + 1}</text>')

    # Shaded polygon between P1 and P0
    poly_pts = []
    for s in range(8):
        px = gx1 + (s / 7.0) * gw1
        py = gy1 + gh1 - (p1_stage_avg[s] / 100.0) * gh1
        poly_pts.append(f"{px:.1f},{py:.1f}")
    for s in reversed(range(8)):
        px = gx1 + (s / 7.0) * gw1
        py = gy1 + gh1 - (p0_stage_avg[s] / 100.0) * gh1
        poly_pts.append(f"{px:.1f},{py:.1f}")
    svg.append(f'<polygon points="{" ".join(poly_pts)}" fill="#10b981" fill-opacity="0.15"/>')

    # Line P0 (Red/Coral)
    p0_pts = [f"{gx1 + (s/7.0)*gw1:.1f},{gy1 + gh1 - (p0_stage_avg[s]/100.0)*gh1:.1f}" for s in range(8)]
    svg.append(f'<polyline points="{" ".join(p0_pts)}" fill="none" stroke="#ef4444" stroke-width="3" stroke-linecap="round"/>')
    for s in range(8):
        px = gx1 + (s / 7.0) * gw1
        py = gy1 + gh1 - (p0_stage_avg[s] / 100.0) * gh1
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#ef4444" stroke="#ffffff" stroke-width="1.8"/>')

    # Line P1 (Emerald Green)
    p1_pts = [f"{gx1 + (s/7.0)*gw1:.1f},{gy1 + gh1 - (p1_stage_avg[s]/100.0)*gh1:.1f}" for s in range(8)]
    svg.append(f'<polyline points="{" ".join(p1_pts)}" fill="none" stroke="#059669" stroke-width="3" stroke-linecap="round"/>')
    for s in range(8):
        px = gx1 + (s / 7.0) * gw1
        py = gy1 + gh1 - (p1_stage_avg[s] / 100.0) * gh1
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#059669" stroke="#ffffff" stroke-width="1.8"/>')

    # Legend 1
    svg.append(f'<rect x="{gx1 + 10}" y="{gy1 + 10}" width="16" height="4" fill="#059669" rx="2"/>')
    svg.append(f'<text x="{gx1 + 32}" y="{gy1 + 15}" font-size="11" font-weight="700" fill="#065f46">P1 Oracle Upper Bound (Cuối: {p1_stage_avg[-1]:.1f}%)</text>')
    svg.append(f'<rect x="{gx1 + 10}" y="{gy1 + 28}" width="16" height="4" fill="#ef4444" rx="2"/>')
    svg.append(f'<text x="{gx1 + 32}" y="{gy1 + 33}" font-size="11" font-weight="700" fill="#991b1b">P0 Shared Lower Bound (Cuối: {p0_stage_avg[-1]:.1f}%)</text>')

    # -------------------------------------------------------------
    # PANEL 2: Task 1 Retention Curve (Right Top)
    # -------------------------------------------------------------
    p2_x, p2_y, p2_w, p2_h = 620, 125, 535, 340
    svg.append(f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.append(f'<text x="{p2_x + 20}" y="{p2_y + 32}" font-size="15" font-weight="700" fill="#0f172a">2. Đường Cong Quên Lãng Của Tác Vụ 1 (Task 1 Retention)</text>')
    svg.append(f'<text x="{p2_x + 20}" y="{p2_y + 50}" font-size="11" fill="#64748b">Mức suy giảm độ chính xác của Tác vụ 1 khi mô hình học tiếp các Tác vụ 2..8</text>')

    gx2, gy2 = p2_x + 55, p2_y + 65
    gw2, gh2 = p2_w - 90, p2_h - 110

    for pct in [20, 40, 60, 80, 100]:
        py = gy2 + gh2 - (pct / 100.0) * gh2
        svg.append(f'<line x1="{gx2}" y1="{py}" x2="{gx2 + gw2}" y2="{py}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx2 - 10}" y="{py + 4}" text-anchor="end" font-size="10" font-weight="600" fill="#94a3b8">{pct}%</text>')

    for s in range(8):
        px = gx2 + (s / 7.0) * gw2
        svg.append(f'<line x1="{px}" y1="{gy2}" x2="{px}" y2="{gy2 + gh2}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{px}" y="{gy2 + gh2 + 20}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">S{s + 1}</text>')

    # Shaded poly for T1
    poly_pts2 = []
    for s in range(8):
        px = gx2 + (s / 7.0) * gw2
        py = gy2 + gh2 - (p1_t1_decay[s] / 100.0) * gh2
        poly_pts2.append(f"{px:.1f},{py:.1f}")
    for s in reversed(range(8)):
        px = gx2 + (s / 7.0) * gw2
        py = gy2 + gh2 - (p0_t1_decay[s] / 100.0) * gh2
        poly_pts2.append(f"{px:.1f},{py:.1f}")
    svg.append(f'<polygon points="{" ".join(poly_pts2)}" fill="#38bdf8" fill-opacity="0.15"/>')

    # Line P0 T1
    p0_t1_pts = [f"{gx2 + (s/7.0)*gw2:.1f},{gy2 + gh2 - (p0_t1_decay[s]/100.0)*gh2:.1f}" for s in range(8)]
    svg.append(f'<polyline points="{" ".join(p0_t1_pts)}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linecap="round"/>')
    for s in range(8):
        px = gx2 + (s / 7.0) * gw2
        py = gy2 + gh2 - (p0_t1_decay[s] / 100.0) * gh2
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.8"/>')

    # Line P1 T1
    p1_t1_pts = [f"{gx2 + (s/7.0)*gw2:.1f},{gy2 + gh2 - (p1_t1_decay[s]/100.0)*gh2:.1f}" for s in range(8)]
    svg.append(f'<polyline points="{" ".join(p1_t1_pts)}" fill="none" stroke="#0284c7" stroke-width="3" stroke-linecap="round"/>')
    for s in range(8):
        px = gx2 + (s / 7.0) * gw2
        py = gy2 + gh2 - (p1_t1_decay[s] / 100.0) * gh2
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#0284c7" stroke="#ffffff" stroke-width="1.8"/>')

    # End labels for T1
    p0_end_x = gx2 + gw2
    p0_end_y = gy2 + gh2 - (p0_t1_decay[-1] / 100.0) * gh2
    p1_end_x = gx2 + gw2
    p1_end_y = gy2 + gh2 - (p1_t1_decay[-1] / 100.0) * gh2

    svg.append(f'<rect x="{p0_end_x - 55}" y="{p0_end_y - 20}" width="50" height="18" fill="#fee2e2" rx="4" stroke="#f87171" stroke-width="1"/>')
    svg.append(f'<text x="{p0_end_x - 30}" y="{p0_end_y - 7}" text-anchor="middle" font-size="10" font-weight="800" fill="#b91c1c">{p0_t1_decay[-1]:.1f}%</text>')

    svg.append(f'<rect x="{p1_end_x - 55}" y="{p1_end_y - 20}" width="50" height="18" fill="#e0f2fe" rx="4" stroke="#38bdf8" stroke-width="1"/>')
    svg.append(f'<text x="{p1_end_x - 30}" y="{p1_end_y - 7}" text-anchor="middle" font-size="10" font-weight="800" fill="#0369a1">{p1_t1_decay[-1]:.1f}%</text>')

    # Legend 2
    svg.append(f'<rect x="{gx2 + 10}" y="{gy2 + 10}" width="16" height="4" fill="#0284c7" rx="2"/>')
    svg.append(f'<text x="{gx2 + 32}" y="{gy2 + 15}" font-size="11" font-weight="700" fill="#0369a1">P1 T1 Retention: Bền vững ở {p1_t1_decay[-1]:.1f}% (Chỉ giảm 11.5%)</text>')
    svg.append(f'<rect x="{gx2 + 10}" y="{gy2 + 28}" width="16" height="4" fill="#dc2626" rx="2"/>')
    svg.append(f'<text x="{gx2 + 32}" y="{gy2 + 33}" font-size="11" font-weight="700" fill="#991b1b">P0 T1 Retention: Quên nặng nề xuống {p0_t1_decay[-1]:.1f}% (Giảm 49.7%)</text>')

    # -------------------------------------------------------------
    # PANEL 3: Final Per-Task Accuracy Bar Comparison (Bottom)
    # -------------------------------------------------------------
    p3_x, p3_y, p3_w, p3_h = 45, 490, 1110, 395
    svg.append(f'<rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="{p3_h}" rx="10" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.append(f'<text x="{p3_x + 25}" y="{p3_y + 35}" font-size="16" font-weight="700" fill="#0f172a">3. So Sánh Độ Chính Xác Từng Tác Vụ Sau Khi Hoàn Thành Cả 8 Tác Vụ (Final Matrix Row 8)</text>')
    svg.append(f'<text x="{p3_x + 25}" y="{p3_y + 55}" font-size="12" fill="#64748b">Cột bên trái (Xanh ngọc): P1 Oracle-Task Prompt | Cột bên phải (Đỏ cam): P0 Shared Prompt | Huy hiệu: Độ chênh lệch tăng trưởng (Δ)</text>')

    gx3, gy3 = p3_x + 60, p3_y + 80
    gw3, gh3 = p3_w - 100, p3_h - 130

    for pct in [20, 40, 60, 80, 100]:
        py = gy3 + gh3 - (pct / 100.0) * gh3
        svg.append(f'<line x1="{gx3}" y1="{py}" x2="{gx3 + gw3}" y2="{py}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{gx3 - 12}" y="{py + 4}" text-anchor="end" font-size="11" font-weight="600" fill="#94a3b8">{pct}%</text>')

    bar_group_w = gw3 / 8.0
    bar_w = 34

    for t in range(8):
        cx = gx3 + (t + 0.5) * bar_group_w
        p1_val = p1_final[t]
        p0_val = p0_final[t]
        diff = p1_val - p0_val

        # P1 Bar
        bh1 = (p1_val / 100.0) * gh3
        bx1 = cx - bar_w - 4
        by1 = gy3 + gh3 - bh1
        svg.append(f'<rect x="{bx1:.1f}" y="{by1:.1f}" width="{bar_w}" height="{bh1:.1f}" rx="5" fill="#10b981"/>')
        svg.append(f'<text x="{bx1 + bar_w/2:.1f}" y="{by1 - 8:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#047857">{p1_val:.1f}%</text>')

        # P0 Bar
        bh0 = (p0_val / 100.0) * gh3
        bx0 = cx + 4
        by0 = gy3 + gh3 - bh0
        svg.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{bar_w}" height="{bh0:.1f}" rx="5" fill="#f87171"/>')
        svg.append(f'<text x="{bx0 + bar_w/2:.1f}" y="{by0 - 8:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#b91c1c">{p0_val:.1f}%</text>')

        # Delta Badge
        badge_y = min(by1, by0) - 28
        svg.append(f'<rect x="{cx - 28:.1f}" y="{badge_y:.1f}" width="56" height="17" rx="4" fill="#dcfce7" stroke="#86efac" stroke-width="1"/>')
        svg.append(f'<text x="{cx:.1f}" y="{badge_y + 12:.1f}" text-anchor="middle" font-size="10.5" font-weight="800" fill="#15803d">+{diff:.1f}%</text>')

        # X-axis Labels
        svg.append(f'<text x="{cx:.1f}" y="{gy3 + gh3 + 22}" text-anchor="middle" font-size="13" font-weight="800" fill="#1e293b">Task {t + 1}</text>')
        svg.append(f'<text x="{cx:.1f}" y="{gy3 + gh3 + 37}" text-anchor="middle" font-size="10.5" fill="#64748b">10 rels (1000 ex)</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def main() -> int:
    with P0_METRICS.open("r", encoding="utf-8") as f:
        p0_data = json.load(f)
    with P1_METRICS.open("r", encoding="utf-8") as f:
        p1_data = json.load(f)

    svg_content = build_comparison_svg(p0_data, p1_data)

    # 1. Save in research-code-lab
    out_dir = REPO_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "p0_vs_p1_comparison.svg"
    with out_svg.open("w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG to: {out_svg}")

    # 2. Save in Knowledge Hub Attachments
    VAULT_ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    vault_svg = VAULT_ATTACHMENTS / "p0-vs-p1-comparison.svg"
    with vault_svg.open("w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG to Vault: {vault_svg}")

    return 0


if __name__ == "__main__":
    main()
