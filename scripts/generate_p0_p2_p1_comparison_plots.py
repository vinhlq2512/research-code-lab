#!/usr/bin/env python3
"""Generate publication-grade 4-panel SVG/PNG dashboard comparing P0, P2b, P2a, and P1.

Visualizes:
1. Average Incremental Accuracy Evolution (Stage 1..8) for P0, P2b, P2a, P1.
2. Task 1 Retention Curve (Stage 1..8) showing how P2a retains 69.1% (vs P0 38.9% and P1 75.6%).
3. P2a Task Router Confusion Matrix (8x8 Heatmap, Router Acc = 64.69%).
4. Final Per-Task Accuracy (T1..T8) 4-Bar Comparison (P0 vs P2b vs P2a vs P1).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
P0_METRICS = REPO_ROOT / "runs" / "p0_shared_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P1_METRICS = REPO_ROOT / "runs" / "p1_oracle_prompt" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P2A_METRICS = REPO_ROOT / "runs" / "p2a_centroid_router" / "fewrel_5shot" / "seed_2021" / "metrics.json"
P2B_METRICS = REPO_ROOT / "runs" / "p2b_learned_router" / "fewrel_5shot" / "seed_2021" / "metrics.json"

VAULT_ATTACHMENTS = Path("/Users/vinhlq2512/Obisidian/Knowledge Hub/06 - Attachments")


def build_p0_p2_p1_svg(p0: dict, p1: dict, p2a: dict, p2b: dict) -> str:
    p0_m = p0["proto_matrix_A"]
    p1_m = p1["proto_matrix_A"]
    p2a_m = p2a["routed_proto_matrix_A"]
    p2b_m = p2b["proto_matrix_A"]
    conf_m = p2a["router_confusion_matrix"]

    p0_stage = [sum(p0_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p1_stage = [sum(p1_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p2a_stage = [sum(p2a_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]
    p2b_stage = [sum(p2b_m[s][: s + 1]) / (s + 1) * 100 for s in range(8)]

    p0_t1 = [p0_m[s][0] * 100 for s in range(8)]
    p1_t1 = [p1_m[s][0] * 100 for s in range(8)]
    p2a_t1 = [p2a_m[s][0] * 100 for s in range(8)]
    p2b_t1 = [p2b_m[s][0] * 100 for s in range(8)]

    p0_final = [p0_m[7][t] * 100 for t in range(8)]
    p1_final = [p1_m[7][t] * 100 for t in range(8)]
    p2a_final = [p2a_m[7][t] * 100 for t in range(8)]
    p2b_final = [p2b_m[7][t] * 100 for t in range(8)]

    W, H = 1280, 1020
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
    svg.append(f'<rect x="30" y="20" width="{W - 60}" height="80" rx="10" fill="#0f172a" filter="url(#shadow)"/>')
    svg.append(
        '<text x="50" y="53" font-size="19" font-weight="800" fill="#ffffff">'
        "ĐỐI SÁNH KHUNG KẸP SANDWICH BASELINES: P0 (SÀN) — P2a/P2b (TASK ROUTER) — P1 (TRẦN ORACLE)"
        "</text>"
    )
    svg.append(
        '<text x="50" y="79" font-size="12.5" font-weight="500" fill="#94a3b8">'
        "Protocol v1 Track A (FewRel 8×10, strict 5-shot, Rehearsal-Free M=0, Seed 2021) | Thực thi trên Google Colab GPU (CUDA)"
        "</text>"
    )
    svg.append(f'<rect x="{W - 315}" y="34" width="265" height="52" rx="6" fill="#1e293b"/>')
    svg.append(
        f'<text x="{W - 182}" y="53" text-anchor="middle" font-size="11" font-weight="600" fill="#38bdf8">'
        "P2a CENTROID ROUTER (NO TASK ID)"
        "</text>"
    )
    svg.append(
        f'<text x="{W - 182}" y="74" text-anchor="middle" font-size="14" font-weight="800" fill="#4ade80">'
        "Acc: 59.35% | Router: 64.69% | Gap: 37.7%"
        "</text>"
    )

    # -------------------------------------------------------------------------
    # PANEL 1: Average Incremental Accuracy Evolution (Top Left)
    # -------------------------------------------------------------------------
    p1_x, p1_y, p1_w, p1_h = 30, 120, 415, 360
    svg.append(
        f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p1_x + 18}" y="{p1_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "1. Độ Chính Xác Trung Bình Qua 8 Stage"
        "</text>"
    )
    svg.append(
        f'<text x="{p1_x + 18}" y="{p1_y + 47}" font-size="11" fill="#64748b">'
        "P2a vượt P0 (+6.53%) và P2b (+3.31%) khi không có Task ID"
        "</text>"
    )

    gx1, gy1, gw1, gh1 = p1_x + 48, p1_y + 65, p1_w - 72, p1_h - 115
    for pct in [40, 55, 70, 85, 100]:
        py = gy1 + gh1 - ((pct - 40) / 60.0) * gh1
        svg.append(f'<line x1="{gx1}" y1="{py:.1f}" x2="{gx1 + gw1}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(
            f'<text x="{gx1 - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#94a3b8">{pct}%</text>'
        )

    for s in range(8):
        px = gx1 + (s / 7.0) * gw1
        svg.append(f'<line x1="{px:.1f}" y1="{gy1}" x2="{px:.1f}" y2="{gy1 + gh1}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(
            f'<text x="{px:.1f}" y="{gy1 + gh1 + 18}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#475569">S{s + 1}</text>'
        )

    series_1 = [
        (p1_stage, "#059669", "P1 Oracle (70.14%)"),
        (p2a_stage, "#2563eb", "P2a Centroid (59.35%)"),
        (p2b_stage, "#d97706", "P2b Learned (56.04%)"),
        (p0_stage, "#ef4444", "P0 Shared (52.83%)"),
    ]
    for vals, color, _ in series_1:
        pts = [f"{gx1 + (s/7.0)*gw1:.1f},{gy1 + gh1 - ((vals[s]-40)/60.0)*gh1:.1f}" for s in range(8)]
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round"/>')
        for s in range(8):
            px = gx1 + (s / 7.0) * gw1
            py = gy1 + gh1 - ((vals[s] - 40) / 60.0) * gh1
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.8" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')

    for idx, (_, color, label) in enumerate(series_1):
        lx = gx1 + (idx % 2) * 175
        ly = gy1 + gh1 + 34 + (idx // 2) * 16
        svg.append(f'<rect x="{lx}" y="{ly - 8}" width="14" height="4" fill="{color}" rx="2"/>')
        svg.append(f'<text x="{lx + 20}" y="{ly - 4}" font-size="10.5" font-weight="700" fill="{color}">{label}</text>')

    # -------------------------------------------------------------------------
    # PANEL 2: Task 1 Retention Curve (Top Middle)
    # -------------------------------------------------------------------------
    p2_x, p2_y, p2_w, p2_h = 460, 120, 415, 360
    svg.append(
        f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p2_x + 18}" y="{p2_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "2. Đường Cong Giữ Lại Tác Vụ 1 (Task 1 Retention)"
        "</text>"
    )
    svg.append(
        f'<text x="{p2_x + 18}" y="{p2_y + 47}" font-size="11" fill="#64748b">'
        "P2a giữ được 69.1% ở Stage 8 (cao hơn P0 tới +30.2%)"
        "</text>"
    )

    gx2, gy2, gw2, gh2 = p2_x + 48, p2_y + 65, p2_w - 72, p2_h - 115
    for pct in [30, 45, 60, 75, 90]:
        py = gy2 + gh2 - ((pct - 30) / 65.0) * gh2
        svg.append(f'<line x1="{gx2}" y1="{py:.1f}" x2="{gx2 + gw2}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(
            f'<text x="{gx2 - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#94a3b8">{pct}%</text>'
        )

    for s in range(8):
        px = gx2 + (s / 7.0) * gw2
        svg.append(f'<line x1="{px:.1f}" y1="{gy2}" x2="{px:.1f}" y2="{gy2 + gh2}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(
            f'<text x="{px:.1f}" y="{gy2 + gh2 + 18}" text-anchor="middle" font-size="10.5" font-weight="600" fill="#475569">S{s + 1}</text>'
        )

    series_2 = [
        (p1_t1, "#059669", "P1 T1 (Cuối: 75.6%)"),
        (p2a_t1, "#2563eb", "P2a T1 (Cuối: 69.1%)"),
        (p2b_t1, "#d97706", "P2b T1 (Cuối: 64.4%)"),
        (p0_t1, "#ef4444", "P0 T1 (Cuối: 38.9%)"),
    ]
    for vals, color, _ in series_2:
        pts = [f"{gx2 + (s/7.0)*gw2:.1f},{gy2 + gh2 - ((vals[s]-30)/65.0)*gh2:.1f}" for s in range(8)]
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linecap="round"/>')
        for s in range(8):
            px = gx2 + (s / 7.0) * gw2
            py = gy2 + gh2 - ((vals[s] - 30) / 65.0) * gh2
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.8" fill="{color}" stroke="#ffffff" stroke-width="1.4"/>')

    for idx, (_, color, label) in enumerate(series_2):
        lx = gx2 + (idx % 2) * 175
        ly = gy2 + gh2 + 34 + (idx // 2) * 16
        svg.append(f'<rect x="{lx}" y="{ly - 8}" width="14" height="4" fill="{color}" rx="2"/>')
        svg.append(f'<text x="{lx + 20}" y="{ly - 4}" font-size="10.5" font-weight="700" fill="{color}">{label}</text>')

    # -------------------------------------------------------------------------
    # PANEL 3: P2a Task Router Confusion Matrix (Top Right)
    # -------------------------------------------------------------------------
    p3_x, p3_y, p3_w, p3_h = 890, 120, 360, 360
    svg.append(
        f'<rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="{p3_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p3_x + 18}" y="{p3_y + 30}" font-size="14" font-weight="700" fill="#0f172a">'
        "3. Ma Trận Nhầm Lẫn Router P2a (Stage 8)"
        "</text>"
    )
    svg.append(
        f'<text x="{p3_x + 18}" y="{p3_y + 47}" font-size="11" fill="#64748b">'
        "Đường chéo chính nổi bật (Router Acc = 64.69%)"
        "</text>"
    )

    hm_x, hm_y, cell_s = p3_x + 48, p3_y + 72, 34
    for k in range(8):
        svg.append(
            f'<text x="{hm_x + k * cell_s + cell_s / 2:.1f}" y="{hm_y - 6}" text-anchor="middle" '
            'font-size="10" font-weight="700" fill="#475569">'
            f"P{k + 1}</text>"
        )
        svg.append(
            f'<text x="{hm_x - 8}" y="{hm_y + k * cell_s + cell_s / 2 + 3.5:.1f}" text-anchor="end" '
            'font-size="10" font-weight="700" fill="#475569">'
            f"T{k + 1}</text>"
        )

    for i in range(8):
        for j in range(8):
            val = conf_m[i][j]
            cx = hm_x + j * cell_s
            cy = hm_y + i * cell_s
            if i == j:
                opacity = min(1.0, max(0.25, val))
                fill = "#2563eb"
                txt_color = "#ffffff"
            else:
                opacity = min(0.85, val * 2.8)
                fill = "#ef4444" if val >= 0.09 else "#94a3b8"
                txt_color = "#1e293b" if val < 0.15 else "#ffffff"
            svg.append(
                f'<rect x="{cx}" y="{cy}" width="{cell_s - 2}" height="{cell_s - 2}" rx="3" '
                f'fill="{fill}" fill-opacity="{opacity:.2f}"/>'
            )
            svg.append(
                f'<text x="{cx + (cell_s - 2) / 2:.1f}" y="{cy + (cell_s - 2) / 2 + 3.5:.1f}" '
                f'text-anchor="middle" font-size="9" font-weight="700" fill="{txt_color}">'
                f"{val * 100:.0f}%</text>"
            )

    # -------------------------------------------------------------------------
    # PANEL 4: Final Per-Task Accuracy 4-Bar Chart (Bottom)
    # -------------------------------------------------------------------------
    p4_x, p4_y, p4_w, p4_h = 30, 500, 1220, 495
    svg.append(
        f'<rect x="{p4_x}" y="{p4_y}" width="{p4_w}" height="{p4_h}" rx="10" fill="#ffffff" '
        'stroke="#e2e8f0" stroke-width="1.5" filter="url(#shadow)"/>'
    )
    svg.append(
        f'<text x="{p4_x + 25}" y="{p4_y + 34}" font-size="15.5" font-weight="700" fill="#0f172a">'
        "4. So Sánh Độ Chính Xác Từng Tác Vụ Sau Khi Học Xong Cả 8 Tác Vụ (P0 vs P2b vs P2a vs P1)"
        "</text>"
    )
    svg.append(
        f'<text x="{p4_x + 25}" y="{p4_y + 53}" font-size="12" fill="#64748b">'
        "Đỏ: P0 Shared (52.83%) | Cam: P2b Learned Router (56.04%) | Xanh dương: P2a Centroid Router (59.35%) | Xanh lá: P1 Oracle (70.14%)"
        "</text>"
    )

    gx4, gy4, gw4, gh4 = p4_x + 60, p4_y + 80, p4_w - 95, p4_h - 140
    for pct in [20, 40, 60, 80, 100]:
        py = gy4 + gh4 - (pct / 100.0) * gh4
        svg.append(f'<line x1="{gx4}" y1="{py:.1f}" x2="{gx4 + gw4}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(
            f'<text x="{gx4 - 12}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-weight="600" fill="#94a3b8">{pct}%</text>'
        )

    group_w = gw4 / 8.0
    bw = 26
    bars_cfg = [
        (p0_final, "#f87171", "#b91c1c", -1.5 * bw - 4.5),
        (p2b_final, "#fbbf24", "#b45309", -0.5 * bw - 1.5),
        (p2a_final, "#3b82f6", "#1d4ed8", 0.5 * bw + 1.5),
        (p1_final, "#10b981", "#047857", 1.5 * bw + 4.5),
    ]

    for t in range(8):
        cx = gx4 + (t + 0.5) * group_w
        for vals, fill_c, txt_c, offset in bars_cfg:
            v = vals[t]
            bh = (v / 100.0) * gh4
            bx = cx + offset - bw / 2.0
            by = gy4 + gh4 - bh
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" rx="4" fill="{fill_c}"/>')
            svg.append(
                f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
                f'font-size="9.5" font-weight="700" fill="{txt_c}">{v:.1f}%</text>'
            )

        gain_p2a = p2a_final[t] - p0_final[t]
        top_y = gy4 + gh4 - (max(p0_final[t], p2b_final[t], p2a_final[t], p1_final[t]) / 100.0) * gh4 - 26
        badge_fill = "#dbeafe" if gain_p2a >= 0 else "#fee2e2"
        badge_txt = "#1e40af" if gain_p2a >= 0 else "#991b1b"
        sign = "+" if gain_p2a >= 0 else ""
        svg.append(f'<rect x="{cx - 34:.1f}" y="{top_y:.1f}" width="68" height="16" rx="4" fill="{badge_fill}"/>')
        svg.append(
            f'<text x="{cx:.1f}" y="{top_y + 11.5:.1f}" text-anchor="middle" font-size="9.5" font-weight="800" fill="{badge_txt}">'
            f"P2a {sign}{gain_p2a:.1f}%</text>"
        )

        svg.append(
            f'<text x="{cx:.1f}" y="{gy4 + gh4 + 22}" text-anchor="middle" font-size="12.5" font-weight="800" fill="#1e293b">'
            f"Task {t + 1}</text>"
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{gy4 + gh4 + 37}" text-anchor="middle" font-size="10" fill="#64748b">'
            f"Router P2a: {conf_m[t][t] * 100:.1f}%</text>"
        )

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> int:
    p0 = json.loads(P0_METRICS.read_text(encoding="utf-8"))
    p1 = json.loads(P1_METRICS.read_text(encoding="utf-8"))
    p2a = json.loads(P2A_METRICS.read_text(encoding="utf-8"))
    p2b = json.loads(P2B_METRICS.read_text(encoding="utf-8"))

    svg_content = build_p0_p2_p1_svg(p0, p1, p2a, p2b)

    out_dir = REPO_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_svg = out_dir / "p0_p2_p1_comparison.svg"
    out_svg.write_text(svg_content, encoding="utf-8")
    print(f"Saved SVG to: {out_svg}")

    VAULT_ATTACHMENTS.mkdir(parents=True, exist_ok=True)
    vault_svg = VAULT_ATTACHMENTS / "p0-p2-p1-comparison.svg"
    vault_svg.write_text(svg_content, encoding="utf-8")
    print(f"Saved SVG to Vault: {vault_svg}")
    return 0


if __name__ == "__main__":
    main()
