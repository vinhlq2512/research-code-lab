from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

from evaluation.performance_matrix import PerformanceMatrix

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# Pure Python Canvas & PNG Writer (Zero Dependencies)
# ==============================================================================

class SimpleCanvas:
    """Minimal 2D RGB Canvas supporting line drawing, rectangles, and points."""

    def __init__(self, width: int, height: int, bg_color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.pixels = [[bg_color for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Bresenham's line algorithm with line thickness."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while True:
            for wx in range(-width // 2, width // 2 + 1):
                for wy in range(-width // 2, width // 2 + 1):
                    self.set_pixel(curr_x + wx, curr_y + wy, color)

            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

    def draw_circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        """Draw filled circle marker."""
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.set_pixel(x, y, color)

    def draw_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int], fill: bool = True) -> None:
        for cy in range(y, y + h):
            for cx in range(x, x + w):
                if fill or cy in {y, y + h - 1} or cx in {x, x + w - 1}:
                    self.set_pixel(cx, cy, color)

    def to_png(self) -> bytes:
        """Encode canvas pixels into a valid PNG binary format using zlib."""
        def make_chunk(tag: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            return length + tag + data + crc

        raw_data = bytearray()
        for y in range(self.height):
            raw_data.append(0)  # Filter type 0: None
            for x in range(self.width):
                r, g, b = self.pixels[y][x]
                raw_data.append(r)
                raw_data.append(g)
                raw_data.append(b)

        png_header = b"\x89PNG\r\n\x1a\n"
        ihdr = make_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0),  # 8-bit truecolor RGB
        )
        idat = make_chunk(b"IDAT", zlib.compress(bytes(raw_data), level=6))
        iend = make_chunk(b"IEND", b"")

        return png_header + ihdr + idat + iend

    def save_png(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(self.to_png())


# ==============================================================================
# Chart Rendering Functions
# ==============================================================================

PALETTE = [
    (31, 119, 180),   # Blue
    (255, 127, 14),   # Orange
    (44, 160, 44),    # Green
    (214, 39, 40),    # Red
    (148, 103, 189),  # Purple
    (140, 86, 75),    # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
]


def render_line_chart_canvas(
    series_list: list[dict[str, Any]],
    x_labels: list[str],
    title: str,
    y_range: tuple[float, float] = (0.0, 1.0),
    width: int = 800,
    height: int = 500,
) -> SimpleCanvas:
    """Render a multi-series line chart on SimpleCanvas."""
    canvas = SimpleCanvas(width, height, bg_color=(250, 250, 252))

    margin_left = 70
    margin_right = 160
    margin_top = 60
    margin_bottom = 60

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Draw plot background and border
    canvas.draw_rect(margin_left, margin_top, plot_w, plot_h, color=(255, 255, 255), fill=True)
    canvas.draw_rect(margin_left, margin_top, plot_w, plot_h, color=(200, 205, 215), fill=False)

    # Gridlines (horizontal)
    y_min, y_max = y_range
    num_grid = 5
    for i in range(num_grid + 1):
        grid_val = y_min + i * (y_max - y_min) / num_grid
        y_pos = int(margin_top + plot_h - (grid_val - y_min) / (y_max - y_min) * plot_h)
        canvas.draw_line(margin_left, y_pos, margin_left + plot_w, y_pos, color=(230, 235, 240), width=1)

    # Coordinate mapping helpers
    num_pts = len(x_labels)

    def to_coords(pt_idx: int, val: float) -> tuple[int, int]:
        px = int(margin_left + (pt_idx / max(1, num_pts - 1)) * plot_w)
        norm_y = (val - y_min) / max(1e-6, y_max - y_min)
        py = int(margin_top + plot_h - norm_y * plot_h)
        return px, py

    # Draw vertical gridlines
    for pt_idx in range(num_pts):
        px, _ = to_coords(pt_idx, 0.0)
        canvas.draw_line(px, margin_top, px, margin_top + plot_h, color=(240, 242, 246), width=1)

    # Plot each series
    for s_idx, s in enumerate(series_list):
        color = s.get("color", PALETTE[s_idx % len(PALETTE)])
        data_points = s["data"]  # list of (x_idx, val)

        # Draw lines
        for i in range(len(data_points) - 1):
            idx0, v0 = data_points[i]
            idx1, v1 = data_points[i + 1]
            x0, y0 = to_coords(idx0, v0)
            x1, y1 = to_coords(idx1, v1)
            canvas.draw_line(x0, y0, x1, y1, color=color, width=2)

        # Draw markers
        for idx, val in data_points:
            cx, cy = to_coords(idx, val)
            canvas.draw_circle(cx, cy, radius=4, color=color)

    # Draw Legend on the right
    legend_x = width - margin_right + 15
    legend_y = margin_top + 10
    for s_idx, s in enumerate(series_list):
        color = s.get("color", PALETTE[s_idx % len(PALETTE)])
        label = s["label"]
        ly = legend_y + s_idx * 24

        # Legend marker bar
        canvas.draw_line(legend_x, ly + 6, legend_x + 18, ly + 6, color=color, width=3)
        canvas.draw_circle(legend_x + 9, ly + 6, radius=3, color=color)

    return canvas


def generate_b0_plots(
    results_dir: Path | str,
    matrix_data: PerformanceMatrix | list[list[float | None]],
    summary_data: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate both publication plots: accuracy_over_tasks and forgetting_curve.

    Outputs:
        - plots/accuracy_over_tasks.png
        - plots/forgetting_curve.png
    """
    out_dir = Path(results_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(matrix_data, PerformanceMatrix):
        matrix = matrix_data.as_list()
    else:
        matrix = matrix_data

    num_tasks = len(matrix)
    task_labels = [f"T{i + 1}" for i in range(num_tasks)]

    # 1. Compute Series for accuracy_over_tasks
    # Average Accuracy (AA_t), Old Task Accuracy (Old_t), New Task Accuracy (New_t)
    aa_points: list[tuple[int, float]] = []
    old_points: list[tuple[int, float]] = []
    new_points: list[tuple[int, float]] = []

    for t in range(num_tasks):
        seen_scores = [matrix[t][j] for j in range(t + 1) if matrix[t][j] is not None]
        if seen_scores:
            aa = float(sum(seen_scores) / len(seen_scores))
            aa_points.append((t, aa))

        new_val = matrix[t][t]
        if new_val is not None:
            new_points.append((t, float(new_val)))

        if t > 0:
            old_scores = [matrix[t][j] for j in range(t) if matrix[t][j] is not None]
            if old_scores:
                old_points.append((t, float(sum(old_scores) / len(old_scores))))

    accuracy_plot_path = out_dir / "accuracy_over_tasks.png"
    forgetting_plot_path = out_dir / "forgetting_curve.png"

    # 2. Compute Series for forgetting_curve (Individual task degradation curves)
    forgetting_series: list[dict[str, Any]] = []
    for j in range(num_tasks - 1):  # Task j performance from stage j to num_tasks - 1
        traj: list[tuple[int, float]] = []
        for t in range(j, num_tasks):
            val = matrix[t][j]
            if val is not None:
                traj.append((t, float(val)))
        if traj:
            forgetting_series.append({
                "label": f"Task {j + 1} (T{j + 1})",
                "data": traj,
                "color": PALETTE[j % len(PALETTE)],
            })

    if HAS_MATPLOTLIB:
        # -------------------------------------------------------------
        # Matplotlib High-Quality Rendering
        # -------------------------------------------------------------
        # Chart 1: Accuracy Over Tasks
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot([p[0] + 1 for p in aa_points], [p[1] * 100 for p in aa_points], marker="o", linewidth=2.5, label="Average Accuracy (Seen Tasks)", color="#1f77b4")
        if new_points:
            ax.plot([p[0] + 1 for p in new_points], [p[1] * 100 for p in new_points], marker="s", linewidth=1.8, linestyle="--", label="New Task Accuracy", color="#2ca02c")
        if old_points:
            ax.plot([p[0] + 1 for p in old_points], [p[1] * 100 for p in old_points], marker="^", linewidth=1.8, linestyle=":", label="Old Tasks Accuracy", color="#d62728")

        ax.set_title("B0 Sequential Fine-Tuning: Accuracy Evolution Over Tasks", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Training Stage (Tasks 1 to 8)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xticks(list(range(1, num_tasks + 1)))
        ax.set_xticklabels(task_labels)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()
        fig.savefig(accuracy_plot_path)
        plt.close(fig)

        # Chart 2: Forgetting Curve
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        for s in forgetting_series:
            x_vals = [p[0] + 1 for p in s["data"]]
            y_vals = [p[1] * 100 for p in s["data"]]
            ax.plot(x_vals, y_vals, marker="o", linewidth=2, label=s["label"])

        ax.set_title("B0 Sequential Fine-Tuning: Catastrophic Forgetting Curves", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Training Stage", fontsize=11)
        ax.set_ylabel("Accuracy on Given Task (%)", fontsize=11)
        ax.set_xticks(list(range(1, num_tasks + 1)))
        ax.set_xticklabels(task_labels)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        fig.savefig(forgetting_plot_path)
        plt.close(fig)

    else:
        # -------------------------------------------------------------
        # Pure Python Standalone Canvas Rendering
        # -------------------------------------------------------------
        # Chart 1: Accuracy Over Tasks
        series_acc = [
            {"label": "Avg Acc", "data": aa_points, "color": (31, 119, 180)},
            {"label": "New Task", "data": new_points, "color": (44, 160, 44)},
            {"label": "Old Tasks", "data": old_points, "color": (214, 39, 40)},
        ]
        canvas1 = render_line_chart_canvas(
            series_list=series_acc,
            x_labels=task_labels,
            title="Accuracy Evolution Over Tasks",
            y_range=(0.0, 1.0),
        )
        canvas1.save_png(accuracy_plot_path)

        # Chart 2: Forgetting Curve
        canvas2 = render_line_chart_canvas(
            series_list=forgetting_series,
            x_labels=task_labels,
            title="Catastrophic Forgetting Curves",
            y_range=(0.0, 1.0),
        )
        canvas2.save_png(forgetting_plot_path)

    # 3. Always generate standalone vector SVG charts and interactive HTML dashboard
    svg_acc_path = out_dir / "accuracy_over_tasks.svg"
    svg_forg_path = out_dir / "forgetting_curve.svg"
    dashboard_html_path = Path(results_dir) / "dashboard.html"

    try:
        from ...generate_plots_svg import generate_svg_and_html
        generate_svg_and_html()
    except Exception:
        pass

    return {
        "accuracy_plot": accuracy_plot_path,
        "forgetting_plot": forgetting_plot_path,
        "accuracy_svg": svg_acc_path,
        "forgetting_svg": svg_forg_path,
        "dashboard_html": dashboard_html_path,
    }
