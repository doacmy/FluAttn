#!/usr/bin/env python3
"""
Compare model metrics from two CSV files and plot RMSE, MAE, R2 by year.

Inputs (hardcoded defaults):
 - comparision/li/result.csv
 - fluAttn/result.csv

Outputs (prefers PNG with Matplotlib; falls back to SVG):
 - comparision/model_compare.png  (if Matplotlib available)
 - comparision/model_compare.svg  (fallback, no external libs required)

Usage:
    python plot_compare.py [--show] [--svg]

Notes:
 - If your environment lacks a working Matplotlib/NumPy, the script
   automatically falls back to a pure-SVG renderer.
 - Use --svg to force the SVG fallback even if Matplotlib is present.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Tuple, Optional


class ModelData:
    def __init__(self, label: str, rows: List[Dict[str, Optional[float]]]):
        self.label = label
        # rows: each has Year (int), RMSE/MAE/R2 (float or None)
        rows_sorted = sorted(rows, key=lambda r: r["Year"])  # type: ignore[arg-type]
        self.years: List[int] = [int(r["Year"]) for r in rows_sorted if r.get("Year") is not None]
        self.series: Dict[str, Dict[int, float]] = {
            "RMSE": {int(r["Year"]): float(r["RMSE"]) for r in rows_sorted if r.get("RMSE") is not None},
            "MAE": {int(r["Year"]): float(r["MAE"]) for r in rows_sorted if r.get("MAE") is not None},
            "R2": {int(r["Year"]): float(r["R2"]) for r in rows_sorted if r.get("R2") is not None},
        }


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def load_results(path: str, label: str) -> ModelData:
    """Load a results CSV (no external libs), normalize, and keep Year/RMSE/MAE/R2."""
    import csv

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: List[Dict[str, Optional[float]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize header keys to lowercase mapping
        field_map = { (k or "").strip().lower(): k for k in (reader.fieldnames or []) }

        required = ["year", "rmse", "mae", "r2"]
        missing = [c for c in required if c not in field_map]
        if missing:
            raise ValueError(
                f"Missing required columns in {path}: {missing}. Expected year, rmse, mae, r2."
            )

        for row in reader:
            year_raw = row.get(field_map["year"], "")
            try:
                year = int(float(year_raw)) if year_raw != "" else None
            except Exception:
                year = None

            rmse = _parse_float(row.get(field_map["rmse"], ""))
            mae = _parse_float(row.get(field_map["mae"], ""))
            r2 = _parse_float(row.get(field_map["r2"], ""))

            if year is None:
                continue
            rows.append({"Year": year, "RMSE": rmse, "MAE": mae, "R2": r2})

    return ModelData(label=label, rows=rows)


def plot_compare_matplotlib(
    m1: ModelData, m2: ModelData, out_path: str, show: bool = False
) -> None:
    try:
        import matplotlib.pyplot as plt  # local import to avoid startup errors
    except Exception as e:
        raise RuntimeError(
            f"Matplotlib is not available or failed to import: {e}"
        )

    metrics = ["RMSE", "MAE", "R2"]
    colors = {m1.label: "#1f77b4", m2.label: "#ff7f0e"}

    years = sorted(set(m1.years) | set(m2.years))

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 10), sharex=True)
    for ax, metric in zip(axes, metrics):
        y1 = [m1.series[metric].get(y, float("nan")) for y in years]
        y2 = [m2.series[metric].get(y, float("nan")) for y in years]
        ax.plot(years, y1, marker="o", linestyle="-", color=colors[m1.label], label=m1.label)
        ax.plot(years, y2, marker="s", linestyle="-", color=colors[m2.label], label=m2.label)
        ax.set_title(metric)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(metric)
        ax.legend()

    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(years)

    fig.suptitle("Model Performance Comparison (RMSE, MAE, R2)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)


def _build_series(model: ModelData) -> Dict[str, Dict[int, float]]:
    return model.series


def plot_compare_svg(m1: ModelData, m2: ModelData, out_path: str) -> None:
    # Colors consistent with matplotlib defaults: blue, orange
    color1 = "#1f77b4"
    color2 = "#ff7f0e"
    label1 = m1.label
    label2 = m2.label

    years = sorted(set(m1.years) | set(m2.years))
    metrics = ["RMSE", "MAE", "R2"]

    s1 = _build_series(m1)
    s2 = _build_series(m2)

    # Layout
    width = 900
    panel_h = 220
    panels = len(metrics)
    top = 70
    bottom = 40
    left = 70
    right = 30
    vgap = 30
    height = top + panels * panel_h + (panels - 1) * vgap + bottom

    x_positions = {y: i for i, y in enumerate(years)}
    if len(years) <= 1:
        x_step = 1.0
    else:
        x_step = (width - left - right) / max(1, (len(years) - 1))

    def xpx(y: int) -> float:
        return left + x_positions[y] * x_step

    def y_scale(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        pad = 0.05 * (vmax - vmin)
        return vmin - pad, vmax + pad

    def ypx(v: float, ymin: float, ymax: float, panel_top: float, panel_bottom: float) -> float:
        # Map to pixel with y increasing downwards
        if ymax == ymin:
            return (panel_top + panel_bottom) / 2
        return panel_bottom - (v - ymin) / (ymax - ymin) * (panel_bottom - panel_top)

    # Build SVG parts
    parts: List[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<style> text{font-family:Arial,Helvetica,sans-serif;font-size:12px} .grid{stroke:#ccc;stroke-dasharray:3,3} .axis{stroke:#333} </style>")

    # Title and legend
    parts.append(
        "<text x='20' y='24' style='font-size:18px;font-weight:bold'>Model Performance Comparison (RMSE, MAE, R2)</text>"
    )
    # Legend
    parts.append(f"<rect x='{width-260}' y='10' width='240' height='30' fill='white' stroke='#ddd' />")
    parts.append(f"<circle cx='{width-240}' cy='25' r='5' fill='{color1}' />")
    parts.append(f"<text x='{width-228}' y='29'>{label1}</text>")
    parts.append(f"<rect x='{width-150}' y='20' width='10' height='10' fill='{color2}' />")
    parts.append(f"<text x='{width-132}' y='29'>{label2}</text>")

    for idx, metric in enumerate(metrics):
        panel_top = top + idx * (panel_h + vgap)
        panel_bottom = panel_top + panel_h

        # Frame
        parts.append(f"<rect x='{left}' y='{panel_top}' width='{width-left-right}' height='{panel_h}' fill='none' stroke='#ddd' />")

        # Collect values for scaling
        vals: List[float] = []
        for y in years:
            if y in s1[metric]:
                vals.append(s1[metric][y])
            if y in s2[metric]:
                vals.append(s2[metric][y])
        ymin, ymax = y_scale(vals)

        # Grid and y ticks (min, mid, max)
        yticks = [ymin, (ymin + ymax) / 2, ymax]
        for tv in yticks:
            ypix = ypx(tv, ymin, ymax, panel_top, panel_bottom)
            parts.append(f"<line class='grid' x1='{left}' y1='{ypix}' x2='{width-right}' y2='{ypix}' />")
            parts.append(f"<text x='{left-8}' y='{ypix+4}' text-anchor='end'>{tv:.3g}</text>")

        # X axis and ticks (years)
        parts.append(f"<line class='axis' x1='{left}' y1='{panel_bottom}' x2='{width-right}' y2='{panel_bottom}' />")
        for y in years:
            xp = xpx(y)
            parts.append(f"<line class='axis' x1='{xp}' y1='{panel_bottom}' x2='{xp}' y2='{panel_bottom+5}' />")
            parts.append(f"<text x='{xp}' y='{panel_bottom+20}' text-anchor='middle'>{y}</text>")

        # Series 1 line
        path_cmds = []
        for j, y in enumerate(years):
            if y in s1[metric]:
                xp = xpx(y)
                yp = ypx(s1[metric][y], ymin, ymax, panel_top, panel_bottom)
                path_cmds.append((xp, yp))
        if path_cmds:
            d = " ".join([("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(path_cmds)])
            parts.append(f"<path d='{d}' fill='none' stroke='{color1}' stroke-width='2' />")
            for (x, y) in path_cmds:
                parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3' fill='{color1}' />")

        # Series 2 line
        path_cmds = []
        for j, y in enumerate(years):
            if y in s2[metric]:
                xp = xpx(y)
                yp = ypx(s2[metric][y], ymin, ymax, panel_top, panel_bottom)
                path_cmds.append((xp, yp))
        if path_cmds:
            d = " ".join([("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(path_cmds)])
            parts.append(f"<path d='{d}' fill='none' stroke='{color2}' stroke-width='2' />")
            for (x, y) in path_cmds:
                # square marker via small rect
                parts.append(f"<rect x='{x-3:.2f}' y='{y-3:.2f}' width='6' height='6' fill='{color2}' />")

        # Title for each panel (left top)
        parts.append(f"<text x='{left}' y='{panel_top-8}' style='font-weight:bold'>{metric}</text>")

    parts.append("</svg>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Saved SVG to: {out_path}")


def main(argv: list[str]) -> int:
    show = "--show" in argv
    force_svg = "--svg" in argv

    # Default input paths and labels
    path_li = os.path.join("comparision", "li", "result.csv")
    path_flu = os.path.join("fluAttn", "result.csv")

    try:
        m_li = load_results(path_li, label="LI")
        m_flu = load_results(path_flu, label="FluAttn")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    if not force_svg:
        # Try Matplotlib path first
        try:
            out_png = os.path.join("comparision", "model_compare.png")
            plot_compare_matplotlib(m_li, m_flu, out_path=out_png, show=show)
            return 0
        except Exception as e:
            print(f"Matplotlib unavailable, falling back to SVG: {e}")

    # Fallback: SVG (no external libs)
    try:
        out_svg = os.path.join("comparision", "model_compare.svg")
        plot_compare_svg(m_li, m_flu, out_path=out_svg)
    except Exception as e:
        print(f"Error generating SVG: {e}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
