#!/usr/bin/env python3
"""
Compare model metrics (RMSE, MAE, R2) by year across multiple models.

Data loading:
- Reads all CSV files under a directory (default: `time_result/`).
- Each CSV filename (without extension) is treated as the model name.
- Each CSV must include columns: year, RMSE, MAE, R2 (case-insensitive).

Output:
- PNG figure saved to: time_result/model_compare.png (by default)

Usage:
  python plot_compare.py [--dir DIR] [--out PATH] [--show]
"""

from __future__ import annotations

import os
import sys
import glob
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
    models: List[ModelData], out_path: str, show: bool = False
) -> None:
    try:
        import matplotlib.pyplot as plt  # local import to avoid startup errors
    except Exception as e:
        raise RuntimeError(
            f"Matplotlib is not available or failed to import: {e}"
        )

    metrics = ["RMSE", "MAE", "R2"]
    # Tab10 palette
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", ">", "<", "*"]

    # Union of years for x-axis ticks
    years: List[int] = sorted({y for m in models for y in m.years})

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9.5, 10.5), sharex=True)
    for ax, metric in zip(axes, metrics):
        for i, m in enumerate(models):
            ys = [m.series[metric].get(y, float("nan")) for y in years]
            ax.plot(
                years,
                ys,
                marker=markers[i % len(markers)],
                linestyle="-",
                color=palette[i % len(palette)],
                label=m.label,
            )
        ax.set_title(metric)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(metric)
        ax.legend(ncol=2, fontsize=9)

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


def _arg_value(argv: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
        return default
    return default


def load_models_from_dir(dir_path: str) -> List[ModelData]:
    pattern = os.path.join(dir_path, "*.csv")
    files = sorted(glob.glob(pattern))
    models: List[ModelData] = []
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        try:
            models.append(load_results(f, label=label))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not models:
        raise ValueError(f"No valid CSV files found in directory: {dir_path}")
    return models


def main(argv: List[str]) -> int:
    show = "--show" in argv
    dir_path = _arg_value(argv, "--dir", "time_result") or "time_result"
    out_path_arg = _arg_value(argv, "--out", None)

    try:
        models = load_models_from_dir(dir_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # Matplotlib-only output (PNG)
    try:
        out_png = out_path_arg or os.path.join(dir_path, "model_compare.png")
        plot_compare_matplotlib(models, out_path=out_png, show=show)
        return 0
    except Exception as e:
        print(f"Matplotlib is required to save PNG: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
