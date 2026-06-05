#!/usr/bin/env python3
"""
Fig. 6: Gaussian robustness curves — gaussian_severity_curves_mean_std.png

Recreates the plot produced by gan_seg.reviewer_study from CSV rows so you can
refresh the figure without re-running the full evaluation suite.

Default input: reports/reviewer_suite/robustness_multiseed.csv
Default output: paper_figures/gaussian_severity_curves_mean_std.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "Hybrid-STEMSeg",
    "SegFormer",
    "UNet",
    "DeepLabV3+",
    "Original GAN (Scratch)",
    "GAN (ResNet-Initialized)",
    "Classical Otsu+Opening",
]


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_gaussian_curves(rows: list[dict]) -> None:
    plt.figure(figsize=(9, 6))
    for model_name in MODEL_ORDER:
        rr = [r for r in rows if r["model"] == model_name and r["corruption"] == "gaussian"]
        if not rr:
            continue
        rr = sorted(rr, key=lambda x: float(x["severity"]))
        xs = np.array([float(r["severity"]) for r in rr], dtype=float)
        ys = np.array([float(r["f1_mean"]) for r in rr], dtype=float)
        es = np.array([float(r["f1_std"]) for r in rr], dtype=float)
        plt.plot(xs, ys, marker="o", label=model_name)
        plt.fill_between(xs, ys - es, ys + es, alpha=0.15)
    plt.xlabel("Gaussian severity")
    plt.ylabel("Detection F1 (centroid match, mean ± std)")
    plt.title("Gaussian Robustness Curves")
    plt.grid(alpha=0.25)
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("reports/reviewer_suite/robustness_multiseed.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_figures"),
        help="Writes gaussian_severity_curves_mean_std.png here.",
    )
    ap.add_argument("--dpi", type=int, default=220)
    cli = ap.parse_args()

    if not cli.csv.is_file():
        raise SystemExit(f"Missing CSV: {cli.csv.resolve()}")

    rows = load_rows(cli.csv)
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = cli.out_dir / "gaussian_severity_curves_mean_std.png"

    plot_gaussian_curves(rows)
    plt.savefig(out_png, dpi=cli.dpi)
    plt.close()

    readme = cli.out_dir / "README_smbfo_fig6.txt"
    readme.write_text(
        f"source_csv: {cli.csv.resolve()}\n"
        "figure: gaussian_severity_curves_mean_std.png (matches reviewer_study plot)\n"
        "Regenerate: python -m gan_seg.export_gaussian_robustness_figure --out-dir paper_figures\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_png.resolve()}")
    print(f"  README: {readme.resolve()}")


if __name__ == "__main__":
    main()
