#!/usr/bin/env python3
"""
Optional Fig. 8: noise_ablation_f1_curve.png in paper_figures/

Reads **mean centroid F1** (per val patch, then averaged — same as reviewer ``clean`` metric)
vs Gaussian σ from CSV produced by ``python -m gan_seg.make_noise_figures``.
At σ=0 this matches **clean validation centroid F1**, not a pooled micro-F1 across patches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gan_seg.make_noise_figures import plot_f1_curves


def _load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "model": r["model"],
                    "noise_std": float(r["noise_std"]),
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                    "f1": float(r["f1"]),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("reports/noise_ablation/noise_ablation_results.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_figures"),
        help="Writes noise_ablation_f1_curve.png here.",
    )
    ap.add_argument("--dpi", type=int, default=220)
    cli = ap.parse_args()

    if not cli.csv.is_file():
        raise SystemExit(
            f"Missing {cli.csv.resolve()}\n"
            "Generate it with real val data, e.g.:\n"
            "  python -m gan_seg.make_noise_figures --outdir reports/noise_ablation\n"
        )

    rows = _load_rows(cli.csv)
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = cli.out_dir / "noise_ablation_f1_curve.png"
    plot_f1_curves(rows, out_png, dpi=cli.dpi)

    readme = cli.out_dir / "README_noise_ablation_fig8.txt"
    readme.write_text(
        f"source_csv: {cli.csv.resolve()}\n"
        "file: noise_ablation_f1_curve.png (same plot as make_noise_figures plot_f1_curves)\n"
        "Regenerate: python -m gan_seg.export_noise_ablation_f1_curve --out-dir paper_figures\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_png.resolve()}")
    print(f"README: {readme.resolve()}")


if __name__ == "__main__":
    main()
