#!/usr/bin/env python3
"""
Supplementary Fig.~9 style: JACS external few-shot adaptation vs shot count K.

Reads mean ± std (across RNG seeds) from:
  reports/jacs_fewshot/jacs_fewshot_grid_aggregate.csv

Output:
  paper_figures/fig9.png  (two panels: detection F1 and IoU)

Run from project root:
  .venv/bin/python gan_seg/export_paper_fig9_jacs_fewshot_curves.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "reports" / "jacs_fewshot" / "jacs_fewshot_grid_aggregate.csv"
DEFAULT_OUT = ROOT / "paper_figures" / "fig9.png"

MODEL_ORDER = ["hybrid-nogan", "deeplabv3plus", "segformer", "unet"]
DISPLAY = {
    "hybrid-nogan": "Hybrid-STEMSeg",
    "deeplabv3plus": "DeepLabV3+",
    "segformer": "SegFormer",
    "unet": "UNet",
}


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_figure(rows: list[dict], out_path: Path, dpi: int) -> None:
    by_model: dict[str, list[tuple[int, float, float, float, float]]] = {m: [] for m in MODEL_ORDER}
    for r in rows:
        m = r["model"].strip().lower()
        if m not in by_model:
            continue
        k = int(r["n_shot"])
        by_model[m].append(
            (
                k,
                float(r["f1_mean"]),
                float(r["f1_std"]),
                float(r["iou_mean"]),
                float(r["iou_std"]),
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    ks_sorted = sorted({t[0] for vs in by_model.values() for t in vs})

    for ax, key, ylabel in zip(
        axes,
        ("f1", "iou"),
        ("Detection F1 (centroid match)", "IoU (mean ± std)"),
    ):
        for mi, model in enumerate(MODEL_ORDER):
            pts = sorted(by_model[model], key=lambda x: x[0])
            if not pts:
                continue
            xs = np.array([p[0] for p in pts], dtype=float)
            if key == "f1":
                ys = np.array([p[1] for p in pts])
                es = np.array([p[2] for p in pts])
            else:
                ys = np.array([p[3] for p in pts])
                es = np.array([p[4] for p in pts])
            lw = 2.4 if model == "hybrid-nogan" else 1.4
            z = 2 if model == "hybrid-nogan" else 1
            (line,) = ax.plot(xs, ys, marker="o", linewidth=lw, zorder=z, label=DISPLAY[model])
            ax.fill_between(xs, ys - es, ys + es, alpha=0.18, color=line.get_color(), zorder=z - 1)
        ax.set_xticks(ks_sorted)
        ax.set_xlabel("Shots $K$ (train atoms per class)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_title("JACS Pt/Fe AC-STEM few-shot adaptation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=220)
    cli = ap.parse_args()

    if not cli.csv.is_file():
        raise SystemExit(
            f"Missing {cli.csv}\n"
            "Run few-shot grid first, e.g.:\n"
            "  python -m gan_seg.run_jacs_fewshot_multiseed\n"
        )

    rows = load_rows(cli.csv)
    plot_figure(rows, cli.out, cli.dpi)
    readme = cli.out.parent / "README_jacs_fig9.txt"
    readme.write_text(
        f"source_csv: {cli.csv.resolve()}\n"
        f"figure: {cli.out.name}\n"
        "Regenerate: .venv/bin/python gan_seg/export_paper_fig9_jacs_fewshot_curves.py\n",
        encoding="utf-8",
    )
    print(f"Wrote {cli.out.resolve()}")


if __name__ == "__main__":
    main()
