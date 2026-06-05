#!/usr/bin/env python3
"""
Export four PNG panels for Fig. dataset_examples (LaTeX):
  smbfo_input_patch.png, smbfo_gaussian_mask.png,
  smbfo_centroid_overlay.png, smbfo_prediction_overlay.png

**Default:** requires real ``data/processed/sm_bfo_com`` val shards (run ``preprocess_dataset``).
Synthetic patches are opt-in via ``--allow-synthetic-demo-only`` and must not be used in a journal.

Hybrid-STEMSeg prediction: ``--checkpoint`` …/hybrid-nogan/gan_seg_best.pt; else Otsu fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk
from skimage.measure import label, regionprops

from gan_seg.dataset_patches import build_atom_mask_from_com
from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.paper_export_requirements import exit_missing_processed_val


def _centroids_from_mask(mask_np: np.ndarray) -> np.ndarray:
    lbl = label(mask_np > 0.5)
    return np.array([p.centroid for p in regionprops(lbl)], dtype=np.float64)


def _otsu_pred(img_np: np.ndarray) -> np.ndarray:
    th = threshold_otsu(img_np)
    pred = (img_np > th).astype(np.float32)
    return binary_opening(pred > 0.5, disk(1)).astype(np.float32)


def _load_hybrid(ckpt_path: Path, device: torch.device):
    from gan_seg.train_benchmark import get_model

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = get_model("hybrid-nogan", device)
    model.load_state_dict(ckpt["G"], strict=False)
    model.eval()
    return model


def _synthetic_patch(h: int = 256, w: int = 256, sigma: float = 3.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = gaussian_filter(rng.standard_normal((h, w)), sigma=12.0).astype(np.float32)
    n_atoms = int(rng.integers(35, 65))
    com = []
    for _ in range(n_atoms):
        cy = float(rng.integers(8, h - 8))
        cx = float(rng.integers(8, w - 8))
        com.append([cy, cx])
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.5**2))
        base += float(rng.uniform(0.8, 1.6)) * g.astype(np.float32)
    com = np.asarray(com, dtype=np.float64)
    base += rng.standard_normal((h, w)).astype(np.float32) * 0.08
    mu, sd = float(base.mean()), float(base.std()) + 1e-6
    z = ((base - mu) / sd).astype(np.float32)
    z = np.clip(z, -6.0, 6.0)
    mask = build_atom_mask_from_com(h, w, com, 0, 0, sigma=sigma)
    x = z[np.newaxis, ...]
    return torch.from_numpy(x.copy()), torch.from_numpy(mask[np.newaxis, ...].copy())


def _save_panel(arr, cmap, vmin, vmax, out: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(out, dpi=dpi, pad_inches=0, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="paper_figures",
        help="Default: paper_figures/ at repo root (four smbfo_*.png for LaTeX).",
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--allow-synthetic-demo-only",
        action="store_true",
        help="DEVELOPMENT ONLY: fake patch — not real STEM data; do not submit in a manuscript.",
    )
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    used_synthetic = False

    if args.allow_synthetic_demo_only:
        img_t, mask_t = _synthetic_patch()
        used_synthetic = True
    else:
        exit_missing_processed_val(args.processed)
        try:
            ds = ShardedPatchDataset(args.processed, split="val")
            if len(ds) == 0:
                raise RuntimeError("empty dataset")
            img_t, mask_t = ds[args.index]
            used_synthetic = False
        except (FileNotFoundError, RuntimeError, IndexError) as exc:
            raise SystemExit(
                f"ERROR: Could not read val patch {args.index} from {args.processed}: {exc}\n"
                "Fix the dataset, or (dev only) pass --allow-synthetic-demo-only.\n"
            ) from exc

    img_np = img_t.numpy()[0]
    mask_np = mask_t.numpy()[0]
    lo, hi = np.percentile(img_np, [2, 98])
    disp = np.clip(img_np, lo, hi)

    ckpt_path = Path(args.checkpoint)
    pred_bin: np.ndarray
    if ckpt_path.is_file():
        model = _load_hybrid(ckpt_path, device)
        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device))
            pred_bin = (logits > 0.0).float()[0, 0].cpu().numpy()
    else:
        pred_bin = _otsu_pred(img_np)

    # 1) Input
    _save_panel(disp, "gray", lo, hi, out_dir / "smbfo_input_patch.png", args.dpi)

    # 2) Target mask (Gaussian blobs → thresholded binary in preprocessing)
    _save_panel(mask_np, "gray", 0.0, 1.0, out_dir / "smbfo_gaussian_mask.png", args.dpi)

    # 3) Centroids on image
    cents = _centroids_from_mask(mask_np)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
    if len(cents):
        ax.scatter(cents[:, 1], cents[:, 0], s=14, c="lime", edgecolors="darkgreen", linewidths=0.4)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(out_dir / "smbfo_centroid_overlay.png", dpi=args.dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    # 4) Prediction overlay (binary Hybrid-STEMSeg or Otsu fallback)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
    rgba = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
    rgba[..., 0] = np.clip(pred_bin, 0, 1)
    rgba[..., 1] = 0.25
    rgba[..., 2] = 0.9
    rgba[..., 3] = (pred_bin > 0.5).astype(np.float32) * 0.42
    ax.imshow(rgba)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(out_dir / "smbfo_prediction_overlay.png", dpi=args.dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    note = out_dir / ("README_smbfo_fig2.txt" if out_dir.name == "paper_figures" else "README_figure_source.txt")
    src = (
        "INTERNAL DEMO ONLY — synthetic (not for publication)"
        if used_synthetic
        else f"real val patch {args.processed}[{args.index}]"
    )
    if ckpt_path.is_file():
        pred_line = f"Hybrid-STEMSeg checkpoint: {ckpt_path.resolve()}"
    else:
        pred_line = "Otsu+opening baseline (no Hybrid-STEMSeg checkpoint found — replace panel when available)"
    note.write_text(
        f"source_patch: {src}\n{pred_line}\n"
        "Regenerate: python -m gan_seg.export_smbfo_figure_panels --processed data/processed/sm_bfo_com "
        "--checkpoint gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt --out-dir paper_figures\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}/smbfo_*.png")
    print(f"  source: {src}")
    print(f"  {pred_line}")
    print(f"  meta: {note}")


if __name__ == "__main__":
    main()
