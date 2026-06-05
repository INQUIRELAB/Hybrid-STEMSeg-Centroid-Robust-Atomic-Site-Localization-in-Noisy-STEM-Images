#!/usr/bin/env python3
"""
Fig. 10: External Pt/Fe qualitative examples → paper_figures/

  jacs_input_frame.png          — HAADF (div8 crop, z-score display for visibility)
  jacs_expert_annotation.png    — Expert-derived Gaussian target mask (same σ as few-shot)
  jacs_hybrid_zeroshot.png      — Hybrid-STEMSeg, Sm-BFO benchmark checkpoint (zero-shot)
  jacs_hybrid_fewshot_k{N}.png — Hybrid-STEMSeg after N-shot (default N=5 → ..._k5.png; matches --n-shot)

Requires JACS ``extracted`` tree (Zenodo) and benchmark + few-shot Hybrid-STEMSeg weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from gan_seg.dataset_patches import build_atom_mask_from_com
from gan_seg.eval_noise_SOTA import load_benchmark_model
from gan_seg.train_benchmark import get_model
from gan_seg.jacs_data import (
    DEFAULT_JACS_EXTRACTED,
    crop_div8,
    discover_pairs,
    load_csv_coords_yx,
    normalize_patch2d,
)

ROOT = Path(__file__).resolve().parents[1]


def _require_extracted(extracted: Path) -> list[dict]:
    if not extracted.is_dir():
        print(
            f"ERROR: JACS extracted folder missing: {extracted}\n"
            "Unpack Zenodo 10.5281/zenodo.5931544 under external_stem_data/jacs_single_atom_TEM/extracted\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    pairs = discover_pairs(extracted)
    if not pairs:
        print(f"ERROR: No image/CSV pairs under {extracted}", file=sys.stderr)
        raise SystemExit(1)
    return pairs


def _display_stem(img_c: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = np.percentile(img_c, [2, 98])
    return np.clip(img_c, lo, hi), lo, hi


def _save_gray(path: Path, arr: np.ndarray, lo: float, hi: float, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(path, dpi=dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def _save_mask(path: Path, mask: np.ndarray, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(path, dpi=dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def _save_pred_overlay(
    path: Path, disp: np.ndarray, lo: float, hi: float, pred: np.ndarray, dpi: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
    rgba = np.zeros((*pred.shape, 4), dtype=np.float32)
    rgba[..., 0] = np.clip(pred, 0, 1)
    rgba[..., 1] = 0.35
    rgba[..., 2] = 0.95
    rgba[..., 3] = (pred > 0.5).astype(np.float32) * 0.45
    ax.imshow(rgba)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(path, dpi=dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", type=str, default=str(DEFAULT_JACS_EXTRACTED))
    ap.add_argument("--pair-index", type=int, default=0, help="Which frame from sorted discover_pairs")
    ap.add_argument("--mask-sigma", type=float, default=2.5)
    ap.add_argument("--benchmark-ckpt", type=str, default="", help="Override Hybrid-STEMSeg zero-shot ckpt")
    ap.add_argument(
        "--fewshot-ckpt",
        type=str,
        default="",
        help="Override few-shot checkpoint (default: checkpoints_jacs_fewshot/n{n}_seed{s}/hybrid-nogan/gan_seg_best.pt)",
    )
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    ap.add_argument("--dpi", type=int, default=220)
    cli = ap.parse_args()

    extracted = Path(cli.extracted)
    pairs = _require_extracted(extracted)
    pairs = sorted(pairs, key=lambda r: (r["category"], r["id"]))
    if cli.pair_index < 0 or cli.pair_index >= len(pairs):
        raise SystemExit(f"--pair-index {cli.pair_index} out of range 0..{len(pairs)-1}")
    rec = pairs[cli.pair_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = tifffile.imread(rec["image"])
    if img.ndim != 2:
        img = np.squeeze(img)
    img = np.asarray(img, dtype=np.float32)
    coords = load_csv_coords_yx(Path(rec["csv"]))
    img_c, coords_c = crop_div8(img, coords)
    if len(coords_c) < 1:
        raise SystemExit("No atoms in crop for this frame; try a different --pair-index")
    h, w = img_c.shape
    mask = build_atom_mask_from_com(h, w, coords_c, 0, 0, sigma=cli.mask_sigma)
    disp, lo, hi = _display_stem(img_c)
    x = torch.from_numpy(normalize_patch2d(img_c)).to(device)

    out = cli.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _save_gray(out / "jacs_input_frame.png", disp, lo, hi, cli.dpi)
    _save_mask(out / "jacs_expert_annotation.png", mask, cli.dpi)
    print(f"Frame: {rec['category']} {rec['id']}")
    print(f"Wrote {out / 'jacs_input_frame.png'}")
    print(f"Wrote {out / 'jacs_expert_annotation.png'}")

    bench = (
        Path(cli.benchmark_ckpt)
        if cli.benchmark_ckpt
        else ROOT / "gan_seg" / "checkpoints_benchmark" / "hybrid-nogan" / "gan_seg_best.pt"
    )
    if not bench.is_file():
        raise SystemExit(
            f"Missing benchmark checkpoint for zero-shot: {bench}\n"
            "Train Sm-BFO Hybrid-STEMSeg or pass --benchmark-ckpt"
        )
    model_zs = load_benchmark_model("hybrid-nogan", str(bench), device)
    with torch.no_grad():
        pred_zs = (model_zs(x.unsqueeze(0)) > 0.0).float()[0, 0].cpu().numpy()
    _save_pred_overlay(out / "jacs_hybrid_zeroshot.png", disp, lo, hi, pred_zs, cli.dpi)
    print(f"Wrote {out / 'jacs_hybrid_zeroshot.png'} (zero-shot {bench})")

    fs_path = (
        Path(cli.fewshot_ckpt)
        if cli.fewshot_ckpt
        else ROOT
        / "gan_seg"
        / "checkpoints_jacs_fewshot"
        / f"n{cli.n_shot}_seed{cli.seed}"
        / "hybrid-nogan"
        / "gan_seg_best.pt"
    )
    if not fs_path.is_file():
        raise SystemExit(
            f"Missing few-shot checkpoint: {fs_path}\n"
            "Run few-shot fine-tuning, e.g.:\n"
            f"  python -m gan_seg.train_jacs_fewshot --model hybrid-nogan --n-shot {cli.n_shot} --seed {cli.seed}\n"
            "Or pass --fewshot-ckpt PATH"
        )
    # Few-shot ckpts may pickle pathlib.Path in ``args`` (requires weights_only=False).
    ckpt_fs = torch.load(str(fs_path), map_location=device, weights_only=False)
    model_fs = get_model("hybrid-nogan", device)
    model_fs.load_state_dict(ckpt_fs["G"], strict=False)
    model_fs.eval()
    with torch.no_grad():
        pred_fs = (model_fs(x.unsqueeze(0)) > 0.0).float()[0, 0].cpu().numpy()
    few_fn = f"jacs_hybrid_fewshot_k{cli.n_shot}.png"
    _save_pred_overlay(out / few_fn, disp, lo, hi, pred_fs, cli.dpi)
    print(f"Wrote {out / few_fn} (few-shot {fs_path})")

    readme = out / "README_jacs_fig10.txt"
    readme.write_text(
        f"pair: {rec['category']} {rec['id']} (pair-index={cli.pair_index})\n"
        f"extracted: {extracted.resolve()}\n"
        f"mask_sigma: {cli.mask_sigma}\n"
        f"zero_shot_ckpt: {bench.resolve()}\n"
        f"few_shot_ckpt: {fs_path.resolve()} (n_shot={cli.n_shot}, seed={cli.seed})\n"
        f"Few-shot panel filename: jacs_hybrid_fewshot_k{cli.n_shot}.png\n"
        "Regenerate: python -m gan_seg.export_paper_fig10_jacs_examples --out-dir paper_figures\n",
        encoding="utf-8",
    )
    print(f"README: {readme.resolve()}")


if __name__ == "__main__":
    main()
