#!/usr/bin/env python3
"""
External validation on JACS / Zenodo single-atom catalysis STEM (Mitchell et al.):
  Zenodo https://doi.org/10.5281/zenodo.5931544

Uses manual atom coordinates (CSV: X=column, Y=row) and HAADF TIFs from the
ground-truth folders. Images are center-cropped to the largest H×W divisible by 8
(HybridUNet requirement); coordinates are filtered to the crop.

Writes reports/jacs_external/jacs_external_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gan_seg.dataset_patches import build_atom_mask_from_com
from gan_seg.eval_cross_domain import centroid_metrics, iou_np
from gan_seg.eval_noise_SOTA import load_benchmark_model
from gan_seg.jacs_data import crop_div8, discover_pairs, load_csv_coords_yx, normalize_patch2d

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED = (
    ROOT
    / "external_stem_data"
    / "jacs_single_atom_TEM"
    / "extracted"
)
REPORT_DIR = ROOT / "reports" / "jacs_external"


def run_eval(args: argparse.Namespace) -> None:
    import tifffile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extracted = Path(args.extracted)
    pairs = discover_pairs(extracted)
    if not pairs:
        raise SystemExit(f"No image/CSV pairs under {extracted}")

    name_to_slug = {
        "Hybrid-STEMSeg": "hybrid-nogan",
        "UNet": "unet",
        "SegFormer": "segformer",
        "DeepLabV3+": "deeplabv3plus",
    }
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    ckpt_dir = ROOT / "gan_seg" / "checkpoints_benchmark"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_doi": "https://doi.org/10.5281/zenodo.5931544",
        "paper_doi": "https://doi.org/10.1021/jacs.1c12466",
        "extracted_root": str(extracted.resolve()),
        "n_pairs": len(pairs),
        "device": str(device),
        "mask_sigma": args.mask_sigma,
        "centroid_match_px": args.centroid_match_px,
        "note": (
            "Sm-BFO-trained checkpoints evaluated zero-shot on real Pt/Fe AC-STEM; "
            "near-zero F1 indicates domain gap, not a loader bug."
        ),
        "pairs": pairs,
    }
    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for display_name in model_names:
        slug = name_to_slug.get(display_name)
        if not slug:
            print(f"[SKIP] unknown model {display_name}")
            continue
        ckpt = ckpt_dir / slug / "gan_seg_best.pt"
        if not ckpt.is_file():
            print(f"[SKIP] missing {ckpt}")
            continue
        model = load_benchmark_model(slug, str(ckpt), device)
        print(f"--- {display_name} ---")

        for rec in pairs:
            img = tifffile.imread(rec["image"])
            if img.ndim != 2:
                img = np.squeeze(img)
                if img.ndim != 2:
                    raise ValueError(f"Expected 2D image {rec['image']}, got {img.shape}")
            img = np.asarray(img, dtype=np.float32)
            coords = load_csv_coords_yx(Path(rec["csv"]))
            if len(coords) == 0:
                continue
            try:
                img_c, coords_c = crop_div8(img, coords, multiple=8)
            except ValueError as e:
                print(f"  [skip] {rec['id']}: {e}")
                continue
            if len(coords_c) < 1:
                print(f"  [skip] {rec['id']}: no atoms in crop")
                continue

            h, w = img_c.shape
            mask = build_atom_mask_from_com(h, w, coords_c, 0, 0, sigma=args.mask_sigma)
            if mask.sum() < 0.5:
                continue

            x = torch.from_numpy(normalize_patch2d(img_c)).to(device)
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                pred = (logits > 0.0).float()[0, 0].cpu().numpy()

            pr, rc, f1 = centroid_metrics(pred, mask, args.centroid_match_px)
            iou = iou_np(pred, mask)
            rows.append(
                {
                    "model": display_name,
                    "category": rec["category"],
                    "image_id": rec["id"],
                    "h": h,
                    "w": w,
                    "n_gt_atoms_crop": len(coords_c),
                    "precision": pr,
                    "recall": rc,
                    "f1": f1,
                    "iou": iou,
                    "mean_pred_fg": float(pred.mean()),
                    "logit_max": float(logits.max().item()),
                }
            )
            print(
                f"  {rec['category']} {rec['id'][:40]:<40} F1={f1:.4f} IoU={iou:.4f} "
                f"atoms={len(coords_c)}"
            )

    out_csv = REPORT_DIR / "jacs_external_metrics.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Aggregate by model x category (+ All)
        agg_path = REPORT_DIR / "jacs_external_summary.csv"
        summary: list[dict] = []
        models = sorted({r["model"] for r in rows})
        cats = sorted({r["category"] for r in rows})

        def _agg(sub: list[dict], m: str, label: str) -> dict:
            return {
                "model": m,
                "category": label,
                "n_images": len(sub),
                "f1_mean": float(np.mean([r["f1"] for r in sub])),
                "iou_mean": float(np.mean([r["iou"] for r in sub])),
                "precision_mean": float(np.mean([r["precision"] for r in sub])),
                "recall_mean": float(np.mean([r["recall"] for r in sub])),
                "mean_pred_fg": float(np.mean([r["mean_pred_fg"] for r in sub])),
                "mean_logit_max": float(np.mean([r["logit_max"] for r in sub])),
                "frac_logit_max_positive": float(np.mean([r["logit_max"] > 0 for r in sub])),
            }

        for m in models:
            for c in cats:
                sub = [r for r in rows if r["model"] == m and r["category"] == c]
                if not sub:
                    continue
                summary.append(_agg(sub, m, c))
            all_sub = [r for r in rows if r["model"] == m]
            if all_sub:
                summary.append(_agg(all_sub, m, "All"))
        with agg_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print(f"Wrote {out_csv} ({len(rows)} rows)")
        print(f"Wrote {agg_path}")
    else:
        print("No rows collected.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    p.add_argument(
        "--models",
        type=str,
        default="Hybrid-STEMSeg,UNet,SegFormer,DeepLabV3+",
    )
    p.add_argument("--mask-sigma", type=float, default=2.5, help="Gaussian disk sigma for GT mask")
    p.add_argument(
        "--centroid-match-px",
        type=float,
        default=10.0,
        help="Max distance for centroid matching (experimental STEM)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
