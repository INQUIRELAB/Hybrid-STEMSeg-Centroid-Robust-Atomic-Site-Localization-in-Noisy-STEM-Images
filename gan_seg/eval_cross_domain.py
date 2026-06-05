#!/usr/bin/env python3
"""
Cross-domain evaluation using:
  1) AtomAI / external Sm-BFO dict (byte-identical to data/SmBFO_composition_series.npy when re-downloaded).
  2) Graphene stacks: pseudo-GT from peak_local_max on **min–max normalized** patches (weak labels;
     cross-material OOD). Raw graphene uses ~1e-2 dynamic range; skimage applies threshold_rel to
     max(image), so normalization is required for sensible peaks.

Writes CSV + qualitative figure under reports/cross_domain/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from gan_seg.dataset_patches import build_atom_mask_from_com
from gan_seg.eval_noise_SOTA import load_benchmark_model

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "cross_domain"


def file_md5(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def centroid_metrics(pred_np: np.ndarray, gt_np: np.ndarray, distance_threshold: float):
    from scipy.spatial.distance import cdist
    from skimage.measure import label, regionprops

    def cents(m):
        lbl = label(m > 0.5)
        return np.array([p.centroid for p in regionprops(lbl)])

    pred_c, gt_c = cents(pred_np), cents(gt_np)
    if len(pred_c) == 0 and len(gt_c) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_c) == 0 or len(gt_c) == 0:
        return 0.0, 0.0, 0.0
    dists = cdist(pred_c, gt_c)
    matched_gt = set()
    matched_pred = set()
    tp = 0
    for p_idx in range(len(pred_c)):
        g_idx = int(np.argmin(dists[p_idx]))
        if dists[p_idx, g_idx] <= distance_threshold and g_idx not in matched_gt:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            tp += 1
    fp = len(pred_c) - len(matched_pred)
    fn = len(gt_c) - len(matched_gt)
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-6)
    return pr, rc, f1


def iou_np(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    inter = np.logical_and(pred_np > 0.5, gt_np > 0.5).sum()
    union = np.logical_or(pred_np > 0.5, gt_np > 0.5).sum()
    return float(inter / union) if union > 0 else 1.0


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    p = patch[np.newaxis, ...].astype(np.float32)
    mu, sd = float(p.mean()), float(p.std()) + 1e-6
    return (p - mu) / sd


def pseudo_gt_mask_from_peaks(
    patch: np.ndarray,
    sigma_smooth: float,
    min_distance: int,
    threshold_rel: float,
    mask_sigma: float,
) -> tuple[np.ndarray | None, int]:
    """
    Peaks on **min–max normalized** patch (0..1), not raw intensities.

    skimage's ``threshold_rel`` uses ``max(image) * threshold_rel``. On raw graphene
    (roughly ±0.08 float32 STEM), that threshold is ~1e-3 and peak picking becomes
    unstable and misaligned with lattice scale. Normalizing per patch makes
    ``threshold_rel`` behave like a fraction of dynamic range (see skimage docs).
    """
    p = patch.astype(np.float64)
    pmin, pmax = float(np.min(p)), float(np.max(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or (pmax - pmin) < 1e-12:
        return None, 0
    p01 = (p - pmin) / (pmax - pmin)
    sm = gaussian_filter(p01, sigma_smooth)
    peaks = peak_local_max(
        sm,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=True,
    )
    n_peaks = int(len(peaks))
    if n_peaks == 0:
        return None, 0
    mask = build_atom_mask_from_com(
        patch.shape[0],
        patch.shape[1],
        peaks.astype(np.float64),
        0,
        0,
        sigma=float(mask_sigma),
    )
    return mask, n_peaks


def eval_smbfo_patches(
    npy_path: Path,
    model,
    device,
    n_samples: int,
    patch_size: int,
    seed: int,
    coord_key: str,
    distance_threshold: float,
):
    raw = np.load(npy_path, allow_pickle=True)[()]
    keys = list(raw.keys())
    rng = np.random.default_rng(seed)
    prs, rcs, f1s, ious = [], [], [], []
    used = 0
    attempts = 0
    max_attempts = n_samples * 20

    while used < n_samples and attempts < max_attempts:
        attempts += 1
        k = keys[int(rng.integers(0, len(keys)))]
        entry = raw[k]
        img = np.asarray(entry["main_image"], dtype=np.float32)
        com = np.asarray(entry[coord_key], dtype=np.float64)
        if com.ndim == 2 and com.shape[1] > 2:
            com = com[:, :2]
        h, w = img.shape
        ps = patch_size
        if h < ps or w < ps:
            continue
        top = int(rng.integers(0, h - ps + 1))
        left = int(rng.integers(0, w - ps + 1))
        patch = img[top : top + ps, left : left + ps].copy()
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        mask = build_atom_mask_from_com(ps, ps, com, top, left, sigma=3.0)
        if mask.sum() < 1.0:
            continue
        x = torch.from_numpy(normalize_patch(patch)).to(device)
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            pred = (logits > 0.0).float()[0, 0].cpu().numpy()
        pr, rc, f1 = centroid_metrics(pred, mask, distance_threshold)
        prs.append(pr)
        rcs.append(rc)
        f1s.append(f1)
        ious.append(iou_np(pred, mask))
        used += 1

    return {
        "n_used": used,
        "precision_mean": float(np.mean(prs)) if prs else float("nan"),
        "recall_mean": float(np.mean(rcs)) if rcs else float("nan"),
        "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
    }


def eval_graphene_patches(
    npy_path: Path,
    model,
    device,
    n_samples: int,
    patch_size: int,
    seed: int,
    distance_threshold: float,
    peak_threshold_rel: float,
    peak_min_distance: int,
):
    raw = np.load(npy_path, allow_pickle=True)[()]
    stack_keys = sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    rng = np.random.default_rng(seed + 1)
    prs, rcs, f1s, ious = [], [], [], []
    pred_fg_fracs, logit_maxes, n_peaks_list = [], [], []
    used = 0
    attempts = 0
    max_attempts = n_samples * 40
    samples_for_viz = []

    while used < n_samples and attempts < max_attempts:
        attempts += 1
        k = stack_keys[int(rng.integers(0, len(stack_keys)))]
        stack = raw[k]["image_data"]
        n_frames = stack.shape[0]
        fi = int(rng.integers(0, n_frames))
        frame = np.asarray(stack[fi], dtype=np.float32)
        h, w = frame.shape
        ps = patch_size
        if h < ps or w < ps:
            continue
        top = int(rng.integers(0, h - ps + 1))
        left = int(rng.integers(0, w - ps + 1))
        patch = frame[top : top + ps, left : left + ps].copy()
        mask, n_peaks = pseudo_gt_mask_from_peaks(
            patch,
            sigma_smooth=1.0,
            min_distance=peak_min_distance,
            threshold_rel=peak_threshold_rel,
            mask_sigma=2.5,
        )
        if mask is None or mask.sum() < 1.0:
            continue
        x = torch.from_numpy(normalize_patch(patch)).to(device)
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            pred = (logits > 0.0).float()[0, 0].cpu().numpy()
        pr, rc, f1 = centroid_metrics(pred, mask, distance_threshold)
        prs.append(pr)
        rcs.append(rc)
        f1s.append(f1)
        ious.append(iou_np(pred, mask))
        pred_fg_fracs.append(float(pred.mean()))
        logit_maxes.append(float(logits.max().item()))
        n_peaks_list.append(n_peaks)
        if len(samples_for_viz) < 6:
            samples_for_viz.append(
                {
                    "patch": patch.copy(),
                    "gt": mask.copy(),
                    "pred": pred.copy(),
                    "f1": f1,
                    "logit_max": float(logits.max().item()),
                    "n_peaks": n_peaks,
                }
            )
        used += 1

    return {
        "n_used": used,
        "precision_mean": float(np.mean(prs)) if prs else float("nan"),
        "recall_mean": float(np.mean(rcs)) if rcs else float("nan"),
        "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "mean_pred_fg_frac": float(np.mean(pred_fg_fracs)) if pred_fg_fracs else float("nan"),
        "mean_logit_max": float(np.mean(logit_maxes)) if logit_maxes else float("nan"),
        "frac_logit_max_positive": float(np.mean([m > 0 for m in logit_maxes]))
        if logit_maxes
        else float("nan"),
        "mean_n_peaks_pseudo_gt": float(np.mean(n_peaks_list)) if n_peaks_list else float("nan"),
        "viz": samples_for_viz,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smbfo-path", type=Path, default=ROOT / "external_stem_data" / "SmBFO_composition_series.npy")
    p.add_argument("--smbfo-ref", type=Path, default=ROOT / "data" / "SmBFO_composition_series.npy")
    p.add_argument("--graphene-path", type=Path, default=ROOT / "data" / "graphene_imgstacks_dict.npy")
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--coord-key",
        type=str,
        default="xy_COM",
        choices=["xy_COM", "xy_atms"],
        help="Sm-BFO mask labels: xy_COM matches default preprocess_dataset (not --use-atms).",
    )
    p.add_argument(
        "--graphene-peak-threshold-rel",
        type=float,
        default=0.42,
        help="Peak min intensity = max(smoothed_01) * this (after per-patch min–max).",
    )
    p.add_argument(
        "--graphene-peak-min-distance",
        type=int,
        default=8,
        help="Min separation between peaks (pixels), ~lattice spacing in 256 patches.",
    )
    p.add_argument(
        "--graphene-centroid-match-px",
        type=float,
        default=8.0,
        help="Max |pred−GT| distance (px) for centroid F1 on graphene.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="Hybrid-STEMSeg,UNet,SegFormer,DeepLabV3+",
        help="Comma-separated benchmark model names",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "device": str(device),
        "smbfo_path": str(args.smbfo_path),
        "smbfo_ref": str(args.smbfo_ref),
        "graphene_path": str(args.graphene_path),
        "smbfo_coord_key": args.coord_key,
        "graphene_pseudo_gt": "peak_local_max on minmax01 + Gaussian-smoothed patch",
        "graphene_peak_threshold_rel": args.graphene_peak_threshold_rel,
        "graphene_peak_min_distance": args.graphene_peak_min_distance,
        "graphene_centroid_match_px": args.graphene_centroid_match_px,
        "n_samples": args.n_samples,
        "patch_size": args.patch_size,
    }
    if args.smbfo_path.is_file() and args.smbfo_ref.is_file():
        manifest["smbfo_md5"] = file_md5(args.smbfo_path)
        manifest["smbfo_ref_md5"] = file_md5(args.smbfo_ref)
        manifest["smbfo_identical"] = manifest["smbfo_md5"] == manifest["smbfo_ref_md5"]

    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    name_to_slug = {
        "Hybrid-STEMSeg": "hybrid-nogan",
        "UNet": "unet",
        "SegFormer": "segformer",
        "DeepLabV3+": "deeplabv3plus",
        "Hybrid-NoTransformer": "hybrid-notransformer",
    }
    ckpt_dir = ROOT / "gan_seg" / "checkpoints_benchmark"

    rows = []
    viz_main = None

    for display_name in model_names:
        slug = name_to_slug.get(display_name)
        if not slug:
            print(f"[SKIP] Unknown model label: {display_name}")
            continue
        ckpt = ckpt_dir / slug / "gan_seg_best.pt"
        if not ckpt.is_file():
            print(f"[SKIP] Missing checkpoint: {ckpt}")
            continue
        model = load_benchmark_model(slug, str(ckpt), device)
        print(f"--- {display_name} ---")

        if args.smbfo_path.is_file():
            r_smbfo = eval_smbfo_patches(
                args.smbfo_path,
                model,
                device,
                n_samples=args.n_samples,
                patch_size=args.patch_size,
                seed=args.seed,
                coord_key=args.coord_key,
                distance_threshold=6.0,
            )
            print(f"  Sm-BFO ({args.smbfo_path.name}): n={r_smbfo['n_used']} F1={r_smbfo['f1_mean']:.4f} IoU={r_smbfo['iou_mean']:.4f}")
            rows.append(
                {
                    "model": display_name,
                    "dataset": "Sm-BFO (AtomAI copy)",
                    "n_patches": r_smbfo["n_used"],
                    "precision": r_smbfo["precision_mean"],
                    "recall": r_smbfo["recall_mean"],
                    "f1": r_smbfo["f1_mean"],
                    "iou": r_smbfo["iou_mean"],
                    "mean_pred_fg_frac": "",
                    "mean_logit_max": "",
                    "frac_logit_max_positive": "",
                    "mean_n_peaks_pseudo_gt": "",
                    "notes": "COM supervision; same file as data/SmBFO if MD5 match",
                }
            )

        if args.graphene_path.is_file():
            r_g = eval_graphene_patches(
                args.graphene_path,
                model,
                device,
                n_samples=args.n_samples,
                patch_size=args.patch_size,
                seed=args.seed,
                distance_threshold=args.graphene_centroid_match_px,
                peak_threshold_rel=args.graphene_peak_threshold_rel,
                peak_min_distance=args.graphene_peak_min_distance,
            )
            print(
                f"  Graphene (pseudo-GT): n={r_g['n_used']} F1={r_g['f1_mean']:.4f} IoU={r_g['iou_mean']:.4f} | "
                f"pred_fg={r_g['mean_pred_fg_frac']:.5f} logit_max_mean={r_g['mean_logit_max']:.3f} "
                f"P(max>0)={r_g['frac_logit_max_positive']:.2f} n_peaks_gt={r_g['mean_n_peaks_pseudo_gt']:.1f}"
            )
            rows.append(
                {
                    "model": display_name,
                    "dataset": "Graphene (peak_local_max pseudo-GT)",
                    "n_patches": r_g["n_used"],
                    "precision": r_g["precision_mean"],
                    "recall": r_g["recall_mean"],
                    "f1": r_g["f1_mean"],
                    "iou": r_g["iou_mean"],
                    "mean_pred_fg_frac": r_g["mean_pred_fg_frac"],
                    "mean_logit_max": r_g["mean_logit_max"],
                    "frac_logit_max_positive": r_g["frac_logit_max_positive"],
                    "mean_n_peaks_pseudo_gt": r_g["mean_n_peaks_pseudo_gt"],
                    "notes": (
                        f"minmax01 peaks; thr_rel={args.graphene_peak_threshold_rel}, "
                        f"min_dist={args.graphene_peak_min_distance}px; weak GT"
                    ),
                }
            )
            if display_name == "Hybrid-STEMSeg" and viz_main is None:
                viz_main = r_g["viz"]

    out_csv = REPORT_DIR / "cross_domain_metrics.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    if viz_main:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, s in zip(axes.ravel(), viz_main):
            ax.imshow(s["patch"], cmap="gray")
            ax.imshow(s["pred"], cmap="Reds", alpha=0.35)
            ax.imshow(s["gt"], cmap="Greens", alpha=0.2)
            ax.set_title(
                f"F1={s['f1']:.2f} maxL={s['logit_max']:.2f} peaks={s['n_peaks']}\nred=pred green=weak-GT"
            )
            ax.axis("off")
        fig.suptitle(
            "Graphene OOD: Hybrid-STEMSeg (min–max pseudo-GT + pred overlay)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "graphene_ood_hybrid_nogan_overlay.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
