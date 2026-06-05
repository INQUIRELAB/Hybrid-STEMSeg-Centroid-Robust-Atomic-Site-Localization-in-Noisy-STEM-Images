#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.eval_noise_SOTA import load_benchmark_model, load_hybrid_model


def apply_corruption(img_t, corruption, severity, rng):
    if corruption == "clean":
        return img_t
    if corruption == "gaussian":
        return img_t + torch.randn_like(img_t, generator=rng) * severity
    if corruption == "blur":
        arr = img_t[0, 0].detach().cpu().numpy()
        out = gaussian_filter(arr, sigma=severity)
        return torch.from_numpy(out).to(img_t.device, dtype=img_t.dtype).unsqueeze(0).unsqueeze(0)
    if corruption == "contrast":
        # Contrast around per-image mean in normalized space.
        mu = img_t.mean(dim=(-1, -2), keepdim=True)
        return (img_t - mu) * severity + mu
    if corruption == "poisson":
        arr = img_t[0, 0].detach().cpu().numpy()
        amin, amax = arr.min(), arr.max()
        if amax - amin < 1e-8:
            return img_t
        norm = (arr - amin) / (amax - amin)
        peak = float(severity)
        noisy = np.random.poisson(np.clip(norm, 0.0, 1.0) * peak) / peak
        out = noisy * (amax - amin) + amin
        return torch.from_numpy(out).to(img_t.device, dtype=img_t.dtype).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported corruption: {corruption}")


def evaluate_model(ds, model, device, corruption, severity, n_samples, distance_threshold, seed):
    tp = 0
    fp = 0
    fn = 0
    fg_fracs = []
    n = min(n_samples, len(ds))
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)
    np.random.seed(seed)

    for i in range(n):
        img, mask = ds[i]
        gt_mask = mask[0].numpy()
        gt_lbl = label(gt_mask > 0.5)
        gt_coords = np.array([p.centroid for p in regionprops(gt_lbl)])

        with torch.no_grad():
            img_t = img.unsqueeze(0).to(device)
            cor = apply_corruption(img_t, corruption, severity, torch_rng)
            logits = model(cor)
            pred = (logits > 0.0).float()[0, 0].cpu().numpy()
        fg_fracs.append(float(pred.mean()))

        pred_lbl = label(pred > 0.5)
        pred_coords = np.array([p.centroid for p in regionprops(pred_lbl)])

        if len(pred_coords) == 0 and len(gt_coords) == 0:
            continue
        if len(pred_coords) == 0:
            fn += len(gt_coords)
            continue
        if len(gt_coords) == 0:
            fp += len(pred_coords)
            continue

        dists = cdist(pred_coords, gt_coords)
        matched_gt = set()
        matched_pred = set()
        for p_idx in range(len(pred_coords)):
            closest_gt = np.argmin(dists[p_idx])
            if dists[p_idx, closest_gt] <= distance_threshold and closest_gt not in matched_gt:
                matched_gt.add(closest_gt)
                matched_pred.add(p_idx)
                tp += 1
        fp += len(pred_coords) - len(matched_pred)
        fn += len(gt_coords) - len(matched_gt)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    fg_mean = float(np.mean(fg_fracs))
    return precision, recall, f1, fg_mean


def model_specs():
    return [
        ("UNet", "bm", "unet", "gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt"),
        ("DeepLabV3+", "bm", "deeplabv3plus", "gan_seg/checkpoints_benchmark/deeplabv3plus/gan_seg_best.pt"),
        ("SegFormer", "bm", "segformer", "gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt"),
        ("Hybrid-STEMSeg", "bm", "hybrid-nogan", "gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt"),
        ("Hybrid-NoTransformer", "bm", "hybrid-notransformer", "gan_seg/checkpoints_benchmark/hybrid-notransformer/gan_seg_best.pt"),
        ("Original GAN (Scratch)", "hy", None, "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"),
        ("GAN (ResNet-Initialized)", "hy", None, "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"),
    ]


def main():
    p = argparse.ArgumentParser(description="Corruption robustness suite for all models.")
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--distance-threshold", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="reports/corruption_suite")
    p.add_argument("--collapse-fg-threshold", type=float, default=0.001, help="Collapse if predicted fg fraction <= this")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset(args.processed, split="val")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = {}
    for name, kind, mname, ckpt in model_specs():
        if not Path(ckpt).is_file():
            print(f"[SKIP] {name}: missing checkpoint")
            continue
        try:
            models[name] = load_benchmark_model(mname, ckpt, device) if kind == "bm" else load_hybrid_model(ckpt, device)
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")

    suites = [
        ("clean", [0.0]),
        ("gaussian", [0.3, 0.6, 1.0, 1.5, 2.0]),
        ("blur", [0.5, 1.0, 1.5, 2.0, 3.0]),
        ("contrast", [0.7, 0.5, 0.3]),
        ("poisson", [30.0, 10.0, 5.0, 2.0]),
    ]

    rows = []
    for model_name, model in models.items():
        for corruption, severities in suites:
            for sev in severities:
                pr, rc, f1, fg = evaluate_model(
                    ds, model, device, corruption, sev, args.n_samples, args.distance_threshold, args.seed
                )
                rows.append(
                    {
                        "model": model_name,
                        "corruption": corruption,
                        "severity": sev,
                        "precision": pr,
                        "recall": rc,
                        "f1": f1,
                        "fg_fraction_mean": fg,
                        "collapsed": int(fg <= args.collapse_fg_threshold),
                    }
                )
                print(f"{model_name:<26} {corruption:<9} sev={sev:<4} f1={f1:.4f} fg={fg:.6f}")

    csv_path = outdir / "corruption_suite_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "corruption", "severity", "precision", "recall", "f1", "fg_fraction_mean", "collapsed"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
