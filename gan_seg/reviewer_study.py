#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.eval_noise_SOTA import load_benchmark_model, load_hybrid_model


def apply_corruption(img_t, corruption, severity, torch_rng, np_rng):
    if corruption == "clean":
        return img_t
    if corruption == "gaussian":
        return img_t + torch.randn_like(img_t, generator=torch_rng) * float(severity)
    if corruption == "blur":
        arr = img_t[0, 0].detach().cpu().numpy()
        out = gaussian_filter(arr, sigma=float(severity))
        return torch.from_numpy(out).to(img_t.device, dtype=img_t.dtype).unsqueeze(0).unsqueeze(0)
    if corruption == "contrast":
        mu = img_t.mean(dim=(-1, -2), keepdim=True)
        return (img_t - mu) * float(severity) + mu
    if corruption == "poisson":
        arr = img_t[0, 0].detach().cpu().numpy()
        amin, amax = arr.min(), arr.max()
        if amax - amin < 1e-8:
            return img_t
        norm = (arr - amin) / (amax - amin)
        peak = float(severity)
        noisy = np_rng.poisson(np.clip(norm, 0.0, 1.0) * peak) / peak
        out = noisy * (amax - amin) + amin
        return torch.from_numpy(out).to(img_t.device, dtype=img_t.dtype).unsqueeze(0).unsqueeze(0)
    if corruption == "mixed":
        # Fixed compound corruption chain for OOD stress-testing.
        z = apply_corruption(img_t, "gaussian", 0.8 * float(severity), torch_rng, np_rng)
        z = apply_corruption(z, "blur", 1.0 * float(severity), torch_rng, np_rng)
        z = apply_corruption(z, "contrast", max(0.2, 1.0 - 0.35 * float(severity)), torch_rng, np_rng)
        z = apply_corruption(z, "poisson", max(2.0, 10.0 / max(float(severity), 0.2)), torch_rng, np_rng)
        return z
    raise ValueError(f"Unsupported corruption: {corruption}")


def centroids_from_mask(mask_np):
    lbl = label(mask_np > 0.5)
    return np.array([p.centroid for p in regionprops(lbl)])


def centroid_f1(pred_np, gt_np, distance_threshold=6.0):
    pred_coords = centroids_from_mask(pred_np)
    gt_coords = centroids_from_mask(gt_np)
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_coords) == 0:
        return 0.0, 0.0, 0.0
    if len(gt_coords) == 0:
        return 0.0, 0.0, 0.0
    dists = cdist(pred_coords, gt_coords)
    matched_gt = set()
    matched_pred = set()
    tp = 0
    for p_idx in range(len(pred_coords)):
        g_idx = int(np.argmin(dists[p_idx]))
        if dists[p_idx, g_idx] <= distance_threshold and g_idx not in matched_gt:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            tp += 1
    fp = len(pred_coords) - len(matched_pred)
    fn = len(gt_coords) - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return precision, recall, f1


def iou_score(pred_np, gt_np):
    inter = np.logical_and(pred_np > 0.5, gt_np > 0.5).sum()
    union = np.logical_or(pred_np > 0.5, gt_np > 0.5).sum()
    return float(inter / union) if union > 0 else 1.0


def otsu_baseline_predict(img_t):
    arr = img_t[0, 0].detach().cpu().numpy()
    th = threshold_otsu(arr)
    pred = (arr > th).astype(np.float32)
    pred = binary_opening(pred > 0.5, disk(1)).astype(np.float32)
    return pred


def evaluate_model_on_dataset(
    ds,
    model,
    device,
    corruption,
    severity,
    seed,
    n_samples,
    distance_threshold,
    classical=False,
):
    np_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)
    n = min(n_samples, len(ds))
    pr_list, rc_list, f1_list, iou_list, fg_list = [], [], [], [], []

    for i in range(n):
        img, mask = ds[i]
        gt_np = mask[0].numpy()
        img_t = img.unsqueeze(0).to(device)
        cor = apply_corruption(img_t, corruption, severity, torch_rng, np_rng)
        if classical:
            pred_np = otsu_baseline_predict(cor)
        else:
            with torch.no_grad():
                pred_np = (model(cor) > 0.0).float()[0, 0].cpu().numpy()

        pr, rc, f1 = centroid_f1(pred_np, gt_np, distance_threshold=distance_threshold)
        iou = iou_score(pred_np, gt_np)
        pr_list.append(pr)
        rc_list.append(rc)
        f1_list.append(f1)
        iou_list.append(iou)
        fg_list.append(float(pred_np.mean()))

    return {
        "precision_mean": float(np.mean(pr_list)),
        "recall_mean": float(np.mean(rc_list)),
        "f1_mean": float(np.mean(f1_list)),
        "iou_mean": float(np.mean(iou_list)),
        "fg_mean": float(np.mean(fg_list)),
        "f1_samples": f1_list,
        "iou_samples": iou_list,
    }


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(vals[idx].mean())
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description="Comprehensive reviewer study runner.")
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seeds", type=str, default="101,202,303")
    p.add_argument("--distance-threshold", type=float, default=6.0)
    p.add_argument("--collapse-fg-threshold", type=float, default=0.001)
    p.add_argument("--outdir", type=str, default="reports/reviewer_suite")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset(args.processed, split="val")

    # Baselines + broader comparator (classical Otsu).
    specs = [
        ("UNet", "bm", "unet", "gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt"),
        ("DeepLabV3+", "bm", "deeplabv3plus", "gan_seg/checkpoints_benchmark/deeplabv3plus/gan_seg_best.pt"),
        ("SegFormer", "bm", "segformer", "gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt"),
        ("Hybrid-STEMSeg", "bm", "hybrid-nogan", "gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt"),
        ("Hybrid-NoTransformer", "bm", "hybrid-notransformer", "gan_seg/checkpoints_benchmark/hybrid-notransformer/gan_seg_best.pt"),
        ("Original GAN (Scratch)", "hy", None, "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"),
        ("GAN (ResNet-Initialized)", "hy", None, "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"),
        ("Classical Otsu+Opening", "classical", None, None),
    ]
    models = {}
    for name, kind, mname, ckpt in specs:
        if kind == "classical":
            models[name] = {"kind": "classical", "model": None}
            continue
        if not Path(ckpt).is_file():
            print(f"[SKIP] {name}: checkpoint missing")
            continue
        try:
            m = load_benchmark_model(mname, ckpt, device) if kind == "bm" else load_hybrid_model(ckpt, device)
            models[name] = {"kind": kind, "model": m}
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}")

    suites = [
        ("clean", [0.0]),
        ("gaussian", [0.3, 0.6, 1.0, 1.5, 2.0]),
        ("blur", [0.5, 1.0, 1.5, 2.0, 3.0]),
        ("contrast", [0.7, 0.5, 0.3]),
        ("poisson", [30.0, 10.0, 5.0, 2.0]),
        ("mixed", [0.5, 1.0, 1.5]),
    ]

    rows = []
    sample_cache = {}  # (model, corruption, severity, seed) -> iou samples
    for model_name, obj in models.items():
        for corruption, severities in suites:
            for sev in severities:
                seed_metrics = []
                for sd in seeds:
                    met = evaluate_model_on_dataset(
                        ds=ds,
                        model=obj["model"],
                        device=device,
                        corruption=corruption,
                        severity=sev,
                        seed=sd,
                        n_samples=args.n_samples,
                        distance_threshold=args.distance_threshold,
                        classical=(obj["kind"] == "classical"),
                    )
                    seed_metrics.append(met)
                    sample_cache[(model_name, corruption, float(sev), sd)] = met["iou_samples"]

                f1_vals = [m["f1_mean"] for m in seed_metrics]
                iou_vals = [m["iou_mean"] for m in seed_metrics]
                fg_vals = [m["fg_mean"] for m in seed_metrics]
                row = {
                    "model": model_name,
                    "corruption": corruption,
                    "severity": float(sev),
                    "f1_mean": float(np.mean(f1_vals)),
                    "f1_std": float(np.std(f1_vals)),
                    "iou_mean": float(np.mean(iou_vals)),
                    "iou_std": float(np.std(iou_vals)),
                    "fg_mean": float(np.mean(fg_vals)),
                    "fg_std": float(np.std(fg_vals)),
                    "collapsed": int(float(np.mean(fg_vals)) <= args.collapse_fg_threshold),
                    "f1_failure": int(float(np.mean(f1_vals)) < 0.1),
                }
                rows.append(row)
                print(f"{model_name:<26} {corruption:<9} sev={sev:<4} f1={row['f1_mean']:.4f}±{row['f1_std']:.4f}")

    csv_path = outdir / "robustness_multiseed.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "corruption", "severity", "f1_mean", "f1_std", "iou_mean", "iou_std", "fg_mean", "fg_std", "collapsed", "f1_failure"],
        )
        w.writeheader()
        w.writerows(rows)

    # AURC over Gaussian severities.
    aurc_rows = []
    for model_name in models.keys():
        rr = [r for r in rows if r["model"] == model_name and r["corruption"] == "gaussian"]
        rr = sorted(rr, key=lambda x: x["severity"])
        if len(rr) < 2:
            continue
        xs = np.array([r["severity"] for r in rr], dtype=float)
        ys = np.array([r["f1_mean"] for r in rr], dtype=float)
        aurc = np.trapz(ys, xs) / (xs.max() - xs.min())
        aurc_rows.append({"model": model_name, "aurc_gaussian_f1": float(aurc)})
    aurc_rows = sorted(aurc_rows, key=lambda x: x["aurc_gaussian_f1"], reverse=True)
    with open(outdir / "aurc_gaussian.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "aurc_gaussian_f1"])
        w.writeheader()
        w.writerows(aurc_rows)

    # Significance tests: main model vs each baseline using paired IoU over gaussian severities and samples.
    main_model = "Hybrid-STEMSeg"
    sig_rows = []
    compare_models = [m for m in models.keys() if m != main_model]
    for other in compare_models:
        for sev in [0.3, 0.6, 1.0, 1.5, 2.0]:
            v_main = []
            v_other = []
            for sd in seeds:
                a = sample_cache.get((main_model, "gaussian", float(sev), sd))
                b = sample_cache.get((other, "gaussian", float(sev), sd))
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                v_main.extend(a[:n])
                v_other.extend(b[:n])
            if len(v_main) < 5:
                continue
            try:
                stat, pval = wilcoxon(v_main, v_other, zero_method="wilcox")
            except Exception:
                pval = np.nan
            delta = float(np.mean(v_main) - np.mean(v_other))
            ci_lo, ci_hi = bootstrap_ci(np.asarray(v_main) - np.asarray(v_other), seed=42)
            sig_rows.append(
                {
                    "main_model": main_model,
                    "other_model": other,
                    "corruption": "gaussian",
                    "severity": sev,
                    "delta_iou_mean": delta,
                    "delta_iou_ci95_lo": ci_lo,
                    "delta_iou_ci95_hi": ci_hi,
                    "wilcoxon_p": float(pval) if pval == pval else "",
                }
            )
    with open(outdir / "significance_wilcoxon.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["main_model", "other_model", "corruption", "severity", "delta_iou_mean", "delta_iou_ci95_lo", "delta_iou_ci95_hi", "wilcoxon_p"],
        )
        w.writeheader()
        w.writerows(sig_rows)

    # Ablation table from available checkpoints.
    ablation_specs = [
        ("Main: Hybrid-STEMSeg", "bm", "hybrid-nogan", "gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt"),
        ("No-Transformer", "bm", "hybrid-notransformer", "gan_seg/checkpoints_benchmark/hybrid-notransformer/gan_seg_best.pt"),
        ("GAN Scratch", "hy", None, "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"),
        ("GAN ResNet-Initialized", "hy", None, "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"),
        ("No-Adv Ablation", "hy", None, "gan_seg/checkpoints_ablation_noadv/gan_seg_best.pt"),
        ("No-Adv v2 Ablation", "hy", None, "gan_seg/checkpoints_ablation_noadv_v2/gan_seg_best.pt"),
        ("Spectral Norm Variant", "hy", None, "gan_seg/checkpoints_spectral_norm/gan_seg_best.pt"),
        ("Sigma1.5 Variant", "hy", None, "gan_seg/checkpoints_sigma15/gan_seg_best.pt"),
    ]
    abl_rows = []
    main_iou = None
    for name, kind, mname, ckpt in ablation_specs:
        if not Path(ckpt).is_file():
            continue
        model = load_benchmark_model(mname, ckpt, device) if kind == "bm" else load_hybrid_model(ckpt, device)
        met = evaluate_model_on_dataset(
            ds=ds,
            model=model,
            device=device,
            corruption="clean",
            severity=0.0,
            seed=seeds[0],
            n_samples=args.n_samples,
            distance_threshold=args.distance_threshold,
            classical=False,
        )
        iou_vals = np.asarray(met["iou_samples"], dtype=float)
        lo, hi = bootstrap_ci(iou_vals, seed=77)
        if name.startswith("Main:"):
            main_iou = float(np.mean(iou_vals))
        abl_rows.append(
            {
                "variant": name,
                "clean_iou_mean": float(np.mean(iou_vals)),
                "clean_iou_ci95_lo": lo,
                "clean_iou_ci95_hi": hi,
            }
        )
    if main_iou is not None:
        for r in abl_rows:
            r["delta_vs_main_iou"] = r["clean_iou_mean"] - main_iou
    with open(outdir / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["variant", "clean_iou_mean", "clean_iou_ci95_lo", "clean_iou_ci95_hi", "delta_vs_main_iou"],
        )
        w.writeheader()
        w.writerows(abl_rows)

    # Generalization evidence: evaluate main model on alternate processed splits.
    main_model_obj = models.get(main_model)
    gen_rows = []
    if main_model_obj is not None:
        for root in ["data/processed/sm_bfo_com", "data/processed/sm_bfo_gan", "data/processed/sm_bfo_sigma15"]:
            if not (Path(root) / "manifest.json").is_file():
                continue
            ds_o = ShardedPatchDataset(root, split="val")
            met = evaluate_model_on_dataset(
                ds=ds_o,
                model=main_model_obj["model"],
                device=device,
                corruption="clean",
                severity=0.0,
                seed=seeds[0],
                n_samples=min(args.n_samples, len(ds_o)),
                distance_threshold=args.distance_threshold,
                classical=False,
            )
            gen_rows.append({"dataset_root": root, "f1_mean": met["f1_mean"], "iou_mean": met["iou_mean"]})
    with open(outdir / "generalization_splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_root", "f1_mean", "iou_mean"])
        w.writeheader()
        w.writerows(gen_rows)

    # Corruption severity curve figure (mean +/- std for top models).
    plt.figure(figsize=(9, 6))
    for model_name in ["Hybrid-STEMSeg", "SegFormer", "UNet", "DeepLabV3+", "Original GAN (Scratch)", "GAN (ResNet-Initialized)", "Classical Otsu+Opening"]:
        rr = [r for r in rows if r["model"] == model_name and r["corruption"] == "gaussian"]
        if not rr:
            continue
        rr = sorted(rr, key=lambda x: x["severity"])
        xs = np.array([r["severity"] for r in rr], dtype=float)
        ys = np.array([r["f1_mean"] for r in rr], dtype=float)
        es = np.array([r["f1_std"] for r in rr], dtype=float)
        plt.plot(xs, ys, marker="o", label=model_name)
        plt.fill_between(xs, ys - es, ys + es, alpha=0.15)
    plt.xlabel("Gaussian severity")
    plt.ylabel("Detection F1 (centroid match, mean ± std)")
    plt.title("Gaussian Robustness Curves")
    plt.grid(alpha=0.25)
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "gaussian_severity_curves_mean_std.png", dpi=220)
    plt.close()

    # Qualitative overlay + error maps for systematic failure cases.
    case_dir = outdir / "failure_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    idx = 7
    img, mask = ds[idx]
    gt = mask[0].numpy()
    img_t = img.unsqueeze(0).to(device)
    pairs = [
        ("Hybrid-STEMSeg", 2.0),
        ("DeepLabV3+", 0.6),
        ("Original GAN (Scratch)", 0.6),
        ("GAN (ResNet-Initialized)", 0.6),
    ]
    rng_t = torch.Generator(device=device)
    rng_t.manual_seed(seeds[0] + idx)
    rng_n = np.random.default_rng(seeds[0] + idx)
    for name, sev in pairs:
        if name not in models:
            continue
        cor = apply_corruption(img_t, "gaussian", sev, rng_t, rng_n)
        if models[name]["kind"] == "classical":
            pred = otsu_baseline_predict(cor)
        else:
            with torch.no_grad():
                pred = (models[name]["model"](cor) > 0.0).float()[0, 0].cpu().numpy()
        noisy_np = cor[0, 0].detach().cpu().numpy()
        fp = np.logical_and(pred > 0.5, gt <= 0.5)
        fn = np.logical_and(pred <= 0.5, gt > 0.5)
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].imshow(noisy_np, cmap="gray")
        ax[0].set_title(f"Input (sev={sev})")
        ax[1].imshow(gt, cmap="gray")
        ax[1].set_title("GT")
        ax[2].imshow(pred, cmap="gray")
        ax[2].set_title(f"{name} Pred")
        err = np.zeros((*gt.shape, 3), dtype=float)
        err[..., 0] = fp.astype(float)  # red FP
        err[..., 1] = fn.astype(float)  # green FN
        ax[3].imshow(err)
        ax[3].set_title("Error Map (R=FP,G=FN)")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        outp = case_dir / f"failure_case_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.png"
        plt.savefig(outp, dpi=220)
        plt.close()

    summary = {
        "outputs": {
            "robustness_multiseed_csv": str(csv_path),
            "aurc_csv": str(outdir / "aurc_gaussian.csv"),
            "significance_csv": str(outdir / "significance_wilcoxon.csv"),
            "ablation_csv": str(outdir / "ablation_table.csv"),
            "generalization_csv": str(outdir / "generalization_splits.csv"),
            "gaussian_curve_png": str(outdir / "gaussian_severity_curves_mean_std.png"),
            "failure_case_dir": str(case_dir),
        },
        "config": {
            "seeds": seeds,
            "n_samples": args.n_samples,
            "collapse_fg_threshold": args.collapse_fg_threshold,
        },
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] Reviewer suite complete. Outputs in {outdir}")


if __name__ == "__main__":
    main()
