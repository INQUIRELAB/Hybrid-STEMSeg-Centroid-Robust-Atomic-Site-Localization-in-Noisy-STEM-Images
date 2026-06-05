#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from gan_seg.dataset_preprocessed import ShardedPatchDataset


def evaluate_centroids_noisy(
    model,
    ds,
    device,
    distance_threshold=6.0,
    noise_std=0.0,
    n_samples=100,
    seed=123,
):
    """
    Mean centroid precision / recall / F1 over patches (same per-patch definition as
    reviewer robustness eval). At noise_std=0 this matches **clean** centroid-F1 on val.

    (Previously this pooled TP/FP/FN across patches before computing F1, which did not
    match Table/report ``mean centroid F1`` at σ=0.)
    """
    from gan_seg.eval_cross_domain import centroid_metrics

    model.eval()
    pr_list: list[float] = []
    rc_list: list[float] = []
    f1_list: list[float] = []
    n = min(n_samples, len(ds))
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for i in range(n):
        img, mask = ds[i]
        gt_mask = mask[0].numpy()

        with torch.no_grad():
            img_t = img.unsqueeze(0).to(device)
            if noise_std > 0:
                noisy_img_t = img_t + torch.randn_like(img_t, generator=rng) * noise_std
            else:
                noisy_img_t = img_t
            from gan_seg.inference_centroid import seg_logits_from_model

            logits = seg_logits_from_model(model, noisy_img_t)
            pred_mask = (logits > 0.0).float()[0, 0].cpu().numpy()

        pr, rc, f1 = centroid_metrics(pred_mask, gt_mask, distance_threshold)
        pr_list.append(pr)
        rc_list.append(rc)
        f1_list.append(f1)

    return (
        float(np.mean(pr_list)),
        float(np.mean(rc_list)),
        float(np.mean(f1_list)),
    )


def load_benchmark_model(model_name, ckpt_path, device):
    from gan_seg.train_benchmark import get_model

    m = get_model(model_name, device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)["G"])
    m.eval()
    return m


def load_hybrid_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, dict) and args.get("centroid_aware", False):
        from gan_seg.inference_centroid import load_centroid_aware_hybrid

        return load_centroid_aware_hybrid(ckpt_path, device)
    if isinstance(args, dict) and args.get("pretrained", False):
        from gan_seg.model import PretrainedHybridGAN

        m = PretrainedHybridGAN(use_transformer=True).to(device)
    else:
        from gan_seg.model import HybridUNetTransformerBinary

        d_model = args.get("d_model", 256) if isinstance(args, dict) else args.d_model
        use_transformer = args.get("use_transformer", True) if isinstance(args, dict) else True
        m = HybridUNetTransformerBinary(d_model=d_model, use_transformer=use_transformer).to(device)
    m.load_state_dict(ckpt["G"])
    m.eval()
    return m


def parse_noise_levels(noise_levels_raw):
    values = [float(x.strip()) for x in noise_levels_raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one noise level is required.")
    return sorted(values)


def failure_noise_level(noise_levels, f1_scores, abs_threshold, relative_ratio):
    base_f1 = f1_scores[0]
    rel_threshold = base_f1 * relative_ratio
    effective_threshold = min(abs_threshold, rel_threshold)
    for n, f1 in zip(noise_levels, f1_scores):
        if f1 <= effective_threshold:
            return n, effective_threshold
    return None, effective_threshold


def main():
    p = argparse.ArgumentParser(description="Noise ablation for segmentation models.")
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--distance-threshold", type=float, default=6.0)
    p.add_argument("--noise-levels", type=str, default="0.0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    p.add_argument("--failure-f1-abs", type=float, default=0.2)
    p.add_argument("--failure-f1-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--include-unetr", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset(args.processed, split="val")
    noise_levels = parse_noise_levels(args.noise_levels)

    model_specs = [
        ("UNet", "bm", "unet", "gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt"),
        ("DeepLabV3+", "bm", "deeplabv3plus", "gan_seg/checkpoints_benchmark/deeplabv3plus/gan_seg_best.pt"),
        ("SegFormer", "bm", "segformer", "gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt"),
        ("Hybrid-STEMSeg", "bm", "hybrid-nogan", "gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt"),
        ("Hybrid-NoTransformer", "bm", "hybrid-notransformer", "gan_seg/checkpoints_benchmark/hybrid-notransformer/gan_seg_best.pt"),
        ("Original GAN (Scratch)", "hy", None, "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"),
        ("GAN (ResNet-Initialized)", "hy", None, "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"),
    ]
    if args.include_unetr:
        model_specs.insert(3, ("UNETR", "bm", "unetr", "gan_seg/checkpoints_benchmark/unetr/gan_seg_best.pt"))

    models = {}
    print("\nLoading models...")
    for name, kind, model_name, ckpt_path in model_specs:
        if not Path(ckpt_path).is_file():
            print(f"[SKIP] {name} checkpoint missing: {ckpt_path}")
            continue
        try:
            if kind == "bm":
                models[name] = load_benchmark_model(model_name, ckpt_path, device)
            else:
                models[name] = load_hybrid_model(ckpt_path, device)
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[SKIP] {name} failed to load: {exc}")

    if not models:
        raise SystemExit("No models available for ablation.")

    print("\n--- Noise Ablation (Centroid Detection Robustness) ---")
    print(f"Samples: {min(args.n_samples, len(ds))}, Distance threshold: {args.distance_threshold}px")
    print(f"Noise sweep: {noise_levels}")
    print("")

    results = {}
    for name, model in models.items():
        model_scores = []
        print(f"Model: {name}")
        print(f"{'noise_std':>9} | {'precision':>9} | {'recall':>9} | {'f1':>9}")
        print("-" * 45)
        for noise_std in noise_levels:
            pr, rc, f1 = evaluate_centroids_noisy(
                model,
                ds,
                device,
                distance_threshold=args.distance_threshold,
                noise_std=noise_std,
                n_samples=args.n_samples,
                seed=args.seed,
            )
            model_scores.append((noise_std, pr, rc, f1))
            print(f"{noise_std:9.3f} | {pr:9.4f} | {rc:9.4f} | {f1:9.4f}")
        results[name] = model_scores
        fail_noise, threshold = failure_noise_level(
            noise_levels=[x[0] for x in model_scores],
            f1_scores=[x[3] for x in model_scores],
            abs_threshold=args.failure_f1_abs,
            relative_ratio=args.failure_f1_ratio,
        )
        if fail_noise is None:
            print(f"Failure point: NOT REACHED (threshold={threshold:.4f})")
        else:
            print(f"Failure point: noise_std={fail_noise:.3f} (threshold={threshold:.4f})")
        print("")

    print("\n=== Failure Noise Summary ===")
    print(f"{'Model':<28} | {'Base F1@0':>9} | {'Fail noise std':>14}")
    print("-" * 58)
    for name, rows in results.items():
        base_f1 = rows[0][3]
        fail_noise, _ = failure_noise_level(
            noise_levels=[x[0] for x in rows],
            f1_scores=[x[3] for x in rows],
            abs_threshold=args.failure_f1_abs,
            relative_ratio=args.failure_f1_ratio,
        )
        fail_txt = f"{fail_noise:.3f}" if fail_noise is not None else "not reached"
        print(f"{name:<28} | {base_f1:9.4f} | {fail_txt:>14}")


if __name__ == "__main__":
    main()
