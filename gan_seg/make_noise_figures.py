#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.eval_noise_SOTA import (
    evaluate_centroids_noisy,
    load_benchmark_model,
    load_hybrid_model,
    parse_noise_levels,
)


def default_model_specs(include_unetr=False):
    specs = [
        ("UNet", "bm", "unet", "gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt"),
        ("DeepLabV3+", "bm", "deeplabv3plus", "gan_seg/checkpoints_benchmark/deeplabv3plus/gan_seg_best.pt"),
        ("SegFormer", "bm", "segformer", "gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt"),
        ("Hybrid-STEMSeg", "bm", "hybrid-nogan", "gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt"),
        ("Hybrid-NoTransformer", "bm", "hybrid-notransformer", "gan_seg/checkpoints_benchmark/hybrid-notransformer/gan_seg_best.pt"),
        ("Original GAN (Scratch)", "hy", None, "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"),
        ("GAN (ResNet-Initialized)", "hy", None, "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"),
    ]
    if include_unetr:
        specs.insert(3, ("UNETR", "bm", "unetr", "gan_seg/checkpoints_benchmark/unetr/gan_seg_best.pt"))
    return specs


def load_models(device, include_unetr=False):
    models = {}
    for name, kind, model_name, ckpt_path in default_model_specs(include_unetr=include_unetr):
        if not Path(ckpt_path).is_file():
            print(f"[SKIP] {name}: missing checkpoint at {ckpt_path}")
            continue
        try:
            if kind == "bm":
                m = load_benchmark_model(model_name, ckpt_path, device)
            else:
                m = load_hybrid_model(ckpt_path, device)
            models[name] = m
            print(f"[OK] Loaded {name}")
        except Exception as exc:
            print(f"[SKIP] {name}: failed to load ({exc})")
    return models


def run_ablation(models, ds, device, noise_levels, n_samples, distance_threshold, seed):
    rows = []
    for model_name, model in models.items():
        for noise_std in noise_levels:
            precision, recall, f1 = evaluate_centroids_noisy(
                model=model,
                ds=ds,
                device=device,
                distance_threshold=distance_threshold,
                noise_std=noise_std,
                n_samples=n_samples,
                seed=seed,
            )
            rows.append(
                {
                    "model": model_name,
                    "noise_std": noise_std,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
            print(f"{model_name:<28} noise={noise_std:>4.2f} f1={f1:.4f}")
    return rows


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "noise_std", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def plot_f1_curves(rows, out_path, dpi: int = 220):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    plt.figure(figsize=(10, 6))
    for model_name, values in by_model.items():
        values = sorted(values, key=lambda x: x["noise_std"])
        xs = [v["noise_std"] for v in values]
        ys = [v["f1"] for v in values]
        plt.plot(xs, ys, marker="o", linewidth=2, label=model_name)
    plt.xlabel("Gaussian noise std (on z-scored patches)")
    plt.ylabel("Detection F1 (centroid match, val mean)")
    plt.title("Noise ablation: detection F1 vs Gaussian σ")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def make_qualitative_figure(model, ds, device, noise_levels, sample_idx, seed, out_path, title):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    img, mask = ds[sample_idx % len(ds)]
    img = img.to(device)
    mask_np = mask[0].cpu().numpy()

    rng = torch.Generator(device=device)
    rng.manual_seed(seed + sample_idx)

    rows = len(noise_levels)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 2.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, noise_std in enumerate(noise_levels):
        img_t = img.unsqueeze(0)
        if noise_std > 0:
            noisy = img_t + torch.randn_like(img_t, generator=rng) * noise_std
        else:
            noisy = img_t
        with torch.no_grad():
            logits = model(noisy)
            pred = (logits > 0.0).float()[0, 0].cpu().numpy()
        noisy_np = noisy[0, 0].detach().cpu().numpy()

        ax0, ax1, ax2 = axes[r]
        ax0.imshow(noisy_np, cmap="gray")
        ax0.set_title(f"Noisy Input (std={noise_std:.2f})")
        ax0.axis("off")
        ax1.imshow(mask_np, cmap="gray")
        ax1.set_title("Ground Truth")
        ax1.axis("off")
        ax2.imshow(pred, cmap="gray")
        ax2.set_title("Prediction")
        ax2.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_failure_noise_comparison(
    models_ordered: list[tuple[str, torch.nn.Module]],
    ds,
    device: torch.device,
    noise_std: float,
    sample_idx: int,
    seed: int,
    out_path_combined: Path,
    out_paths_by_name: dict[str, Path] | None = None,
    dpi: int = 220,
) -> None:
    """
    One Gaussian severity on normalized patches: **identical noisy input** for every model.

    Saves a compact combined figure (n_models × 3: noisy | GT | prediction). Optionally
    saves separate 1×3 strips per model (same σ, same patch) for narrow subfigures.
    """
    if not models_ordered:
        return

    for _, m in models_ordered:
        m.eval()

    img, mask = ds[sample_idx % len(ds)]
    img = img.to(device)
    mask_np = mask[0].cpu().numpy()

    rng = torch.Generator(device=device)
    rng.manual_seed(seed + sample_idx)
    img_b = img.unsqueeze(0)
    if noise_std > 0:
        noisy = img_b + torch.randn_like(img_b, generator=rng) * float(noise_std)
    else:
        noisy = img_b
    noisy_np = noisy[0, 0].detach().cpu().numpy()

    preds: list[np.ndarray] = []
    for _, model in models_ordered:
        with torch.no_grad():
            pred = (model(noisy) > 0.0).float()[0, 0].cpu().numpy()
        preds.append(pred)

    n = len(models_ordered)
    out_path_combined.parent.mkdir(parents=True, exist_ok=True)

    # Combined grid
    fig, axes = plt.subplots(n, 3, figsize=(9.2, 2.35 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, ((name, _), pred) in enumerate(zip(models_ordered, preds)):
        axes[i, 0].imshow(noisy_np, cmap="gray")
        axes[i, 0].set_ylabel(name, fontsize=9)
        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    axes[0, 0].set_title(f"Noisy input (σ={noise_std:g})")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    plt.tight_layout()
    plt.savefig(out_path_combined, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Optional single-row PNGs (same tensors)
    if out_paths_by_name:
        for i, ((name, _), pred) in enumerate(zip(models_ordered, preds)):
            outp = out_paths_by_name.get(name)
            if outp is None:
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig, axr = plt.subplots(1, 3, figsize=(9.2, 2.6))
            axr[0].imshow(noisy_np, cmap="gray")
            axr[0].set_title(f"Noisy (σ={noise_std:g})")
            axr[1].imshow(mask_np, cmap="gray")
            axr[1].set_title("Ground truth")
            axr[2].imshow(pred, cmap="gray")
            axr[2].set_title("Prediction")
            for ax in axr:
                ax.axis("off")
            fig.suptitle(name, fontsize=11, y=1.02)
            plt.tight_layout()
            plt.savefig(outp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def _sigma_filename_tag(sigma: float) -> str:
    s = float(sigma)
    if s == 0.0:
        return "0"
    t = f"{s:.4f}".rstrip("0").rstrip(".")
    return t.replace(".", "p")


def _model_filename_slug(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .replace("/", "-")
    )


def _failure_progression_base_tensors(
    models_ordered: list[tuple[str, torch.nn.Module]],
    ds,
    device: torch.device,
    sample_idx: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    for _, m in models_ordered:
        m.eval()
    img, mask = ds[sample_idx % len(ds)]
    img = img.to(device)
    mask_np = mask[0].cpu().numpy()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + sample_idx)
    img_b = img.unsqueeze(0)
    eps = torch.randn_like(img_b, generator=rng)
    return img_b, mask_np, eps


def make_failure_progression_separate_pngs(
    models_ordered: list[tuple[str, torch.nn.Module]],
    ds,
    device: torch.device,
    noise_stds: list[float],
    sample_idx: int,
    seed: int,
    out_dir: Path,
    dpi: int = 220,
    prefix: str = "noise_failure",
    write_by_sigma: bool = True,
    write_by_model: bool = True,
) -> list[Path]:
    """
    One PNG per σ (all models, n×3) and/or one PNG per model (σ down the rows, R×3).
    Uses shared ε across severities: ``noisy = x + σ * eps``.
    """
    written: list[Path] = []
    if not models_ordered or not noise_stds:
        return written

    img_b, mask_np, eps = _failure_progression_base_tensors(
        models_ordered, ds, device, sample_idx, seed
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(models_ordered)
    R = len(noise_stds)

    if write_by_sigma:
        row_h = min(1.75, max(0.82, 28.0 / max(n, 1)))
        for sigma in noise_stds:
            sigma = float(sigma)
            if sigma > 0:
                noisy = img_b + eps * sigma
            else:
                noisy = img_b.clone()
            noisy_np = noisy[0, 0].detach().cpu().numpy()
            preds: list[np.ndarray] = []
            for _, model in models_ordered:
                with torch.no_grad():
                    pred = (model(noisy) > 0.0).float()[0, 0].cpu().numpy()
                preds.append(pred)

            fig, axes = plt.subplots(n, 3, figsize=(9.2, row_h * n))
            if n == 1:
                axes = np.expand_dims(axes, axis=0)
            for i, ((name, _), pred) in enumerate(zip(models_ordered, preds)):
                axes[i, 0].imshow(noisy_np, cmap="gray")
                axes[i, 0].set_ylabel(name, fontsize=9)
                if i == 0:
                    axes[i, 0].set_title(f"Noisy input (σ={sigma:g})")
                axes[i, 1].imshow(mask_np, cmap="gray")
                axes[i, 2].imshow(pred, cmap="gray")
                for j in range(3):
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
            axes[0, 1].set_title("Ground truth")
            axes[0, 2].set_title("Prediction")
            tag = _sigma_filename_tag(sigma)
            outp = out_dir / f"{prefix}_sigma_{tag}_all_models.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(outp)

    if write_by_model:
        row_h = min(2.0, max(0.88, 22.0 / max(R, 1)))
        noisy_nps: list[np.ndarray] = []
        preds_by_sigma: list[list[np.ndarray]] = []
        for sigma in noise_stds:
            sigma = float(sigma)
            if sigma > 0:
                noisy = img_b + eps * sigma
            else:
                noisy = img_b.clone()
            noisy_nps.append(noisy[0, 0].detach().cpu().numpy())
            row_preds: list[np.ndarray] = []
            for _, model in models_ordered:
                with torch.no_grad():
                    pred = (model(noisy) > 0.0).float()[0, 0].cpu().numpy()
                row_preds.append(pred)
            preds_by_sigma.append(row_preds)

        for mi, (name, _) in enumerate(models_ordered):
            fig, axes = plt.subplots(R, 3, figsize=(9.2, row_h * R))
            if R == 1:
                axes = np.expand_dims(axes, axis=0)
            for r, sigma in enumerate(noise_stds):
                sigma = float(sigma)
                noisy_np = noisy_nps[r]
                pred = preds_by_sigma[r][mi]
                axes[r, 0].imshow(noisy_np, cmap="gray")
                axes[r, 0].set_ylabel(f"σ={sigma:g}", fontsize=9)
                if r == 0:
                    axes[r, 0].set_title("Noisy input")
                axes[r, 1].imshow(mask_np, cmap="gray")
                axes[r, 2].imshow(pred, cmap="gray")
                for j in range(3):
                    axes[r, j].set_xticks([])
                    axes[r, j].set_yticks([])
            axes[0, 1].set_title("Ground truth")
            axes[0, 2].set_title("Prediction")
            fig.suptitle(name, fontsize=11, y=1.01)
            slug = _model_filename_slug(name)
            outp = out_dir / f"{prefix}_{slug}_sigma_progression.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(outp)

    return written


def make_failure_noise_comparison_multi_sigma(
    models_ordered: list[tuple[str, torch.nn.Module]],
    ds,
    device: torch.device,
    noise_stds: list[float],
    sample_idx: int,
    seed: int,
    out_path_combined: Path,
    dpi: int = 220,
    row_height_in: float | None = None,
) -> None:
    """
    Same validation patch and **shared** Gaussian direction ``eps`` for every severity:
    ``noisy = x + σ * eps``. For each σ, rows are (model × 3): noisy | GT | prediction.
    Stacking σ from low to high shows increasing corruption for all models on comparable noise.
    """
    if not models_ordered or not noise_stds:
        return

    img_b, mask_np, eps = _failure_progression_base_tensors(
        models_ordered, ds, device, sample_idx, seed
    )

    n = len(models_ordered)
    R = len(noise_stds)
    total_rows = n * R
    if row_height_in is None:
        row_height_in = min(1.75, max(0.82, 34.0 / max(total_rows, 1)))
    fig, axes = plt.subplots(total_rows, 3, figsize=(9.2, row_height_in * total_rows))
    if total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    row_base = 0
    for sigma in noise_stds:
        sigma = float(sigma)
        if sigma > 0:
            noisy = img_b + eps * sigma
        else:
            noisy = img_b.clone()
        noisy_np = noisy[0, 0].detach().cpu().numpy()

        preds: list[np.ndarray] = []
        for _, model in models_ordered:
            with torch.no_grad():
                pred = (model(noisy) > 0.0).float()[0, 0].cpu().numpy()
            preds.append(pred)

        for i, ((name, _), pred) in enumerate(zip(models_ordered, preds)):
            r = row_base + i
            axes[r, 0].imshow(noisy_np, cmap="gray")
            axes[r, 0].set_ylabel(name, fontsize=9)
            if i == 0:
                axes[r, 0].set_title(f"Noisy input (σ={sigma:g})")
            axes[r, 1].imshow(mask_np, cmap="gray")
            axes[r, 2].imshow(pred, cmap="gray")
            for j in range(3):
                axes[r, j].set_xticks([])
                axes[r, j].set_yticks([])

        row_base += n

    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("Prediction")
    out_path_combined.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_combined, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Generate noise ablation plots and qualitative prediction figures.")
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument("--noise-levels", type=str, default="0.0,0.3,0.6,1.0,1.5,2.0,3.0,4.0,5.0")
    p.add_argument(
        "--qual-noise-levels",
        type=str,
        default="0.0,0.3,0.6,1.0,1.5,2.0,3.0",
        help="Include high std (e.g. 3.0) to match Gaussian robustness narrative.",
    )
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--distance-threshold", type=float, default=6.0)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--include-unetr", action="store_true")
    p.add_argument("--outdir", type=str, default="reports/noise_ablation")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset(args.processed, split="val")
    noise_levels = parse_noise_levels(args.noise_levels)
    qual_noise_levels = parse_noise_levels(args.qual_noise_levels)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = load_models(device=device, include_unetr=args.include_unetr)
    if not models:
        raise SystemExit("No models were loaded. Check checkpoint paths.")

    rows = run_ablation(
        models=models,
        ds=ds,
        device=device,
        noise_levels=noise_levels,
        n_samples=args.n_samples,
        distance_threshold=args.distance_threshold,
        seed=args.seed,
    )
    csv_path = outdir / "noise_ablation_results.csv"
    plot_path = outdir / "noise_ablation_f1_curve.png"
    save_csv(rows, csv_path)
    plot_f1_curves(rows, plot_path)
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {plot_path}")

    # Qualitative figures focused on "our model" (noise-resilient Hybrid-STEMSeg),
    # plus GAN variants for comparison.
    if "Hybrid-STEMSeg" in models:
        out_path = outdir / "our_model_main_hybrid_stemseg_noise_examples.png"
        make_qualitative_figure(
            model=models["Hybrid-STEMSeg"],
            ds=ds,
            device=device,
            noise_levels=qual_noise_levels,
            sample_idx=args.sample_idx,
            seed=args.seed,
            out_path=out_path,
            title="Our Main Model (Hybrid-STEMSeg): Noisy Input vs Prediction",
        )
        print(f"[SAVED] {out_path}")

    if "Hybrid-NoTransformer" in models:
        out_path = outdir / "our_model_hybrid_notransformer_noise_examples.png"
        make_qualitative_figure(
            model=models["Hybrid-NoTransformer"],
            ds=ds,
            device=device,
            noise_levels=qual_noise_levels,
            sample_idx=args.sample_idx,
            seed=args.seed,
            out_path=out_path,
            title="Hybrid-NoTransformer (no bottleneck Transformer): Noisy Input vs Prediction",
        )
        print(f"[SAVED] {out_path}")

    if "Original GAN (Scratch)" in models:
        out_path = outdir / "our_model_original_gan_noise_examples.png"
        make_qualitative_figure(
            model=models["Original GAN (Scratch)"],
            ds=ds,
            device=device,
            noise_levels=qual_noise_levels,
            sample_idx=args.sample_idx,
            seed=args.seed,
            out_path=out_path,
            title="Our Model (Original GAN): Noisy Input vs Prediction",
        )
        print(f"[SAVED] {out_path}")

    if "GAN (ResNet-Initialized)" in models:
        out_path = outdir / "our_model_pretrained_gan_noise_examples.png"
        make_qualitative_figure(
            model=models["GAN (ResNet-Initialized)"],
            ds=ds,
            device=device,
            noise_levels=qual_noise_levels,
            sample_idx=args.sample_idx,
            seed=args.seed,
            out_path=out_path,
            title="Our Model (GAN ResNet-Initialized): Noisy Input vs Prediction",
        )
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
