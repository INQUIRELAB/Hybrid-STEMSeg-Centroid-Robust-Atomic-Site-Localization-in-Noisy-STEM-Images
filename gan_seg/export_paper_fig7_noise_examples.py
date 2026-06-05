#!/usr/bin/env python3
"""
Fig. 7 (compact): one Gaussian severity σ, same noisy patch for every model.

Writes:
  noise_failure_cases_all_models.png — grid (n_models × 3) at one σ
  noise_failure_progression/ — per-σ and per-model PNGs when using --failure-noise-levels (default)
  noise_failure_cases_all_models_progression.png — optional tall grid (--failure-progression-combined)
  Plus legacy filenames (each a single short row 1×3) for optional LaTeX subfigures:
    our_model_main_hybrid_stemseg_noise_examples.png  (Hybrid-STEMSeg)
    our_model_hybrid_notransformer_noise_examples.png (Hybrid-NoTransformer)
    our_model_original_gan_noise_examples.png
    our_model_pretrained_gan_noise_examples.png

Real ``data/processed/sm_bfo_com`` val shards required unless ``--allow-synthetic-demo-only``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gan_seg.export_smbfo_figure_panels import _synthetic_patch
from gan_seg.eval_noise_SOTA import parse_noise_levels
from gan_seg.make_noise_figures import (
    default_model_specs,
    load_models,
    make_failure_noise_comparison,
    make_failure_noise_comparison_multi_sigma,
    make_failure_progression_separate_pngs,
)
from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.paper_export_requirements import exit_missing_processed_val


class _SinglePatchDataset(Dataset):
    def __init__(self, img: torch.Tensor, mask: torch.Tensor):
        self.img = img
        self.mask = mask

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return self.img.clone(), self.mask.clone()


def _build_dataset(processed: str, allow_synthetic_demo: bool):
    if allow_synthetic_demo:
        img, mask = _synthetic_patch()
        return _SinglePatchDataset(img, mask), "INTERNAL DEMO ONLY — synthetic (not for publication)"
    exit_missing_processed_val(processed)
    try:
        ds = ShardedPatchDataset(processed, split="val")
        if len(ds) == 0:
            raise RuntimeError("empty")
        return ds, f"real val {processed}"
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(
            f"ERROR: Could not open val split at {processed}: {exc}\n"
            "Fix the dataset, or (dev only) pass --allow-synthetic-demo-only.\n"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    ap.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    ap.add_argument(
        "--failure-noise-std",
        type=float,
        default=1.5,
        help="Used when --failure-noise-levels is omitted: single Gaussian σ for all models.",
    )
    ap.add_argument(
        "--failure-noise-levels",
        type=str,
        default=None,
        help=(
            "Comma-separated σ values (low→high). Uses shared noise direction σ·ε per level. "
            "Writes separate PNGs under noise_failure_progression/ (per-σ all-models + per-model rows). "
            "Optional tall montage: --failure-progression-combined."
        ),
    )
    ap.add_argument(
        "--failure-progression-combined",
        action="store_true",
        help="Also write noise_failure_cases_all_models_progression.png (single tall figure).",
    )
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument(
        "--no-per-model-pngs",
        action="store_true",
        help="Only write noise_failure_cases_all_models.png (omit four legacy filenames).",
    )
    ap.add_argument(
        "--allow-synthetic-demo-only",
        action="store_true",
        help="DEVELOPMENT ONLY: fake patch — not for journal use.",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "Include every loaded checkpoint from default_model_specs (UNet, DeepLabV3+, SegFormer, "
            "hybrids, GANs). Default grid is only the four hybrid/GAN rows used in narrow subfigures."
        ),
    )
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, src_note = _build_dataset(cli.processed, cli.allow_synthetic_demo_only)
    out_dir = cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    want_order: list[tuple[str, str]] = [
        ("Hybrid-STEMSeg", "our_model_main_hybrid_stemseg_noise_examples.png"),
        ("Hybrid-NoTransformer", "our_model_hybrid_notransformer_noise_examples.png"),
        ("Original GAN (Scratch)", "our_model_original_gan_noise_examples.png"),
        ("GAN (ResNet-Initialized)", "our_model_pretrained_gan_noise_examples.png"),
    ]

    models = load_models(device=device, include_unetr=False)
    if cli.all_models:
        ordered = [(n, models[n]) for n, _, _, _ in default_model_specs(include_unetr=False) if n in models]
    else:
        ordered = [(n, models[n]) for n, _ in want_order if n in models]
    if not ordered:
        raise SystemExit("No Fig.7 models loaded (missing checkpoints).")

    combined = out_dir / "noise_failure_cases_all_models.png"
    if cli.failure_noise_levels:
        noise_stds = parse_noise_levels(cli.failure_noise_levels)
        prog_dir = out_dir / "noise_failure_progression"
        separate_paths = make_failure_progression_separate_pngs(
            ordered,
            ds,
            device,
            noise_stds=noise_stds,
            sample_idx=cli.sample_idx,
            seed=cli.seed,
            out_dir=prog_dir,
            dpi=cli.dpi,
            prefix="noise_failure",
            write_by_sigma=True,
            write_by_model=True,
        )
        written = [str(p.relative_to(out_dir)) for p in sorted(separate_paths)]
        if cli.failure_progression_combined:
            combined_prog = out_dir / "noise_failure_cases_all_models_progression.png"
            make_failure_noise_comparison_multi_sigma(
                ordered,
                ds,
                device,
                noise_stds=noise_stds,
                sample_idx=cli.sample_idx,
                seed=cli.seed,
                out_path_combined=combined_prog,
                dpi=cli.dpi,
            )
            written.append(combined_prog.name)
        readme_extra = (
            f"failure_noise_levels (shared ε, noisy = x + σ·ε): {noise_stds}\n"
            f"Separate figures directory: {prog_dir.name}/\n"
            "Regenerate: python -m gan_seg.export_paper_fig7_noise_examples --out-dir paper_figures "
            f'--failure-noise-levels "{cli.failure_noise_levels}"'
            + (" --all-models" if cli.all_models else "")
            + (" --failure-progression-combined" if cli.failure_progression_combined else "")
            + "\n"
        )
    else:
        per_model: dict[str, Path] | None = None
        if not cli.no_per_model_pngs:
            per_model = {n: out_dir / fname for n, fname in want_order if n in models}

        make_failure_noise_comparison(
            ordered,
            ds,
            device,
            noise_std=cli.failure_noise_std,
            sample_idx=cli.sample_idx,
            seed=cli.seed,
            out_path_combined=combined,
            out_paths_by_name=per_model,
            dpi=cli.dpi,
        )

        written = [combined.name]
        if per_model:
            written.extend(sorted({p.name for p in per_model.values()}))
        readme_extra = (
            f"failure_noise_std (single σ for all models, shared noisy input): {cli.failure_noise_std}\n"
            "Combined grid recommended for one figure; per-model PNGs are 1×3 strips at the same σ.\n"
            "Regenerate: python -m gan_seg.export_paper_fig7_noise_examples --out-dir paper_figures "
            "--failure-noise-std 1.5\n"
        )

    readme = out_dir / "README_smbfo_fig7.txt"
    readme.write_text(
        f"source_patch: {src_note}\n"
        f"sample_idx: {cli.sample_idx}  seed: {cli.seed}\n"
        f"files_written: {written}\n"
        + readme_extra,
        encoding="utf-8",
    )
    if cli.failure_noise_levels:
        for line in written:
            print(f"[SAVED] {out_dir / line}")
    else:
        print(f"[SAVED] {combined}")
        if not cli.no_per_model_pngs:
            per_model_out = {n: out_dir / fname for n, fname in want_order if n in models}
            for p in per_model_out.values():
                if p.is_file():
                    print(f"[SAVED] {p}")
    print(f"README: {readme.resolve()}")


if __name__ == "__main__":
    main()
