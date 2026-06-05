#!/usr/bin/env python3
"""
Fig. 5 (corruption protocol): four PNGs in paper_figures/
  smbfo_clean_example.png, smbfo_gaussian_example.png,
  smbfo_poisson_example.png, smbfo_mixed_example.png

Corruptions match gan_seg.reviewer_study.apply_corruption (evaluated at test time).
Default severities align with reviewer_study suites: Gaussian {0.3..2.0}, Poisson
peak list, Mixed compound chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.export_smbfo_figure_panels import _synthetic_patch
from gan_seg.paper_export_requirements import exit_missing_processed_val
from gan_seg.reviewer_study import apply_corruption


def _load_patch(
    processed: str, index: int, allow_synthetic_demo: bool
) -> tuple[torch.Tensor, str]:
    if allow_synthetic_demo:
        img_t, _ = _synthetic_patch()
        return img_t, "INTERNAL DEMO ONLY — synthetic (not for publication)"
    exit_missing_processed_val(processed)
    try:
        ds = ShardedPatchDataset(processed, split="val")
        if len(ds) == 0:
            raise RuntimeError("empty")
        img_t, _ = ds[index]
        return img_t, f"real val patch {processed}[{index}]"
    except (FileNotFoundError, RuntimeError, IndexError) as exc:
        raise SystemExit(
            f"ERROR: Could not read val patch {index} from {processed}: {exc}\n"
            "Fix the dataset, or (dev only) pass --allow-synthetic-demo-only.\n"
        ) from exc


def _to_4d(img_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(1,H,W) -> (1,1,H,W) on device."""
    if img_t.dim() == 3:
        return img_t.unsqueeze(0).to(device)
    return img_t.to(device)


def _display_range(clean_2d: np.ndarray, corrupted_list: list[np.ndarray]) -> tuple[float, float]:
    stack = np.stack([clean_2d] + corrupted_list, axis=0)
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.0))
    pad = max(hi - lo, 1e-6) * 0.25
    return lo - pad, hi + pad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_com")
    p.add_argument("--out-dir", type=str, default="paper_figures")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument(
        "--allow-synthetic-demo-only",
        action="store_true",
        help="DEVELOPMENT ONLY: fake patch — not for journal use.",
    )
    p.add_argument("--gaussian-severity", type=float, default=1.0)
    p.add_argument("--poisson-severity", type=float, default=10.0)
    p.add_argument("--mixed-severity", type=float, default=1.0)
    p.add_argument("--rng-seed", type=int, default=4242)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_t, src = _load_patch(args.processed, args.index, args.allow_synthetic_demo_only)
    x = _to_4d(img_t, device)
    clean_np = x[0, 0].detach().cpu().numpy()

    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(args.rng_seed)
    np_rng = np.random.default_rng(args.rng_seed)

    g_t = apply_corruption(x, "gaussian", args.gaussian_severity, torch_rng, np_rng)
    # Independent draws for Poisson (reseed so Gaussian noise differs from panel 2 chain)
    torch_rng.manual_seed(args.rng_seed + 1)
    np_rng = np.random.default_rng(args.rng_seed + 1)
    p_t = apply_corruption(x, "poisson", args.poisson_severity, torch_rng, np_rng)
    torch_rng.manual_seed(args.rng_seed + 2)
    np_rng = np.random.default_rng(args.rng_seed + 2)
    m_t = apply_corruption(x, "mixed", args.mixed_severity, torch_rng, np_rng)

    gauss_np = g_t[0, 0].detach().cpu().numpy()
    poiss_np = p_t[0, 0].detach().cpu().numpy()
    mix_np = m_t[0, 0].detach().cpu().numpy()

    lo, hi = _display_range(clean_np, [gauss_np, poiss_np, mix_np])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(arr: np.ndarray, name: str) -> None:
        disp = np.clip(arr, lo, hi)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(disp, cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig.savefig(out_dir / name, dpi=args.dpi, pad_inches=0, bbox_inches="tight")
        plt.close(fig)

    save(clean_np, "smbfo_clean_example.png")
    save(gauss_np, "smbfo_gaussian_example.png")
    save(poiss_np, "smbfo_poisson_example.png")
    save(mix_np, "smbfo_mixed_example.png")

    readme = out_dir / "README_smbfo_fig5.txt"
    readme.write_text(
        f"source_patch: {src}\n"
        f"corruptions (reviewer_study.apply_corruption): "
        f"Gaussian sigma={args.gaussian_severity}, "
        f"Poisson peak={args.poisson_severity}, "
        f"Mixed severity={args.mixed_severity} (compound chain)\n"
        f"display: shared grayscale window from 1–99% range across all four panels (+ padding)\n"
        f"Regenerate: python -m gan_seg.export_smbfo_corruption_panels --out-dir paper_figures\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_dir}/smbfo_clean_example.png, smbfo_gaussian_example.png, ...")
    print(f"  source: {src}")
    print(f"  README: {readme}")


if __name__ == "__main__":
    main()
