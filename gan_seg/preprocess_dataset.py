#!/usr/bin/env python3
"""
Preprocess Sm-BFO composition library for training:
  - Clean NaN/Inf in STEM images
  - Split compositions (keys) into train / val — no random patch leakage across splits
  - Extract random crops, per-patch z-score normalization, binary COM masks
  - Write sharded .npz + manifest.json

Example:
  python -m gan_seg.preprocess_dataset \\
    --source data/SmBFO_composition_series.npy \\
    --out data/processed/sm_bfo_gan \\
    --patch-size 256 --patches-per-key-train 400 --patches-per-key-val 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gan_seg.centroid_targets import (
    build_centroid_targets_from_centers,
    centers_from_patch_com,
)
from gan_seg.dataset_patches import build_atom_mask_from_com


def clean_image(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """float32 HxW, NaN/Inf -> 0. Returns (array, had_invalid)."""
    a = np.asarray(img, dtype=np.float32)
    bad = np.isnan(a).any() or np.isinf(a).any()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a, bad


def sample_patch(
    img: np.ndarray,
    com: np.ndarray,
    patch_size: int,
    sigma: float,
    rng: np.random.Generator,
    max_atoms: int = 64,
    centroid_targets: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]:
    """One normalized (1,H,W) image patch and (1,H,W) binary mask (+ optional centroid targets)."""
    h, w = img.shape
    ps = patch_size
    # A few crops can have tiny std -> extreme z-scores. Resample a few times,
    # then fall back to clamping.
    for _ in range(10):
        top = int(rng.integers(0, h - ps + 1))
        left = int(rng.integers(0, w - ps + 1))
        patch = img[top : top + ps, left : left + ps].copy()
        sd = float(patch.std())
        if sd >= 1e-3:
            break
    mask = build_atom_mask_from_com(ps, ps, com, top, left, sigma)
    p = patch[np.newaxis, ...]
    mu, sd = float(p.mean()), float(p.std()) + 1e-6
    p = ((p - mu) / sd).astype(np.float32)
    # Clamp extreme values to reduce distribution shift between shards.
    p = np.clip(p, -6.0, 6.0)
    m = mask.astype(np.float32)[np.newaxis, ...]
    if not centroid_targets:
        return p, m
    centers = centers_from_patch_com(com, top, left, ps, ps)
    ct = build_centroid_targets_from_centers(centers, ps, ps, sigma, max_atoms)
    return {
        "image": p,
        "mask": m,
        "heatmap": ct["heatmap"],
        "offset": ct["offset"],
        "offset_mask": ct["offset_mask"],
        "centroids": ct["centroids"],
        "centroid_valid": ct["centroid_valid"],
    }


def com_density_per_mpix(entry: Dict) -> float:
    """COM count per megapixel for stratified splitting."""
    img = np.asarray(entry["main_image"])
    h, w = img.shape
    ncom = int(np.asarray(entry["xy_atms"]).shape[0])
    return float(ncom / (h * w / 1e6))


def stratified_split_by_density(
    raw: Dict,
    keys: List[str],
    n_val: int,
    seed: int,
    force_train_keys: List[str] | None = None,
) -> Tuple[List[str], List[str], List[Tuple[str, float]]]:
    """
    Split keys into train/val by stratifying on COM density.
    Also supports forcing certain keys into train to avoid validation-only regimes.
    """
    force_train = set(force_train_keys or [])
    keys2 = [k for k in keys if k not in force_train]
    dens = [(k, com_density_per_mpix(raw[k])) for k in keys2]
    dens.sort(key=lambda t: t[1])  # low -> high
    if n_val <= 0:
        return keys, [], dens
    if n_val >= len(dens):
        n_val = max(1, len(dens) - 1)
    rng = np.random.default_rng(seed)
    # pick approximately evenly across density with random tie-break jitter
    positions = np.linspace(0, len(dens) - 1, n_val, dtype=int).tolist()
    positions = sorted(set(positions))
    while len(positions) < n_val:
        positions.append(int(rng.integers(0, len(dens))))
        positions = sorted(set(positions))
    positions = positions[:n_val]
    val_set = {dens[i][0] for i in positions}
    train_keys = [k for k in keys if k not in val_set]
    # forced train keys always included
    for k in force_train:
        if k in keys and k not in train_keys:
            train_keys.append(k)
    train_keys = sorted(train_keys)
    val_keys = sorted(list(val_set))
    return train_keys, val_keys, dens


def write_shards(
    out_root: Path,
    split_name: str,
    patches: List[Tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]],
    shard_size: int,
    centroid_targets: bool = False,
) -> Tuple[List[str], List[int]]:
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_shards: List[str] = []
    counts: List[int] = []
    for start in range(0, len(patches), shard_size):
        chunk = patches[start : start + shard_size]
        if centroid_targets:
            xs = np.stack([c["image"] for c in chunk], axis=0)
            ys = np.stack([c["mask"] for c in chunk], axis=0)
            hms = np.stack([c["heatmap"] for c in chunk], axis=0)
            offs = np.stack([c["offset"] for c in chunk], axis=0)
            offm = np.stack([c["offset_mask"] for c in chunk], axis=0)
            cents = np.stack([c["centroids"] for c in chunk], axis=0)
            cval = np.stack([c["centroid_valid"] for c in chunk], axis=0)
        else:
            xs = np.stack([c[0] for c in chunk], axis=0)
            ys = np.stack([c[1] for c in chunk], axis=0)
        idx = len(rel_shards)
        name = f"shard_{idx:05d}.npz"
        path = out_dir / name
        if centroid_targets:
            np.savez_compressed(
                path,
                images=xs,
                masks=ys,
                heatmaps=hms,
                offsets=offs,
                offset_masks=offm,
                centroids=cents,
                centroid_valid=cval,
            )
        else:
            np.savez_compressed(path, images=xs, masks=ys)
        rel = f"{split_name}/{name}"
        rel_shards.append(rel)
        counts.append(xs.shape[0])
    return rel_shards, counts


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Sm-BFO for GAN/seg training")
    p.add_argument("--source", type=str, default="data/SmBFO_composition_series.npy")
    p.add_argument("--out", type=str, default="data/processed/sm_bfo_gan")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--sigma", type=float, default=3.0, help="COM Gaussian sigma (pixels)")
    p.add_argument("--patches-per-key-train", type=int, default=400)
    p.add_argument("--patches-per-key-val", type=int, default=100)
    p.add_argument(
        "--val-key-fraction",
        type=float,
        default=0.22,
        help="Fraction of composition keys held out for validation (rounded, at least 1 if n>=2)",
    )
    p.add_argument(
        "--use-atms",
        action="store_true",
        help="Use xy_atms instead of xy_COM",
    )
    p.add_argument("--shard-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force-train-keys",
        type=str,
        default="Sm_7_0",
        help="Comma-separated composition keys to force into train split (default: Sm_7_0)",
    )
    p.add_argument(
        "--centroid-targets",
        action="store_true",
        help="Store heatmap, offset, and padded centroid sets in each shard (for centroid-aware training).",
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=64,
        help="Max atoms per patch when --centroid-targets is set.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"Missing source: {src.resolve()}")

    ps = args.patch_size
    if ps % 8 != 0:
        raise SystemExit("--patch-size must be divisible by 8")

    raw: Dict = np.load(src, allow_pickle=True)[()]
    keys = sorted(raw.keys())
    n = len(keys)
    if n < 2:
        raise SystemExit("Need at least 2 composition keys for train/val split")

    n_val = max(1, int(round(args.val_key_fraction * n)))
    n_val = min(n_val, n - 1)
    force_train_keys = [k.strip() for k in args.force_train_keys.split(",") if k.strip()]
    train_keys, val_keys, dens_table = stratified_split_by_density(
        raw, keys, n_val=n_val, seed=args.seed, force_train_keys=force_train_keys
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    keys_with_nan: List[str] = []
    for k in keys:
        im, bad = clean_image(raw[k]["main_image"])
        if bad:
            keys_with_nan.append(k)

    def collect_for_key_list(
        key_list: List[str], n_per_key: int, rng: np.random.Generator
    ) -> List[Tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]]:
        patches: List[Tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray]] = []
        for k in key_list:
            entry = raw[k]
            img, _ = clean_image(entry["main_image"])
            com_key = "xy_atms" if getattr(args, "use_atms", False) else "xy_COM"
            com = np.asarray(entry[com_key], dtype=np.float64)
            h, w = img.shape
            if h < ps or w < ps:
                raise ValueError(f"Key {k}: image {h}x{w} smaller than patch {ps}")
            for _ in range(n_per_key):
                patches.append(
                    sample_patch(
                        img,
                        com,
                        ps,
                        args.sigma,
                        rng,
                        max_atoms=args.max_atoms,
                        centroid_targets=args.centroid_targets,
                    )
                )
        return patches

    rng_tr = np.random.default_rng(args.seed + 1)
    rng_va = np.random.default_rng(args.seed + 2)
    train_patches = collect_for_key_list(
        train_keys, args.patches_per_key_train, rng_tr
    )
    val_patches = collect_for_key_list(val_keys, args.patches_per_key_val, rng_va)

    train_shards, train_counts = write_shards(
        out_root,
        "train",
        train_patches,
        args.shard_size,
        centroid_targets=args.centroid_targets,
    )
    val_shards, val_counts = write_shards(
        out_root,
        "val",
        val_patches,
        args.shard_size,
        centroid_targets=args.centroid_targets,
    )

    manifest = {
        "version": 2 if args.centroid_targets else 1,
        "source": str(src.resolve()),
        "patch_size": ps,
        "sigma": args.sigma,
        "centroid_targets": bool(args.centroid_targets),
        "max_atoms": int(args.max_atoms),
        "seed": args.seed,
        "split_strategy": "stratified_by_com_density",
        "forced_train_keys": force_train_keys,
        "com_density_table_sorted": dens_table,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "keys_with_nan_cleaned": keys_with_nan,
        "patches_per_key_train": args.patches_per_key_train,
        "patches_per_key_val": args.patches_per_key_val,
        "splits": {
            "train": {"shards": train_shards, "counts": train_counts},
            "val": {"shards": val_shards, "counts": val_counts},
        },
        "total_train": len(train_patches),
        "total_val": len(val_patches),
    }
    man_path = out_root / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}")
    print(
        f"Train: {len(train_patches)} patches ({len(train_keys)} keys), "
        f"Val: {len(val_patches)} patches ({len(val_keys)} keys)"
    )
    print(f"Output directory: {out_root.resolve()}")


if __name__ == "__main__":
    main()
