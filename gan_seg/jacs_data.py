"""Shared JACS Zenodo dataset helpers (Mitchell et al., DOI 10.5281/zenodo.5931544)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JACS_EXTRACTED = (
    ROOT / "external_stem_data" / "jacs_single_atom_TEM" / "extracted"
)


def load_csv_coords_yx(csv_path: Path) -> np.ndarray:
    """Rows (y, x) = (row, col) for build_atom_mask_from_com; CSV columns X,Y."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return np.zeros((0, 2), dtype=np.float64)
        norm = {k.strip(): k for k in reader.fieldnames}
        xk = norm.get("X")
        yk = norm.get("Y")
        if xk is None or yk is None:
            raise ValueError(f"Expected X,Y columns in {csv_path}, got {reader.fieldnames}")
        for row in reader:
            x = float(row[xk].strip())
            y = float(row[yk].strip())
            rows.append([y, x])
    return np.asarray(rows, dtype=np.float64)


def crop_div8(
    img: np.ndarray, coords_yx: np.ndarray, multiple: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    h, w = int(img.shape[0]), int(img.shape[1])
    ch = (h // multiple) * multiple
    cw = (w // multiple) * multiple
    if ch < multiple or cw < multiple:
        raise ValueError(f"Image too small after crop: {h}x{w}")
    top = (h - ch) // 2
    left = (w - cw) // 2
    img_c = img[top : top + ch, left : left + cw].copy()
    cy = coords_yx[:, 0] - top
    cx = coords_yx[:, 1] - left
    inside = (cy >= 0) & (cy < ch) & (cx >= 0) & (cx < cw)
    coords_c = np.column_stack([cy[inside], cx[inside]])
    return img_c, coords_c


def normalize_patch2d(patch: np.ndarray) -> np.ndarray:
    p = patch[np.newaxis, ...].astype(np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = float(p.mean()), float(p.std()) + 1e-6
    return (p - mu) / sd


def discover_pairs(extracted: Path) -> list[dict]:
    img_dir = extracted / "01_groundtruth_labeled_images_manual"
    pt_csv = extracted / "02_groundtruth_atomic_positions_manual" / "Pt-Catalyst"
    fe_csv = extracted / "02_groundtruth_atomic_positions_manual" / "Fe-Catalyst"
    pairs: list[dict] = []

    for tif_path in sorted(img_dir.glob("*.tif")):
        name = tif_path.name
        if "Fe" in name or name.endswith("_labeled.tif"):
            if not name.endswith("_labeled.tif"):
                continue
            stem = name[: -len("_labeled.tif")]
            cpath = fe_csv / f"{stem}_results_updated.csv"
            cat = "Fe"
        else:
            parts = tif_path.stem.split("_")
            pid = parts[0]
            if not pid.isdigit():
                continue
            cpath = pt_csv / f"{pid}_Results.csv"
            cat = "Pt"
        if not cpath.is_file():
            continue
        pairs.append(
            {
                "category": cat,
                "image": str(tif_path.resolve()),
                "csv": str(cpath.resolve()),
                "id": tif_path.stem,
            }
        )
    return pairs


def few_shot_train_val_split(
    pairs: list[dict],
    n_shot: int,
    seed: int,
    min_val: int = 1,
) -> tuple[list[dict], list[dict]]:
    if n_shot < 1:
        raise ValueError("n_shot must be >= 1")
    if len(pairs) - n_shot < min_val:
        raise ValueError(
            f"Need at least {n_shot + min_val} pairs for train+val; have {len(pairs)}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    tr = order[:n_shot].tolist()
    va = order[n_shot:].tolist()
    train = [pairs[i] for i in tr]
    val = [pairs[i] for i in va]
    return train, val
