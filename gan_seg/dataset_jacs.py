"""PyTorch Dataset for JACS manual STEM + atom CSV (full frame, div8 crop)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from gan_seg.dataset_patches import build_atom_mask_from_com
from gan_seg.jacs_data import crop_div8, load_csv_coords_yx, normalize_patch2d


class JacsSTEMDataset(Dataset):
    """One sample = one cropped HAADF frame + binary mask from manual CSV."""

    def __init__(self, records: list[dict], mask_sigma: float = 2.5):
        self.records = records
        self.mask_sigma = mask_sigma

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img = tifffile.imread(rec["image"])
        if img.ndim != 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D {rec['image']}, got {img.shape}")
        img = np.asarray(img, dtype=np.float32)
        coords = load_csv_coords_yx(Path(rec["csv"]))
        if len(coords) == 0:
            raise ValueError(f"No coords: {rec['csv']}")
        img_c, coords_c = crop_div8(img, coords)
        if len(coords_c) < 1:
            raise ValueError(f"No atoms in crop: {rec['id']}")
        h, w = img_c.shape
        mask = build_atom_mask_from_com(h, w, coords_c, 0, 0, sigma=self.mask_sigma)
        x = normalize_patch2d(img_c)
        y = mask[np.newaxis, ...].astype(np.float32)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
