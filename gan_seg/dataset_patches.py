"""Random patches from Sm-BFO `stem_smbfo` arrays + binary masks from `xy_COM`."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def build_atom_mask_from_com(
    patch_h: int,
    patch_w: int,
    xy_com: np.ndarray,
    top: int,
    left: int,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Binary mask (1 = atom / foreground) from centers of mass inside the crop.
    xy_com: (N, 2) with rows (y, x) in full-image pixel coordinates.
    """
    mask = np.zeros((patch_h, patch_w), dtype=np.float32)
    if xy_com is None or len(xy_com) == 0:
        return mask
    y_all, x_all = xy_com[:, 0], xy_com[:, 1]
    inside = (
        (y_all >= top)
        & (y_all < top + patch_h)
        & (x_all >= left)
        & (x_all < left + patch_w)
    )
    yy, xx = np.ogrid[:patch_h, :patch_w]
    sig2 = 2.0 * (sigma ** 2)
    for y0, x0 in zip(y_all[inside], x_all[inside]):
        cy, cx = float(y0 - top), float(x0 - left)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sig2)
        mask = np.maximum(mask, g)
    return (mask >= 0.25).astype(np.float32)


class SmBFOPatchDataset(Dataset):
    """
    Loads `SmBFO_composition_series.npy` dict; each item is a random crop from
    a randomly chosen composition key.
    """

    def __init__(
        self,
        npy_path: str,
        patch_size: int = 256,
        keys: Optional[list] = None,
        sigma: float = 3.0,
        seed: int = 0,
    ):
        super().__init__()
        raw = np.load(npy_path, allow_pickle=True)[()]
        self.keys = list(raw.keys()) if keys is None else [k for k in keys if k in raw]
        if not self.keys:
            raise ValueError("No valid composition keys")
        self.data: Dict[str, dict] = {k: raw[k] for k in self.keys}
        self.patch_size = patch_size
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.keys[int(self.rng.integers(0, len(self.keys)))]
        entry = self.data[k]
        img = np.asarray(entry["main_image"], dtype=np.float32)
        com = np.asarray(entry["xy_atms"], dtype=np.float64)
        h, w = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"Image {h}x{w} smaller than patch {ps}")
        top = int(self.rng.integers(0, h - ps + 1))
        left = int(self.rng.integers(0, w - ps + 1))
        patch = img[top : top + ps, left : left + ps].copy()
        # Several Sm-BFO frames contain NaNs in the public array; zero them for stable training.
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        mask = build_atom_mask_from_com(ps, ps, com, top, left, self.sigma)
        p = patch[np.newaxis, ...]
        mu, sd = float(p.mean()), float(p.std()) + 1e-6
        p = (p - mu) / sd
        return torch.from_numpy(p), torch.from_numpy(mask[np.newaxis, ...])
