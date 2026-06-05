"""Stack sharded NPZ patches into dense numpy arrays (for AtomAI-style trainers)."""

from __future__ import annotations

import numpy as np

from gan_seg.dataset_preprocessed import ShardedPatchDataset


def load_split_stack(processed_root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (images, masks) as float32 arrays shaped (N, 1, H, W).
    """
    ds = ShardedPatchDataset(processed_root, split=split)
    n = len(ds)
    if n == 0:
        raise ValueError(f"empty split={split!r} under {processed_root}")
    x0, y0 = ds[0]
    c, h, w = x0.shape
    images = np.zeros((n, c, h, w), dtype=np.float32)
    masks = np.zeros((n, c, h, w), dtype=np.float32)
    for i in range(n):
        x, y = ds[i]
        images[i] = x.numpy()
        masks[i] = y.numpy()
    return images, masks
