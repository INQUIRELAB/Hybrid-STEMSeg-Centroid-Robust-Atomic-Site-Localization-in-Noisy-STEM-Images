"""Load preprocessed sharded patch archives (see preprocess_dataset.py)."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedPatchDataset(Dataset):
    """
    Reads `manifest.json` plus train/val `.npz` shards written by preprocess_dataset.
    Each shard stores `images` (N,1,H,W) float32 (per-patch normalized) and
    `masks` (N,1,H,W) float32 in {0,1}.
    """

    def __init__(self, processed_root: str, split: str = "train"):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(processed_root).resolve()
        man_path = self.root / "manifest.json"
        if not man_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {man_path}")
        with open(man_path, "r", encoding="utf-8") as f:
            self.manifest: Dict = json.load(f)
        sp = self.manifest["splits"][split]
        self.shard_relpaths: List[str] = sp["shards"]
        self.shard_counts: List[int] = sp["counts"]
        if len(self.shard_relpaths) != len(self.shard_counts):
            raise ValueError("manifest shards/counts mismatch")
        self._cum: List[int] = [0]
        for c in self.shard_counts:
            self._cum.append(self._cum[-1] + c)
        self._total = self._cum[-1]
        self._cache_path: str | None = None
        self._cache_x: np.ndarray | None = None
        self._cache_y: np.ndarray | None = None

    def __len__(self) -> int:
        return self._total

    def _load_shard(self, relpath: str) -> Tuple[np.ndarray, np.ndarray]:
        path = self.root / relpath
        if not path.is_file():
            raise FileNotFoundError(path)
        z = np.load(path)
        return z["images"], z["masks"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        shard_i = bisect.bisect_right(self._cum, idx) - 1
        offset = idx - self._cum[shard_i]
        rel = self.shard_relpaths[shard_i]
        if rel != self._cache_path:
            self._cache_x, self._cache_y = self._load_shard(rel)
            self._cache_path = rel
        x = self._cache_x[offset]
        y = self._cache_y[offset]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
