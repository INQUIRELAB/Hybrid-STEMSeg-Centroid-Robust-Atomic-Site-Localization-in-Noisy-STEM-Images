"""Save and load reproducible experiment configs (JSON)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_date_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def build_run_name(
    dataset: str,
    model: str,
    decoder: str = "multitask",
    seed: int = 42,
    date: str | None = None,
) -> str:
    """e.g. sm_bfo_centroid_hybrid_centroid_multitask_seed42_20260524"""
    ds = Path(dataset).name if "/" in dataset or "\\" in dataset else dataset
    ds = ds.replace("data/processed/", "").replace("processed/", "")
    d = date or utc_date_stamp()
    return f"{ds}_{model}_{decoder}_seed{seed}_{d}"


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str | Path, config: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return p


def atomic_torch_save(obj: Any, path: str | Path) -> Path:
    """Write checkpoint atomically (tmp + replace) and fsync for power-loss safety."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(p)
    return p


def mask_train_resume_state(
    log_path: Path,
    val_key: str = "val_loss",
) -> tuple[int, float, int]:
    """
    Read train_log.jsonl for mask-only / benchmark training.
    Returns (start_epoch, best_val, epochs_no_improve) for early-stopping continuity.
    """
    import math

    if not log_path.is_file():
        return 0, math.inf, 0
    records: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return 0, math.inf, 0

    def val_of(r: dict[str, Any]) -> float:
        if val_key in r:
            return float(r[val_key])
        val = r.get("val", {})
        if isinstance(val, dict) and "loss_total" in val:
            return float(val["loss_total"])
        raise KeyError(f"No validation loss in record keys: {list(r.keys())}")

    best_val = math.inf
    epochs_no_improve = 0
    for r in records:
        v = val_of(r)
        if v < best_val - 1e-4:
            best_val = v
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
    start_epoch = int(records[-1]["epoch"]) + 1
    return start_epoch, best_val, epochs_no_improve


def centroid_resume_from_checkpoint(
    ckpt: dict[str, Any],
    log_path: Path,
) -> tuple[int, int, float, int]:
    """
    Resume training from a centroid-aware checkpoint.

    Returns (start_epoch, step_in_epoch, best_val, epochs_no_improve).

    - Mid-epoch checkpoint (step_in_epoch > 0): continue that epoch from the next step.
    - End-of-epoch checkpoint (step_in_epoch == 0): start the following epoch.
    """
    import math

    log_start, best_val, epochs_no_improve = mask_train_resume_state(log_path)
    ckpt_epoch = int(ckpt.get("epoch", -1))
    step = int(ckpt.get("step_in_epoch", 0) or 0)

    if "best_val" in ckpt:
        best_val = float(ckpt["best_val"])
    if "epochs_no_improve" in ckpt:
        epochs_no_improve = int(ckpt["epochs_no_improve"])

    if step > 0:
        return ckpt_epoch, step, best_val, epochs_no_improve

    start_epoch = ckpt_epoch + 1 if ckpt_epoch >= 0 else 0
    if log_path.is_file():
        start_epoch = max(start_epoch, log_start)
    else:
        start_epoch = max(start_epoch, 0)
    if best_val == math.inf and "val_loss" in ckpt:
        best_val = float(ckpt["val_loss"])
    return start_epoch, 0, best_val, epochs_no_improve


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge; nested dicts merged one level deep."""
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out
