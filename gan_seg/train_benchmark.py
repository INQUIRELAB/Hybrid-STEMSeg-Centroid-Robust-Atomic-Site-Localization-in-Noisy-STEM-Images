#!/usr/bin/env python3
"""
Train mask-only segmentation baselines on a preprocessed Sm-BFO shard root.

Default for fair xy_atms study: data/processed/sm_bfo_centroid (xy_atms masks).

Saves under gan_seg/checkpoints_runs/<run_name>/ without overwriting
gan_seg/checkpoints_benchmark/ (sm_bfo_com legacy).

Example:
  python -m gan_seg.train_benchmark \\
    --config experiments/configs/xy_atms/unet_seed42.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.experiment_io import (
    build_run_name,
    load_config,
    mask_train_resume_state,
    merge_config,
    save_config,
)
from gan_seg.losses import CombinedSegLoss
from gan_seg.model import HybridUNetTransformerBinary


def parse_args():
    p = argparse.ArgumentParser(description="Train mask-only benchmark model")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--processed", type=str, default="")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--date-stamp", type=str, default="")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--checkpoints-base", type=str, default="gan_seg/checkpoints_runs")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-weight", type=float, default=18.0)
    p.add_argument("--lambda-dice", type=float, default=0.5)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = use config training.num_workers or 4)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default="",
        help="Resume from checkpoint (default: <save_dir>/gan_seg_last.pt)",
    )
    p.add_argument("--no-pin-memory", action="store_true")
    return p.parse_args()


def get_model(name: str, device: torch.device) -> nn.Module:
    if name == "unet":
        import segmentation_models_pytorch as smp

        return smp.Unet("resnet34", in_channels=1, classes=1).to(device)
    if name == "deeplabv3plus":
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus("resnet34", in_channels=1, classes=1).to(device)
    if name == "unetr":
        from monai.networks.nets import UNETR

        return UNETR(
            in_channels=1,
            out_channels=1,
            img_size=(256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)
    if name == "segformer":
        from transformers import SegformerForSemanticSegmentation

        class SegFormerWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/mit-b0",
                    num_labels=1,
                    num_channels=1,
                    ignore_mismatched_sizes=True,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.m(pixel_values=x)
                return torch.nn.functional.interpolate(
                    out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False
                )

        return SegFormerWrapper().to(device)
    if name == "hybrid-nogan":
        return HybridUNetTransformerBinary(d_model=256, use_transformer=True).to(device)
    if name == "hybrid-notransformer":
        return HybridUNetTransformerBinary(d_model=256, use_transformer=False).to(device)
    raise ValueError(f"Unknown model: {name}")


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = load_config(args.config) if args.config else {}
    model_name = args.model or cfg.get("model", {}).get("name", "")
    if not model_name:
        raise SystemExit("--model or config.model.name is required")

    dataset_cli: dict[str, Any] = {"label_semantics": "xy_atms"}
    if args.processed:
        dataset_cli["processed_root"] = args.processed
    cli = {
        "model": {"name": model_name},
        "dataset": dataset_cli,
        "training": {
            "seed": args.seed,
            "batch_size": args.batch,
            "epochs": args.epochs,
            "lr": args.lr,
            "patience": args.patience,
            "log_every_steps": args.log_every_steps,
        },
        "loss": {"pos_weight": args.pos_weight, "lambda_dice": args.lambda_dice},
        "output": {"checkpoints_base": args.checkpoints_base},
        "run_name": args.run_name or None,
    }
    cfg = merge_config(cfg, {k: v for k, v in cli.items() if v is not None})

    proc = cfg.setdefault("dataset", {}).get(
        "processed_root", "data/processed/sm_bfo_centroid"
    )
    seed = int(cfg.setdefault("training", {}).get("seed", 42))
    if not cfg.get("run_name"):
        cfg["run_name"] = build_run_name(
            proc,
            model_name.replace("-", "_"),
            "mask",
            seed,
            args.date_stamp or None,
        )
    return cfg


def resolve_save_dir(cfg: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.save:
        return Path(args.save)
    base = Path(cfg.get("output", {}).get("checkpoints_base", "gan_seg/checkpoints_runs"))
    return base / cfg["run_name"]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    save_dir = resolve_save_dir(cfg, args)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    processed = cfg["dataset"]["processed_root"]
    full_config = {**cfg, "save_dir": str(save_dir.resolve()), "task": "mask_only_segmentation"}
    save_config(save_dir / "config.json", full_config)
    print(f"[config] {save_dir / 'config.json'}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["training"]["seed"]))

    train_ds = ShardedPatchDataset(processed, split="train")
    val_ds = ShardedPatchDataset(processed, split="val")
    batch_size = int(cfg["training"]["batch_size"])
    if args.num_workers >= 0:
        num_workers = args.num_workers
    else:
        num_workers = int(cfg["training"].get("num_workers", 4))
    pin_memory = torch.cuda.is_available() and not args.no_pin_memory
    loader_kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kw
    )

    model = get_model(model_name, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loss_block = cfg.get("loss", {})
    loss_fn = CombinedSegLoss(
        pos_weight=float(loss_block.get("pos_weight", 18.0)),
        dice_weight=float(loss_block.get("lambda_dice", 0.5)),
    ).to(device)

    log_path = save_dir / "train_log.jsonl"
    best_val = math.inf
    epochs_no_improve = 0
    start_epoch = 0
    log_every = int(cfg["training"].get("log_every_steps", 100))

    resume_path = args.resume
    if resume_path:
        ckpt_path = save_dir / "gan_seg_last.pt" if resume_path == "last" else Path(resume_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["G"])
        start_epoch, best_val, epochs_no_improve = mask_train_resume_state(log_path)
        if start_epoch <= int(ckpt.get("epoch", start_epoch - 1)):
            start_epoch = int(ckpt["epoch"]) + 1
        print(
            f"[resume] {ckpt_path} -> epoch {start_epoch}, best_val={best_val:.6f}, "
            f"epochs_no_improve={epochs_no_improve}, num_workers={num_workers}",
            flush=True,
        )

    max_epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, max_epochs):
        model.train()
        sum_loss = 0.0
        n_steps = 0
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            n_steps += 1
            if i % log_every == 0:
                print(
                    f"[{model_name}] epoch {epoch} step {i} loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        sum_vloss = 0.0
        n_val = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                loss = loss_fn(model(img), mask)
                sum_vloss += loss.item() * img.size(0)
                n_val += img.size(0)
        val_loss = sum_vloss / max(1, n_val)
        train_loss = sum_loss / max(1, n_steps)
        print(
            f"[{model_name}] epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        append_jsonl(log_path, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        scheduler.step(val_loss)
        ckpt = {
            "G": model.state_dict(),
            "epoch": epoch,
            "config": full_config,
            "args": {
                "model": model_name,
                "processed": processed,
                "label_semantics": "xy_atms",
            },
            "val_seg": val_loss,
        }
        torch.save(ckpt, save_dir / "gan_seg_last.pt")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, save_dir / "gan_seg_best.pt")
            print(f"  -> new best val {best_val:.6f}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(cfg["training"]["patience"]):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    print(f"Done. {save_dir / 'gan_seg_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
