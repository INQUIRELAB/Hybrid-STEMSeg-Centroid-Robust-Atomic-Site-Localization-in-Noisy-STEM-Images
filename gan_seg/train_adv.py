#!/usr/bin/env python3
"""
Train HybridUNetTransformerBinary + PatchDiscriminator (adversarial binary seg).

Regularization / stability (defaults):
  - Segmentation: pos-weighted BCE + soft Dice (class imbalance + better masks)
  - Adversarial: warmup (no adv) + linear ramp; D updated every d_every steps
  - Adam weight decay on G and D; optional H/V flips on train patches

Example:
  python -m gan_seg.train_adv --processed data/processed/sm_bfo_gan --epochs 100 --patience 12
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gan_seg.dataset_patches import SmBFOPatchDataset
from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.losses import CombinedSegLoss
from gan_seg.model import HybridUNetTransformerBinary, PatchDiscriminator


def effective_lambda_adv(epoch: int, warmup: int, ramp: int, lam_max: float) -> float:
    if lam_max <= 0:
        return 0.0
    if epoch < warmup:
        return 0.0
    if ramp <= 0:
        return lam_max
    t = (epoch - warmup + 1) / float(ramp)
    return lam_max * max(0.0, min(1.0, t))


def augment_flips(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand((), device=img.device) < 0.5:
        img = torch.flip(img, (-1,))
        mask = torch.flip(mask, (-1,))
    if torch.rand((), device=img.device) < 0.5:
        img = torch.flip(img, (-2,))
        mask = torch.flip(mask, (-2,))
    return img, mask


@torch.no_grad()
def validate_epoch(
    G: nn.Module,
    loader: DataLoader,
    device: torch.device,
    seg_loss_fn: CombinedSegLoss,
    lambda_adv: float,
    D: nn.Module,
    bce_logits: nn.BCEWithLogitsLoss,
) -> tuple[float, float, float]:
    """Mean (val_seg BCE+Dice, val_g total, val_d) over validation loader."""
    G.eval()
    D.eval()
    sum_seg = 0.0
    sum_g = 0.0
    sum_d = 0.0
    n = 0
    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)
        bs = img.size(0)
        fake_logits = G(img)
        loss_seg = seg_loss_fn(fake_logits, mask)
        fake_prob = torch.sigmoid(fake_logits)
        loss_adv = torch.zeros((), device=device)
        if lambda_adv > 0:
            pred_fake = D(img, fake_prob)
            loss_adv = bce_logits(pred_fake, torch.ones_like(pred_fake))
        loss_g = loss_seg + lambda_adv * loss_adv
        pred_fake_d = D(img, fake_prob.detach())
        pred_real = D(img, mask)
        loss_d = 0.5 * (
            bce_logits(pred_real, torch.ones_like(pred_real) * 0.9)
            + bce_logits(pred_fake_d, torch.zeros_like(pred_fake_d))
        )
        sum_seg += loss_seg.item() * bs
        sum_g += loss_g.item() * bs
        sum_d += loss_d.item() * bs
        n += bs
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return sum_seg / n, sum_g / n, sum_d / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=str, default="")
    p.add_argument("--data", type=str, default="data/SmBFO_composition_series.npy")
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--lambda-adv", type=float, default=0.05, help="Max adversarial weight after ramp")
    p.add_argument("--adv-warmup-epochs", type=int, default=4, help="Epochs with lambda_adv=0")
    p.add_argument(
        "--adv-ramp-epochs",
        type=int,
        default=6,
        help="Linearly ramp lambda_adv from 0 to --lambda-adv over these epochs after warmup",
    )
    p.add_argument(
        "--d-every",
        type=int,
        default=2,
        help="Update discriminator every this many train steps (>=1)",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=18.0, help="BCE pos_weight for rare foreground")
    p.add_argument("--dice-weight", type=float, default=0.5, help="Multiplier on soft Dice term")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save", type=str, default="gan_seg/checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--no-augment", action="store_true", help="Disable random H/V flips")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained resnet backbone")
    return p.parse_args()


def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    device = torch.device(args.device)
    proc_root: Path | None = None

    if args.processed:
        proc_root = Path(args.processed)
        man = proc_root / "manifest.json"
        if not man.is_file():
            raise SystemExit(
                f"Missing processed manifest: {man.resolve()}\n"
                f"Run: python -m gan_seg.preprocess_dataset --out {proc_root}"
            )
        with open(man, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        ps = int(manifest["patch_size"])
    else:
        data_path = Path(args.data)
        if not data_path.is_file():
            raise SystemExit(f"Missing data file: {data_path.resolve()}")
        ps = args.patch
        if ps % 8 != 0:
            raise SystemExit("patch size must be divisible by 8")

    if ps % 8 != 0:
        raise SystemExit("patch size must be divisible by 8")

    if args.processed:
        assert proc_root is not None
        ds = ShardedPatchDataset(str(proc_root), split="train")
        val_ds = ShardedPatchDataset(str(proc_root), split="val")
    else:
        ds = SmBFOPatchDataset(str(data_path), patch_size=ps)
        val_ds = None

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    track_best = val_loader is not None
    use_early_stop = args.patience > 0 and track_best
    if args.patience > 0 and not track_best:
        print(
            "Early stopping disabled: no validation set (use --processed with val shards).",
            flush=True,
        )

    if getattr(args, "pretrained", False):
        from gan_seg.model import PretrainedHybridGAN
        G = PretrainedHybridGAN(use_transformer=True).to(device)
    else:
        G = HybridUNetTransformerBinary(d_model=args.d_model).to(device)
    D = PatchDiscriminator().to(device)

    wd = args.weight_decay
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=wd)
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=wd)

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=5)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode='min', factor=0.5, patience=5)

    seg_loss_fn = CombinedSegLoss(pos_weight=args.pos_weight, dice_weight=args.dice_weight).to(
        device
    )
    bce_logits = nn.BCEWithLogitsLoss()

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_g = math.inf
    epochs_no_improve = 0
    best_epoch = -1
    d_every = max(1, args.d_every)

    for epoch in range(args.epochs):
        lam = effective_lambda_adv(
            epoch, args.adv_warmup_epochs, args.adv_ramp_epochs, args.lambda_adv
        )
        print(
            f"epoch {epoch}: lambda_adv={lam:.5f} (warmup {args.adv_warmup_epochs}, ramp {args.adv_ramp_epochs}, max {args.lambda_adv})",
            flush=True,
        )

        G.train()
        D.train()
        loss_d_report = 0.0

        for i, (img, mask) in enumerate(loader):
            img = img.to(device)
            mask = mask.to(device)
            if not args.no_augment:
                img, mask = augment_flips(img, mask)

            train_d = i % d_every == 0
            if train_d:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_logits_d = G(img)
                fake_prob_d = torch.sigmoid(fake_logits_d)
                pred_real = D(img, mask)
                pred_fake = D(img, fake_prob_d.detach())
                loss_d_real = bce_logits(pred_real, torch.ones_like(pred_real) * 0.9)
                loss_d_fake = bce_logits(pred_fake, torch.zeros_like(pred_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                opt_d.step()
                loss_d_report = loss_d.item()

            opt_g.zero_grad(set_to_none=True)
            fake_logits = G(img)
            fake_prob = torch.sigmoid(fake_logits)
            loss_seg = seg_loss_fn(fake_logits, mask)
            if lam > 0:
                pred_fake_g = D(img, fake_prob)
                loss_adv = bce_logits(pred_fake_g, torch.ones_like(pred_fake_g))
                loss_g = loss_seg + lam * loss_adv
            else:
                loss_adv = torch.zeros((), device=device)
                loss_g = loss_seg
            loss_g.backward()
            opt_g.step()

            if i % 50 == 0:
                adv_val = loss_adv.item() if lam > 0 else 0.0
                print(
                    f"epoch {epoch} step {i}  "
                    f"loss_g {loss_g.item():.4f} (seg {loss_seg.item():.4f} adv {adv_val:.4f})  "
                    f"loss_d {loss_d_report:.4f}  lam {lam:.4f}",
                    flush=True,
                )

            if args.steps > 0 and (i + 1) >= args.steps:
                break

        val_seg = val_g = val_d = float("nan")
        if val_loader is not None:
            val_seg, val_g, val_d = validate_epoch(
                G, val_loader, device, seg_loss_fn, lam, D, bce_logits
            )
            print(
                f"epoch {epoch} done  val_seg {val_seg:.4f} (bce+dice)  val_g {val_g:.4f}  val_d {val_d:.4f}",
                flush=True,
            )
            if not math.isnan(val_g):
                scheduler_g.step(val_g)
                scheduler_d.step(val_g)

        last_ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "epoch": epoch,
            "val_seg": val_seg,
            "val_g": val_g,
            "lambda_adv_effective": lam,
            "args": vars(args),
        }
        torch.save(last_ckpt, save_dir / "gan_seg_last.pt")

        if track_best and not math.isnan(val_g):
            if val_g < best_val_g - args.min_delta:
                best_val_g = val_g
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(last_ckpt, save_dir / "gan_seg_best.pt")
                print(f"  -> new best val_g {best_val_g:.6f} (epoch {epoch})", flush=True)
            elif use_early_stop:
                # Only accumulate patience drops after the critic has fully activated
                if epoch > (args.adv_warmup_epochs + args.adv_ramp_epochs):
                    epochs_no_improve += 1
                print(
                    f"  -> no improvement ({epochs_no_improve}/{args.patience} vs best val_g {best_val_g:.6f} @ epoch {best_epoch})",
                    flush=True,
                )

        if use_early_stop and epochs_no_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch} (best val_g {best_val_g:.6f} at epoch {best_epoch})",
                flush=True,
            )
            break

    print("done.", flush=True)


if __name__ == "__main__":
    main()
