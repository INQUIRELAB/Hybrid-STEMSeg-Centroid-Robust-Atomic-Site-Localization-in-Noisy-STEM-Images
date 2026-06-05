#!/usr/bin/env python3
"""
Few-shot fine-tuning on JACS manual Pt/Fe STEM (Zenodo 10.5281/zenodo.5931544).

Loads Sm-BFO pretrained benchmark weights (--init-ckpt), trains on N random labeled
frames, evaluates centroid F1 / IoU on the remaining held-out frames.

Uses batch_size=1 (variable H×W after div8 crop). UNETR is not supported (fixed 256).

Multi-seed: pass --seeds 0,1,2,3,4 (single --model) to train each seed and write
mean ± std to reports/jacs_fewshot/. Or use gan_seg.run_jacs_fewshot_multiseed for
all benchmark models at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gan_seg.dataset_jacs import JacsSTEMDataset
from gan_seg.eval_cross_domain import centroid_metrics, iou_np
from gan_seg.jacs_data import DEFAULT_JACS_EXTRACTED, discover_pairs, few_shot_train_val_split, normalize_patch2d
from gan_seg.jacs_data import crop_div8, load_csv_coords_yx
from gan_seg.losses import CombinedSegLoss
from gan_seg.train_benchmark import get_model

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "jacs_fewshot"
DEFAULT_INIT = ROOT / "gan_seg" / "checkpoints_benchmark"


def set_batchnorm_eval(model: nn.Module) -> None:
    """Use pretrained BN stats (no batch-variance) — required for batch_size=1 on DeepLabV3+."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def load_pretrained(model: torch.nn.Module, ckpt_path: Path, device: torch.device, strict: bool = False) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    inc = model.load_state_dict(ckpt["G"], strict=strict)
    if inc.missing_keys or inc.unexpected_keys:
        print("load_state_dict:", inc)


@torch.no_grad()
def evaluate_records(
    model: torch.nn.Module,
    records: list[dict],
    device: torch.device,
    mask_sigma: float,
    centroid_match_px: float,
) -> list[dict]:
    import tifffile

    from gan_seg.dataset_patches import build_atom_mask_from_com

    model.eval()
    rows = []
    for rec in records:
        img = tifffile.imread(rec["image"])
        if img.ndim != 2:
            img = np.squeeze(img)
        img = np.asarray(img, dtype=np.float32)
        coords = load_csv_coords_yx(Path(rec["csv"]))
        try:
            img_c, coords_c = crop_div8(img, coords)
        except ValueError:
            continue
        if len(coords_c) < 1:
            continue
        h, w = img_c.shape
        mask = build_atom_mask_from_com(h, w, coords_c, 0, 0, sigma=mask_sigma)
        x = torch.from_numpy(normalize_patch2d(img_c)).to(device)
        logits = model(x.unsqueeze(0))
        pred = (logits > 0.0).float()[0, 0].cpu().numpy()
        pr, rc, f1 = centroid_metrics(pred, mask, centroid_match_px)
        iou = iou_np(pred, mask)
        rows.append(
            {
                "category": rec["category"],
                "image_id": rec["id"],
                "f1": f1,
                "iou": iou,
                "precision": pr,
                "recall": rc,
                "mean_pred_fg": float(pred.mean()),
                "logit_max": float(logits.max().item()),
            }
        )
    return rows


def run_fewshot_training(args: argparse.Namespace, quiet: bool = False) -> dict[str, Any]:
    """Run one few-shot experiment; return summary dict (includes val metrics)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = discover_pairs(args.extracted)
    if len(pairs) < 6:
        raise SystemExit(f"Need >= 6 JACS pairs, found {len(pairs)}")

    train_recs, val_recs = few_shot_train_val_split(pairs, args.n_shot, args.seed)
    train_ds = JacsSTEMDataset(train_recs, mask_sigma=args.mask_sigma)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    init_path = args.init_ckpt
    if init_path is None:
        init_path = DEFAULT_INIT / args.model / "gan_seg_best.pt"
    if not init_path.is_file():
        raise SystemExit(f"Missing init checkpoint: {init_path}")

    save_dir = args.save / f"n{args.n_shot}_seed{args.seed}" / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model, device)
    load_pretrained(model, init_path, device, strict=args.strict_init)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_fn = CombinedSegLoss(pos_weight=args.pos_weight, dice_weight=args.dice_weight).to(device)

    pre_rows = evaluate_records(model, val_recs, device, args.mask_sigma, args.centroid_match_px)
    pre_f1 = float(np.mean([r["f1"] for r in pre_rows])) if pre_rows else 0.0
    pre_iou = float(np.mean([r["iou"] for r in pre_rows])) if pre_rows else 0.0
    if not quiet:
        print(f"[baseline val] n={len(pre_rows)} mean F1={pre_f1:.4f} IoU={pre_iou:.4f}")

    best_val = math.inf
    epochs_no_improve = 0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_bn:
            set_batchnorm_eval(model)
        sum_loss = 0.0
        n_steps = 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            n_steps += 1
        train_loss = sum_loss / max(1, n_steps)

        model.eval()
        sum_v = 0.0
        nv = 0
        with torch.no_grad():
            for img, mask in DataLoader(
                JacsSTEMDataset(val_recs, mask_sigma=args.mask_sigma),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            ):
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                sum_v += loss_fn(logits, mask).item()
                nv += 1
        val_loss = sum_v / max(1, nv)
        scheduler.step(val_loss)
        if not quiet:
            print(
                f"[{args.model}] epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                flush=True,
            )

        ckpt = {
            "G": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "train_records": train_recs,
            "val_records": val_recs,
            "val_seg": val_loss,
        }
        torch.save(ckpt, save_dir / "gan_seg_last.pt")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt, save_dir / "gan_seg_best.pt")
            if not quiet:
                print(f"  -> best val_loss {best_val:.6f}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                if not quiet:
                    print(f"Early stop at epoch {epoch}", flush=True)
                break

    best_path = save_dir / "gan_seg_best.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["G"])
    post_rows = evaluate_records(model, val_recs, device, args.mask_sigma, args.centroid_match_px)
    post_f1 = float(np.mean([r["f1"] for r in post_rows])) if post_rows else 0.0
    post_iou = float(np.mean([r["iou"] for r in post_rows])) if post_rows else 0.0
    if not quiet:
        print(f"[after few-shot] n_val={len(post_rows)} mean F1={post_f1:.4f} IoU={post_iou:.4f}")

    summary = {
        "model": args.model,
        "n_shot": args.n_shot,
        "seed": args.seed,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "train_ids": [r["id"] for r in train_recs],
        "init_ckpt": str(init_path),
        "save_dir": str(save_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "val_baseline_mean_f1": pre_f1,
        "val_baseline_mean_iou": pre_iou,
        "val_finetuned_mean_f1": post_f1,
        "val_finetuned_mean_iou": post_iou,
        "dataset_doi": "https://doi.org/10.5281/zenodo.5931544",
    }
    (REPORT_DIR / f"summary_{args.model}_n{args.n_shot}_s{args.seed}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    csv_path = REPORT_DIR / f"val_metrics_{args.model}_n{args.n_shot}_s{args.seed}.csv"
    if post_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(post_rows[0].keys()))
            w.writeheader()
            w.writerows(post_rows)
    if not quiet:
        print(f"Wrote {save_dir}/gan_seg_best.pt")
        print(f"Wrote {csv_path}")

    return summary


def _parse_seeds(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extracted", type=Path, default=DEFAULT_JACS_EXTRACTED)
    p.add_argument("--model", type=str, required=True, choices=[
        "unet", "deeplabv3plus", "segformer", "hybrid-nogan", "hybrid-notransformer",
    ])
    p.add_argument("--n-shot", type=int, default=5, help="Training labeled frames")
    p.add_argument("--seed", type=int, default=42, help="Used if --seeds not set")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. 0,1,2,3,4). Trains --model once per seed; writes multiseed aggregate.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pos-weight", type=float, default=18.0)
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--mask-sigma", type=float, default=2.5)
    p.add_argument("--centroid-match-px", type=float, default=10.0)
    p.add_argument("--init-ckpt", type=Path, default=None)
    p.add_argument(
        "--save",
        type=Path,
        default=ROOT / "gan_seg" / "checkpoints_jacs_fewshot",
    )
    p.add_argument("--strict-init", action="store_true")
    p.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--quiet-multiseed",
        action="store_true",
        help="Less logging when running multiple seeds (only final aggregate printed).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = _parse_seeds(args.seeds) if args.seeds else []

    if not seeds:
        run_fewshot_training(args, quiet=False)
        return

    summaries: list[dict] = []
    per_run_rows: list[dict] = []
    quiet = args.quiet_multiseed or len(seeds) > 1
    for s in seeds:
        args.seed = s
        if not quiet:
            print(f"\n========== seed {s} ==========\n", flush=True)
        summ = run_fewshot_training(args, quiet=quiet)
        summaries.append(summ)
        per_run_rows.append(
            {
                "model": summ["model"],
                "n_shot": summ["n_shot"],
                "seed": summ["seed"],
                "n_val": summ["n_val"],
                "baseline_f1": summ["val_baseline_mean_f1"],
                "finetuned_f1": summ["val_finetuned_mean_f1"],
                "baseline_iou": summ["val_baseline_mean_iou"],
                "finetuned_iou": summ["val_finetuned_mean_iou"],
                "best_epoch": summ["best_epoch"],
            }
        )

    f1s = [float(s["val_finetuned_mean_f1"]) for s in summaries]
    ious = [float(s["val_finetuned_mean_iou"]) for s in summaries]
    agg = {
        "model": args.model,
        "n_shot": args.n_shot,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0,
        "baseline_f1_mean": float(np.mean([s["val_baseline_mean_f1"] for s in summaries])),
        "baseline_iou_mean": float(np.mean([s["val_baseline_mean_iou"] for s in summaries])),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_path = REPORT_DIR / f"multiseed_per_run_{args.model}_n{args.n_shot}.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)

    agg_path = REPORT_DIR / f"multiseed_aggregate_{args.model}_n{args.n_shot}.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    agg_csv = REPORT_DIR / f"multiseed_aggregate_{args.model}_n{args.n_shot}.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)

    print(
        f"[multiseed] {args.model} n_shot={args.n_shot} seeds={seeds} | "
        f"F1 = {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f} | "
        f"IoU = {agg['iou_mean']:.4f} ± {agg['iou_std']:.4f}",
        flush=True,
    )
    print(f"Wrote {per_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {agg_csv}")


if __name__ == "__main__":
    main()
