#!/usr/bin/env python3
"""Save one val patch: input image, ground-truth mask, predicted probability map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.model import HybridUNetTransformerBinary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=str, default="data/processed/sm_bfo_gan_v2")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="gan_seg/checkpoints_ablation_noadv_v2/gan_seg_best.pt",
    )
    p.add_argument("--out", type=str, default="gan_seg/example_gt_pred.png")
    p.add_argument("--index", type=int, default=0, help="Index in val set")
    p.add_argument("--d-model", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    d_model = ckpt.get("args", {}).get("d_model", args.d_model)

    G = HybridUNetTransformerBinary(d_model=d_model).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    ds = ShardedPatchDataset(args.processed, split="val")
    img, mask = ds[args.index]
    with torch.no_grad():
        logits = G(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)
    prob = prob.cpu().numpy()[0, 0]
    img_np = img.numpy()[0]
    mask_np = mask.numpy()[0]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    # Normalized patch: show with robust scaling for display
    lo, hi = np.percentile(img_np, [2, 98])
    axes[0].imshow(np.clip(img_np, lo, hi), cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Input (patch, z-scored)")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")

    axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Prediction (sigmoid prob)")
    axes[2].axis("off")

    axes[3].imshow(np.clip(img_np, lo, hi), cmap="gray", vmin=lo, vmax=hi)
    axes[3].imshow(prob, cmap="cool", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("Overlay: pred on image")
    axes[3].axis("off")

    epoch = ckpt.get("epoch", "?")
    val_seg = ckpt.get("val_seg", float("nan"))
    fig.suptitle(
        f"Checkpoint epoch={epoch}  val_seg(bce+dice)={val_seg:.4f}  val index={args.index}",
        fontsize=10,
    )
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
