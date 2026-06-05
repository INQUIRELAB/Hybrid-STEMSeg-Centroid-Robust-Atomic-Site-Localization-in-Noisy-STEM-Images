#!/usr/bin/env python3
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from gan_seg.dataset_preprocessed import ShardedPatchDataset

def get_model(name, device):
    if name == "unet":
        import segmentation_models_pytorch as smp
        return smp.Unet('resnet34', in_channels=1, classes=1).to(device)
    elif name == "deeplabv3plus":
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3Plus('resnet34', in_channels=1, classes=1).to(device)
    elif name == "unetr":
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
    elif name == "segformer":
        from transformers import SegformerForSemanticSegmentation
        import torch.nn as nn
        class SegFormerWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=1, num_channels=1, ignore_mismatched_sizes=True)
            def forward(self, x):
                out = self.m(pixel_values=x)
                return torch.nn.functional.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return SegFormerWrapper().to(device)
    elif name == "hybrid-nogan":
        from gan_seg.model import HybridUNetTransformerBinary
        return HybridUNetTransformerBinary(d_model=256, use_transformer=True).to(device)
    elif name == "hybrid-notransformer":
        from gan_seg.model import HybridUNetTransformerBinary
        return HybridUNetTransformerBinary(d_model=256, use_transformer=False).to(device)

@torch.no_grad()
def evaluate_benchmark(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    
    model = get_model(args["model"], device)
    model.load_state_dict(ckpt["G"])
    model.eval()
    
    val_ds = ShardedPatchDataset(args["processed"], split="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    tp = 0
    cp = 0
    inter = 0
    union = 0

    for img, mask in val_loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        preds = (logits > 0.0).float()
        
        tp += mask.numel()
        cp += (preds == mask).sum().item()
        inter += (preds * mask).sum().item()
        union += (preds + mask > 0).sum().item()

    acc = cp / tp
    iou = inter / union if union > 0 else 0
    print(f"[{args['model']}] Pixel Accuracy: {acc:.4f} | IoU: {iou:.4f}")

if __name__ == "__main__":
    evaluate_benchmark(sys.argv[1])
