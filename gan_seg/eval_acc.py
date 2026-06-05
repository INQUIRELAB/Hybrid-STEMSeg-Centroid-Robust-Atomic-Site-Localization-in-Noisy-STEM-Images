import torch
import numpy as np
from torch.utils.data import DataLoader
from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.model import HybridUNetTransformerBinary

@torch.no_grad()
def main():
    import sys
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "gan_seg/checkpoints_adversarial_v2_small/gan_seg_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    if args.get("pretrained", False):
        from gan_seg.model import PretrainedHybridGAN
        G = PretrainedHybridGAN(use_transformer=True).to(device)
    else:
        from gan_seg.model import HybridUNetTransformerBinary
        G = HybridUNetTransformerBinary(d_model=args["d_model"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    
    proc_root = args["processed"]
    val_ds = ShardedPatchDataset(str(proc_root), split="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    total_pixels = 0
    correct_pixels = 0
    intersection = 0
    union = 0
    
    print("Evaluating over validation set...")
    for img, mask in val_loader:
        img = img.to(device)
        mask = mask.to(device)
        
        logits = G(img)
        preds = (logits > 0.0).float()
        
        total_pixels += mask.numel()
        correct_pixels += (preds == mask).sum().item()
        
        intersection += (preds * mask).sum().item()
        union += (preds + mask > 0).sum().item()
        
    acc = correct_pixels / total_pixels
    iou = intersection / union if union > 0 else 0
    
    print(f"Pixel Accuracy: {acc:.4f}")
    print(f"IoU: {iou:.4f}")

if __name__ == "__main__":
    main()
