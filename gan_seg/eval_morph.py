import torch
import numpy as np
import scipy.ndimage as ndimage
from torch.utils.data import DataLoader
from gan_seg.dataset_preprocessed import ShardedPatchDataset
from gan_seg.model import HybridUNetTransformerBinary

@torch.no_grad()
def evaluate_accuracy_morph(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating Morphological Cleanup!")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt['args']
    
    G = HybridUNetTransformerBinary(d_model=args['d_model']).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    
    proc_root = "data/processed/sm_bfo_com"
    val_ds = ShardedPatchDataset(str(proc_root), split="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    
    total_pixels = 0
    correct_pixels = 0
    intersection = 0
    union = 0
    
    for img, mask in val_loader:
        img = img.to(device)
        mask = mask.to(device)
        
        logits = G(img)
        preds = (logits > 0.0).cpu().numpy().astype(float)
        mask_np = mask.cpu().numpy()
        
        # Morphological Closing & Opening to clean up the predictions
        for b in range(preds.shape[0]):
            struct = ndimage.generate_binary_structure(2, 2)
            c = ndimage.binary_closing(preds[b, 0], structure=struct, iterations=2)
            c = ndimage.binary_opening(c, structure=struct, iterations=1)
            preds[b, 0] = c
            
        preds_t = torch.from_numpy(preds).to(device)
        
        total_pixels += mask.numel()
        correct_pixels += (preds_t == mask).sum().item()
        
        intersection += (preds_t * mask).sum().item()
        union += (preds_t + mask > 0).sum().item()
        
    acc = correct_pixels / total_pixels
    iou = intersection / union if union > 0 else 0
    print(f"[Morphological Cleaning] Pixel Accuracy: {acc:.4f}, IoU: {iou:.4f}")

if __name__ == "__main__":
    evaluate_accuracy_morph("gan_seg/checkpoints_adversarial_v2_small/gan_seg_best.pt")
