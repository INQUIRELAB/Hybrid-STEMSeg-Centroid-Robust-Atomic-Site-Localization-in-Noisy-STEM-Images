import torch
import numpy as np
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from gan_seg.dataset_preprocessed import ShardedPatchDataset

def evaluate_centroids(model, ds, device, distance_threshold=6.0):
    TP = 0
    FP = 0
    FN = 0
    
    # Evaluate 100 validation patches for speed (~3200 atoms total)
    for i in range(100):
        img, mask = ds[i]
        
        # Ground Truth Centroids
        gt_mask = mask[0].numpy()
        lbl_gt = label(gt_mask > 0.5)
        gt_coords = np.array([p.centroid for p in regionprops(lbl_gt)])
        
        # Predicted Centroids
        with torch.no_grad():
            img_t = img.unsqueeze(0).to(device)
            logits = model(img_t)
            pred_mask = (logits > 0.0).float()[0, 0].cpu().numpy()
            
        lbl_pred = label(pred_mask > 0.5)
        pred_coords = np.array([p.centroid for p in regionprops(lbl_pred)])
        
        if len(pred_coords) == 0 and len(gt_coords) == 0:
            continue
        if len(pred_coords) == 0:
            FN += len(gt_coords)
            continue
        if len(gt_coords) == 0:
            FP += len(pred_coords)
            continue
            
        dists = cdist(pred_coords, gt_coords)
        matched_gt = set()
        matched_pred = set()
        
        for p_idx in range(len(pred_coords)):
            closest_gt = np.argmin(dists[p_idx])
            dist = dists[p_idx, closest_gt]
            if dist <= distance_threshold and closest_gt not in matched_gt:
                matched_gt.add(closest_gt)
                matched_pred.add(p_idx)
                TP += 1
                
        FP += len(pred_coords) - len(matched_pred)
        FN += len(gt_coords) - len(matched_gt)
        
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    return precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset('data/processed/sm_bfo_com', split='val')
    
    def load_bm(name, p):
        from gan_seg.train_benchmark import get_model
        m = get_model(name, device)
        m.load_state_dict(torch.load(p, map_location=device)["G"])
        m.eval()
        return m

    def load_acc(p):
        ckpt = torch.load(p, map_location=device)
        args = ckpt.get("args", {})
        if isinstance(args, dict) and args.get("pretrained", False):
            from gan_seg.model import PretrainedHybridGAN
            m = PretrainedHybridGAN(use_transformer=True).to(device)
        else:
            from gan_seg.model import HybridUNetTransformerBinary
            d = args.get("d_model", 256) if isinstance(args, dict) else args.d_model
            m = HybridUNetTransformerBinary(d_model=d).to(device)
        m.load_state_dict(ckpt["G"])
        m.eval()
        return m

    print("Loading models onto GPU...")
    models = {
        "UNet": load_bm('unet', 'gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt'),
        "SegFormer": load_bm('segformer', 'gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt'),
        "Hybrid-STEMSeg": load_bm('hybrid-nogan', 'gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt'),
        "Original GAN (Scratch)": load_acc('gan_seg/checkpoints_final_100ep/gan_seg_last.pt'),
        "GAN (ResNet-Initialized)": load_acc('gan_seg/checkpoints_final_pretrained/gan_seg_best.pt')
    }
    
    print("\nExtracted Morphological Centroids F1-Score:")
    print(f"{'Model Architecture':<25} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 62)
    for name, m in models.items():
        p, r, f1 = evaluate_centroids(m, ds, device, distance_threshold=6.0)
        print(f"{name:<25} | {p:.4f}    | {r:.4f}    | {f1:.4f}")

if __name__ == '__main__':
    main()
