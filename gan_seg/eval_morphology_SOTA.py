import torch
import numpy as np
from skimage.measure import label, regionprops
from gan_seg.dataset_preprocessed import ShardedPatchDataset

def evaluate_morphology(model, ds, device):
    eccentricities = []
    areas = []
    
    for i in range(100):
        img, mask = ds[i]
        with torch.no_grad():
            img_t = img.unsqueeze(0).to(device)
            logits = model(img_t)
            pred_mask = (logits > 0.0).float()[0, 0].cpu().numpy()
            
        lbl_pred = label(pred_mask > 0.5)
        for p in regionprops(lbl_pred):
            eccentricities.append(p.eccentricity)
            areas.append(p.area)
            
    mean_ecc = np.mean(eccentricities) if eccentricities else 1.0
    std_area = np.std(areas) if areas else 0.0
    mean_area = np.mean(areas) if areas else 0.0
    # Coefficient of Variation for Area (StdDev / Mean) -> normalized variance
    cov_area = (std_area / mean_area) if mean_area > 0 else 0.0
    
    return mean_ecc, cov_area

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

    # Ground Truth Baseline
    gt_eccs = []
    gt_areas = []
    for i in range(100):
        _, mask = ds[i]
        gt_mask = mask[0].numpy()
        lbl_gt = label(gt_mask > 0.5)
        for p in regionprops(lbl_gt):
            gt_eccs.append(p.eccentricity)
            gt_areas.append(p.area)
            
    gt_mean_ecc = np.mean(gt_eccs)
    gt_cov_area = np.std(gt_areas) / np.mean(gt_areas)
    
    print("\n--- Structural Microscopy Metrics ---")
    print(f"{'Model Architecture':<25} | {'Mean Eccentricity (Lower=Rounder)':<35} | {'Area Variance Coefficient':<25}")
    print("-" * 95)
    print(f"{'GROUND TRUTH (Target)':<25} | {gt_mean_ecc:.4f}                              | {gt_cov_area:.4f}")
    print("-" * 95)

    models = {
        "UNet": load_bm('unet', 'gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt'),
        "SegFormer": load_bm('segformer', 'gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt'),
        "GAN (ResNet-Initialized)": load_acc('gan_seg/checkpoints_final_pretrained/gan_seg_best.pt'),
        "Hybrid-STEMSeg": load_bm('hybrid-nogan', 'gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt'),
        "Original GAN (Scratch)": load_acc('gan_seg/checkpoints_final_100ep/gan_seg_last.pt')
    }
    
    for name, m in models.items():
        ecc, cov = evaluate_morphology(m, ds, device)
        print(f"{name:<25} | {ecc:.4f}                              | {cov:.4f}")

if __name__ == '__main__':
    main()
