import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from gan_seg.dataset_preprocessed import ShardedPatchDataset
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ShardedPatchDataset('data/processed/sm_bfo_com', split='val')

    idx = random.randint(0, len(ds)-1) # Pick a random shard in the validation set
    img, mask = ds[idx]
    img_t = img.unsqueeze(0).to(device)

    def load_benchmark_model(name, ckpt_path):
        from gan_seg.train_benchmark import get_model
        model = get_model(name, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["G"])
        model.eval()
        return model

    def load_acc_model(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        args = ckpt.get("args", {})
        if isinstance(args, dict) and args.get("pretrained", False):
            from gan_seg.model import PretrainedHybridGAN
            model = PretrainedHybridGAN(use_transformer=True).to(device)
            model.load_state_dict(ckpt["G"])
        else:
            from gan_seg.model import HybridUNetTransformerBinary
            d_model = args.get("d_model", 256) if isinstance(args, dict) else args.d_model
            model = HybridUNetTransformerBinary(d_model=d_model).to(device)
            model.load_state_dict(ckpt["G"])
        model.eval()
        return model

    @torch.no_grad()
    def get_pred(model, x):
        logits = model(x)
        pred = (logits > 0.0).float()
        return pred[0, 0].cpu().numpy()

    print("Loading UNet...")
    unet = load_benchmark_model('unet', 'gan_seg/checkpoints_benchmark/unet/gan_seg_best.pt')
    p_unet = get_pred(unet, img_t)
    
    print("Loading SegFormer...")
    segformer = load_benchmark_model('segformer', 'gan_seg/checkpoints_benchmark/segformer/gan_seg_best.pt')
    p_segformer = get_pred(segformer, img_t)
    
    print("Loading Hybrid No-GAN...")
    hybrid_nogan = load_benchmark_model('hybrid-nogan', 'gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt')
    p_hybrid_nogan = get_pred(hybrid_nogan, img_t)
    
    print("Loading Original GAN...")
    orig_gan = load_acc_model('gan_seg/checkpoints_final_100ep/gan_seg_best.pt')
    p_orig = get_pred(orig_gan, img_t)
    
    print("Loading GAN (ResNet-Initialized)...")
    pre_gan = load_acc_model('gan_seg/checkpoints_final_pretrained/gan_seg_best.pt')
    p_pre = get_pred(pre_gan, img_t)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    axes[0].imshow(img[0].numpy(), cmap='gray')
    axes[0].set_title('Raw Input Image')
    axes[0].axis('off')

    axes[1].imshow(mask[0].numpy(), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(p_unet, cmap='gray')
    axes[2].set_title('Standard U-Net\n(0.88 IoU)')
    axes[2].axis('off')

    axes[3].imshow(p_segformer, cmap='gray')
    axes[3].set_title('SegFormer\n(0.84 IoU)')
    axes[3].axis('off')

    axes[4].imshow(p_hybrid_nogan, cmap='gray')
    axes[4].set_title('HybridUNet [No GAN]\n(0.85 IoU)')
    axes[4].axis('off')

    axes[5].imshow(p_orig, cmap='gray')
    axes[5].set_title('Hybrid GAN [From Scratch]\n(0.60 IoU)')
    axes[5].axis('off')

    axes[6].imshow(p_pre, cmap='gray')
    axes[6].set_title('GAN [ResNet-Initialized]\n(0.70 IoU)')
    axes[6].axis('off')

    axes[7].axis('off') # empty plot for grid symmetry

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print('Plot saved to model_comparison.png')

if __name__ == '__main__':
    main()
