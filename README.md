# Hybrid STEM segmentation (Sm-BFO) — reproduction bundle

Mask-only **Hybrid-STEMSeg** (Hybrid U-Net + Transformer; checkpoint key `hybrid-nogan`) and benchmark baselines on simulated Sm-BFO, trained with **center-of-mass (xy_COM)** Gaussian atom masks (`sm_bfo_com`).

This repository slice is for the **Hybrid-STEMSeg** line of work (mask-only segmentation on `sm_bfo_com`). It intentionally **does not** include the separate **centroid-aware / xy_atms** research. See `EXCLUDED.md`.

## Dataset (download + local preprocess)

Processed patches are **not** in this repo (~1 GB; train shards exceed GitHub’s per-file limit). Use the public raw dataset and our split recipe:

| Step | Action |
|------|--------|
| **Download** | [Zenodo 4876786](https://zenodo.org/record/4876786) · [DOI 10.13139/ORNLNCCS/1773704](https://doi.org/10.13139/ORNLNCCS/1773704) |
| **Split recipe** | `data/split_recipe_sm_bfo_com.json` (train/val composition keys, seed 42) |
| **Full guide** | **`data/README.md`** |

```bash
python scripts/download_sm_bfo_dataset.py
python -m gan_seg.preprocess_dataset \
  --source data/SmBFO_composition_series.npy \
  --out data/processed/sm_bfo_com \
  --patch-size 256 --sigma 3.0 \
  --patches-per-key-train 400 --patches-per-key-val 100 \
  --val-key-fraction 0.22 --force-train-keys Sm_7_0 --seed 42
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Data (required before eval/train)
python scripts/download_sm_bfo_dataset.py
python -m gan_seg.preprocess_dataset \
  --source data/SmBFO_composition_series.npy \
  --out data/processed/sm_bfo_com \
  --patch-size 256 --sigma 3.0 \
  --patches-per-key-train 400 --patches-per-key-val 100 \
  --val-key-fraction 0.22 --force-train-keys Sm_7_0 --seed 42

# 2) Benchmark metrics (IoU + mask-derived centroid F1)
./evaluate_all.sh

# 3) Noise / robustness figures (optional, GPU)
python -m gan_seg.make_noise_figures --processed data/processed/sm_bfo_com
python -m gan_seg.reviewer_study --processed data/processed/sm_bfo_com
```

More detail: **`REPRODUCE.md`**.

## What is included

| Component | Location |
|-----------|----------|
| Hybrid + baseline weights (COM-trained) | `gan_seg/checkpoints_benchmark/` |
| GAN ablation checkpoints | `gan_seg/checkpoints_final_100ep/`, `checkpoints_final_pretrained/` |
| Dataset links + split recipe | `data/README.md`, `data/split_recipe_sm_bfo_com.json` |
| Download script | `scripts/download_sm_bfo_dataset.py` |
| Training / eval / figure scripts | `gan_seg/`, `evaluate_all.sh` |
| Published metric tables | `reports/reviewer_suite/`, `reports/noise_ablation/`, … |
| COM-era configs | `experiments/configs/com/` |

## Citation

If you use this code, cite your manuscript and note that detection F1 is a **post-hoc** mask-centroid metric (not a centroid-regression training objective). See `paper_figures/PUBLICATION_NOTICE.txt`.

## License

Add your license file before public release. Third-party weights (e.g. SegFormer `nvidia/mit-b0`, smp ResNet-34 backbones) remain subject to their upstream licenses.
