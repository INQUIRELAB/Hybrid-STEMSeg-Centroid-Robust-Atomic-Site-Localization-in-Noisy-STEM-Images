# Hybrid STEM segmentation (Sm-BFO) — reproduction bundle

Mask-only **Hybrid-STEMSeg** (Hybrid U-Net + Transformer; checkpoint key `hybrid-nogan`) and benchmark baselines on simulated Sm-BFO, trained with **center-of-mass (xy_COM)** Gaussian atom masks (`sm_bfo_com`).

This repository slice is for the **Hybrid-STEMSeg** line of work (mask-only segmentation on `sm_bfo_com`). It intentionally **does not** include the separate **centroid-aware / xy_atms** research (heatmap–offset multitask, fair xy_atms tables, GemNet/AtomAI centroid heads, or recent geometry analyses). See `EXCLUDED.md`.

## Dataset

| | |
|---|---|
| **Processed (in repo)** | `data/processed/sm_bfo_com/` — train/val patches + xy_COM masks |
| **Raw (download)** | [Zenodo 4876786](https://zenodo.org/record/4876786) · [DOI 10.13139/ORNLNCCS/1773704](https://doi.org/10.13139/ORNLNCCS/1773704) |

```bash
python scripts/download_sm_bfo_dataset.py   # → data/SmBFO_composition_series.npy
```

Full provenance, citations, and preprocess steps: **`data/README.md`**.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: download raw data and rebuild shards (see data/README.md)
# python scripts/download_sm_bfo_dataset.py
# python -m gan_seg.preprocess_dataset --source data/SmBFO_composition_series.npy --out data/processed/sm_bfo_com

# 2) Reproduce benchmark metrics (IoU + mask-derived centroid F1)
./evaluate_all.sh

# 3) Noise / robustness figures (optional, GPU recommended)
python -m gan_seg.make_noise_figures --processed data/processed/sm_bfo_com
python -m gan_seg.reviewer_study --processed data/processed/sm_bfo_com
```

Full step-by-step commands: **`REPRODUCE.md`**.

## What is included

| Component | Location |
|-----------|----------|
| Hybrid + baseline weights (COM-trained) | `gan_seg/checkpoints_benchmark/` |
| GAN ablation checkpoints | `gan_seg/checkpoints_final_100ep/`, `checkpoints_final_pretrained/` |
| Preprocessed val/train shards | `data/processed/sm_bfo_com/` |
| Training / eval / figure scripts | `gan_seg/`, `evaluate_all.sh` |
| Published metric tables | `reports/reviewer_suite/`, `reports/noise_ablation/`, … |
| COM-era configs | `experiments/configs/com/` |

## Citation

If you use this code, cite your manuscript and note that detection F1 is a **post-hoc** mask-centroid metric (not a centroid-regression training objective). See `paper_figures/PUBLICATION_NOTICE.txt`.

## License

Add your license file before public release. Third-party weights (e.g. SegFormer `nvidia/mit-b0`, smp ResNet-34 backbones) remain subject to their upstream licenses.
