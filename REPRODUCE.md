# Reproduction guide (Hybrid-STEMSeg, xy_COM / sm_bfo_com)

## Environment

- Python 3.10+
- CUDA GPU recommended for training and figure regeneration
- ~2 GB disk for processed shards; +~2 GB if you add the raw `SmBFO_composition_series.npy`

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

1. **Preprocessed shards** (included): `data/processed/sm_bfo_com/` with `manifest.json`, train/val `.npz` shards.
2. **Raw source** (public download, ~2 GB):

| Resource | URL |
|----------|-----|
| Zenodo record | https://zenodo.org/record/4876786 |
| Direct `.npy` | https://zenodo.org/record/4876786/files/composition_series_dict_full.npy |
| Dataset DOI | https://doi.org/10.13139/ORNLNCCS/1773704 |
| Paper | https://doi.org/10.1038/s41524-020-00396-2 |

```bash
python scripts/download_sm_bfo_dataset.py
# or: python scripts/download_atomai_smbfo.py  (AtomAI wrapper, same file)
```

Rebuild COM masks (no centroid-aware shard fields):

```bash
python -m gan_seg.preprocess_dataset \
  --source data/SmBFO_composition_series.npy \
  --out data/processed/sm_bfo_com \
  --patch-size 256 --sigma 3.0 \
  --patches-per-key-train 400 --patches-per-key-val 100 \
  --seed 42
```

Do **not** pass `--centroid-targets` or `--use-atms` for this pipeline.

## Train Hybrid-STEMSeg (mask-only)

From repo root:

```bash
python -m gan_seg.train_benchmark \
  --config experiments/configs/com/hybrid_stemseg_sm_bfo_com_seed42.json
```

Other baselines: `unet`, `deeplabv3plus`, `segformer`, `hybrid-notransformer` — same config pattern with `"name"` changed, or use shipped weights under `gan_seg/checkpoints_benchmark/`.

## Train adversarial Hybrid GAN (optional ablation)

```bash
python -m gan_seg.train_adv --processed data/processed/sm_bfo_gan --epochs 100
```

Shipped checkpoints: `gan_seg/checkpoints_final_100ep/`, `checkpoints_final_pretrained/`.

## Evaluation

| Task | Command |
|------|---------|
| Mask IoU / pixel acc (benchmark ckpt) | `python -m gan_seg.eval_benchmark gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt` |
| Centroid F1 table (all models) | `python -m gan_seg.eval_centroids` |
| All-in-one script | `./evaluate_all.sh` |
| Noise F1 curve | `python -m gan_seg.make_noise_figures` |
| Multi-corruption reviewer suite | `python -m gan_seg.reviewer_study` |
| Corruption suite CSV | `python -m gan_seg.eval_corruption_suite` |

Detection F1: threshold mask → connected components → centroids → greedy match within **6 px** (see `gan_seg/eval_centroids.py`).

## Regenerate paper figures (Sm-BFO)

Requires real `data/processed/sm_bfo_com` val shards.

```bash
python -m gan_seg.export_smbfo_figure_panels \
  --processed data/processed/sm_bfo_com \
  --checkpoint gan_seg/checkpoints_benchmark/hybrid-nogan/gan_seg_best.pt \
  --out-dir paper_figures

python -m gan_seg.make_noise_figures --processed data/processed/sm_bfo_com
python -m gan_seg.export_gaussian_robustness_figure
```

See `paper_figures/README_*.txt` for figure-specific notes.

## JACS few-shot (optional, large checkpoints)

Scripts are included; **few-shot checkpoints are not** in this bundle (~5 GB). After downloading external TEM data:

```bash
python scripts/download_jacs_single_atom_tem_dataset.py
python -m gan_seg.train_jacs_fewshot ...
```

See `gan_seg/export_paper_fig10_jacs_examples.py` and `reports/jacs_fewshot/`.

## What this bundle does not reproduce

Listed in **`EXCLUDED.md`** (centroid-aware multitask, xy_atms fair eval, GemNet, geometry analysis, etc.).
