# Sm-BFO dataset — download, split, and preprocess

**No processed shards are stored in this GitHub repository** (they are ~1 GB and individual train files exceed GitHub’s 100 MB file limit). Follow the steps below to obtain the public raw data and build `data/processed/sm_bfo_com/` locally.

## 1. Download the raw dataset

Sm-doped BiFeO₃ combinatorial STEM library (Ziatdinov et al., *npj Comput. Mater.* 2020):

| Resource | URL |
|----------|-----|
| **Zenodo record** | https://zenodo.org/record/4876786 |
| **Direct download** | https://zenodo.org/record/4876786/files/composition_series_dict_full.npy |
| **Dataset DOI** | https://doi.org/10.13139/ORNLNCCS/1773704 |
| **Paper** | https://doi.org/10.1038/s41524-020-00396-2 |

**Please cite** [10.13139/ORNLNCCS/1773704](https://doi.org/10.13139/ORNLNCCS/1773704) when using this data.

### Option A — script (recommended)

```bash
python scripts/download_sm_bfo_dataset.py
```

Writes `data/SmBFO_composition_series.npy` (~2.1 GB) and `data/dataset_manifest.json`.

### Option B — manual

```bash
mkdir -p data
wget -O data/SmBFO_composition_series.npy \
  'https://zenodo.org/record/4876786/files/composition_series_dict_full.npy'
```

### Raw file format

- NumPy pickled `dict[str, dict]`
- Keys: composition ids, e.g. `Sm_0_0`, `Sm_7_3`, … (14 compositions in the public release)
- Per key, fields used here:
  - `main_image` — 2D float32 STEM image
  - `xy_COM` — `(N, 2)` atom centers as **(y, x)** in pixel coordinates

## 2. Train / validation split (composition keys)

Splits are at the **composition-key** level (no random patch leakage across train and val). Strategy: **stratified by COM density** with `seed=42`, `val_key_fraction=0.22`, and `Sm_7_0` forced into train.

**Exact keys used for shipped benchmark checkpoints** (also in `data/split_recipe_sm_bfo_com.json`):

| Split | Composition keys | Patches per key | Total patches |
|-------|------------------|-----------------|---------------|
| **Train** | `Sm_0_0`, `Sm_0_1`, `Sm_10_0`, `Sm_10_1`, `Sm_13_0`, `Sm_13_1`, `Sm_20_0`, `Sm_7_0`, `Sm_7_1`, `Sm_7_2`, `Sm_7_4` | 400 | **4400** |
| **Val** | `Sm_0_2`, `Sm_20_1`, `Sm_7_3` | 100 | **300** |

To use a different split, change `--val-key-fraction`, `--force-train-keys`, or `--seed` in preprocess (you must retrain models for comparable numbers).

## 3. Build processed patches

From the repository root:

```bash
python -m gan_seg.preprocess_dataset \
  --source data/SmBFO_composition_series.npy \
  --out data/processed/sm_bfo_com \
  --patch-size 256 \
  --sigma 3.0 \
  --patches-per-key-train 400 \
  --patches-per-key-val 100 \
  --val-key-fraction 0.22 \
  --force-train-keys Sm_7_0 \
  --seed 42
```

**Do not** pass `--centroid-targets` or `--use-atms` for this Hybrid-STEMSeg release.

### What preprocess produces

```
data/processed/sm_bfo_com/
  manifest.json          # split keys, counts, preprocess settings
  train/shard_*.npz      # 9 shards, 4400 patches
  val/shard_00000.npz    # 300 patches
```

Each `.npz` shard:

| Array | Shape | Description |
|-------|-------|-------------|
| `images` | `(N, 1, 256, 256)` | Per-patch z-score normalized STEM |
| `masks` | `(N, 1, 256, 256)` | Binary Gaussian atom masks from **xy_COM** (σ = 3 px) |

NaN/Inf in source images are zeroed; affected keys are listed in `manifest.json` under `keys_with_nan_cleaned`.

## 4. Verify before training / eval

```bash
# Should print train/val totals matching split_recipe_sm_bfo_com.json
python -c "
import json
from pathlib import Path
m = json.loads(Path('data/processed/sm_bfo_com/manifest.json').read_text())
print('train', m['total_train'], 'val', m['total_val'])
print('train keys', m['train_keys'])
print('val keys', m['val_keys'])
"

./evaluate_all.sh   # after checkpoints are in place
```

## 5. Disk space

| Artifact | Approx. size |
|----------|----------------|
| Raw `.npy` | ~2.1 GB |
| Processed `sm_bfo_com/` | ~1.0 GB |

You may delete the raw `.npy` after preprocessing if you only need the shard pipeline.
