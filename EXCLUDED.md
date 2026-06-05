# Intentionally excluded from this upload

These belong to **separate centroid-aware / xy_atms research** and are not part of this GitHub bundle.

## Code not shipped

- `gan_seg/train_centroid_aware.py`, `model_centroid.py`, `dataset_centroid.py`
- `gan_seg/centroid_losses.py`, `centroid_inference.py`, `inference_centroid.py`
- `gan_seg/eval_centroid_aware.py`, `eval_xy_atms_fair.py`, `eval_label_domain_cross.py`
- `gan_seg/run_robustness_xy_atms.py`, `gan_seg/analyze_hybrid_nogan_geometry.py`, `gan_seg/geometry_metrics.py`
- `gan_seg/models_atomai_centroid.py`, `gan_seg/train_atomai_xy_atms.py`, `gan_seg/train_atomsegnet_xy_atms.py`
- `gan_seg/gemnet_*.py`, `third_party/gemnet_pytorch/`
- `experiments/configs/xy_atms/*` (fair xy_atms grid)
- `scripts/run_centroid_aware_nohup.sh`, `run_xy_atms_*.sh`, `run_atomai_*_centroid_*.sh`

## Checkpoints not shipped

- `gan_seg/checkpoints_centroid_aware/`
- `gan_seg/checkpoints_runs/sm_bfo_centroid_*`
- AtomAI / AtomSegNet / GemNet xy_atms runs

## Reports not shipped

- `reports/centroid_aware/`
- `reports/runs/xy_atms_*`, `label_domain_cross_*`, `hybrid_nogan_geometry_analysis/`

## Data not shipped

- `data/processed/sm_bfo_centroid/` (xy_atms expert labels)

If you need the centroid-aware line, use the full private project tree or a future dedicated release.
