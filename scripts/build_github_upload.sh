#!/usr/bin/env bash
# Build github_upload/ — Hybrid-STEMSeg (mask / COM) reproduction bundle.
# Excludes centroid-aware training, xy_atms fair study, GemNet/AtomAI multitask, and
# recent geometry / centroid analyses.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/github_upload"
STAGING="${ROOT}/.github_upload_staging"

rm -rf "${STAGING}"
mkdir -p "${STAGING}"

echo "==> Staging code and configs"
mkdir -p "${STAGING}/gan_seg" "${STAGING}/scripts" "${STAGING}/experiments/configs/com"
mkdir -p "${STAGING}/data" "${STAGING}/reports" "${STAGING}/paper_figures" "${STAGING}/figures/smbfo_panels"

# --- gan_seg Python (mask-only / hybrid pipeline) ---
GAN_SEG_PY=(
  __init__.py
  model.py
  transformer_bottleneck.py
  losses.py
  dataset_preprocessed.py
  dataset_patches.py
  preprocess_dataset.py
  centroid_targets.py
  experiment_io.py
  model_labels.py
  train_benchmark.py
  train_adv.py
  eval_benchmark.py
  eval_acc.py
  eval_centroids.py
  eval_noise_SOTA.py
  eval_morphology_SOTA.py
  eval_morph.py
  eval_corruption_suite.py
  reviewer_study.py
  make_noise_figures.py
  export_paper_fig7_noise_examples.py
  export_smbfo_figure_panels.py
  export_smbfo_corruption_panels.py
  export_gaussian_robustness_figure.py
  export_noise_ablation_f1_curve.py
  eval_cross_domain.py
  eval_jacs_external.py
  train_jacs_fewshot.py
  run_jacs_fewshot_multiseed.py
  jacs_fewshot_pairwise_stats.py
  generate_jacs_fewshot_supp_tables.py
  export_paper_fig9_jacs_fewshot_curves.py
  export_paper_fig10_jacs_examples.py
  dataset_jacs.py
  jacs_data.py
  paper_export_requirements.py
  plot_comparison.py
  visualize_gt_pred.py
  stack_npz.py
)
for f in "${GAN_SEG_PY[@]}"; do
  cp "${ROOT}/gan_seg/${f}" "${STAGING}/gan_seg/${f}"
done

# --- Checkpoints (benchmark + GAN ablations; no centroid_aware / checkpoints_runs) ---
echo "==> Staging checkpoints (~500 MB benchmarks + GAN)"
for d in unet deeplabv3plus segformer hybrid-nogan hybrid-notransformer; do
  if [[ -d "${ROOT}/gan_seg/checkpoints_benchmark/${d}" ]]; then
    mkdir -p "${STAGING}/gan_seg/checkpoints_benchmark"
    cp -a "${ROOT}/gan_seg/checkpoints_benchmark/${d}" \
      "${STAGING}/gan_seg/checkpoints_benchmark/${d}"
  fi
done
for d in checkpoints_final_100ep checkpoints_final_pretrained; do
  if [[ -d "${ROOT}/gan_seg/${d}" ]]; then
    cp -a "${ROOT}/gan_seg/${d}" "${STAGING}/gan_seg/${d}"
  fi
done

# --- Processed COM dataset (required for main Sm-BFO results) ---
echo "==> Staging data/processed/sm_bfo_com"
if [[ -d "${ROOT}/data/processed/sm_bfo_com" ]]; then
  mkdir -p "${STAGING}/data/processed"
  cp -a "${ROOT}/data/processed/sm_bfo_com" "${STAGING}/data/processed/sm_bfo_com"
else
  echo "WARN: missing data/processed/sm_bfo_com — run preprocess after adding source .npy"
fi

# Source .npy is large; copy only if RELEASE_INCLUDE_SOURCE=1
if [[ "${RELEASE_INCLUDE_SOURCE:-0}" == "1" && -f "${ROOT}/data/SmBFO_composition_series.npy" ]]; then
  echo "==> Staging source SmBFO_composition_series.npy (RELEASE_INCLUDE_SOURCE=1)"
  cp "${ROOT}/data/SmBFO_composition_series.npy" "${STAGING}/data/"
fi

# --- Scripts ---
SCRIPTS=(
  download_atomai_smbfo.py
  download_jacs_single_atom_tem_dataset.py
)
for f in "${SCRIPTS[@]}"; do
  [[ -f "${ROOT}/scripts/${f}" ]] && cp "${ROOT}/scripts/${f}" "${STAGING}/scripts/${f}"
done
cp "${ROOT}/scripts/build_github_upload.sh" "${STAGING}/scripts/"
cp "${ROOT}/evaluate_all.sh" "${STAGING}/"

# --- Experiment configs (COM only) ---
cp -a "${ROOT}/experiments/configs/com" "${STAGING}/experiments/configs/"
[[ -f "${ROOT}/experiments/configs/README.txt" ]] && \
  cp "${ROOT}/experiments/configs/README.txt" "${STAGING}/experiments/configs/"

# --- Reports (pre-centroid-aware study outputs) ---
echo "==> Staging reports"
for sub in reviewer_suite noise_ablation corruption_suite cross_domain jacs_external jacs_fewshot; do
  if [[ -d "${ROOT}/reports/${sub}" ]]; then
    mkdir -p "${STAGING}/reports"
    cp -a "${ROOT}/reports/${sub}" "${STAGING}/reports/${sub}"
  fi
done

# --- Paper figure assets (README + generated PNGs from COM pipeline) ---
for f in "${ROOT}"/paper_figures/README*.txt "${ROOT}"/paper_figures/PUBLICATION_NOTICE.txt; do
  [[ -f "$f" ]] && cp "$f" "${STAGING}/paper_figures/"
done
shopt -s nullglob
pngs=("${ROOT}"/paper_figures/*.png)
shopt -u nullglob
if ((${#pngs[@]} > 0)); then
  cp "${pngs[@]}" "${STAGING}/paper_figures/"
fi
[[ -f "${ROOT}/figures/smbfo_panels/README_figure_source.txt" ]] && \
  cp "${ROOT}/figures/smbfo_panels/README_figure_source.txt" "${STAGING}/figures/smbfo_panels/"

# --- Root metadata from github_release/ ---
echo "==> Staging README / requirements"
for f in README.md REPRODUCE.md EXCLUDED.md requirements.txt .gitignore; do
  [[ -f "${ROOT}/github_release/${f}" ]] && cp "${ROOT}/github_release/${f}" "${STAGING}/${f}"
done
[[ -f "${ROOT}/github_release/data/README.md" ]] && \
  cp "${ROOT}/github_release/data/README.md" "${STAGING}/data/README.md"

# Promote staging -> OUT
rm -rf "${OUT}"
mv "${STAGING}" "${OUT}"
mkdir -p "${OUT}/docs"

echo "==> Done: ${OUT}"
du -sh "${OUT}" "${OUT}/gan_seg/checkpoints_benchmark" "${OUT}/data/processed/sm_bfo_com" 2>/dev/null || true
