JSON configs for reproducible training runs.

Example (does not overwrite legacy checkpoints):
  python -m gan_seg.train_centroid_aware \
    --config experiments/configs/sm_bfo_centroid_hybrid_centroid_multitask_seed42.json

Outputs:
  gan_seg/checkpoints_runs/<run_name>/config.json
  gan_seg/checkpoints_runs/<run_name>/train_log.jsonl
  gan_seg/checkpoints_runs/<run_name>/gan_seg_best.pt

Loss (default): lambda_set=0 (set head off). See config "loss" block.
