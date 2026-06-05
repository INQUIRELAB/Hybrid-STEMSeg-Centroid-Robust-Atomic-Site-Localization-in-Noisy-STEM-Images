#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python executable not found or not executable: $PYTHON_BIN"
    echo "Create/fix venv first, then re-run."
    exit 1
fi

run_eval_acc() {
    local ckpt="$1"
    if [ -f "$ckpt" ]; then
        "$PYTHON_BIN" -m gan_seg.eval_acc "$ckpt"
    else
        echo "Missing checkpoint: $ckpt"
    fi
}

run_eval_benchmark() {
    local ckpt="$1"
    if [ -f "$ckpt" ]; then
        "$PYTHON_BIN" -m gan_seg.eval_benchmark "$ckpt"
    else
        echo "Missing checkpoint: $ckpt"
    fi
}

echo "Original Hybrid GAN (From Scratch):"
run_eval_acc "gan_seg/checkpoints_final_100ep/gan_seg_last.pt"

echo "---"
for m in unet deeplabv3plus segformer hybrid-nogan hybrid-notransformer; do
    echo "Benchmark $m:"
    run_eval_benchmark "gan_seg/checkpoints_benchmark/$m/gan_seg_best.pt"
    echo "---"
done

echo "SOTA Hybrid GAN (ResNet-34 Initialized):"
run_eval_acc "gan_seg/checkpoints_final_pretrained/gan_seg_last.pt"
