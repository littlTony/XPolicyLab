#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: bash train_sparkarena.sh DexBench tianji_wuji tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7 [--run-id ID | --run-suffix SUFFIX] [--output-dir PATH] [--dry-run]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${STARVLA_PYTHON:-/personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python}"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

# Preparation is deliberately CPU-only, including real native-loader samples.
CONFIG=$(CUDA_VISIBLE_DEVICES= "$PYTHON" "$SCRIPT_DIR/training/prepare_sparkarena.py" "$@")
echo "[StarVLA/DexBench-EE58] prepared config: $CONFIG"
if [[ " $* " == *" --dry-run "* ]]; then
    echo "[StarVLA/DexBench-EE58] CPU checks passed. Training was not started."
    exit 0
fi

export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export ACCELERATE_DEEPSPEED_ZERO_STAGE=2
unset ACCELERATE_DEEPSPEED_ZERO3_INIT ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="$6"
exec "$PYTHON" -m accelerate.commands.launch \
    --use_deepspeed --zero_stage 2 --gradient_accumulation_steps 1 --gradient_clipping 1.0 \
    --num_processes 8 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
    --main_process_port "${STARVLA_MASTER_PORT:-29517}" \
    "$SCRIPT_DIR/training/train_entry.py" --config_yaml "$CONFIG"
