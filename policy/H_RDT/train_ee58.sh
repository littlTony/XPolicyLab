#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 || $# -gt 7 || ( $# -eq 7 && "$7" != "--dry-run" ) ]]; then
  echo "Usage: bash train_ee58.sh DexBench <ckpt_name> tianji_marvin_wuji ee <seed> 0,1,2,3,4,5,6,7 [--dry-run]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HRDT_PYTHON:-/personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python}"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
CONFIG=$(CUDA_VISIBLE_DEVICES= "$PYTHON" "$SCRIPT_DIR/training/prepare_ee58.py" "$@")
echo "H_RDT EE58 config: $CONFIG"
if [[ "${7:-}" == "--dry-run" ]]; then
  echo "CPU data/header checks complete; read preflight.json for language readiness. GPU training was not started."
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$6"
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
exec "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes=8 \
  --num_machines=1 --main_process_port="${HRDT_MASTER_PORT:-42528}" \
  "$SCRIPT_DIR/training/train_entry.py" --config "$CONFIG"
