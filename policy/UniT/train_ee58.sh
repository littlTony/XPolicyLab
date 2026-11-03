#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: bash train_ee58.sh [ckpt_name=tianji_wuji] 0,1,2,3,4,5,6,7 [--dry-run]" >&2
  exit 2
fi
if [[ "$1" == *,* ]]; then
  [[ $# -le 2 ]] || { echo 'Too many arguments after GPU IDs' >&2; exit 2; }
  CKPT_NAME=tianji_wuji
  GPU_IDS="$1"
  MODE="${2:-}"
else
  CKPT_NAME="$1"
  GPU_IDS="${2:?Missing GPU IDs}"
  MODE="${3:-}"
fi
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then echo "Only --dry-run is accepted" >&2; exit 2; fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${UNIT_PYTHON:-/personal/miniconda3/envs/unit_torch26_recovery/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
PREPARED=$(CUDA_VISIBLE_DEVICES= "$PYTHON" "$ROOT/training/prepare_ee58.py" "$CKPT_NAME" "$GPU_IDS")
echo "[UniT EE58] $PREPARED"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "[UniT EE58] CPU preflight passed; no model/GPU training started."
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
exec "$PYTHON" "$ROOT/training/run_pipeline.py" --prepared "$PREPARED"
