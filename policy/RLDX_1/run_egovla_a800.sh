#!/usr/bin/env bash
set -euo pipefail

# A800_01 production entry point. The API key stays in the host-only env file.
mode="${1:-start}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_ENV_PATH="${WANDB_ENV_PATH:-/personal/xiangpc/training_0908-real/wandb.env}"

[[ -r "${WANDB_ENV_PATH}" ]] || {
  echo "missing W&B environment: ${WANDB_ENV_PATH}" >&2
  exit 1
}
set -a
# shellcheck disable=SC1090
source "${WANDB_ENV_PATH}"
set +a
: "${WANDB_API_KEY:?WANDB_API_KEY is missing from ${WANDB_ENV_PATH}}"

export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-XPolicyLab-RLDX1-EgoVLA-raw-action}"
if [[ "${mode}" == "smoke" ]]; then
  default_run_name="EgoVLA-RLDX1-rawaction-relarm-bs64-A800_01-smoke-20260921"
else
  default_run_name="EgoVLA-RLDX1-rawaction-relarm-bs64-80k-A800_01-20260921"
fi
export RLDX_RUN_NAME="${RLDX_RUN_NAME:-${default_run_name}}"
export WANDB_NAME="${WANDB_NAME:-${RLDX_RUN_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RLDX_RUN_NAME}}"
export WANDB_RESUME=allow

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_STEPS=80000
export SAVE_STEPS=10000
export SAVE_TOTAL_LIMIT=8
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export RLDX_VIDEO_BACKEND=torchcodec

exec bash "${SCRIPT_DIR}/train_egovla_joint38.sh" "${mode}"
