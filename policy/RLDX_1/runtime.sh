#!/usr/bin/env bash
NATIVE_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_XPL_ROOT="$(cd "${NATIVE_POLICY_DIR}/../.." && pwd)"
export NATIVE_POLICY_DIR NATIVE_XPL_ROOT
export NATIVE_ROOT="${NATIVE_POLICY_DIR}/RLDX-1"
export NATIVE_PYTHON="/personal/xiangpc/0811_Xpolicylab_bench/RLDX_1/policy/RLDX_1/RLDX-1/.venv/bin/python"
export PATH="$(dirname "${NATIVE_PYTHON}"):${PATH}"
export PYTHONPATH="${NATIVE_POLICY_DIR}:${NATIVE_ROOT}:${NATIVE_XPL_ROOT}/..:${NATIVE_XPL_ROOT}:${PYTHONPATH:-}"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export GR00T_COSMOS_MODEL=/personal/xiangpc/0811_Xpolicylab_bench/GrootN17/pretrain_model/Cosmos-Reason2-2B
export HF_HOME=/personal/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/hf_home
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
