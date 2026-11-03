#!/usr/bin/env bash
FASTWAM_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="${FASTWAM_POLICY_DIR}/FastWAM"
FASTWAM_XPL_ROOT="$(cd "${FASTWAM_POLICY_DIR}/../.." && pwd)"
export FASTWAM_ROOT
export FASTWAM_PYTHON="${FASTWAM_PYTHON:-/personal/miniconda3/envs/fastwam/bin/python}"
export PYTHONPATH="${FASTWAM_XPL_ROOT}/..:${FASTWAM_XPL_ROOT}:${FASTWAM_ROOT}:${FASTWAM_ROOT}/src:${PYTHONPATH:-}"
export PATH="$(dirname "${FASTWAM_PYTHON}"):${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH=/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/pretrain_model
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
