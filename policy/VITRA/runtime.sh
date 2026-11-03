#!/usr/bin/env bash
VITRA_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITRA_ROOT="${VITRA_POLICY_DIR}/VITRA"
VITRA_XPL_ROOT="$(cd "${VITRA_POLICY_DIR}/../.." && pwd)"
export VITRA_PYTHON="${VITRA_PYTHON:-/personal/miniconda3/envs/vitra_xpolicylab_0813/bin/python}"
export PATH="$(dirname "${VITRA_PYTHON}"):${PATH}"
export PYTHONPATH="${VITRA_XPL_ROOT}/..:${VITRA_XPL_ROOT}:${VITRA_ROOT}:${PYTHONPATH:-}"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
