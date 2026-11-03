#!/usr/bin/env bash
OPENWAM_POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="${OPENWAM_POLICY_DIR}/OpenWAM"
OPENWAM_XPL_ROOT="$(cd "${OPENWAM_POLICY_DIR}/../.." && pwd)"
export OPENWAM_ROOT
export OPENWAM_PYTHON="${OPENWAM_PYTHON:-/personal/miniconda3/envs/openwam/bin/python}"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export PYTHONPATH="${OPENWAM_XPL_ROOT}/..:${OPENWAM_XPL_ROOT}:${OPENWAM_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${OPENWAM_PYTHON}"):${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
if [[ ! -x "${OPENWAM_PYTHON}" ]]; then
    echo "OpenWAM interpreter unavailable: ${OPENWAM_PYTHON}" >&2
    return 1
fi
