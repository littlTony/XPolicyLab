#!/usr/bin/env bash
# Source this file; bind the copied source tree before any editable installs.
PI05_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_OPENPI_ROOT="${PI05_DIR}/openpi"
PI05_XPL_ROOT="$(cd "${PI05_DIR}/../.." && pwd)"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
export PI05_PYTHON_BIN="${PI05_PYTHON_BIN:-/personal/xiangpc/0811_Xpolicylab_bench/pi05/policy/Pi_05/openpi/.venv-egovla/bin/python}"
export PYTHONPATH="${PI05_XPL_ROOT}/..:${PI05_XPL_ROOT}:${PI05_OPENPI_ROOT}/src:${PI05_OPENPI_ROOT}/packages/openpi-client/src:${PYTHONPATH:-}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/personal/wenwei/DexBench/data_preprocess/data}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PI05_DIR}/.cache/openpi}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [[ ! -x "${PI05_PYTHON_BIN}" ]]; then
    echo "Pi0.5 Python unavailable: ${PI05_PYTHON_BIN}; set PI05_PYTHON_BIN" >&2
    return 1
fi
