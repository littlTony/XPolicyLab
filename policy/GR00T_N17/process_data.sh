#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
data=/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21
"${NATIVE_PYTHON}" /personal/wenwei/DexBench/data_preprocess/prepare_native_ee58.py
if ! "${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/prepare_relative_stats.py" --check; then
    nice -n 10 "${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/prepare_relative_stats.py" --workers 8
fi
"${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/prepare_relative_stats.py" --check
exec "${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/test_dexbench.py"
