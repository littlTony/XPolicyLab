#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
cd "${VITRA_POLICY_DIR}"
root="${VITRA_LEROBOT_ROOT:-/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks}"
if [[ ! -s "${root}/meta/vitra/manifest.json" || "${VITRA_RECOMPUTE_STATS:-0}" == 1 ]]; then
    "${VITRA_PYTHON}" prepare_dexbench.py --root "${root}" --workers "${VITRA_PREP_WORKERS:-8}"
fi
exec "${VITRA_PYTHON}" validate_dexbench.py --root "${root}"
