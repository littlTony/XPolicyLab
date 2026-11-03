#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
cd "${FASTWAM_POLICY_DIR}"
"${FASTWAM_PYTHON}" prepare_stats.py
if [[ "${1:-}" == --text-cache ]]; then
    cd "${FASTWAM_ROOT}"
    exec "${FASTWAM_PYTHON}" scripts/precompute_text_embeds.py task=dexbench_ee58 +overwrite=false
fi
