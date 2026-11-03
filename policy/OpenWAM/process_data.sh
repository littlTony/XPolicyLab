#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
exec "${OPENWAM_PYTHON}" "${OPENWAM_POLICY_DIR}/prepare_stats.py"
