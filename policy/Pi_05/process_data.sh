#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
"${PI05_PYTHON_BIN}" "${PI05_DIR}/check_contract.py"
exec "${PI05_PYTHON_BIN}" "${PI05_DIR}/prepare_stats.py"
