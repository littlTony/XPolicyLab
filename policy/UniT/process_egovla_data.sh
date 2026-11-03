#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UNIT_ENV="${UNIT_ENV:-/personal/miniconda3/envs/unit_torch26_recovery}"
PYTHON_BIN="${PYTHON_BIN:-${UNIT_ENV}/bin/python}"
PREPARE_SCRIPT="${MODEL_ROOT}/data_scripts/prepare_egovla_three_view_v21.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "error: UniT Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
    echo "error: missing EgoVLA preparation script: ${PREPARE_SCRIPT}" >&2
    exit 1
fi

# The preparation script performs the Deprecated inventory gate first, then
# publishes the audited three-view LeRobot v2.1 tree under UniT/data.  The
# output keeps real Insert-And-Unload wrist streams and black fallback wrists
# with camera_mask provenance for the remaining active episodes.
exec "${PYTHON_BIN}" "${PREPARE_SCRIPT}" "$@"
