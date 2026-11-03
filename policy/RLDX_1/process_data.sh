#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"
CONVERTER="${XPL_ROOT}/scripts/convert_spark0_bench_to_lerobot_v21_joint54.py"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${UPSTREAM}/.venv/bin/python}"

if [[ ! -f "${CONVERTER}" ]]; then
    echo "Missing Spark converter: ${CONVERTER}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing RLDX conversion Python: ${PYTHON_BIN}" >&2
    echo "Run ${SCRIPT_DIR}/install.sh or set RLDX_PYTHON_BIN." >&2
    exit 1
fi

# With no arguments, the converter reads /mnt/xspark-data/tjy/spark0_bench
# and writes data/Spark0_bench_lerobotV21_joint54.  Any provided flags are
# forwarded verbatim so conversion remains owned by the repository script.
exec "${PYTHON_BIN}" "${CONVERTER}" "$@"
