#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${SCRIPT_DIR}/UniT"

if [[ ! -f "${UPSTREAM_ROOT}/examples/environment_setup.sh" ]]; then
    echo "error: official UniT checkout is missing at ${UPSTREAM_ROOT}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    for candidate in /personal/miniconda3/bin/conda /root/miniconda3/bin/conda /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            export PATH="$(dirname "${candidate}"):${PATH}"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available" >&2
    exit 1
fi

# The upstream installer is the source of truth for Python/CUDA package pins.
# Set PIP_INDEX_URL before running this script if a regional mirror is desired.
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-unit}"
bash "${UPSTREAM_ROOT}/examples/environment_setup.sh"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
python -m pip install -e "${XPL_ROOT}"

echo "UniT and XPolicyLab installed in conda env: ${CONDA_ENV_NAME}"
