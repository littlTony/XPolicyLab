#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"

if [[ ! -f "${UPSTREAM}/pyproject.toml" ]]; then
    echo "Missing upstream clone: ${UPSTREAM}" >&2
    exit 1
fi
bash "${SCRIPT_DIR}/apply_rldx_local_patches.sh"

UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" ]] && command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
fi
if [[ -z "${UV_BIN}" && -x /root/.local/bin/uv ]]; then
    UV_BIN=/root/.local/bin/uv
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
    echo "uv is required; set UV_BIN to its executable path and rerun." >&2
    exit 1
fi

# Official standard (non-Blackwell) RLDX-1 installation recipe.  A800 is
# SM80, so the upstream uv environment is the appropriate path.
cd "${UPSTREAM}"
"${UV_BIN}" sync --python 3.10
"${UV_BIN}" pip install -e .
"${UV_BIN}" pip install -e "${XPL_ROOT}"

"${UV_BIN}" run python -c "import rldx; print('rldx', rldx.__version__)"
"${UV_BIN}" run python -c "import XPolicyLab; print('xpolicylab import ok')"
echo "Policy environment: ${UPSTREAM}/.venv"
