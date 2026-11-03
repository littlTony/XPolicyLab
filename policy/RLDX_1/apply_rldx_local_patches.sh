#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT:-${SCRIPT_DIR}/RLDX-1}"
PATCH_FILE="${SCRIPT_DIR}/patches/rldx_inference_precision.patch"
EXPECTED_COMMIT="cf67c31"

if [[ ! -d "${UPSTREAM_ROOT}/.git" && ! -f "${UPSTREAM_ROOT}/.git" ]]; then
    echo "RLDX submodule is not initialized: ${UPSTREAM_ROOT}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -r "${PATCH_FILE}" ]]; then
    echo "Missing local RLDX patch: ${PATCH_FILE}" >&2
    exit 1
fi

current_commit="$(git -C "${UPSTREAM_ROOT}" rev-parse --short=7 HEAD)"
if [[ "${current_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unexpected RLDX submodule commit: ${current_commit} (expected ${EXPECTED_COMMIT})" >&2
    exit 1
fi

if git -C "${UPSTREAM_ROOT}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "READY local RLDX inference patch already applied"
    exit 0
fi
if ! git -C "${UPSTREAM_ROOT}" apply --check "${PATCH_FILE}"; then
    echo "Local RLDX inference patch does not apply cleanly" >&2
    exit 1
fi

git -C "${UPSTREAM_ROOT}" apply "${PATCH_FILE}"
echo "DONE applied local RLDX inference patch"
