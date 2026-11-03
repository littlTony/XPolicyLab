#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-${DEFAULT_BENCH_ROOT}}"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"

# Debug mode uses the lightweight XPL client directly and does not need a
# simulator checkout.  Keep the simulator entrypoint guard for sim/real eval.
if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" && ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "[CLIENT][ERROR] Missing simulator entrypoint: ${BENCH_ROOT}/scripts/eval_policy.sh" >&2
    echo "[CLIENT][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi

if [[ -z "${XPL_IMPORT_ROOT:-}" && ! -d "${BENCH_ROOT}/XPolicyLab/policy/${policy_name}" ]]; then
    XPL_IMPORT_ROOT="${XPL_ROOT}/.."
fi
export XPL_IMPORT_ROOT

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] eval main root=${BENCH_ROOT}"

PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}" \
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
    "${policy_server_port}" "${bench_name}" "${task_name}" \
    "${env_cfg_type}" "${policy_name}" "${additional_info}" \
    "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
