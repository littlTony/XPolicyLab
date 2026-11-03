#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
    echo "usage: $0 bench task checkpoint env_cfg_type action_type seed policy_gpu env_gpu policy_env eval_env" >&2
    exit 2
fi

bench_name="$1"
task_name="$2"
ckpt_name="$3"
env_cfg_type="$4"
action_type="$5"
seed="$6"
policy_gpu_id="$7"
env_gpu_id="$8"
policy_conda_env="$9"
eval_env_conda_env="${10}"

[[ -n "${bench_name}" && -n "${task_name}" && -n "${ckpt_name}" ]] || {
    echo "[ERROR] bench, task, and checkpoint must be non-empty" >&2
    exit 2
}
if [[ "${action_type}" != "joint" ]]; then
    echo "[ERROR] RDT_1B supports only action_type=joint (got ${action_type@Q})" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    map_file="${SCRIPT_DIR}/rdt/configs/egovla_joint38.py"
    [[ -f "${map_file}" ]] || { echo "[ERROR] EgoVLA 38D adapter map is missing: ${map_file}" >&2; exit 2; }
fi
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

# These values are inherited by both the policy server and the benchmark
# client. They make the checkpoint and ABI explicit instead of relying on
# comma-delimited legacy additional_info parsing.
export XPOLICYLAB_ROOT="${XPL_ROOT}"
export EGOVLA_CHECKPOINT="${ckpt_name}"
export EGOVLA_ACTION_TYPE="${action_type}"
export EGOVLA_ENV_CFG_TYPE="${env_cfg_type}"
export EGOVLA_REQUESTED_SEED="${seed}"
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    # EgoVLA is single-view. Wrist slots are black images, matching training.
    # Set RDT_CAMERA_MODE=main_replicated only for the old duplicated-wrist run.
    export RDT_CAMERA_MODE="${RDT_CAMERA_MODE:-black_wrist}"
    # RDT evaluates one simulator environment at a time. The generic debug
    # batch path creates ten serial model calls and is needlessly expensive.
    export RDT_EVAL_BATCH="${RDT_EVAL_BATCH:-false}"
    # The benchmark manifest remains deny-by-default. Set this explicitly to
    # 1 for the local adapter path after reviewing the checkpoint/ABI.
    export EGOVLA_RDT_EVAL_FORCE="${EGOVLA_RDT_EVAL_FORCE:-0}"
fi

policy_server_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
policy_server_ip="${RDT_POLICY_SERVER_HOST:-localhost}"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] stopping policy server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] RDT_1B eval: task=${task_name}, env=${env_cfg_type}, action=${action_type}"
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    default_ctrl_freq=30
    default_image_size="[384, 384]"
else
    default_ctrl_freq=25
    default_image_size="[640, 480]"
fi
echo "[MAIN] checkpoint=${ckpt_name}, camera_mode=${RDT_CAMERA_MODE:-real_wrist}, ctrl_freq=${RDT_CTRL_FREQ:-${default_ctrl_freq}}, image_size=${RDT_IMAGE_SIZE:-${default_image_size}}"
echo "[MAIN] server=${policy_server_ip}:${policy_server_port}, EVAL_ENV_TYPE=${EVAL_ENV_TYPE:-sim}"

if [[ "${RDT_EVAL_DRY_RUN:-0}" == "1" ]]; then
    echo "[MAIN][DRY-RUN] validating policy-server and environment-client launch contracts"
    bash "${SERVER_SCRIPT}" \
        "${bench_name}" \
        "${task_name}" \
        "${ckpt_name}" \
        "${env_cfg_type}" \
        "${action_type}" \
        "${seed}" \
        "${policy_gpu_id}" \
        "${policy_conda_env}" \
        "${policy_server_port}" \
        "${policy_server_ip}"
    bash "${CLIENT_SCRIPT}" \
        "${bench_name}" \
        "${task_name}" \
        "${ckpt_name}" \
        "${env_cfg_type}" \
        "${action_type}" \
        "${seed}" \
        "${env_gpu_id}" \
        "${eval_env_conda_env}" \
        "${additional_info}" \
        "${policy_server_port}" \
        "${policy_server_ip}"
    echo "[MAIN][DRY-RUN] RDT_1B eval preflight passed"
    exit 0
fi

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" &
SERVER_PID=$!

wait_timeout="${RDT_POLICY_SERVER_TIMEOUT:-1200}"
[[ "${wait_timeout}" =~ ^[0-9]+$ ]] || { echo "[ERROR] RDT_POLICY_SERVER_TIMEOUT must be an integer" >&2; exit 2; }
bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "RDT_1B policy server" "${wait_timeout}"

echo "[MAIN] start environment client via ${EVAL_MAIN_ROOT:-${EGOVLA_WORKSPACE_ROOT:-auto}}"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
