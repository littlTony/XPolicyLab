#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 10 ]]; then
    echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

if [[ "${action_type}" != "ee" ]]; then
    echo "[ERROR] VITRA must be evaluated with action_type=ee" >&2
    exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ ! -d "${XPOLICYLAB_ROOT}/policy/VITRA" ]]; then
    echo "[ERROR] resolved XPOLICYLAB_ROOT is not the VITRA checkout: ${XPOLICYLAB_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${XPOLICYLAB_ROOT}/env_cfg/ego_h1_inspire.yml" && ! -f "${XPOLICYLAB_ROOT}/../env_cfg/ego_h1_inspire.yml" ]]; then
    echo "[ERROR] resolved VITRA checkout has no env_cfg/ego_h1_inspire.yml: ${XPOLICYLAB_ROOT}" >&2
    exit 2
fi
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Split jobs keep VITRA weights on an A800. The render node must only run Isaac.
if [[ "${EGOVLA_COMPONENT:-}" == "policy" ]]; then
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    policy_server_ip="0.0.0.0"
    echo "[MAIN] start VITRA policy server, GPU=${policy_gpu_id}, bind=${policy_server_ip}:${policy_server_port}"
    bash "${SERVER_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
        "${policy_server_port}" "${policy_server_ip}" &
    SERVER_PID=$!
    bash "${UTILS_DIR}/wait_for_policy_server.sh" \
        "127.0.0.1" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
    wait "${SERVER_PID}"
    exit $?
fi

if [[ "${EGOVLA_COMPONENT:-}" == "environment" ]]; then
    policy_server_ip="${EGOVLA_POLICY_SERVER_HOST:?EGOVLA_POLICY_SERVER_HOST is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    echo "[MAIN] start VITRA simulator client, GPU=${env_gpu_id}, server=${policy_server_ip}:${policy_server_port}"
    ready=0
    for _ in $(seq 1 600); do
        if timeout 1 bash -c "</dev/tcp/${policy_server_ip}/${policy_server_port}" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "${ready}" != "1" ]]; then
        echo "[MAIN] VITRA policy server did not listen within 1200 seconds" >&2
        exit 1
    fi
    bash "${CLIENT_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
        "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
        "${additional_info}" "${policy_server_port}" "${policy_server_ip}"
    echo "[MAIN] eval finished"
    exit 0
fi

echo "[MAIN] start server, policy_server_port=${policy_server_port}"
bash "${SERVER_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" \
    "${policy_server_port}" &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"
bash "${CLIENT_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
    "${additional_info}" "${policy_server_port}" "${policy_server_ip}"

echo "[MAIN] eval finished"
