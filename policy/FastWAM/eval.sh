#!/bin/bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"

# Contract: 10 positional args. `task_name` is the simulator task; `ckpt_name`
# is the full experiment/run directory name under checkpoints/ (exp_path).
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo -e "\033[31m[CLEANUP] kill server PID=${SERVER_PID}\033[0m"
        kill "${SERVER_PID}" 2>/dev/null || true
        # Official EgoVLA suites relaunch eval.sh for the next room/table
        # immediately. Wait for this 15–28G model to leave VRAM first.
        for _ in $(seq 1 30); do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo -e "\033[31m[CLEANUP] server still alive after 30s, SIGKILL ${SERVER_PID}\033[0m"
            kill -9 "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Split jobs run the 15–28G policy on an A800 and Isaac on a render card.
# Without this branch the environment side loads the model on a 24G GPU and OOMs.
if [[ "${EGOVLA_COMPONENT:-}" == "policy" ]]; then
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    policy_server_ip="0.0.0.0"
    echo -e "\033[32m[MAIN] start FastWAM policy server, GPU=${policy_gpu_id}, bind=${policy_server_ip}:${policy_server_port}\033[0m"
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
    bash "${UTILS_DIR}/wait_for_policy_server.sh" "127.0.0.1" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
    wait "${SERVER_PID}"
    exit $?
fi

if [[ "${EGOVLA_COMPONENT:-}" == "environment" ]]; then
    policy_server_ip="${EGOVLA_POLICY_SERVER_HOST:?EGOVLA_POLICY_SERVER_HOST is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?EGOVLA_POLICY_SERVER_PORT is required}"
    echo -e "\033[32m[MAIN] start FastWAM simulator client, GPU=${env_gpu_id}, server=${policy_server_ip}:${policy_server_port}\033[0m"
    ready=0
    for _ in $(seq 1 600); do
        if timeout 1 bash -c "</dev/tcp/${policy_server_ip}/${policy_server_port}" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "${ready}" != "1" ]]; then
        echo "[MAIN] FastWAM policy server did not listen within 1200 seconds" >&2
        exit 1
    fi
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
    echo -e "\033[33m[MAIN] eval finished\033[0m"
    exit 0
fi

echo -e "\033[32m[MAIN] start server, policy_server_port=${policy_server_port}\033[0m"
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
echo -e "\033[32m[MAIN] server PID=${SERVER_PID}\033[0m"

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

echo -e "\033[32m[MAIN] start client, server=${policy_server_ip}:${policy_server_port}\033[0m"
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

echo -e "\033[33m[MAIN] eval finished\033[0m"
