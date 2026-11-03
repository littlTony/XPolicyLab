#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim's AppLauncher must not prompt on the Web worker's non-interactive
# stdin.  Preserve an explicit caller choice, but default the private runner
# to the accepted EULA required by the audited runtime.
: "${OMNI_KIT_ACCEPT_EULA:=Y}"
export OMNI_KIT_ACCEPT_EULA
: "${ACCEPT_EULA:=Y}"
export ACCEPT_EULA

if [[ $# -ne 10 ]]; then
    echo "Usage: bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> joint <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>" >&2
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
policy_env_or_uv_path=$9
eval_env_conda_env=${10}

if [[ "${action_type}" != "joint" ]]; then
    echo "[RLDX_1] Spark integration supports action_type=joint only." >&2
    exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RLDX_POLICY_ROOT="${XPL_ROOT}/policy/RLDX_1"
RLDX_ADAPTER_ROOT="${SCRIPT_DIR}/RLDX-1"
RLDX_UPSTREAM_ROOT="${RLDX_POLICY_ROOT}/RLDX-1"
POLICY_SITE_PACKAGES="${RLDX_POLICY_SITE_PACKAGES:-${RLDX_SITE_PACKAGES:-${RLDX_UPSTREAM_ROOT}/.venv/lib/python3.10/site-packages}}"
export RLDX_POLICY_SITE_PACKAGES="${POLICY_SITE_PACKAGES}"
UTILS_DIR="${XPL_ROOT}/utils"

# The benchmark checkout is separate from this private XPolicyLab clone.  Set
# its default before deriving the bridge path so custom EVAL_MAIN_ROOT values
# and the two child scripts all observe one canonical root.
if [[ -z "${EVAL_MAIN_ROOT:-}" ]]; then
    if [[ "${bench_name}" == "EgoVLA" && -d "/mnt/xspark-data/xiangpc/EgoVLA benchmark" ]]; then
        export EVAL_MAIN_ROOT="/mnt/xspark-data/xiangpc/EgoVLA benchmark"
    elif [[ "${bench_name}" != "EgoVLA" && -d "/personal/wenwei/xspark-0-simeval" ]]; then
        export EVAL_MAIN_ROOT="/personal/wenwei/xspark-0-simeval"
    fi
fi

# EgoVLA keeps its official bridge.  DexBench / SparkArena talks to
# xspark-0-simeval/scripts/eval_policy.sh and must not require that bridge.
if [[ "${bench_name}" == "EgoVLA" ]]; then
    if [[ -n "${EGOVLA_BRIDGE_ROOT:-}" ]]; then
        BRIDGE_ROOT="$(cd "${EGOVLA_BRIDGE_ROOT}" && pwd -P)"
    elif [[ -d "${EVAL_MAIN_ROOT:-}/integration" ]]; then
        BRIDGE_ROOT="$(cd "${EVAL_MAIN_ROOT}/integration" && pwd -P)"
    else
        BRIDGE_ROOT=""
    fi
    if [[ -n "${BRIDGE_ROOT}" && ! -d "${BRIDGE_ROOT}/src/egovla_xpolicy" ]]; then
        echo "[MAIN][ERROR] bridge root lacks integration/src/egovla_xpolicy: ${BRIDGE_ROOT}" >&2
        exit 1
    fi
else
    BRIDGE_ROOT=""
    unset EGOVLA_BRIDGE_ROOT || true
fi

# The RLDX venv (numpy/h5py) must stay off the Isaac/Luminis client path.
# DexBench only exports adapter roots here; the policy server rebuilds its
# own PYTHONPATH from RLDX_POLICY_SITE_PACKAGES.
if [[ "${bench_name}" == "EgoVLA" ]]; then
    private_pythonpath="${XPL_ROOT}:${RLDX_POLICY_ROOT}:${POLICY_SITE_PACKAGES}:${RLDX_UPSTREAM_ROOT}"
    if [[ -n "${BRIDGE_ROOT}" ]]; then
        private_pythonpath="${private_pythonpath}:${BRIDGE_ROOT}/src"
    fi
    if [[ -n "${PYTHONPATH:-}" ]]; then
        private_pythonpath="${private_pythonpath}:${PYTHONPATH}"
    fi
    export PYTHONPATH="${private_pythonpath}"
else
    export PYTHONPATH="${XPL_ROOT}:${RLDX_POLICY_ROOT}"
fi
export RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_ROOT}"
export EGOVLA_POLICY_ADAPTER_ROOT="${XPL_ROOT}"
export RLDX_UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT}"
export RLDX_BRIDGE_BOOTSTRAP="${SCRIPT_DIR}/egovla_bridge_bootstrap.py"
export RLDX_PRIVATE_MANIFEST="${SCRIPT_DIR}/rldx_capabilities.json"
if [[ -n "${BRIDGE_ROOT}" ]]; then
    export EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}"
fi

export XPOLICYLAB_ROOT="${XPL_ROOT}"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

if [[ ! -f "${XPL_ROOT}/XPolicyLab.py" || ! -d "${RLDX_POLICY_ROOT}" || ! -f "${RLDX_POLICY_ROOT}/model.py" || ! -f "${RLDX_POLICY_ROOT}/deploy.py" ]]; then
    echo "[MAIN][ERROR] private RLDX checkout is incomplete: ${XPL_ROOT}" >&2
    exit 1
fi
if [[ ! -r "${SERVER_SCRIPT}" || ! -r "${CLIENT_SCRIPT}" ]]; then
    echo "[MAIN][ERROR] private eval entrypoints are missing under ${SCRIPT_DIR}" >&2
    exit 1
fi

echo "[MAIN] XPL_ROOT=${XPL_ROOT}"
echo "[MAIN] PYTHONPATH=${PYTHONPATH}"

# Eval Web launches policy/environment independently. Same-node evaluation
# (no EGOVLA_COMPONENT) keeps the localhost server+client pair below.
component="${EGOVLA_COMPONENT:-}"
if [[ -n "${component}" ]]; then
    if [[ "${bench_name}" != "DexBench" && "${bench_name}" != "EgoVLA" ]]; then
        echo "[MAIN][ERROR] RLDX split components require DexBench or EgoVLA" >&2
        exit 2
    fi
    policy_server_host="${EGOVLA_POLICY_SERVER_HOST:?policy server host is required}"
    policy_server_port="${EGOVLA_POLICY_SERVER_PORT:?policy server port is required}"
    if [[ ! "${policy_server_port}" =~ ^[0-9]+$ ]] || (( policy_server_port < 1 || policy_server_port > 65535 )); then
        echo "[MAIN][ERROR] invalid policy server port" >&2
        exit 2
    fi
    case "${component}" in
        policy)
            echo "[MAIN] split RLDX server GPU=${policy_gpu_id}, bind=${policy_server_host}:${policy_server_port}"
            exec bash "${SERVER_SCRIPT}" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env_or_uv_path}" \
                "${policy_server_port}" "${policy_server_host}"
            ;;
        environment)
            echo "[MAIN] split RLDX client GPU=${env_gpu_id}, server=${policy_server_host}:${policy_server_port}"
            ready=0
            # Remote endpoint: do not use the local LISTEN check in
            # wait_for_policy_server.sh.
            for (( attempt=0; attempt<600; attempt++ )); do
                if timeout 1 bash -c 'exec 3<>"/dev/tcp/$1/$2"' _ "${policy_server_host}" "${policy_server_port}" 2>/dev/null; then
                    ready=1
                    break
                fi
                sleep 2
            done
            if [[ "${ready}" != "1" ]]; then
                echo "[MAIN][ERROR] remote RLDX policy server did not become ready" >&2
                exit 1
            fi
            additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
            exec bash "${CLIENT_SCRIPT}" \
                "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
                "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" \
                "${additional_info}" "${policy_server_port}" "${policy_server_host}"
            ;;
        *)
            echo "[MAIN][ERROR] unsupported RLDX component: ${component}" >&2
            exit 2
            ;;
    esac
fi

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

echo "[MAIN] start RLDX_1 server, policy_server_port=${policy_server_port}"
bash "${SERVER_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_env_or_uv_path}" \
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
