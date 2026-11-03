#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 11 ]]; then
    echo "usage: $0 bench task checkpoint env_cfg_type action_type seed env_gpu eval_env additional_info port [host]" >&2
    exit 2
fi

bench_name="$1"
task_name="$2"
ckpt_name="$3"
env_cfg_type="$4"
action_type="$5"
seed="$6"
env_gpu_id="$7"
eval_env_conda_env="$8"
additional_info="$9"
policy_server_port="${10}"
policy_server_ip="${11:-localhost}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

if [[ -n "${EVAL_MAIN_ROOT:-}" ]]; then
    BENCH_ROOT="${EVAL_MAIN_ROOT}"
elif [[ -n "${EGOVLA_WORKSPACE_ROOT:-}" ]]; then
    BENCH_ROOT="${EGOVLA_WORKSPACE_ROOT}"
elif [[ -d "/personal/xiangpc/EgoVLA benchmark" ]]; then
    BENCH_ROOT="/personal/xiangpc/EgoVLA benchmark"
else
    BENCH_ROOT="${MODEL_ROOT}"
fi
if [[ ! -d "${BENCH_ROOT}" ]]; then
    echo "[ERROR] benchmark workspace does not exist: ${BENCH_ROOT}" >&2
    exit 1
fi
BENCH_ROOT="$(cd "${BENCH_ROOT}" && pwd)"
if [[ "${EVAL_ENV_TYPE:-sim}" != "debug" && ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "[ERROR] benchmark eval hook is missing: ${BENCH_ROOT}/scripts/eval_policy.sh; set EVAL_MAIN_ROOT" >&2
    exit 1
fi

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Make the ABI/provenance visible to both the benchmark bridge and the model
# server. The bridge's published RDT capability is deny-by-default because
# this adapter is local; the RDT wrapper opts in explicitly for this path.
export EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}"
export XPOLICYLAB_ROOT="${XPL_ROOT}"
export EGOVLA_CHECKPOINT="${ckpt_name}"
export EGOVLA_ACTION_TYPE="${action_type}"
export EGOVLA_ENV_CFG_TYPE="${env_cfg_type}"
export EGOVLA_REQUESTED_SEED="${seed}"
export EGOVLA_EVAL_CONDA_ENV="${eval_env_conda_env}"
if [[ "${policy_name}" == "ACT" ]]; then
    export EGOVLA_ACT_CAMERA_MODE="${EGOVLA_ACT_CAMERA_MODE:-contract}"
else
    # The benchmark bridge reserves this variable for ACT and rejects
    # non-contract ACT modes for other policies. RDT applies its released-data
    # camera contract inside model.encode_obs via RDT_CAMERA_MODE.
    unset EGOVLA_ACT_CAMERA_MODE || true
fi
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    export RDT_CAMERA_MODE="${RDT_CAMERA_MODE:-black_wrist}"
    export RDT_EVAL_BATCH="${RDT_EVAL_BATCH:-false}"
    export EGOVLA_RDT_EVAL_FORCE="${EGOVLA_RDT_EVAL_FORCE:-0}"
    if [[ -d "${BENCH_ROOT}/Ego_Humanoid_Manipulation_Benchmark" ]]; then
        export EGOVLA_ROOT="${BENCH_ROOT}/Ego_Humanoid_Manipulation_Benchmark"
    fi
    if [[ -d "${BENCH_ROOT}/EgoVLA_Release" ]]; then
        export EGOVLA_RELEASE_ROOT="${BENCH_ROOT}/EgoVLA_Release"
    fi
fi

# Resolve the YAML interpreter once so setup_env_client never depends on a
# bare `python` alias or an initialized shell function.
# shellcheck source=../../utils/conda_env.sh
source "${UTILS_DIR}/conda_env.sh"
yaml_python="${XPL_YAML_PYTHON:-$(xpl_python_for_env "${eval_env_conda_env}" 2>/dev/null || true)}"
if [[ -z "${yaml_python}" || ! -x "${yaml_python}" ]]; then
    echo "[ERROR] no Python interpreter for eval environment ${eval_env_conda_env@Q}; set XPL_YAML_PYTHON" >&2
    exit 1
fi
export XPL_YAML_PYTHON="${yaml_python}"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] benchmark_root=${BENCH_ROOT}, xpolicy_root=${XPL_ROOT}, camera_mode=${RDT_CAMERA_MODE:-real_wrist}, eval_batch=${RDT_EVAL_BATCH:-deploy.yml}"
echo "[CLIENT] checkpoint=${ckpt_name}"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
