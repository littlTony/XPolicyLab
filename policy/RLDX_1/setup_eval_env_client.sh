#!/bin/bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    exec "${NATIVE_PYTHON}" "${NATIVE_XPL_ROOT}/utils/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name RLDX_1 --protocol ws \
        --host "${policy_server_ip}" --port "${policy_server_port}" --eval_batch true \
        --eval_episode_num "${NATIVE_DEBUG_EPISODES:-1}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Web eval sets EVAL_MAIN_ROOT to the simulator checkout (DexBench / EgoVLA).
# Falling back to this policy tree's parent only works for a full bench clone.
BENCH_ROOT="${EVAL_MAIN_ROOT:-$(cd "${XPL_ROOT}/.." && pwd)}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# Keep the env-client import path on this adapter checkout so
# ``XPolicyLab.policy.RLDX_1.deploy`` resolves even when EVAL_MAIN_ROOT is
# the shared DexBench / EgoVLA simulator tree.
export XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-${XPL_ROOT}}"

echo -e "\033[34m[CLIENT] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[34m[CLIENT] server=${policy_server_ip}:${policy_server_port}\033[0m"
echo -e "\033[34m[CLIENT] BENCH_ROOT=${BENCH_ROOT}\033[0m"

# Do not set CUDA_VISIBLE_DEVICES here.  The evaluator passes physical GPU
# ids; masking them remaps cuda:N and breaks Isaac / EgoVLA AppLauncher.
exec env \
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
