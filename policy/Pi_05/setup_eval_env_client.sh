#!/usr/bin/env bash
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
if [[ "${action_type}" != ee ]]; then
    echo '[CLIENT] DexBench Pi0.5 requires action_type=ee' >&2
    exit 2
fi
if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
    # Exercise the copied adapter and decoder, rather than importing a stale
    # XPolicyLab helper from the separately maintained simulator checkout.
    exec "${PI05_PYTHON_BIN}" "${PI05_XPL_ROOT}/utils/debug_env_client.py" \
        --bench_name "${bench_name}" --task_name "${task_name}" \
        --env_cfg_type "${env_cfg_type}" --policy_name Pi_05 --protocol ws \
        --host "${policy_server_ip}" --port "${policy_server_port}" --eval_batch true \
        --eval_episode_num "${PI05_DEBUG_EPISODES:-1}"
fi
if [[ ! -f "${EVAL_MAIN_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "Simulator entrypoint missing: ${EVAL_MAIN_ROOT}/scripts/eval_policy.sh" >&2
    exit 1
fi
# The unchanged simulator helper consumes the adapter's standard action dict.
if ! command -v conda >/dev/null 2>&1; then
    export PATH="/personal/miniconda3/bin:${PATH}"
fi
export PATH="$(dirname "${PI05_PYTHON_BIN}"):${PATH}"
bash "${PI05_XPL_ROOT}/utils/setup_env_client.sh" \
    "${PI05_XPL_ROOT}/utils" "${PI05_DIR}/deploy.yml" "${eval_env_conda_env}" \
    "${policy_server_port}" "${bench_name}" "${task_name}" "${env_cfg_type}" \
    Pi_05 "${additional_info}" "${EVAL_MAIN_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
