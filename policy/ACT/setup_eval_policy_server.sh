#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: $0 bench task checkpoint env_cfg_type action_type seed gpu_id policy_env port [host]" >&2
    exit 2
fi
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

if [[ ! "${policy_gpu_id}" =~ ^[0-9]+$ || ! "${seed}" =~ ^[0-9]{1,10}$ || ! "${policy_server_port}" =~ ^[0-9]{1,5}$ ]]; then
    echo "[ERROR] gpu_id, seed and port must be nonnegative decimal integers" >&2
    exit 2
fi
if (( 10#${seed} > 4294967295 || 10#${policy_server_port} < 1 || 10#${policy_server_port} > 65535 )); then
    echo "[ERROR] seed must fit uint32 and port must be in 1..65535" >&2
    exit 2
fi
if [[ -z "${bench_name//[[:space:]]/}" || -z "${task_name//[[:space:]]/}" || -z "${policy_server_host//[[:space:]]/}" ]]; then
    echo "[ERROR] benchmark, task and host must be nonempty" >&2
    exit 2
fi

if [[ "${policy_conda_env}" == auto ]]; then
    policy_conda_env=/personal/miniconda3/envs/rdt_1b_sm120
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

if [[ -x "${policy_conda_env%/}/bin/python" ]]; then
    POLICY_PYTHON="${policy_conda_env%/}/bin/python"
elif [[ -f "${policy_conda_env}" && -x "${policy_conda_env}" ]]; then
    POLICY_PYTHON="${policy_conda_env}"
else
    CONDA_EXE="${CONDA_EXE:-/personal/miniconda3/bin/conda}"
    source "$("${CONDA_EXE}" info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    POLICY_PYTHON="$(command -v python)"
fi
export PATH="$(dirname "${POLICY_PYTHON}"):${PATH}"
export PYTHONPATH="${ACT_DEPENDENCY_PATH:-/personal/wenwei/Spark0-eval/server_scripts/.runtime_act_dependencies}:${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"
export XPL_ENV_CFG_ROOT="${XPL_ENV_CFG_ROOT:-/personal/xiangpc/0811_Xpolicylab_bench/env_cfg}"
export TORCH_HOME="${TORCH_HOME:-/personal/xiangpc/egovla_ops/act_joint_12task_20260901/cache/torch}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

action_dim=$(
    "${POLICY_PYTHON}" -c "
import sys
from XPolicyLab.utils.process_data import get_action_dim
print(get_action_dim(sys.argv[1]))
" "${env_cfg_type}"
)
export ACT_ACTION_DIM="${action_dim}"

# ckpt_name is the checkpoint directory under checkpoints/ (full run dir name).
if [[ "${ckpt_name}" == /* ]]; then
    ckpt_dir="${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/checkpoints/${ckpt_name}" ]]; then
    ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/${ckpt_name}" ]]; then
    ckpt_dir="${SCRIPT_DIR}/${ckpt_name}"
else
    ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] ckpt_dir=${ckpt_dir}"

# Milestones keep saved architecture/statistics at the task-run root.
# Prepare only their runtime config/filenames, then use the original server.
if [[ -f "${ckpt_dir}/../config_0908-real.json" || -f "${ckpt_dir}/config_0908-real.json" || -f "$(dirname "${ckpt_dir}")/../config_0908-real.json" ]]; then
    if [[ "${env_cfg_type}" != tianji_marvin_wuji || "${action_type}" != joint || "${action_dim}" != 54 ]]; then
        echo "[ERROR] This ACT campaign requires tianji_marvin_wuji joint54" >&2
        exit 2
    fi
    yaml_file="$("${POLICY_PYTHON}" "${SCRIPT_DIR}/prepare_eval_checkpoint.py" \
        "${ckpt_dir}" "${yaml_file}" "${task_name}" \
        "${ACT_EVAL_RUNTIME_ROOT:-/personal/wenwei/Spark0-eval/server_scripts/.runtime_act}")"
    ckpt_dir="$(dirname "${yaml_file}")"
fi
server_args=(
    --config_path "${yaml_file}" --overrides
    port="${policy_server_port}" host="${policy_server_host}" protocol=ws
    bench_name="${bench_name}" task_name="${task_name}" ckpt_name="${ckpt_name}"
    ckpt_dir="${ckpt_dir}" env_cfg_type="${env_cfg_type}" seed="${seed}"
    policy_name="${policy_name}" action_type="${action_type}" action_dim="${action_dim}"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '[NATIVE DRY RUN] model=%s\n' "${SCRIPT_DIR}/model.py"
    printf '%q ' "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" "${server_args[@]}"
    printf '\n'
    exit 0
fi
exec env PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" "${server_args[@]}"
