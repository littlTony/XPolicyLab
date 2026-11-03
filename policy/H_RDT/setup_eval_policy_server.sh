#!/bin/bash
set -euo pipefail
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
checkpoint_path=${11:-""}
config_path=${12:-""}
lang_embedding_path=${13:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
if [[ "${policy_conda_env}" == /* && -x "${policy_conda_env}" && ! -d "${policy_conda_env}" ]]; then
    POLICY_PYTHON="${policy_conda_env}"
elif [[ -x "${policy_conda_env}/bin/python" ]]; then
    POLICY_PYTHON="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    POLICY_PYTHON=python
fi
export PYTHONPATH="${BENCH_ROOT}:${PYTHONPATH:-}"
EE58=0
if [[ "${bench_name}" == "DexBench" && "${env_cfg_type}" == "tianji_marvin_wuji" && "${action_type}" == "ee" ]]; then
    EE58=1
fi

resolve_checkpoint_path() {
    local explicit_path="$1"
    local default_dir="$2"

    if [[ -n "${explicit_path}" ]]; then
        echo "${explicit_path}"
        return
    fi

    if [[ -f "${default_dir}/pytorch_model.bin" || -f "${default_dir}/model.safetensors" || -f "${default_dir}/config.json" ]]; then
        echo "${default_dir}"
        return
    fi

    if [[ ! -d "${default_dir}" ]]; then
        echo "${default_dir}"
        return
    fi

    local matches=()
    shopt -s nullglob
    matches=("${default_dir}"/checkpoint-*)
    shopt -u nullglob

    if (( ${#matches[@]} == 1 )); then
        echo "${matches[0]}"
        return
    fi

    if (( ${#matches[@]} == 0 )); then
        echo "[ERROR] No checkpoint-* found under ${default_dir}" >&2
    else
        echo "[ERROR] Multiple checkpoint-* directories found under ${default_dir}; pass checkpoint_path explicitly." >&2
    fi
    exit 1
}

if (( EE58 )); then
    action_dim=$("${POLICY_PYTHON}" - <<'PY'
from XPolicyLab.utils.process_data import get_robot_action_dim_info
dims = get_robot_action_dim_info('tianji_marvin_wuji')
assert dims['ee_dim'] == [20, 20]
print(len(dims['arm_dim']) * (3 + 4) + sum(dims['ee_dim']))
PY
)
else
    action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
fi

# Shared checkpoint-dir precedence (POLICY_DIR = policy dir,
# CKPT_ROOT = its checkpoints dir):
#   1. ckpt_name as an absolute path
#   2. ckpt_name as a relative path (POLICY_DIR-relative)
#   3. 5-tuple concat run-dir under checkpoints/
#   4. checkpoints/<ckpt_name> verbatim (backward compatible)
# The positional checkpoint_path override still wins inside
# resolve_checkpoint_path() below.
POLICY_DIR="${SCRIPT_DIR}"
CKPT_ROOT="${SCRIPT_DIR}/checkpoints"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
if [[ "${ckpt_name}" == /* ]]; then
    checkpoint_dir="${ckpt_name}"
elif [[ "${ckpt_name}" == */* ]]; then
    checkpoint_dir="${POLICY_DIR}/${ckpt_name}"
elif [[ -d "${CKPT_ROOT}/${run_dir_name}" ]]; then
    checkpoint_dir="${CKPT_ROOT}/${run_dir_name}"
else
    checkpoint_dir="${CKPT_ROOT}/${ckpt_name}"
fi
if (( EE58 )); then
    if [[ -n "${lang_embedding_path}" ]]; then
        echo "EE58 language is checkpoint-bound; do not pass a legacy task_name.pt override." >&2
        exit 2
    fi
    checkpoint_path=$("${POLICY_PYTHON}" - "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" "${checkpoint_path}" <<'PY'
from pathlib import Path
import sys
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
policy, bench, ckpt, env, action, seed, explicit = sys.argv[1:]
cfg = dict(bench_name=bench, ckpt_name=ckpt, env_cfg_type=env, action_type=action, seed=seed)
if explicit:
    cfg['checkpoint_path'] = explicit
path = resolve_checkpoint_root(cfg, Path(policy)/'checkpoints', policy_dir=policy)
start = path.parent if path.is_file() else path
if not any((candidate/'ee58_runtime.json').is_file() for candidate in (start, *list(start.parents)[:3])):
    raise FileNotFoundError('No EE58 checkpoint contract; refusing legacy/default 14D config')
print(path)
PY
)
else
checkpoint_path="$(resolve_checkpoint_path "${checkpoint_path}" "${checkpoint_dir}")"
# Prefer the config copied into the checkpoint dir by train.sh; fall back to data/;
# pass config_path explicitly if neither matches.
if [[ -z "${config_path}" ]]; then
    if [[ -f "${checkpoint_dir}/hrdt_finetune_xpolicy.yaml" ]]; then
        config_path="${checkpoint_dir}/hrdt_finetune_xpolicy.yaml"
    else
        config_path="${SCRIPT_DIR}/data/${ckpt_name}/hrdt_finetune_xpolicy.yaml"
    fi
fi
lang_embedding_path="${lang_embedding_path:-${SCRIPT_DIR}/H_RDT/datasets/xpolicylab/lang_embeddings/${task_name}.pt}"
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] checkpoint_path=${checkpoint_path}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${POLICY_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            checkpoint_path="${checkpoint_path}" \
            config_path="${config_path}" \
            lang_embedding_path="${lang_embedding_path}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            ee58_instruction_key="${HRDT_INSTRUCTION_KEY:-null}"
