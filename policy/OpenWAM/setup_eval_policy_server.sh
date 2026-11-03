#!/bin/bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
if [[ "${action_type}" != ee ]]; then echo "OpenWAM requires ee" >&2; exit 2; fi
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${XPL_ROOT}/policy/${policy_name}"
yaml_file="${POLICY_DIR}/deploy.yml"

# OpenWAM source root: default is the vendored copy inside this policy dir.
# Weights are NOT vendored; they resolve under policy/OpenWAM/checkpoints/.
OPENWAM_ROOT="${OPENWAM_ROOT:-${POLICY_DIR}/OpenWAM}"
CKPT_ROOT="${POLICY_DIR}/checkpoints"
run_dir_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"

# Let the shared checkpoint resolver in Model select absolute/path/run roots.
ckpt_dir="${OPENWAM_CKPT_DIR:-}"

allow_dummy_policy="${OPENWAM_ALLOW_DUMMY_POLICY:-false}"
# Default: OpenWAM's own deploy defaults; model.py re-forces the
# correctness-critical keys (dit_cache/compile/decode_video off, sync).
openwam_deploy_config="${OPENWAM_DEPLOY_CONFIG:-${OPENWAM_ROOT}/configs/deploy.yaml}"



echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] ckpt_dir: ${ckpt_dir}\033[0m"
echo -e "\033[33m[SERVER] openwam_deploy_config: ${openwam_deploy_config}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port}\033[0m"

# Resolve the policy python: accept either a conda env prefix path (preferred —
# robust to broken conda base configs) or an env name for `conda activate`.
if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="python"
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

# Scope env vars to this server process only; never export them in the
# orchestrator, otherwise they leak into the env client.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${OPENWAM_ROOT}:${BENCH_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            openwam_root="${OPENWAM_ROOT}" \
            ckpt_dir="${ckpt_dir}" \
            openwam_deploy_config="${openwam_deploy_config}" \
            allow_dummy_policy="${allow_dummy_policy}"
