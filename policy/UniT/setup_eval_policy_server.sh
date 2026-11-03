#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${SCRIPT_DIR}/UniT"
policy_name="$(basename "${SCRIPT_DIR}")"
# Set DEPLOY_CONFIG for an adapter-specific runtime schema.  When the helper is
# invoked directly (without eval.sh), select the EgoVLA contract from its
# benchmark/robot pair; all other calls retain the Spark default.
if [[ -n "${DEPLOY_CONFIG:-}" ]]; then
    yaml_file="${DEPLOY_CONFIG}"
elif [[ ( "${bench_name}" == "EgoVLA_benchmark" || "${bench_name}" == "EgoVLA" ) && "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    case "${UNIT_EGOVLA_VIEW_MODE:-three_view}" in
        single_view) yaml_file="${SCRIPT_DIR}/deploy_egovla_singleview.yml" ;;
        three_view|multiview) yaml_file="${SCRIPT_DIR}/deploy_egovla.yml" ;;
        *) echo "error: UNIT_EGOVLA_VIEW_MODE must be single_view or three_view" >&2; exit 2 ;;
    esac
else
    yaml_file="${SCRIPT_DIR}/deploy.yml"
fi

if [[ ! -f "${yaml_file}" ]]; then
    echo "error: missing deploy config: ${yaml_file}" >&2
    exit 1
fi
if [[ "${UNIT_DRY_RUN:-0}" != "1" && ! -f "${UPSTREAM_ROOT}/gr00t/model/policy_unit.py" ]]; then
    echo "error: official UniT checkout is missing under ${UPSTREAM_ROOT}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    for candidate in /personal/miniconda3/bin/conda /root/miniconda3/bin/conda /opt/conda/bin/conda; do
        if [[ -x "${candidate}" ]]; then
            export PATH="$(dirname "${candidate}"):${PATH}"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda is not available" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
eval "$(conda shell.bash hook)"
conda activate "${policy_conda_env}"
PYTHON_BIN="$(command -v python)"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${UPSTREAM_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
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
            action_type="${action_type}"
