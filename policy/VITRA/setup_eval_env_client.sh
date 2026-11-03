#!/usr/bin/env bash
set -euo pipefail

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
dry_run=${12:-}

[[ "${action_type}" == ee ]] || { echo '[ERROR] VITRA environment client requires action_type=ee' >&2; exit 2; }
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
requested_eval_root="${EVAL_MAIN_ROOT:-}"
source "${SCRIPT_DIR}/runtime.sh"
XPOLICYLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPOLICYLAB_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# Resolve the requested simulator workspace before validating its selected
# environment. The copied adapter is also used outside the old EgoVLA checkout.
BENCH_ROOT=""
for candidate in "${requested_eval_root}" "${EGOVLA_WORKSPACE_ROOT:-}" \
    "${EGOVLA_ROOT:-}" "${EVAL_MAIN_ROOT:-}"; do
    [[ -n "${candidate}" ]] || continue
    candidate="$(readlink -f "${candidate}")"
    if [[ -f "${candidate}/scripts/eval_policy.sh" ]]; then
        BENCH_ROOT="${candidate}"
        break
    fi
done
[[ -n "${BENCH_ROOT}" ]] || { echo '[ERROR] cannot resolve simulator workspace containing scripts/eval_policy.sh' >&2; exit 2; }
ENV_CFG_PATH="${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml"
[[ -f "${ENV_CFG_PATH}" ]] || { echo "[ERROR] missing simulator environment config: ${ENV_CFG_PATH}" >&2; exit 2; }

if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    EVAL_RUNTIME_PYTHON="${eval_env_conda_env}/bin/python"
elif [[ -x "${eval_env_conda_env}" ]]; then
    EVAL_RUNTIME_PYTHON="${eval_env_conda_env}"
elif command -v conda >/dev/null 2>&1; then
    EVAL_RUNTIME_PYTHON="$(conda run --no-capture-output -n "${eval_env_conda_env}" python -c 'import sys; print(sys.executable)' 2>/dev/null)"
else
    EVAL_RUNTIME_PYTHON=""
fi
[[ -n "${EVAL_RUNTIME_PYTHON}" && -x "${EVAL_RUNTIME_PYTHON}" ]] || { echo "[ERROR] cannot resolve eval runtime Python: ${eval_env_conda_env}" >&2; exit 2; }
if ! YAML_SOURCE="$("${EVAL_RUNTIME_PYTHON}" -c 'import yaml; print(yaml.__file__)' 2>/dev/null)"; then
    echo "[ERROR] PyYAML missing from eval runtime: ${EVAL_RUNTIME_PYTHON}" >&2
    exit 2
fi
echo "[CLIENT] eval_runtime_python=${EVAL_RUNTIME_PYTHON} yaml_source=${YAML_SOURCE}"
echo "[CLIENT] benchmark_root=${BENCH_ROOT} env_cfg=${ENV_CFG_PATH}"

if [[ "${dry_run}" == --dry-run ]]; then
    echo "[CLIENT] dry-run adapter_root=${XPOLICYLAB_ROOT} policy=${policy_name} action_type=${action_type} env_cfg_type=${env_cfg_type}"
    exit 0
fi

export EVAL_MAIN_ROOT="${BENCH_ROOT}"
export EGOVLA_EVAL_PYTHON="${EVAL_RUNTIME_PYTHON}"
export EGOVLA_EVAL_CONDA_ENV="${eval_env_conda_env}"
export XPOLICYLAB_ROOT
export EGOVLA_POLICY_ADAPTER_ROOT="${XPOLICYLAB_ROOT}"
exec bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
    "${policy_server_port}" "${bench_name}" "${task_name}" \
    "${env_cfg_type}" "${policy_name}" "${additional_info}" \
    "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
