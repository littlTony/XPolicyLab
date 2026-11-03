#!/bin/bash
set -euo pipefail

: "${OMNI_KIT_ACCEPT_EULA:=Y}"
export OMNI_KIT_ACCEPT_EULA
: "${ACCEPT_EULA:=Y}"
export ACCEPT_EULA
if [[ -n "${EVAL_WEB_HEADLESS_BIN:-}" && -d "${EVAL_WEB_HEADLESS_BIN}" ]]; then
    export PATH="${EVAL_WEB_HEADLESS_BIN}:${PATH}"
fi

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
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BENCH_ROOT="/personal/xiangpc/EgoVLA benchmark"
if [[ ! -d "${DEFAULT_BENCH_ROOT}" ]]; then
    DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
fi
BENCH_ROOT="${EVAL_MAIN_ROOT:-${DEFAULT_BENCH_ROOT}}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${PI05_DEPLOY_CONFIG:-${XPL_ROOT}/policy/${policy_name}/deploy.yml}"
if [[ "${yaml_file}" != /* ]]; then
    yaml_file="${SCRIPT_DIR}/${yaml_file}"
fi
if [[ ! -f "${yaml_file}" ]]; then
    echo "[CLIENT][ERROR] Deploy config not found: ${yaml_file}" >&2
    exit 1
fi

if [[ "${action_type}" != "joint" ]]; then
    echo "[CLIENT][ERROR] Pi_05 EgoVLA adapter supports action_type=joint only." >&2
    exit 2
fi

# The debug helper calls conda itself.  Make the private HPC installation
# visible in clean non-interactive SSH sessions without disturbing PATH.
if ! command -v conda >/dev/null 2>&1 && [[ -x /mnt/nfs/miniconda3/bin/conda ]]; then
    export PATH="/mnt/nfs/miniconda3/bin:${PATH}"
fi

# Resolve the requested environment mode before deciding whether conda is
# required.  The simulator bridge has its own direct Python runtime and does
# not need conda's shell function.
source "${UTILS_DIR}/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"
if [[ "${eval_env_mode}" == "debug" ]] && ! command -v conda >/dev/null 2>&1; then
    echo "[CLIENT][ERROR] conda is required by the debug eval client; set PATH or install conda." >&2
    exit 1
fi
if [[ "${eval_env_mode}" != "debug" && ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "[CLIENT][ERROR] Missing simulator entrypoint: ${BENCH_ROOT}/scripts/eval_policy.sh" >&2
    echo "[CLIENT][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] eval main root=${BENCH_ROOT}"

# DexBench / SparkArena uses the generic sim helper (conda + eval_policy.sh).
# The EgoVLA bridge flags below are not accepted by xspark-0-simeval.
if [[ "${bench_name}" != "EgoVLA" && "${eval_env_mode}" != "debug" ]]; then
    PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
        "${policy_server_port}" "${bench_name}" "${task_name}" \
        "${env_cfg_type}" "${policy_name}" "${additional_info}" \
        "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
    exit $?
fi

if [[ "${eval_env_mode}" == "debug" ]]; then
    # Keep the bundled synthetic client for transport and shape checks.
    PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
        "${policy_server_port}" "${bench_name}" "${task_name}" \
        "${env_cfg_type}" "${policy_name}" "${additional_info}" \
        "${BENCH_ROOT}" "${seed}" "${env_gpu_id}" "${policy_server_ip}"
    exit $?
fi

# The generic helper does not pass --xpolicy-root, --checkpoint, or --force;
# it would select the benchmark's old Pi_05 (32D) adapter and fail the
# deny-by-default capability gate.  Bind the private adapter explicitly.
eval_batch="$(awk -F: '/^[[:space:]]*eval_batch[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' "${yaml_file}")"
eval_batch="${eval_batch:-false}"
yaml_protocol="$(awk -F: '/^[[:space:]]*protocol[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' "${yaml_file}")"
yaml_protocol="${yaml_protocol:-ws}"
export EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}"
export EGOVLA_BRIDGE_ROOT="${BENCH_ROOT}/integration"
export XPOLICYLAB_ROOT="${XPL_ROOT}"
export PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${BENCH_ROOT}:${BENCH_ROOT}/integration/src:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}"

if [[ -n "${PI05_EVAL_CONDA_ENV:-}" && -x "${PI05_EVAL_CONDA_ENV}/bin/python" ]]; then
    export EGOVLA_EVAL_CONDA_ENV="${PI05_EVAL_CONDA_ENV}"
elif [[ -x "${eval_env_conda_env}/bin/python" ]]; then
    export EGOVLA_EVAL_CONDA_ENV="${eval_env_conda_env}"
elif [[ -x "${BENCH_ROOT}/.runtime/conda/egovla-isaaclab-1.2.0/bin/python" ]]; then
    export EGOVLA_EVAL_CONDA_ENV="${BENCH_ROOT}/.runtime/conda/egovla-isaaclab-1.2.0"
fi

bridge_args=(
    --bench_name "${bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --checkpoint "${ckpt_name}"
    --action-type "${action_type}"
    --host "${policy_server_ip}"
    --port "${policy_server_port}"
    --protocol "${yaml_protocol}"
    --benchmark-protocol "${EGOVLA_PROTOCOL:-official}"
    --eval_batch "${eval_batch}"
    --root_dir "${BENCH_ROOT}"
    --xpolicy-root "${XPL_ROOT}"
    # eval.sh receives physical GPU ids. Legacy Web snapshots may contain
    # stale EVAL_WEB_PHYSICAL_ENV_GPU=1 after moving a job to GPUs 2/3 or 4/5;
    # the explicit command argument remains the assigned environment GPU.
    --device_id "${env_gpu_id}"
    --additional_info "${additional_info}"
    --seed "${seed}"
    --force
)
if [[ -n "${EVAL_NUM:-}" ]]; then
    bridge_args+=(--episodes "${EVAL_NUM}")
fi
echo "[CLIENT] Isaac physical GPU=${env_gpu_id}; CUDA namespace unmasked"
exec env -u CUDA_VISIBLE_DEVICES "${BENCH_ROOT}/scripts/eval_policy.sh" "${bridge_args[@]}"
