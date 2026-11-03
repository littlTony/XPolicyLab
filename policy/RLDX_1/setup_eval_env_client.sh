#!/usr/bin/env bash
set -euo pipefail

# The client starts IsaacLab/Isaac Sim in a non-interactive Web worker.  Keep
# the environment deterministic and avoid an EOF from the EULA prompt while
# retaining an explicit caller override.
: "${OMNI_KIT_ACCEPT_EULA:=Y}"
export OMNI_KIT_ACCEPT_EULA
: "${ACCEPT_EULA:=Y}"
export ACCEPT_EULA

if [[ $# -lt 10 ]]; then
    echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <env_gpu_id> <eval_env> <additional_info> <port> [host]" >&2
    exit 2
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

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
if [[ "${policy_name}" != "RLDX_1" ]]; then
    echo "[CLIENT][ERROR] this entrypoint must resolve to policy/RLDX_1, got ${policy_name}" >&2
    exit 1
fi
RLDX_POLICY_ROOT="${XPL_ROOT}/policy/${policy_name}"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

# DexBench / SparkArena talks to xspark-0-simeval.  Keep the EgoVLA bridge
# and bootstrap below this branch; they are not present on the sim checkout.
if [[ "${bench_name}" != "EgoVLA" ]]; then
    BENCH_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
    if [[ ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
        echo "[CLIENT][ERROR] Missing simulator entrypoint: ${BENCH_ROOT}/scripts/eval_policy.sh" >&2
        echo "[CLIENT][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
        exit 1
    fi
    echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
    echo "[CLIENT] eval main root=${BENCH_ROOT}"
    # Drop the policy venv from PYTHONPATH.  Isaac/Luminis crashes if it
    # imports numpy/h5py from RLDX-1/.venv (0807 site-packages).
    PYTHONPATH="${XPL_ROOT}:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab" \
    bash "${UTILS_DIR}/setup_env_client.sh" \
        "${UTILS_DIR}" "${yaml_file}" "${eval_env_conda_env}" \
        "${policy_server_port}" "${bench_name}" "${task_name}" "${env_cfg_type}" \
        "${policy_name}" "${additional_info}" "${BENCH_ROOT}" "${seed}" \
        "${env_gpu_id}" "${policy_server_ip}"
    exit $?
fi

DEFAULT_BENCH_ROOT="/mnt/xspark-data/xiangpc/EgoVLA benchmark"
if [[ ! -d "${DEFAULT_BENCH_ROOT}" ]]; then
    DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
fi
BENCH_ROOT="${EVAL_MAIN_ROOT:-${EGOVLA_WORKSPACE_ROOT:-${DEFAULT_BENCH_ROOT}}}"
BRIDGE_ROOT="${EGOVLA_BRIDGE_ROOT:-${BENCH_ROOT}/integration}"

if [[ ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "[CLIENT][ERROR] Missing simulator entrypoint: ${BENCH_ROOT}/scripts/eval_policy.sh" >&2
    echo "[CLIENT][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi
if [[ ! -d "${BRIDGE_ROOT}/src/egovla_xpolicy" ]]; then
    echo "[CLIENT][ERROR] bridge root lacks integration/src/egovla_xpolicy: ${BRIDGE_ROOT}" >&2
    exit 1
fi

DEFAULT_EVAL_PYTHON_BIN="${BENCH_ROOT}/.runtime/conda/egovla-isaaclab-1.2.0/bin/python"

resolve_python_bin() {
    local reference candidate
    for reference in "${RLDX_EVAL_PYTHON_BIN:-}" "${EVAL_PYTHON_BIN:-}" "${eval_env_conda_env}"; do
        [[ -z "${reference}" ]] && continue
        for candidate in \
            "${reference}" \
            "${reference}/bin/python" \
            "${reference}/bin/python3" \
            "${reference}/.venv/bin/python" \
            "${reference}/.venv/bin/python3"; do
            if [[ -f "${candidate}" && -x "${candidate}" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done
    done
    if [[ -f "${DEFAULT_EVAL_PYTHON_BIN}" && -x "${DEFAULT_EVAL_PYTHON_BIN}" ]]; then
        printf '%s\n' "${DEFAULT_EVAL_PYTHON_BIN}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    return 1
}

if ! EVAL_PYTHON_BIN="$(resolve_python_bin)"; then
    echo "[CLIENT][ERROR] Unable to locate an evaluation Python executable. Set RLDX_EVAL_PYTHON_BIN or pass an environment path." >&2
    exit 1
fi

RLDX_UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT:-${RLDX_POLICY_ROOT}/RLDX-1}"
POLICY_SITE_PACKAGES="${RLDX_POLICY_SITE_PACKAGES:-${RLDX_SITE_PACKAGES:-${RLDX_UPSTREAM_ROOT}/.venv/lib/python3.10/site-packages}}"
export RLDX_POLICY_SITE_PACKAGES="${POLICY_SITE_PACKAGES}"
BRIDGE_BOOTSTRAP="${RLDX_BRIDGE_BOOTSTRAP:-${SCRIPT_DIR}/egovla_bridge_bootstrap.py}"
PRIVATE_MANIFEST="${RLDX_PRIVATE_MANIFEST:-${SCRIPT_DIR}/rldx_capabilities.json}"

# The private checkout root contains the XPolicyLab shim and policy package.
# It must be first so the benchmark's older XPolicyLab checkout cannot shadow
# the private RLDX_1 adapter.
XPL_PACKAGE_ROOT="${RLDX_XPOLICYLAB_PACKAGE_ROOT:-${XPL_ROOT}}"
if [[ "${XPL_PACKAGE_ROOT}" != "${XPL_ROOT}" ]]; then
    echo "[CLIENT][ERROR] private package root must equal XPL_ROOT: ${XPL_PACKAGE_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${XPL_PACKAGE_ROOT}/XPolicyLab.py" && ! -d "${XPL_PACKAGE_ROOT}/XPolicyLab" ]]; then
    echo "[CLIENT][ERROR] Invalid private XPolicyLab package root: ${XPL_PACKAGE_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${XPL_PACKAGE_ROOT}/policy/${policy_name}" && ! -d "${XPL_PACKAGE_ROOT}/XPolicyLab/policy/${policy_name}" ]]; then
    echo "[CLIENT][ERROR] Missing private adapter package XPolicyLab.policy.${policy_name}" >&2
    exit 1
fi
if [[ ! -r "${BRIDGE_BOOTSTRAP}" ]]; then
    echo "[CLIENT][ERROR] private bridge bootstrap is missing: ${BRIDGE_BOOTSTRAP}" >&2
    exit 1
fi
if [[ ! -r "${PRIVATE_MANIFEST}" ]]; then
    echo "[CLIENT][ERROR] private RLDX capability manifest is missing: ${PRIVATE_MANIFEST}" >&2
    exit 1
fi

client_pythonpath="${XPL_ROOT}:${RLDX_POLICY_ROOT}:${POLICY_SITE_PACKAGES}:${RLDX_UPSTREAM_ROOT}:${BRIDGE_ROOT}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
    client_pythonpath="${client_pythonpath}:${PYTHONPATH}"
fi

echo "[CLIENT] XPL_ROOT=${XPL_ROOT}"
echo "[CLIENT] PYTHONPATH=${client_pythonpath}"

# Fail closed before starting the simulator if Python resolves a stale
# benchmark package instead of the private adapter.  The bootstrap removes the
# implicit cwd entry and purges any pre-imported flat XPolicyLab module, then
# verifies the four private policy imports and their origins.
import_check_log="/tmp/rldx_client_import_check.$$.log"
if ! RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_PACKAGE_ROOT}" \
    OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}" \
    ACCEPT_EULA="${ACCEPT_EULA:-Y}" \
    EGOVLA_POLICY_ADAPTER_ROOT="${XPL_PACKAGE_ROOT}" \
    EGOVLA_POLICY_NAME="${policy_name}" \
    RLDX_UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT}" \
    EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}" \
    EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}" \
    PYTHONPATH="${client_pythonpath}" "${EVAL_PYTHON_BIN}" \
    "${BRIDGE_BOOTSTRAP}" --check-only \
    >"${import_check_log}" 2>&1 <<'PY'
# The Python body is intentionally empty; the private bootstrap is the gate.
pass
PY
then
    cat "${import_check_log}" >&2 || true
    rm -f "${import_check_log}"
    echo "[CLIENT][ERROR] XPolicyLab.policy.${policy_name} is not importable with the private PYTHONPATH." >&2
    exit 1
fi
cat "${import_check_log}"
rm -f "${import_check_log}"

eval_batch="$(awk -F: '/^[[:space:]]*eval_batch[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' "${yaml_file}")"
eval_batch="${eval_batch:-false}"
eval_env_mode="${EVAL_ENV_TYPE:-sim}"
case "${eval_env_mode}" in
    debug)
        echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
        echo "[CLIENT] debug Python=${EVAL_PYTHON_BIN}"
        echo "[CLIENT] private XPolicyLab root=${XPL_PACKAGE_ROOT}"
        export PYTHONPATH="${client_pythonpath}"
        export XPOLICYLAB_ROOT="${XPL_PACKAGE_ROOT}"
        exec "${EVAL_PYTHON_BIN}" "${XPL_ROOT}/debug_env_client.py" \
            --bench_name "${bench_name}" \
            --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" \
            --policy_name "${policy_name}" \
            --protocol ws \
            --host "${policy_server_ip}" \
            --port "${policy_server_port}" \
            --eval_batch "${eval_batch}" \
            --eval_episode_num "${RLDX_DEBUG_EPISODES:-1}" \
            --obs_encoded "${DEBUG_OBS_ENCODED:-0}"
        ;;
    sim)
        echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
        echo "[CLIENT] EgoVLA root=${BENCH_ROOT}"
        echo "[CLIENT] eval Python=${EVAL_PYTHON_BIN}"
        eval_prefix=""
        case "${EVAL_PYTHON_BIN}" in
            */bin/python|*/bin/python3|*/bin/python3.*)
                eval_prefix="${EVAL_PYTHON_BIN%/bin/*}"
                ;;
        esac
        export EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}"
        export EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}"
        export XPOLICYLAB_ROOT="${XPL_PACKAGE_ROOT}"
        export RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_PACKAGE_ROOT}"
        export EGOVLA_POLICY_ADAPTER_ROOT="${XPL_PACKAGE_ROOT}"
        export EGOVLA_POLICY_NAME="${policy_name}"
        export RLDX_UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT}"
        export PYTHONPATH="${client_pythonpath}"
        if [[ -n "${eval_prefix}" && -x "${eval_prefix}/bin/python" ]]; then
            export EGOVLA_EVAL_CONDA_ENV="${eval_prefix}"
        elif [[ -n "${EGOVLA_EVAL_CONDA_ENV:-}" && -x "${EGOVLA_EVAL_CONDA_ENV}/bin/python" ]]; then
            : # Preserve an explicitly supplied bridge runtime prefix.
        else
            unset EGOVLA_EVAL_CONDA_ENV || true
        fi

        # Invoke through the private bootstrap.  It purges a benchmark
        # XPolicyLab package already imported from cwd before entering the
        # otherwise unchanged bridge CLI.  No --force escape hatch is used.
        bridge_args=(
            --episodes "${EGOVLA_EPISODES:-${EVAL_NUM:-1}}"
            --num-envs "${EGOVLA_NUM_ENVS:-1}"
            --request-timeout-s "${EGOVLA_REQUEST_TIMEOUT_S:-600}"
        )
        if [[ -n "${EGOVLA_OUTPUT_DIR:-}" ]]; then
            bridge_args+=(--output-dir "${EGOVLA_OUTPUT_DIR}")
        fi
        if [[ -n "${EGOVLA_OFFICIAL_SPLIT:-}" ]]; then
            bridge_args+=(
                --official-split "${EGOVLA_OFFICIAL_SPLIT}"
                --official-episodes "${EGOVLA_OFFICIAL_EPISODES:-${EGOVLA_EPISODES:-1}}"
                --official-trials "${EGOVLA_OFFICIAL_TRIALS:-1}"
            )
        fi
        if [[ -n "${EGOVLA_ROOM_IDX:-}" ]]; then
            bridge_args+=(--room-idx "${EGOVLA_ROOM_IDX}")
        fi
        if [[ -n "${EGOVLA_TABLE_IDX:-}" ]]; then
            bridge_args+=(--table-idx "${EGOVLA_TABLE_IDX}")
        fi
        cd "${XPL_PACKAGE_ROOT}"
        exec env \
            OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}" \
            ACCEPT_EULA="${ACCEPT_EULA:-Y}" \
            PYTHONPATH="${client_pythonpath}" \
            EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}" \
            EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}" \
            XPOLICYLAB_ROOT="${XPL_PACKAGE_ROOT}" \
            RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_PACKAGE_ROOT}" \
            EGOVLA_POLICY_ADAPTER_ROOT="${XPL_PACKAGE_ROOT}" \
            EGOVLA_POLICY_NAME="${policy_name}" \
            RLDX_UPSTREAM_ROOT="${RLDX_UPSTREAM_ROOT}" \
            "${EVAL_PYTHON_BIN}" "${BRIDGE_BOOTSTRAP}" \
            env-client \
            --manifest "${PRIVATE_MANIFEST}" \
            --bench_name "${bench_name}" \
            --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" \
            --policy_name "${policy_name}" \
            --checkpoint "${ckpt_name}" \
            --action-type "${action_type}" \
            --host "${policy_server_ip}" \
            --port "${policy_server_port}" \
            --transport ws \
            --benchmark-protocol "${EGOVLA_PROTOCOL:-official}" \
            --eval_batch "${eval_batch}" \
            --root_dir "${BENCH_ROOT}" \
            --xpolicy-root "${XPL_PACKAGE_ROOT}" \
            --device_id "${env_gpu_id}" \
            --additional_info "${additional_info}" \
            --seed "${seed}" \
            "${bridge_args[@]}"
        ;;
    real|real_world)
        echo "[CLIENT][ERROR] RLDX_1 private adapter has no real-world client; use EVAL_ENV_TYPE=sim or debug." >&2
        exit 2
        ;;
    *)
        echo "[CLIENT][ERROR] Unknown EVAL_ENV_TYPE=${eval_env_mode}; expected debug, sim, or real_world." >&2
        exit 2
        ;;
esac
