#!/bin/bash
set -euo pipefail
if (( $# < 9 || $# > 10 )); then
    echo 'Usage: setup_eval_policy_server.sh <bench> <task|auto> <checkpoint/run> <env_cfg> ee <seed> <gpu> <uv|project_path> <port> [host]' >&2
    exit 2
fi
source "$(dirname "$(readlink -f "$0")")/runtime.sh"

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
if [[ "${action_type}" != ee ]]; then
    echo '[SERVER] DexBench Pi0.5 requires action_type=ee' >&2
    exit 2
fi
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=$9
policy_server_host=${10:-"localhost"}
checkpoint_num="${PI05_CHECKPOINT_NUM:-null}"
if [[ "${checkpoint_num}" != null && ! "${checkpoint_num}" =~ ^[0-9]+$ ]]; then
    echo 'PI05_CHECKPOINT_NUM must be a numeric finalized checkpoint or null for latest' >&2
    exit 2
fi

# Keep the proven 30% pool on 24GiB cards, but do not reserve 24GiB merely
# because the same model runs on an 80GiB card. Explicit operator overrides win.
# This controls the JAX pool, not all CUDA/context allocations in the process.
if [[ -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
    if ! pi05_gpu_memory_mib="$(nvidia-smi --id="${policy_gpu_id}" \
        --query-gpu=memory.total --format=csv,noheader,nounits)"; then
        echo "[SERVER][ERROR] Cannot determine Pi GPU capacity; set XLA_PYTHON_CLIENT_MEM_FRACTION explicitly if needed." >&2
        exit 1
    fi
    pi05_gpu_memory_mib="${pi05_gpu_memory_mib//[[:blank:]]/}"
    if [[ ! "${pi05_gpu_memory_mib}" =~ ^[0-9]+$ ]] || (( pi05_gpu_memory_mib <= 0 )); then
        echo "[SERVER][ERROR] Pi requires the capacity of one selected GPU; got: ${pi05_gpu_memory_mib}" >&2
        exit 1
    fi
    XLA_PYTHON_CLIENT_MEM_FRACTION="$(awk -v total="${pi05_gpu_memory_mib}" \
        'BEGIN { fraction=10240/total; if (fraction>0.3) fraction=0.3; printf "%.6f", fraction }')"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION
echo "[SERVER] JAX memory fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION}, physical GPU=${policy_gpu_id}, preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-${DEFAULT_BENCH_ROOT}}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${PI05_DEPLOY_CONFIG:-${XPL_ROOT}/policy/${policy_name}/deploy.yml}"
if [[ "${yaml_file}" != /* ]]; then
    yaml_file="${SCRIPT_DIR}/${yaml_file}"
fi
if [[ ! -f "${yaml_file}" ]]; then
    echo "[SERVER][ERROR] Deploy config not found: ${yaml_file}" >&2
    exit 1
fi
if [[ ! -f "${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" ]]; then
    echo "[SERVER][ERROR] Missing simulator env config: ${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" >&2
    echo "[SERVER][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] eval main root=${BENCH_ROOT}"

# Resolve YAML/configuration with a local interpreter. Non-interactive SSH
# sessions on the rented node do not put conda on PATH, and debug evaluation
# must not depend on an unrelated user's shell initialization.
if [[ -n "${PI05_CONFIG_PYTHON:-}" && -x "${PI05_CONFIG_PYTHON}" ]]; then
    YAML_PYTHON="${PI05_CONFIG_PYTHON}"
elif [[ -n "${PI05_PYTHON_BIN:-}" && -x "${PI05_PYTHON_BIN}" ]]; then
    YAML_PYTHON="${PI05_PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    YAML_PYTHON="${CONDA_BASE}/bin/python"
elif [[ -x /mnt/nfs/miniconda3/bin/python ]]; then
    YAML_PYTHON=/mnt/nfs/miniconda3/bin/python
elif command -v python3 >/dev/null 2>&1; then
    YAML_PYTHON="$(command -v python3)"
else
    echo "[SERVER][ERROR] No Python interpreter available to read deploy.yml." >&2
    exit 1
fi

resolve_uv_env() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        "${YAML_PYTHON}" - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${SCRIPT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8")) or {}
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    else
        "${YAML_PYTHON}" - <<PYENV
from pathlib import Path
script_dir = Path("${SCRIPT_DIR}")
path = Path("${raw_path}").expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
    fi
}

policy_uv_env_path="$(resolve_uv_env "${policy_uv_env}")"

# Keep the project source and its Python runtime separate. The old adapter
# assumed ``.venv`` (a symlink to another checkout); EgoVLA is installed in the
# private ``.venv-egovla`` environment. An explicit environment variable has
# highest precedence for one-off debugging without changing deploy.yml.
configured_python="$("${YAML_PYTHON}" - <<PYENV
import yaml
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8")) or {}
print(cfg.get("policy_python_bin") or "")
PYENV
)"
if [[ -n "${PI05_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PI05_PYTHON_BIN}"
elif [[ -n "${configured_python}" ]]; then
    PYTHON_BIN="${configured_python}"
else
    if [[ -x "${policy_uv_env_path}/.venv-egovla/bin/python" ]]; then
        PYTHON_BIN="${policy_uv_env_path}/.venv-egovla/bin/python"
    else
        PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
    fi
fi
if [[ "${PYTHON_BIN}" != /* ]]; then
    PYTHON_BIN="${policy_uv_env_path}/${PYTHON_BIN}"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[SERVER][ERROR] Policy Python not found or not executable: ${PYTHON_BIN}" >&2
    echo "[SERVER][ERROR] Set PI05_PYTHON_BIN or run: bash ${SCRIPT_DIR}/install.sh" >&2
    exit 1
fi

OPENPI_SRC="${policy_uv_env_path}/src"
OPENPI_CLIENT_SRC="${policy_uv_env_path}/packages/openpi-client/src"
echo "[SERVER] Using project: ${policy_uv_env_path}"
echo "[SERVER] Using python: ${PYTHON_BIN}"

# ``XPolicyLab.py`` makes this standalone clone importable as the
# ``XPolicyLab`` package. Keep it ahead of the shared simulator checkout so
# the selected policy never resolves to another checkout's implementation.
PYTHONPATH_PARTS=("${XPL_ROOT}/.." "${XPL_ROOT}" "${BENCH_ROOT}" "${BENCH_ROOT}/XPolicyLab")
if [[ -d "${OPENPI_SRC}" ]]; then
    PYTHONPATH_PARTS+=("${OPENPI_SRC}")
fi
if [[ -d "${OPENPI_CLIENT_SRC}" ]]; then
    PYTHONPATH_PARTS+=("${OPENPI_CLIENT_SRC}")
fi

exec env \
    PYTHONUNBUFFERED=1 \
    JAX_COMPILATION_CACHE_DIR="${PI05_JAX_CACHE_DIR:-${SCRIPT_DIR}/.cache/jax}" \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
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
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_num="${checkpoint_num}"
