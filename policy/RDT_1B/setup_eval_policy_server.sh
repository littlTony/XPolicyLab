#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
    echo "usage: $0 bench task checkpoint env_cfg_type action_type seed policy_gpu policy_env port [host]" >&2
    exit 2
fi

bench_name="$1"
task_name="$2"
ckpt_name="$3"
env_cfg_type="$4"
action_type="$5"
seed="$6"
policy_gpu_id="$7"
policy_conda_env="$8"
policy_server_port="$9"
policy_server_host="${10:-localhost}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
POLICY_NAME="$(basename "${SCRIPT_DIR}")"
YAML_FILE="${XPL_ROOT}/policy/${POLICY_NAME}/deploy.yml"
MAP_FILE="${XPL_ROOT}/policy/${POLICY_NAME}/rdt/configs/egovla_joint38.py"

if [[ ! -f "${YAML_FILE}" ]]; then
    echo "[ERROR] deployment config is missing: ${YAML_FILE}" >&2
    exit 1
fi
if [[ "${action_type}" != "joint" ]]; then
    echo "[ERROR] RDT_1B supports only action_type=joint (got ${action_type@Q})" >&2
    exit 2
fi
if [[ "${ckpt_name}" == /* && ! -e "${ckpt_name}" ]]; then
    echo "[ERROR] checkpoint path does not exist: ${ckpt_name}" >&2
    exit 1
fi

action_dim="$(bash "${UTILS_DIR}/get_action_dim.sh" "${MODEL_ROOT}" "${env_cfg_type}" | tr -d '[:space:]')"
if [[ ! "${action_dim}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] could not resolve numeric action dimension for ${env_cfg_type}: ${action_dim@Q}" >&2
    exit 1
fi
if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    [[ -f "${MAP_FILE}" ]] || { echo "[ERROR] missing EgoVLA 38D map: ${MAP_FILE}" >&2; exit 1; }
    [[ "${action_dim}" == "38" ]] || { echo "[ERROR] ego_h1_inspire must resolve to 38D, got ${action_dim}" >&2; exit 1; }
fi

if [[ -n "${EVAL_MAIN_ROOT:-}" ]]; then
    BENCH_ROOT="${EVAL_MAIN_ROOT}"
elif [[ -n "${EGOVLA_WORKSPACE_ROOT:-}" ]]; then
    BENCH_ROOT="${EGOVLA_WORKSPACE_ROOT}"
elif [[ -d "/personal/xiangpc/EgoVLA benchmark" ]]; then
    BENCH_ROOT="/personal/xiangpc/EgoVLA benchmark"
else
    BENCH_ROOT="${MODEL_ROOT}"
fi
if [[ -d "${BENCH_ROOT}" ]]; then BENCH_ROOT="$(cd "${BENCH_ROOT}" && pwd)"; fi

# shellcheck source=../../utils/conda_env.sh
source "${UTILS_DIR}/conda_env.sh"
if ! xpl_activate_env "${policy_conda_env}"; then
    echo "[ERROR] cannot activate policy environment ${policy_conda_env@Q}; set XPL_CONDA_SH/RDT_CONDA_BASE or pass an absolute prefix" >&2
    exit 1
fi
policy_python="$(xpl_python_for_env "${policy_conda_env}" 2>/dev/null || true)"
if [[ -z "${policy_python}" || ! -x "${policy_python}" ]]; then
    echo "[ERROR] cannot find policy Python for ${policy_conda_env@Q}" >&2
    exit 1
fi

# Optional completeness gate prevents accidentally loading a still-being-written
# training directory. Leave it opt-in so legacy checkpoints with custom names
# remain usable.
if [[ "${RDT_EVAL_REQUIRE_COMPLETE:-0}" == "1" && "${ckpt_name}" == /* ]]; then
    if [[ -d "${ckpt_name}" ]]; then
        has_config=0; has_weights=0
        # Check through the standard checkpoint symlink as well as a direct
        # directory; plain find does not traverse a symlink operand.
        find -L "${ckpt_name}" -maxdepth 3 -type f -name 'config.json' -print -quit | grep -q . && has_config=1 || true
        find -L "${ckpt_name}" -maxdepth 3 -type f \( -name 'pytorch_model.bin' -o -name 'pytorch_model' -o -name 'model.safetensors' \) -print -quit | grep -q . && has_weights=1 || true
        if [[ "${has_config}" != 1 || "${has_weights}" != 1 ]]; then
            echo "[ERROR] checkpoint directory is not complete (need config + model weights): ${ckpt_name}" >&2
            exit 1
        fi
    elif [[ ! -f "${ckpt_name}" ]]; then
        echo "[ERROR] checkpoint target is neither a file nor directory: ${ckpt_name}" >&2
        exit 1
    fi
fi

camera_mode="${RDT_CAMERA_MODE:-}"
if [[ -z "${camera_mode}" ]]; then
    if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then camera_mode="black_wrist"; else camera_mode="real_wrist"; fi
fi
camera_mode="${camera_mode,,}"
camera_mode="${camera_mode//-/_}"
case "${camera_mode}" in
    contract)
        if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then camera_mode="black_wrist"; else camera_mode="real_wrist"; fi
        ;;
    black_wrist|main_replicated|real_wrist) ;;
    *) echo "[ERROR] RDT_CAMERA_MODE must be black_wrist, main_replicated, or real_wrist (got ${camera_mode@Q})" >&2; exit 2 ;;
esac

ctrl_freq="${RDT_CTRL_FREQ:-}"
if [[ -z "${ctrl_freq}" ]]; then
    if [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then ctrl_freq=30; else ctrl_freq=25; fi
fi
[[ "${ctrl_freq}" =~ ^[0-9]+$ ]] || { echo "[ERROR] RDT_CTRL_FREQ must be an integer" >&2; exit 2; }

if [[ -n "${RDT_IMAGE_SIZE:-}" ]]; then
    image_size="${RDT_IMAGE_SIZE}"
elif [[ "${env_cfg_type}" == "ego_h1_inspire" ]]; then
    image_size="[384, 384]"
else
    # SparkArena / Tianji training HDF5 is 640x480 and is letterboxed, not stretched.
    image_size="[640, 480]"
fi

lang_embed_dir="${RDT_LANG_EMBED_DIR:-}"
if [[ -z "${lang_embed_dir}" && "${env_cfg_type}" != "ego_h1_inspire" ]]; then
    if [[ -d "${SCRIPT_DIR}/lang_embeds_0908_7task" ]]; then
        lang_embed_dir="${SCRIPT_DIR}/lang_embeds_0908_7task"
    fi
fi

overrides=(
    "port=${policy_server_port}"
    "host=${policy_server_host}"
    "bench_name=${bench_name}"
    "task_name=${task_name}"
    "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}"
    "seed=${seed}"
    "policy_name=${POLICY_NAME}"
    "action_type=${action_type}"
    "action_dim=${action_dim}"
    "camera_mode=${camera_mode}"
    "ctrl_freq=${ctrl_freq}"
    "image_size=${image_size}"
)
if [[ -n "${lang_embed_dir}" ]]; then
    overrides+=("lang_embed_dir=${lang_embed_dir}")
fi
if [[ -n "${RDT_CHECKPOINT_NUM:-}" ]]; then overrides+=("checkpoint_num=${RDT_CHECKPOINT_NUM}"); fi
if [[ -n "${RDT_PROMPT:-}" ]]; then overrides+=("prompt=${RDT_PROMPT}"); fi
if [[ -n "${EGOVLA_INSTRUCTION:-}" && -z "${RDT_PROMPT:-}" ]]; then overrides+=("prompt=${EGOVLA_INSTRUCTION}"); fi
if [[ -n "${RDT_EVAL_BATCH:-}" ]]; then overrides+=("eval_batch=${RDT_EVAL_BATCH}"); fi

policy_pythonpath="${MODEL_ROOT}:${XPL_ROOT}"
if [[ -d "${BENCH_ROOT}" && "${BENCH_ROOT}" != "${MODEL_ROOT}" ]]; then policy_pythonpath+=":${BENCH_ROOT}"; fi
if [[ -n "${PYTHONPATH:-}" ]]; then policy_pythonpath+=":${PYTHONPATH}"; fi
export XPOLICYLAB_ROOT="${XPL_ROOT}"
export EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}"

echo "[SERVER] policy=${POLICY_NAME}, task=${task_name}, action_dim=${action_dim}, camera_mode=${camera_mode}, ctrl_freq=${ctrl_freq}, image_size=${image_size}"
echo "[SERVER] checkpoint=${ckpt_name}"
echo "[SERVER] policy_python=${policy_python}, XPolicyLab_root=${XPL_ROOT}"

if [[ "${RDT_EVAL_DRY_RUN:-0}" == "1" ]]; then
    echo "[SERVER][DRY-RUN] PYTHONPATH=${policy_pythonpath}"
    printf '[SERVER][DRY-RUN] overrides:'
    printf ' %q' "${overrides[@]}"
    printf '\n'
    exit 0
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    RDT_T5_DEVICE="${RDT_T5_DEVICE:-cpu}" \
    PYTHONPATH="${policy_pythonpath}" \
    "${policy_python}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${YAML_FILE}" \
        --overrides "${overrides[@]}"
