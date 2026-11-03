#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_ids>" >&2
    echo "Example gpu_ids: 0,1,2,3,4,5,6,7" >&2
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_ids=$6

if [[ "${action_type}" != "joint" ]]; then
    echo "[RLDX_1] Spark integration supports action_type=joint only." >&2
    exit 2
fi
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
if [[ ${#gpu_array[@]} -ne 8 ]]; then
    echo "[RLDX_1] This recipe requires exactly 8 visible GPUs; got ${gpu_ids}." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"
DEFAULT_DATASET_PATH="${XPL_ROOT}/data/spark0_bench_7task_0908"
DATASET_PATH="${RLDX_DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
SOURCE_BASE_MODEL_PATH="${RLDX_SOURCE_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a}"
BASE_MODEL_PATH="${RLDX_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-sparkarena-relarm}"
MODALITY_CONFIG_PATH="${SCRIPT_DIR}/spark_joint54_relative_config.py"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-relarm-${seed}"
checkpoints_root="${SCRIPT_DIR}/checkpoints"
ckpt_dir="${checkpoints_root}/${ckpt_setting}"

if [[ ! -x "${UPSTREAM}/.venv/bin/python" ]]; then
    echo "Missing RLDX uv environment. Run ${SCRIPT_DIR}/install.sh first." >&2
    exit 1
fi
TORCHRUN="${UPSTREAM}/.venv/bin/torchrun"
if [[ ! -x "${TORCHRUN}" ]]; then
    echo "Missing RLDX torchrun executable: ${TORCHRUN}" >&2
    exit 1
fi
if [[ ! -d "${SOURCE_BASE_MODEL_PATH}" ]]; then
    echo "Missing source base model: ${SOURCE_BASE_MODEL_PATH}" >&2
    echo "Set RLDX_SOURCE_BASE_MODEL_PATH or restore the documented pretrain_model symlink." >&2
    exit 1
fi
for required in meta/info.json meta/episodes.jsonl meta/tasks.jsonl meta/stats.json meta/modality.json; do
    if [[ ! -f "${DATASET_PATH}/${required}" ]]; then
        echo "Dataset is incomplete; missing ${DATASET_PATH}/${required}" >&2
        exit 1
    fi
done
if ! "${UPSTREAM}/.venv/bin/python" - <<'PY'
import wandb
raise SystemExit(0 if getattr(wandb.api, "api_key", None) else 1)
PY
then
    echo "W&B is not authenticated; export WANDB_API_KEY or run wandb login." >&2
    exit 1
fi

mkdir -p "${checkpoints_root}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_PROJECT="${WANDB_PROJECT:-spark0-bench-rldx1}"
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"
export PYTHONPATH="${UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RLDX_SOURCE_BASE_MODEL_PATH="${SOURCE_BASE_MODEL_PATH}" \
RLDX_RELATIVE_BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
RLDX_UPSTREAM_PATH="${UPSTREAM}" \
RLDX_PYTHON_BIN="${UPSTREAM}/.venv/bin/python" \
    bash "${SCRIPT_DIR}/prepare_sparkarena_relative_base_overlay.sh"

"${UPSTREAM}/.venv/bin/python" \
    "${XPL_ROOT}/data_scripts/compute_sparkarena_7task_relative_stats.py" \
    --dataset "${DATASET_PATH}" \
    --modality-config "${MODALITY_CONFIG_PATH}"

echo "[RLDX_1] checkpoints will be written under ${ckpt_dir}"
echo "[RLDX_1] action contract: arms relative to current qpos; hands absolute"
cd "${UPSTREAM}"
exec "${TORCHRUN}" \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    rldx/experiment/launch_train.py \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG_PATH}" \
    --n-cog-tokens 64 \
    --video-length 4 \
    --video-stride 2 \
    --global-batch-size 64 \
    --gradient-accumulation-steps 1 \
    --num-gpus 8 \
    --learning-rate 1e-4 \
    --max-steps 100000 \
    --save-steps 5000 \
    --save-total-limit 20 \
    --output-dir "${checkpoints_root}" \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --experiment-name "${ckpt_setting}"
