#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids>" >&2
    echo "gpu_ids must list eight devices, for example: 0,1,2,3,4,5,6,7" >&2
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_ids=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_ROOT="${SCRIPT_DIR}/UniT"
DATA_ROOT="${DATA_ROOT:-${XPL_ROOT}/data/Spark0_bench_lerobotV20_joint54}"

# Exact joint54 run recipe. Upstream UniT's --batch-size is per device.
NUM_GPUS=8
GLOBAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DUAL_GRADIENT_CHECKPOINTING="${DUAL_GRADIENT_CHECKPOINTING:-0}"
DUAL_DEEPSPEED_CONFIG="${DUAL_DEEPSPEED_CONFIG:-}"
DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY="${DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY:-0}"
DUAL_TRAINING_BF16="${DUAL_TRAINING_BF16:-1}"
TOKENIZER_STEPS=40000
PRETRAIN_STEPS="${PRETRAIN_STEPS:-80000}"
SAVE_STEPS=10000
if ! [[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: gradient accumulation steps must be a positive integer" >&2
    exit 1
fi
if ! [[ "${PRETRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: PRETRAIN_STEPS must be a positive integer" >&2
    exit 1
fi
TOKENIZER_SAVE_TOTAL_LIMIT=$(((TOKENIZER_STEPS + SAVE_STEPS - 1) / SAVE_STEPS))
PRETRAIN_SAVE_TOTAL_LIMIT=$(((PRETRAIN_STEPS + SAVE_STEPS - 1) / SAVE_STEPS))
if (( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRADIENT_ACCUMULATION_STEPS) != 0 )); then
    echo "error: global batch size is not divisible by GPU count and accumulation" >&2
    exit 1
fi
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRADIENT_ACCUMULATION_STEPS))
dual_gradient_checkpointing_args=()
dual_deepspeed_args=()
dual_deepspeed_boundary_args=()
dual_bf16_args=()
case "${DUAL_GRADIENT_CHECKPOINTING}" in
    0) ;;
    1) dual_gradient_checkpointing_args+=(--gradient-checkpointing) ;;
    *) echo "error: DUAL_GRADIENT_CHECKPOINTING must be 0 or 1" >&2; exit 1 ;;
esac
if [[ -n "${DUAL_DEEPSPEED_CONFIG}" ]]; then
    if [[ ! -f "${DUAL_DEEPSPEED_CONFIG}" ]]; then
        echo "error: DeepSpeed config does not exist: ${DUAL_DEEPSPEED_CONFIG}" >&2
        exit 1
    fi
    dual_deepspeed_args+=(--deepspeed "${DUAL_DEEPSPEED_CONFIG}")
fi
case "${DUAL_TRAINING_BF16}" in
    1) ;;
    0) dual_bf16_args+=(--no-bf16) ;;
    *) echo "error: DUAL_TRAINING_BF16 must be 0 or 1" >&2; exit 1 ;;
esac
case "${DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY}" in
    0) ;;
    1)
        if [[ -z "${DUAL_DEEPSPEED_CONFIG}" ]]; then
            echo "error: DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY=1 requires DeepSpeed" >&2
            exit 1
        fi
        dual_deepspeed_boundary_args+=(--deepspeed-force-sync-boundary) ;;
    *) echo "error: DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY must be 0 or 1" >&2; exit 1 ;;
esac

if [[ "${action_type}" != "joint" ]]; then
    echo "error: UniT Spark adapter supports action_type=joint only" >&2
    exit 1
fi
if [[ "${seed}" != "42" ]]; then
    echo "error: official UniT trainers hard-code seed=42; pass seed 42 so the checkpoint name is truthful" >&2
    exit 1
fi
if [[ ! -f "${XPL_ROOT}/../env_cfg/${env_cfg_type}.yml" ]]; then
    echo "error: missing benchmark env config: ${XPL_ROOT}/../env_cfg/${env_cfg_type}.yml" >&2
    exit 1
fi
if [[ ! -f "${UPSTREAM_ROOT}/scripts/groot_tokenizer_train.py" || ! -f "${UPSTREAM_ROOT}/scripts/dual_system_train.py" ]]; then
    echo "error: official UniT checkout is incomplete at ${UPSTREAM_ROOT}" >&2
    exit 1
fi
if (( PER_DEVICE_BATCH_SIZE < 1 )); then
    echo "error: per-device batch must be at least 1, got ${PER_DEVICE_BATCH_SIZE}" >&2
    exit 1
fi

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
if [[ ${#gpu_array[@]} -ne ${NUM_GPUS} ]]; then
    echo "error: expected ${NUM_GPUS} GPU ids, got ${#gpu_array[@]} in '${gpu_ids}'" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"

UNIT_ENV="${UNIT_ENV:-/personal/miniconda3/envs/unit_torch26_recovery}"
PYTHON_BIN="${PYTHON_BIN:-${UNIT_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${UNIT_ENV}/bin/torchrun}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "error: UniT Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "error: UniT torchrun is not executable: ${TORCHRUN_BIN}" >&2
    exit 1
fi
if [[ -n "${DUAL_DEEPSPEED_CONFIG}" ]] && ! "${PYTHON_BIN}" -c 'import deepspeed' >/dev/null 2>&1; then
    echo "error: DUAL_DEEPSPEED_CONFIG requires DeepSpeed in ${UNIT_ENV}" >&2
    exit 1
fi
export PYTHONPATH="${UPSTREAM_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${XPL_ROOT}/pretrain_model/hf_home}"
CHECKPOINT_VERIFIER="${SCRIPT_DIR}/verify_hf_checkpoint.py"
if [[ ! -f "${CHECKPOINT_VERIFIER}" ]]; then
    echo "error: missing checkpoint verifier: ${CHECKPOINT_VERIFIER}" >&2
    exit 1
fi

task_names=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)
dataset_paths=()
data_configs=()
embodiment_tags=()
data_splits=()
dataset_weights=()
for task_name in "${task_names[@]}"; do
    dataset_path="${DATA_ROOT}/${task_name}"
    if [[ ! -f "${dataset_path}/meta/info.json" ]]; then
        echo "error: missing converted LeRobot v2.0 dataset: ${dataset_path}" >&2
        exit 1
    fi
    dataset_paths+=("${dataset_path}")
    data_configs+=("spark0_dual_arm_dexterous_joint")
    embodiment_tags+=("new_embodiment")
    data_splits+=("[:-10]")
    dataset_weights+=("1.0")
done

# The official split reserves each task's last ten episodes.  Fail before an
# expensive launch if any converted task is too small or not LeRobot v2.0.
"${PYTHON_BIN}" - "${dataset_paths[@]}" <<'PY'
import json
import pathlib
import sys

expected_groups = {
    "right_arm": 7,
    "right_hand": 20,
    "left_arm": 7,
    "left_hand": 20,
}
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    info = json.loads((path / "meta" / "info.json").read_text())
    version = str(info.get("codebase_version", info.get("version", "")))
    episodes = int(info.get("total_episodes", 0))
    if not version.startswith("v2.0"):
        raise SystemExit(f"{path}: expected LeRobot v2.0, got {version!r}")
    if episodes <= 10:
        raise SystemExit(f"{path}: {episodes} episodes cannot support official split [:-10]")
    for feature in ("observation.state", "action"):
        spec = info.get("features", {}).get(feature, {})
        if spec.get("shape") != [54]:
            raise SystemExit(f"{path}: {feature} must be exactly 54D, got {spec.get('shape')}")
        names = spec.get("names")
        if "waist" in json.dumps(names).lower():
            raise SystemExit(f"{path}: forbidden waist name in {feature}")
    modality = json.loads((path / "meta" / "modality.json").read_text())
    metadata = json.loads((path / "meta" / "metadata.json").read_text())
    if metadata.get("embodiment_tag") != "new_embodiment" or metadata.get("embodiment_id") != 31:
        raise SystemExit(f"{path}: expected new_embodiment id 31")
    if metadata.get("data_config") != "spark0_dual_arm_dexterous_joint":
        raise SystemExit(f"{path}: wrong data config {metadata.get('data_config')!r}")
    for kind in ("state", "action"):
        groups = modality.get(kind, {})
        if list(groups) != list(expected_groups):
            raise SystemExit(f"{path}: {kind} group order/keys mismatch: {list(groups)}")
        widths = {name: int(groups[name]["end"]) - int(groups[name]["start"]) for name in groups}
        if widths != expected_groups:
            raise SystemExit(f"{path}: {kind} widths mismatch: {widths}")
    if "waist" in "".join(p.read_text() for p in (path / "meta").iterdir() if p.is_file()).lower():
        raise SystemExit(f"{path}: forbidden waist field in metadata")
print("validated six native Spark0 LeRobot v2.0 joint54 datasets and held-out splits")
PY

# W&B 0.18.0's core GPU sampler can crash from an internal concurrent map access.
# Disable only W&B system/GPU telemetry; online training metrics remain enabled.
export WANDB__DISABLE_STATS=true

# W&B credentials are supplied only through the environment or a prior
# `wandb login`; never place an API key in this script or the repository.
if ! "${PYTHON_BIN}" - <<'PY'
import wandb
raise SystemExit(0 if getattr(wandb.api, "api_key", None) else 1)
PY
then
    echo "error: W&B is not authenticated; export WANDB_API_KEY or run 'wandb login'" >&2
    exit 1
fi
export WANDB_PROJECT="${WANDB_PROJECT:-XPolicyLab-UniT-Spark0}"

# Keep this mean/std recipe isolated from the retired SinCos/wo-flip run.  In
# particular, checkpoint-40000 from the old tokenizer must never be reused.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-joint54-meanstd-${seed}"
OUTPUT_ROOT="${SCRIPT_DIR}/checkpoints/${ckpt_setting}"
TOKENIZER_OUTPUT_DIR="${OUTPUT_ROOT}/tokenizer"
PRETRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/pretrain"
LOCAL_CONFIG_DIR="${OUTPUT_ROOT}/runtime_configs"
QWEN_PATH="${UNIT_QWEN_PATH:-${XPL_ROOT}/pretrain_model/Qwen2.5-VL-3B-Instruct}"
DINOV2_PATH="${UNIT_DINOV2_PATH:-${XPL_ROOT}/pretrain_model/dinov2-large}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_local_configs.py" \
    --upstream-root "${UPSTREAM_ROOT}" \
    --output-dir "${LOCAL_CONFIG_DIR}" \
    --qwen-path "${QWEN_PATH}" \
    --dinov2-path "${DINOV2_PATH}"
TOKENIZER_BASE_CONFIG="${TOKENIZER_BASE_CONFIG:-${LOCAL_CONFIG_DIR}/gr00t_tokenizer_local_weights.json}"
PRETRAIN_BASE_CONFIG="${PRETRAIN_BASE_CONFIG:-${LOCAL_CONFIG_DIR}/gr00t_dual_system_local_weights.json}"
TOKENIZER_CKPT="${TOKENIZER_OUTPUT_DIR}/checkpoint-${TOKENIZER_STEPS}"
PRETRAIN_CKPT="${PRETRAIN_OUTPUT_DIR}/checkpoint-${PRETRAIN_STEPS}"

if [[ "${OUTPUT_ROOT}" != *joint54* ]]; then
    echo "error: refusing checkpoint output path without explicit joint54 identity: ${OUTPUT_ROOT}" >&2
    exit 1
fi
"${PYTHON_BIN}" - "${TOKENIZER_BASE_CONFIG}" "${PRETRAIN_BASE_CONFIG}" <<'PY'
import json
import pathlib
import sys

tokenizer = json.loads(pathlib.Path(sys.argv[1]).read_text())
dual = json.loads(pathlib.Path(sys.argv[2]).read_text())
required_id = 31
dual_capacity = dual["action_head_cfg"].get("max_num_embodiments")
dual_capacity = 32 if dual_capacity is None else int(dual_capacity)
capacities = {
    "tokenizer.action_encoder_cfg": int(tokenizer["action_encoder_cfg"]["max_num_embodiments"]),
    "tokenizer.action_decoder_cfg": int(tokenizer["action_decoder_cfg"]["max_num_embodiments"]),
    "dual.action_head_cfg": dual_capacity,
}
invalid = {name: value for name, value in capacities.items() if required_id >= value}
if invalid:
    raise SystemExit(f"new_embodiment id 31 is outside configured capacities: {invalid}")
print(f"validated new_embodiment id 31 capacities: {capacities}")
PY

echo "[UniT] global_batch_size=${GLOBAL_BATCH_SIZE} per_device_batch_size=${PER_DEVICE_BATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
echo "[UniT] dual_gradient_checkpointing=${DUAL_GRADIENT_CHECKPOINTING}"
echo "[UniT] dual_deepspeed_config=${DUAL_DEEPSPEED_CONFIG:-disabled}"
echo "[UniT] dual_deepspeed_force_sync_boundary=${DUAL_DEEPSPEED_FORCE_SYNC_BOUNDARY}"
echo "[UniT] dual_training_arguments_bf16=${DUAL_TRAINING_BF16}"
echo "[UniT] tokenizer_save_total_limit=${TOKENIZER_SAVE_TOTAL_LIMIT} pretrain_save_total_limit=${PRETRAIN_SAVE_TOTAL_LIMIT}"

checkpoint_complete() {
    "${PYTHON_BIN}" "${CHECKPOINT_VERIFIER}" \
        --checkpoint "$1" --expected-step "$2" --quiet
}

mkdir -p "${OUTPUT_ROOT}"
cd "${UPSTREAM_ROOT}"

resume_tokenizer=()
if checkpoint_complete "${TOKENIZER_CKPT}" "${TOKENIZER_STEPS}"; then
    echo "[UniT] tokenizer checkpoint already complete: ${TOKENIZER_CKPT}"
else
    if [[ -d "${TOKENIZER_OUTPUT_DIR}" ]] && compgen -G "${TOKENIZER_OUTPUT_DIR}/checkpoint-*" >/dev/null; then
        if [[ "${RESUME_JOINT54:-0}" != "1" ]]; then
            echo "error: partial joint54 tokenizer checkpoints exist; set RESUME_JOINT54=1 explicitly to resume" >&2
            exit 1
        fi
        resume_tokenizer+=(--resume)
    fi
    echo "========== Stage 1: official tokenizer (${TOKENIZER_STEPS} steps) =========="
    WANDB_RUN_NAME="${ckpt_setting}-tokenizer" \
    IS_TORCHRUN=1 "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NUM_GPUS}" --nnodes=1 \
        scripts/groot_tokenizer_train.py \
        --dataset-path "${dataset_paths[@]}" \
        --data-config "${data_configs[@]}" \
        --embodiment_tag "${embodiment_tags[@]}" \
        --data_splits "${data_splits[@]}" \
        --dataset_weights "${dataset_weights[@]}" \
        --num-gpus "${NUM_GPUS}" \
        --batch-size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --max-steps "${TOKENIZER_STEPS}" \
        --save-steps "${SAVE_STEPS}" \
        --save-total-limit "${TOKENIZER_SAVE_TOTAL_LIMIT}" \
        --report_to wandb \
        --use_dino \
        --base_model_path "${TOKENIZER_BASE_CONFIG}" \
        --output_dir "${TOKENIZER_OUTPUT_DIR}" \
        "${resume_tokenizer[@]}"
fi
if ! "${PYTHON_BIN}" "${CHECKPOINT_VERIFIER}" \
    --checkpoint "${TOKENIZER_CKPT}" --expected-step "${TOKENIZER_STEPS}"; then
    echo "error: tokenizer stage did not produce a complete ${TOKENIZER_STEPS}-step checkpoint" >&2
    exit 1
fi

resume_pretrain=()
if checkpoint_complete "${PRETRAIN_CKPT}" "${PRETRAIN_STEPS}"; then
    echo "[UniT] dual-system checkpoint already complete: ${PRETRAIN_CKPT}"
else
    if [[ -d "${PRETRAIN_OUTPUT_DIR}" ]] && compgen -G "${PRETRAIN_OUTPUT_DIR}/checkpoint-*" >/dev/null; then
        if [[ "${RESUME_JOINT54:-0}" != "1" ]]; then
            echo "error: partial joint54 dual checkpoints exist; set RESUME_JOINT54=1 explicitly to resume" >&2
            exit 1
        fi
        resume_pretrain+=(--resume)
    fi
    echo "========== Stage 2: official dual-system (${PRETRAIN_STEPS} steps) =========="
    WANDB_RUN_NAME="${ckpt_setting}-dual-system" \
    IS_TORCHRUN=1 "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NUM_GPUS}" --nnodes=1 \
        scripts/dual_system_train.py \
        --dataset-path "${dataset_paths[@]}" \
        --data-config "${data_configs[@]}" \
        --embodiment_tag "${embodiment_tags[@]}" \
        --data_splits "${data_splits[@]}" \
        --dataset-weights "${dataset_weights[@]}" \
        --num-gpus "${NUM_GPUS}" \
        --batch-size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        "${dual_gradient_checkpointing_args[@]}" \
        "${dual_deepspeed_args[@]}" \
        "${dual_deepspeed_boundary_args[@]}" \
        "${dual_bf16_args[@]}" \
        --max-steps "${PRETRAIN_STEPS}" \
        --save-steps "${SAVE_STEPS}" \
        --save-total-limit "${PRETRAIN_SAVE_TOTAL_LIMIT}" \
        --report_to wandb \
        --base_model_path "${PRETRAIN_BASE_CONFIG}" \
        --bridge_type vision_lang_obs \
        --detach_vl_for_action \
        --use_image_type_embedding \
        --ignore_lang_prefix \
        --compute-bridge-loss \
        --tune-llm \
        --no-tune-visual \
        --tune-projector \
        --tune-diffusion-model \
        --no-tune-bridge-visual \
        --groot_tokenizer_path "${TOKENIZER_CKPT}" \
        --tune-bridge-embedding \
        --enable_imagenet_preprocessing \
        --select_layer 36 \
        --learning_rate 5e-5 \
        --output_dir "${PRETRAIN_OUTPUT_DIR}" \
        "${resume_pretrain[@]}"
fi
if ! "${PYTHON_BIN}" "${CHECKPOINT_VERIFIER}" \
    --checkpoint "${PRETRAIN_CKPT}" --expected-step "${PRETRAIN_STEPS}"; then
    echo "error: dual-system stage did not produce a complete ${PRETRAIN_STEPS}-step checkpoint" >&2
    exit 1
fi

ln -sfn "${PRETRAIN_CKPT}" "${OUTPUT_ROOT}/latest"
echo "[UniT] training complete: ${PRETRAIN_CKPT}"
