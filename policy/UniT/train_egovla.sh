#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: bash train_egovla.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids>" >&2
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
DATA_ROOT="${UNIT_EGOVLA_DATA_ROOT:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21_three_view}"

# Requested run contract: 8 GPUs × batch 8 = global batch 64, no accumulation.
NUM_GPUS=8
GLOBAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TOKENIZER_STEPS="${TOKENIZER_STEPS:-40000}"
MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
# ZeRO-2 with torch autocast is the validated path on this UniT checkout.  It
# keeps compute in BF16 while retaining FP32 parameter/gradient hooks; native
# DeepSpeed BF16 can be selected explicitly with DUAL_DEEPSPEED_CONFIG and
# DUAL_TRAINING_BF16=1 after an environment-specific compatibility test.
DUAL_DEEPSPEED_CONFIG="${DUAL_DEEPSPEED_CONFIG:-${SCRIPT_DIR}/configs/deepspeed_zero2.json}"
DUAL_TRAINING_BF16="${DUAL_TRAINING_BF16:-0}"
DATA_CONFIG=egovla_h1_joint38_threeview
EMBODIMENT_TAG=new_embodiment

[[ "${action_type}" == joint ]] || { echo "error: action_type must be joint" >&2; exit 1; }
[[ "${env_cfg_type}" == ego_h1_inspire ]] || { echo "error: env_cfg_type must be ego_h1_inspire" >&2; exit 1; }
[[ "${seed}" == 42 ]] || { echo "error: UniT trainers use seed 42" >&2; exit 1; }
for name in GRADIENT_ACCUMULATION_STEPS TOKENIZER_STEPS MAX_STEPS SAVE_STEPS DATALOADER_WORKERS; do
    value=${!name}
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "error: ${name} must be a positive integer" >&2; exit 1; }
done
(( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRADIENT_ACCUMULATION_STEPS) == 0 )) || { echo "error: incompatible batch/accumulation" >&2; exit 1; }
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRADIENT_ACCUMULATION_STEPS))
(( PER_DEVICE_BATCH_SIZE == 8 )) || { echo "error: per-device batch must be 8, got ${PER_DEVICE_BATCH_SIZE}" >&2; exit 1; }
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
[[ ${#gpu_array[@]} -eq ${NUM_GPUS} ]] || { echo "error: expected 8 GPU ids" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${gpu_ids}

UNIT_ENV="${UNIT_ENV:-/personal/miniconda3/envs/unit_torch26_recovery}"
PYTHON_BIN="${PYTHON_BIN:-${UNIT_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${UNIT_ENV}/bin/torchrun}"
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "error: UniT environment is incomplete: ${UNIT_ENV}" >&2; exit 1; }
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
[[ -f "${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" ]] || { echo "error: missing ${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" >&2; exit 1; }
[[ -f "${BENCH_ROOT}/env_cfg/sim/egovla_single.yml" ]] || { echo "error: missing simulation config" >&2; exit 1; }

# Fail before loading the 3B model if the converter did not publish the exact
# active release and the audited three-camera contract.
"${PYTHON_BIN}" - "${DATA_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
marker = root / "meta" / "unit_egovla_preparation.json"
if not marker.is_file(): raise SystemExit(f"missing preparation manifest: {root}")
info = json.loads((root / "meta" / "info.json").read_text())
for key, expected in {"codebase_version":"v2.1", "total_episodes":1903, "total_frames":510546, "total_tasks":12, "total_videos":5709}.items():
    if info.get(key) != expected: raise SystemExit(f"info.{key}={info.get(key)!r}, expected {expected!r}")
video_features = [k for k,v in info.get("features",{}).items() if isinstance(v,dict) and v.get("dtype") == "video"]
expected_videos = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
if video_features != expected_videos: raise SystemExit(f"expected three camera videos, got {video_features}")
if info.get("features",{}).get("observation.camera_mask",{}).get("shape") != [3]:
    raise SystemExit("observation.camera_mask must be present with shape [3]")
modality = json.loads((root / "meta" / "modality.json").read_text())
expected_video_aliases = {"cam_head":"observation.images.cam_high", "cam_left_wrist":"observation.images.cam_left_wrist", "cam_right_wrist":"observation.images.cam_right_wrist"}
actual_video_aliases = {k:v.get("original_key") for k,v in modality.get("video",{}).items() if isinstance(v,dict)}
if actual_video_aliases != expected_video_aliases:
    raise SystemExit(f"three-view modality mismatch: {actual_video_aliases}")
expected = {"left_arm":(0,7), "left_hand":(7,19), "right_arm":(19,26), "right_hand":(26,38)}
for section in ("state", "action"):
    groups = modality.get(section,{})
    if list(groups) != list(expected): raise SystemExit(f"{section} group order mismatch: {list(groups)}")
    for name,(start,end) in expected.items():
        if (groups[name].get("start"),groups[name].get("end")) != (start,end): raise SystemExit(f"{section}.{name} range mismatch")
prep = json.loads(marker.read_text())
contract = prep.get("filter_contract",{})
for key, expected_value in {"deprecated_excluded":100,"active_episodes":1903,"active_frames":510546}.items():
    if contract.get(key) != expected_value: raise SystemExit(f"preparation {key} mismatch")
view = prep.get("view_contract", {})
if (view.get("three_view") is not True or view.get("single_view") is not False
        or view.get("video_features") != expected_videos
        or view.get("real_wrist_episodes") != 900
        or view.get("black_fallback_wrist_episodes") != 1003):
    raise SystemExit(f"preparation is not the audited three-view contract: {view!r}")
print(f"EGOVLA_DATA_OK episodes=1903 frames=510546 videos=5709 real_wrist=900 black_fallback=1003 deprecated_excluded=100 root={root}")
PY

export PYTHONPATH="${UPSTREAM_ROOT}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${XPL_ROOT}/pretrain_model/hf_home}"
export WANDB__DISABLE_STATS=true
if ! "${PYTHON_BIN}" - <<'PY'
import wandb
raise SystemExit(0 if getattr(wandb.api, "api_key", None) else 1)
PY
then
    echo "error: W&B is not authenticated; export WANDB_API_KEY or run wandb login" >&2
    exit 1
fi
export WANDB_PROJECT="${WANDB_PROJECT:-XPolicyLab-UniT-EgoVLA}"

# Keep the standard XPolicyLab run-directory ABI.  The multiview schema is
# encoded by DATA_CONFIG and the caller's ckpt_name, while the resolver expects
# exactly {bench}-{ckpt}-{env}-{action}-{seed}.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
OUTPUT_ROOT="${SCRIPT_DIR}/checkpoints/${ckpt_setting}"
TOKENIZER_OUTPUT_DIR="${OUTPUT_ROOT}/tokenizer"
PRETRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/pretrain"
LOCAL_CONFIG_DIR="${OUTPUT_ROOT}/runtime_configs"
QWEN_PATH="${UNIT_QWEN_PATH:-${XPL_ROOT}/pretrain_model/Qwen2.5-VL-3B-Instruct}"
DINOV2_PATH="${UNIT_DINOV2_PATH:-${XPL_ROOT}/pretrain_model/dinov2-large}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_local_configs.py" --upstream-root "${UPSTREAM_ROOT}" --output-dir "${LOCAL_CONFIG_DIR}" --qwen-path "${QWEN_PATH}" --dinov2-path "${DINOV2_PATH}"
TOKENIZER_BASE_CONFIG="${TOKENIZER_BASE_CONFIG:-${LOCAL_CONFIG_DIR}/gr00t_tokenizer_local_weights.json}"
PRETRAIN_BASE_CONFIG="${PRETRAIN_BASE_CONFIG:-${LOCAL_CONFIG_DIR}/gr00t_dual_system_local_weights.json}"
TOKENIZER_CKPT="${TOKENIZER_OUTPUT_DIR}/checkpoint-${TOKENIZER_STEPS}"
PRETRAIN_CKPT="${PRETRAIN_OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
CONTRACT_SCRIPT="${UNIT_EGOVLA_CONTRACT_SCRIPT:-${XPL_ROOT}/data_scripts/write_egovla_contract.py}"

checkpoint_complete() { "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_hf_checkpoint.py" --checkpoint "$1" --expected-step "$2" --quiet; }
mkdir -p "${OUTPUT_ROOT}"
cd "${UPSTREAM_ROOT}"
echo "[UniT/EgoVLA] global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${PER_DEVICE_BATCH_SIZE} accumulation=${GRADIENT_ACCUMULATION_STEPS} tokenizer_steps=${TOKENIZER_STEPS} max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"

if [[ "${UNIT_TRAIN_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[UniT/EgoVLA] preflight-only: no training process launched"
    exit 0
fi

resume_tokenizer=()
if ! checkpoint_complete "${TOKENIZER_CKPT}" "${TOKENIZER_STEPS}"; then
    if [[ -d "${TOKENIZER_OUTPUT_DIR}" ]] && compgen -G "${TOKENIZER_OUTPUT_DIR}/checkpoint-*" >/dev/null && [[ "${RESUME_EGOVLA:-0}" != 1 ]]; then
        echo "error: partial tokenizer output exists; set RESUME_EGOVLA=1 to resume" >&2; exit 1
    fi
    [[ -d "${TOKENIZER_OUTPUT_DIR}" ]] && compgen -G "${TOKENIZER_OUTPUT_DIR}/checkpoint-*" >/dev/null && resume_tokenizer+=(--resume)
    WANDB_RUN_NAME="${ckpt_setting}-tokenizer" IS_TORCHRUN=1 "${TORCHRUN_BIN}" --standalone --nproc_per_node=${NUM_GPUS} --nnodes=1 scripts/groot_tokenizer_train.py \
        --dataset-path "${DATA_ROOT}" --data-config "${DATA_CONFIG}" --embodiment_tag "${EMBODIMENT_TAG}" --data_splits "[:]" --dataset_weights 1.0 \
        --num-gpus ${NUM_GPUS} --batch-size ${PER_DEVICE_BATCH_SIZE} --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} --dataloader-num-workers ${DATALOADER_WORKERS} \
        --max-steps ${TOKENIZER_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit $(( (TOKENIZER_STEPS + SAVE_STEPS - 1) / SAVE_STEPS )) --report_to wandb --use_dino --no-use-multi-scenario-training \
        --base_model_path "${TOKENIZER_BASE_CONFIG}" --output_dir "${TOKENIZER_OUTPUT_DIR}" "${resume_tokenizer[@]}"
fi
checkpoint_complete "${TOKENIZER_CKPT}" "${TOKENIZER_STEPS}" || { echo "error: tokenizer checkpoint missing" >&2; exit 1; }

resume_pretrain=()
if ! checkpoint_complete "${PRETRAIN_CKPT}" "${MAX_STEPS}"; then
    if [[ -d "${PRETRAIN_OUTPUT_DIR}" ]] && compgen -G "${PRETRAIN_OUTPUT_DIR}/checkpoint-*" >/dev/null && [[ "${RESUME_EGOVLA:-0}" != 1 ]]; then
        echo "error: partial dual output exists; set RESUME_EGOVLA=1 to resume" >&2; exit 1
    fi
    [[ -d "${PRETRAIN_OUTPUT_DIR}" ]] && compgen -G "${PRETRAIN_OUTPUT_DIR}/checkpoint-*" >/dev/null && resume_pretrain+=(--resume)
    dual_gradient_checkpointing_args=()
    [[ "${DUAL_GRADIENT_CHECKPOINTING:-1}" == 1 ]] && dual_gradient_checkpointing_args+=(--gradient-checkpointing)
    [[ "${DUAL_GRADIENT_CHECKPOINTING:-1}" =~ ^[01]$ ]] || { echo "error: DUAL_GRADIENT_CHECKPOINTING must be 0/1" >&2; exit 1; }
    dual_bf16_args=()
    [[ "${DUAL_TRAINING_BF16}" == 0 ]] && dual_bf16_args+=(--no-bf16)
    [[ "${DUAL_TRAINING_BF16}" =~ ^[01]$ ]] || { echo "error: DUAL_TRAINING_BF16 must be 0/1" >&2; exit 1; }
    dual_deepspeed_args=()
    [[ -f "${DUAL_DEEPSPEED_CONFIG}" ]] || { echo "error: missing DeepSpeed config: ${DUAL_DEEPSPEED_CONFIG}" >&2; exit 1; }
    dual_deepspeed_args+=(--deepspeed "${DUAL_DEEPSPEED_CONFIG}")
    WANDB_RUN_NAME="${ckpt_setting}-dual-system" IS_TORCHRUN=1 "${TORCHRUN_BIN}" --standalone --nproc_per_node=${NUM_GPUS} --nnodes=1 scripts/dual_system_train.py \
        --dataset-path "${DATA_ROOT}" --data-config "${DATA_CONFIG}" --embodiment_tag "${EMBODIMENT_TAG}" --data_splits "[:]" --dataset-weights 1.0 \
        --num-gpus ${NUM_GPUS} --batch-size ${PER_DEVICE_BATCH_SIZE} --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} --dataloader-num-workers ${DATALOADER_WORKERS} \
        "${dual_gradient_checkpointing_args[@]}" "${dual_deepspeed_args[@]}" "${dual_bf16_args[@]}" --max-steps ${MAX_STEPS} --save-steps ${SAVE_STEPS} --save-total-limit $(( (MAX_STEPS + SAVE_STEPS - 1) / SAVE_STEPS )) \
        --report_to wandb --base_model_path "${PRETRAIN_BASE_CONFIG}" --bridge_type vision_lang_obs --detach_vl_for_action --use_image_type_embedding --ignore_lang_prefix --compute-bridge-loss \
        --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model --no-tune-bridge-visual --groot_tokenizer_path "${TOKENIZER_CKPT}" --tune-bridge-embedding \
        --enable_imagenet_preprocessing --select_layer 36 --learning_rate "${LEARNING_RATE:-5e-5}" --output_dir "${PRETRAIN_OUTPUT_DIR}" "${resume_pretrain[@]}"
fi
checkpoint_complete "${PRETRAIN_CKPT}" "${MAX_STEPS}" || { echo "error: dual checkpoint missing" >&2; exit 1; }
ln -sfn "${PRETRAIN_CKPT}" "${OUTPUT_ROOT}/latest"
if [[ ! -f "${CONTRACT_SCRIPT}" ]]; then
    echo "error: missing EgoVLA contract writer: ${CONTRACT_SCRIPT}" >&2
    exit 1
fi
"${PYTHON_BIN}" "${CONTRACT_SCRIPT}" --run-root "${OUTPUT_ROOT}" --data-root "${DATA_ROOT}" --checkpoint "${PRETRAIN_CKPT}" --runtime-config-dir "${LOCAL_CONFIG_DIR}" --bench-name "${bench_name}" --ckpt-name "${ckpt_name}" --env-cfg-type "${env_cfg_type}" --action-type "${action_type}" --seed "${seed}" --tokenizer-steps "${TOKENIZER_STEPS}" --max-steps "${MAX_STEPS}"
echo "[UniT/EgoVLA] training complete: ${PRETRAIN_CKPT}"
