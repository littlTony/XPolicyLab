#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/personal/xiangpc/training_hrdt_sparkarena_samplerfix_0921}"
EXPECTED_HOSTNAME=instance-ic9spp2o-2
EXPECTED_HOST_IP=10.189.16.2
WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/H_RDT
BASE_CODE=/personal/xiangpc/training_0908-real/hrdt_0908-real/code_0908-real
HRDT="$RUN_ROOT/H_RDT"
DATA="$WORKSPACE/data/spark0_bench_7task_0908"
POLICY_DATA="$HRDT/data/sparkarena_7task_samplerfix"
MANIFEST="$POLICY_DATA/data_manifest.json"
OUTPUT="$RUN_ROOT/checkpoints/hrdt-sparkarena-joint54-samplerfix-s42-bs64-80k"
PRETRAIN="$HRDT/pretrain_model/hrdt_pretrain_500k/pytorch_model.bin"
PRETRAIN_SHA256=e9eaff94ac1c1d039a193f04627b8c27794e826ca1eebd3de5eab60435254be2

MAX_TRAIN_STEPS=80000
PER_DEVICE_BATCH=8
WORLD_SIZE=8
GRAD_ACCUM=1
GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH * WORLD_SIZE * GRAD_ACCUM))
SAMPLE_COUNT=$((MAX_TRAIN_STEPS * GLOBAL_BATCH_SIZE))
MASTER_PORT="${MASTER_PORT:-42521}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/.launch.lock"
if ! flock -n 9; then
  echo "[error] Another launcher already holds $RUN_ROOT/.launch.lock" >&2
  exit 2
fi

RUN_ROOT="$RUN_ROOT" bash "$RUN_ROOT/preflight.sh"

if [[ "${START_HRDT_TRAIN:-0}" != "1" ]]; then
  echo "Preflight passed. Set START_HRDT_TRAIN=1 only after the 8×A800 allocation is confirmed."
  exit 0
fi

actual_hostname="$(hostname)"
if [[ "$actual_hostname" != "$EXPECTED_HOSTNAME" ]]; then
  echo "[error] Refusing to train on host $actual_hostname; expected $EXPECTED_HOSTNAME." >&2
  exit 2
fi
if ! hostname -I | tr ' ' '\n' | grep -Fxq "$EXPECTED_HOST_IP"; then
  echo "[error] Refusing to train: host does not own $EXPECTED_HOST_IP." >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "[error] Exactly 8 GPU ids are required." >&2
  exit 2
fi
for gpu_id in "${gpu_ids[@]}"; do
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id")"
  if [[ "$gpu_name" != *"A800"* ]]; then
    echo "[error] GPU $gpu_id is not an A800: $gpu_name" >&2
    exit 2
  fi
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "[error] At least one GPU already has a compute process. Nothing was killed." >&2
  exit 2
fi

if [[ -z "$RESUME_FROM_CHECKPOINT" ]]; then
  if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
    echo "[error] Fresh-run output is not empty: $OUTPUT" >&2
    exit 2
  fi
else
  if [[ "$RESUME_FROM_CHECKPOINT" != "latest" && "$RESUME_FROM_CHECKPOINT" != "$OUTPUT"/* ]]; then
    echo "[error] Resume checkpoint must be 'latest' or live under $OUTPUT" >&2
    exit 2
  fi
fi

if [[ -f /root/.config/hrdt-sparkarena-samplerfix/wandb.env ]]; then
  source /root/.config/hrdt-sparkarena-samplerfix/wandb.env
elif [[ -f /root/.config/training_0908-real/wandb.env ]]; then
  source /root/.config/training_0908-real/wandb.env
elif [[ -f /personal/xiangpc/training_0908-real/wandb.env ]]; then
  source /personal/xiangpc/training_0908-real/wandb.env
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_API_KEY is required for the official run." >&2
  exit 3
fi

mkdir -p "$RUN_ROOT/logs" "$OUTPUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PATH=/personal/miniconda3/envs/hrdt/bin:$PATH
export PYTHONPATH="/personal/xiangpc/training_0908-real/hrdt_0908-real/python_packages_0908-real:$HRDT:$BASE_CODE:$BASE_CODE/Xpolicylab"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4

export WANDB_PROJECT="${WANDB_PROJECT:-xpolicylab-HRDT-SparkArena}"
export WANDB_NAME="${WANDB_NAME:-hrdt-sparkarena-joint54-samplerfix-s42-80k}"
export WANDB_RUN_GROUP=hrdt_sparkarena_samplerfix
export WANDB_RUN_ID="${WANDB_RUN_ID:-hrdt_samplerfix_s42_0921}"
export WANDB_RESUME=allow
export WANDB_DIR="$RUN_ROOT/logs"
export WANDB_MODE=online

export XPOLICY_HRDT_RUN_TAG=sparkarena_samplerfix_s42
export XPOLICY_HRDT_SOURCE_ROOT="$POLICY_DATA/raw"
export XPOLICY_HRDT_RAW_BENCH_NAME=spark0_bench_7tasks
export XPOLICY_HRDT_ENV_CFG_TYPE=tianji_marvin_wuji
export XPOLICY_HRDT_ACTION_TYPE=joint
export XPOLICY_HRDT_STAT_PATH="$POLICY_DATA/stats.json"
export XPOLICY_HRDT_DATASET_MODE=multi_task
export XPOLICY_HRDT_TASKS=all
export XPOLICY_HRDT_SAMPLE_SEMANTICS=state_action
export XPOLICY_HRDT_LANG_EMBEDDING_DIR="$POLICY_DATA/lang_embeddings"
export XPOLICY_HRDT_INDEXED_SAMPLING=1
export XPOLICY_HRDT_SAMPLE_SEED=42
export XPOLICY_HRDT_SAMPLE_COUNT="$SAMPLE_COUNT"
export XPOLICY_HRDT_DATA_MANIFEST="$MANIFEST"
export XPOLICY_HRDT_EXPECT_TASKS=click_mouse,collect_objects,dual_bottles_pick,hammer_beat,put_food_in_microwave,retrieve_gap,stack_bowls
export XPOLICY_HRDT_PRETRAIN_SHA256="$PRETRAIN_SHA256"

CODE_COMMIT="$(git -C "$RUN_ROOT" rev-parse HEAD)"
if ! git -C "$RUN_ROOT" diff --quiet --ignore-submodules -- || \
   ! git -C "$RUN_ROOT" diff --cached --quiet --ignore-submodules -- || \
   [[ -n "$(git -C "$RUN_ROOT" ls-files --others --exclude-standard)" ]]; then
  echo "[error] Training code checkout is not clean: $RUN_ROOT" >&2
  exit 2
fi
export XPOLICY_HRDT_CODE_COMMIT="$CODE_COMMIT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

args=(
  --pretrained_vision_encoder_name_or_path=dino-siglip
  --deepspeed=configs/zero1.json
  --config_path="$RUN_ROOT/hrdt_sparkarena_7task.yaml"
  --output_dir="$OUTPUT"
  --train_batch_size="$PER_DEVICE_BATCH"
  --sample_batch_size=8
  --gradient_accumulation_steps="$GRAD_ACCUM"
  --max_train_steps="$MAX_TRAIN_STEPS"
  --checkpointing_period=10000
  --checkpoints_total_limit=8
  --lr_scheduler=constant_with_warmup
  --learning_rate=1e-4
  --mixed_precision=bf16
  --dataloader_num_workers=4
  --dataset_type=finetune
  --report_to=wandb
  --sample_period=-1
  --upsample_rate=1
  --precomp_lang_embed
  --training_mode=lang
  --mode=finetune
  --task_name=spark0_bench_7tasks
  --dataset_name=xpolicylab
  --seed=42
  --pretrained_backbone_path="$PRETRAIN"
)
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  args+=(--resume_from_checkpoint="$RESUME_FROM_CHECKPOINT")
fi

cd "$HRDT"
exec python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes="$WORLD_SIZE" \
  --main_process_port="$MASTER_PORT" \
  main.py "${args[@]}"
