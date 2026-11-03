#!/usr/bin/env bash
set -euo pipefail
ROOT=/personal/xiangpc/training_0908-SparkArena/unit_0908-SparkArena
exec 9>"$ROOT/pipeline_0908-SparkArena.lock"
flock -n 9 || { echo '0908-SparkArena UniT pipeline already active'; exit 0; }
CODE=$ROOT/code_0908-SparkArena
POLICY=$CODE/policy/UniT
UPSTREAM=$POLICY/UniT
PYTHON=/personal/miniconda3/envs/unit_torch26_recovery/bin/python
source /root/.config/training_0908-real/wandb.env
export WANDB_PROJECT=xpolicylab-0908-SparkArena WANDB_RUN_GROUP=0908-SparkArena WANDB_MODE=online WANDB_RESUME=allow
export WANDB_DIR=$ROOT/logs_0908-SparkArena
export SPARK_UNIT_ROOT=$ROOT
export PYTHONPATH=$ROOT/runtime_packages_0908-SparkArena:$UPSTREAM:$POLICY:$CODE
export USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=$CODE/pretrain_model/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB__DISABLE_STATS=true IS_TORCHRUN=1
export TORCH_HOME=$ROOT/cache_0908-SparkArena/torch TORCH_EXTENSIONS_DIR=$ROOT/cache_0908-SparkArena/torch_extensions
mkdir -p "$WANDB_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"
cd "$UPSTREAM"
DATA=$ROOT/data/SparkArena_joint54_3view_0908-SparkArena
SHARED=/personal/xiangpc/training_0908-SparkArena/data/SparkArena_joint54_3view_0908-SparkArena
while [[ ! -f "$SHARED/conversion_complete_0908-SparkArena.json" ]]; do
 echo '0908-SparkArena waiting for fully verified 600-episode three-view conversion'
 sleep 30
done
"$PYTHON" "$ROOT/adapt_data_0908_SparkArena.py"
"$PYTHON" "$ROOT/probe_batch_0908_SparkArena.py"
COMMON=(--dataset-path "$DATA" --data-config sparkarena_3view_joint54 --embodiment-tag new_embodiment --data-split '[:]' --num-gpus 8 --dataloader-num-workers 4 --dataloader-prefetch-factor 2 --save-steps 10000 --report-to wandb)
TOK=$ROOT/checkpoints_0908-SparkArena/tokenizer_0908-SparkArena
POL=$ROOT/checkpoints_0908-SparkArena/policy_0908-SparkArena
GATE=$POLICY/checkpoint_gate_0908_SparkArena.py
if ! "$PYTHON" "$GATE" "$TOK/checkpoint-40000" 40000 require_end >/dev/null 2>&1; then
 export SPARK_UNIT_STAGE=tokenizer WANDB_NAME=unit-tokenizer-3view-0908-SparkArena WANDB_RUN_NAME=unit-tokenizer-3view-0908-SparkArena WANDB_RUN_ID=unit_tok_0908sparkarena
 RESUME=()
 if compgen -G "$TOK/checkpoint-*" >/dev/null; then RESUME=(--resume-checkpoint "$("$PYTHON" "$POLICY/select_resume_0908_SparkArena.py" "$TOK")"); fi
 "$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29615 "$POLICY/launch_0908_SparkArena.py" "${COMMON[@]}" --batch-size 8 --gradient-accumulation-steps 1 --max-steps 40000 --save-total-limit 4 --use-dino --base-model-path "$ROOT/runtime_configs_0908-SparkArena/gr00t_tokenizer_local_weights.json" --output-dir "$TOK" "${RESUME[@]}" > "$ROOT/logs_0908-SparkArena/tokenizer_0908-SparkArena.log" 2>&1
fi
"$PYTHON" "$GATE" "$TOK/checkpoint-40000" 40000 require_end
if ! "$PYTHON" "$GATE" "$POL/checkpoint-80000" 80000 require_end >/dev/null 2>&1; then
 export SPARK_UNIT_STAGE=policy WANDB_NAME=unit-policy-3view-0908-SparkArena WANDB_RUN_NAME=unit-policy-3view-0908-SparkArena WANDB_RUN_ID=unit_policy_0908sparkarena
 RESUME=()
 if compgen -G "$POL/checkpoint-*" >/dev/null; then RESUME=(--resume-checkpoint "$("$PYTHON" "$POLICY/select_resume_0908_SparkArena.py" "$POL")"); fi
 "$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29615 "$POLICY/launch_0908_SparkArena.py" "${COMMON[@]}" --batch-size 4 --gradient-accumulation-steps 2 --gradient-checkpointing --deepspeed "$POLICY/configs/deepspeed_zero2.json" --no-bf16 --max-steps 80000 --save-total-limit 8 --base-model-path "$ROOT/runtime_configs_0908-SparkArena/gr00t_dual_system_local_weights.json" --bridge-type vision_lang_obs --detach-vl-for-action --use-image-type-embedding --ignore-lang-prefix --compute-bridge-loss --tune-llm --no-tune-visual --tune-projector --tune-diffusion-model --no-tune-bridge-visual --groot-tokenizer-path "$TOK/checkpoint-40000" --tune-bridge-embedding --enable-imagenet-preprocessing --select-layer 36 --learning-rate 5e-5 --output-dir "$POL" "${RESUME[@]}" > "$ROOT/logs_0908-SparkArena/policy_0908-SparkArena.log" 2>&1
fi
"$PYTHON" "$GATE" "$POL/checkpoint-80000" 80000 require_end
echo '0908-SparkArena UniT tokenizer 40000 and policy 80000 complete'
