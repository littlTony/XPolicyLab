#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
[[ $# == 6 ]] || { echo 'Usage: train.sh <bench> <run> tianji_marvin_wuji ee <seed> <GPU IDs>' >&2; exit 2; }
bench=$1; run=$2; robot=$3; action=$4; seed=$5; gpu_ids=$6
[[ "${robot}" == tianji_marvin_wuji && "${action}" == ee ]] || exit 2
export NUM_GPUS="$(awk -F, '{print NF}' <<<"${gpu_ids}")"
effective_batch="${DEXBENCH_GLOBAL_BATCH_SIZE:-256}"
accum="${DEXBENCH_GRAD_ACCUM:-${NATIVE_ACCUMULATION:-1}}"
[[ "${effective_batch}" =~ ^[1-9][0-9]*$ && "${accum}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid global batch or accumulation' >&2; exit 2; }
[[ "${effective_batch}" == 128 || "${effective_batch}" == 256 ]] || { echo 'Effective batch must be128 or256' >&2; exit 2; }
denominator=$((NUM_GPUS * accum))
[[ $((effective_batch % denominator)) == 0 ]] || { echo 'Effective batch must divide exactly across GPUs and accumulation' >&2; exit 2; }
micro_global=$((effective_batch / accum))
per_device_batch=$((micro_global / NUM_GPUS))
echo "[LAUNCH] policy=GR00T_N17 effective_batch_size=${effective_batch} world_size=${NUM_GPUS} per_device_batch_size=${per_device_batch} gradient_accumulation_steps=${accum} max_optimizer_steps=60000"
data=/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21
output="${NATIVE_POLICY_DIR}/checkpoints/${bench}-${run}-${robot}-ee-${seed}"
[[ ! -e "${output}" ]] || { echo "Output exists: ${output}" >&2; exit 2; }
# This is a target-policy contract, not the physical absolute-data marker.
# Check all current source/statistics bindings before creating a training run.
"${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/prepare_relative_stats.py" --check
mkdir -p "${output}"
"${NATIVE_PYTHON}" "${NATIVE_POLICY_DIR}/prepare_relative_stats.py" --check --write-policy-contract "${output}/action_contract.json"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
# Both upstream launchers divide global_batch_size by GPU count only.
# Account for accumulation explicitly to preserve the requested effective batch.
cd "${NATIVE_ROOT}"
if [[ "$(basename "${NATIVE_POLICY_DIR}")" == GR00T_N17 ]]; then
    export MAX_STEPS=60000 GLOBAL_BATCH_SIZE="${micro_global}" USE_WANDB=0
    exec bash examples/finetune.sh \
        --base-model-path /personal/xiangpc/0811_Xpolicylab_bench/GrootN17/pretrain_model/GR00T-N1.7-3B \
        --dataset-path "${data}" --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${NATIVE_POLICY_DIR}/dexbench_config.py" \
        --output-dir "${output}" --experiment-name "${run}" -- \
        --gradient_accumulation_steps "${accum}"
else
    exec torchrun --standalone --nproc_per_node="${NUM_GPUS}" rldx/experiment/launch_train.py \
        --base-model-path /personal/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/RLDX-1-PT-1592013a \
        --dataset-path "${data}" --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "${NATIVE_POLICY_DIR}/dexbench_config.py" \
        --n-cog-tokens 64 --video-length 4 --dataloader-num-workers 8 \
        --global-batch-size "${micro_global}" --gradient-accumulation-steps "${accum}" \
        --num-gpus "${NUM_GPUS}" --max-steps 60000 --save-steps 1000 --save-total-limit 5 \
        --output-dir "${output}" --experiment-name "${run}"
fi
