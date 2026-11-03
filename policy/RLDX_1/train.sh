#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
[[ $# == 6 ]] || { echo 'Usage: train.sh <bench> <run> tianji_marvin_wuji ee <seed> <GPU IDs>' >&2; exit 2; }
bench=$1; run=$2; robot=$3; action=$4; seed=$5; gpu_ids=$6
[[ "${robot}" == tianji_marvin_wuji && "${action}" == ee ]] || exit 2
# Native launchers divide global_batch_size by GPU count, not accumulation.
# Keep the historical effective256 unless the scheduler explicitly requests128.
export NUM_GPUS="$(awk -F, '{print NF}' <<<"${gpu_ids}")"
global_batch="${DEXBENCH_GLOBAL_BATCH_SIZE:-256}"
accum="${DEXBENCH_GRAD_ACCUM:-${NATIVE_ACCUMULATION:-1}}"
[[ "${global_batch}" =~ ^[1-9][0-9]*$ && "${accum}" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch and accumulation must be positive integers' >&2; exit 2; }
[[ $((global_batch % accum)) == 0 ]] || { echo 'Effective batch must divide accumulation exactly' >&2; exit 2; }
micro_global=$((global_batch / accum))
[[ "${NUM_GPUS}" -gt 0 && "${micro_global}" -ge "${NUM_GPUS}" && $((micro_global % NUM_GPUS)) == 0 ]] || { echo 'Per-GPU microbatch must be a positive integer' >&2; exit 2; }
echo "[batch] effective=${global_batch} gpus=${NUM_GPUS} per_gpu=$((micro_global / NUM_GPUS)) accumulation=${accum} optimizer_steps=60000"
data=/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21
output="${NATIVE_POLICY_DIR}/checkpoints/${bench}-${run}-${robot}-ee-${seed}"
[[ ! -e "${output}" ]] || { echo "Output exists: ${output}" >&2; exit 2; }
# Validate current numeric source before creating a run or launching workers.
"${NATIVE_PYTHON}" /personal/wenwei/DexBench/data_preprocess/prepare_native_ee58.py --check
mkdir -p "${output}"
cp "${data}/meta/action_contract.json" "${output}/action_contract.json"
cp "${data}/meta/source_fingerprint.json" "${output}/source_fingerprint.json"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
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
    exec "${NATIVE_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" rldx/experiment/launch_train.py \
        --base-model-path /personal/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/RLDX-1-PT-1592013a \
        --dataset-path "${data}" --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "${NATIVE_POLICY_DIR}/dexbench_config.py" \
        --n-cog-tokens 64 --video-length 4 --dataloader-num-workers 8 \
        --global-batch-size "${micro_global}" --gradient-accumulation-steps "${accum}" \
        --num-gpus "${NUM_GPUS}" --max-steps 60000 --save-steps 1000 --save-total-limit 5 \
        --output-dir "${output}" --experiment-name "${run}"
fi
