#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
[[ $# == 6 ]] || { echo 'Usage: train.sh <bench> <run> tianji_marvin_wuji ee <seed> <GPU IDs>' >&2; exit 2; }
bench=$1; run=$2; robot=$3; action=$4; seed=$5; gpu_ids=$6
[[ "${robot}" == tianji_marvin_wuji && "${action}" == ee ]] || exit 2
nproc="$(awk -F, '{print NF}' <<<"${gpu_ids}")"
export DEXBENCH_WORLD_SIZE="${nproc}"
effective_batch="${DEXBENCH_GLOBAL_BATCH_SIZE:-256}"
[[ "${effective_batch}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid effective global batch' >&2; exit 2; }
if [[ -n "${DEXBENCH_GRAD_ACCUM:-}" ]]; then
    accum="${DEXBENCH_GRAD_ACCUM}"
    [[ "${accum}" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid gradient accumulation' >&2; exit 2; }
    denominator=$((nproc * accum))
    [[ $((effective_batch % denominator)) == 0 ]] || { echo 'Effective batch must divide exactly across GPUs and accumulation' >&2; exit 2; }
    export VITRA_PER_DEVICE_BATCH=$((effective_batch / denominator))
fi
output="${VITRA_POLICY_DIR}/checkpoints/${bench}-${run}-${robot}-ee-${seed}"
[[ ! -e "${output}" ]] || { echo "Output exists: ${output}" >&2; exit 2; }
mkdir -p "${output}"
python "${VITRA_POLICY_DIR}/make_train_config.py" \
    --base-config "${VITRA_ROOT}/vitra/configs/spark0_finetune.json" \
    --output-config "${output}/train_config.json" --output-root "${output}" \
    --workspace-root "${VITRA_XPL_ROOT}/.." --data-root "${VITRA_LEROBOT_ROOT:-/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks}" \
    --representation wuji20 --bench-name "${bench}" --ckpt-name "${run}" \
    --run-date "${VITRA_RUN_DATE:-2026-10-22}" --seed "${seed}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export WANDB_MODE="${WANDB_MODE:-offline}"
cd "${VITRA_ROOT}"
exec torchrun --standalone --nproc_per_node="${nproc}" scripts/train.py --config "${output}/train_config.json"
