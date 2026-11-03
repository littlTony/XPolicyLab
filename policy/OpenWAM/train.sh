#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo 'Usage: bash train.sh <bench> <run> <env_cfg> ee <seed> <GPU IDs> [Hydra overrides...]' >&2
    exit 1
fi
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
bench=$1; run=$2; env_cfg=$3; action=$4; seed=$5; gpu_ids=$6
shift 6
if [[ "${action}" != ee || "${env_cfg}" != tianji_marvin_wuji ]]; then
    echo 'Expected absolute EE + two Wuji20 hands: tianji_marvin_wuji ee' >&2
    exit 2
fi
"${OPENWAM_PYTHON}" "${OPENWAM_POLICY_DIR}/check_contract.py"
output="${OPENWAM_POLICY_DIR}/checkpoints/${bench}-${run}-${env_cfg}-ee-${seed}"
if [[ -e "${output}" ]]; then
    echo "Output exists: ${output}; use another run name or an explicit resume workflow" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NPROC_PER_NODE="$(awk -F, '{print NF}' <<<"${gpu_ids}")"
export MASTER_PORT="${MASTER_PORT:-29519}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${OPENWAM_ROOT}"
exec bash scripts/train.sh dataloader=dexbench \
    "training.output_path=${output}" "project.seed=${seed}" \
    "training.finetune_ckpt_path=/personal/xiangpc/0912_Xpolicylab_bench/openwam/pretrain_model/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors" \
    training.resume_ckpt_path=null "$@"
