#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> ee <seed> <gpu_ids> [train.py options]" >&2
    exit 1
fi
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
bench_name=$1; ckpt_name=$2; env_cfg_type=$3; action_type=$4; seed=$5; gpu_ids=$6
shift 6
if [[ "${action_type}" != ee ]]; then
    echo 'DexBench Pi0.5 trains absolute EE xyz+wxyz + absolute Wuji20 joints; action_type must be ee' >&2
    exit 2
fi
export OPENPI_TRAIN_CONFIG_NAME="${OPENPI_TRAIN_CONFIG_NAME:-pi05_wuji_marvin_54d}"
export OPENPI_LEROBOT_REPO_ID="${OPENPI_LEROBOT_REPO_ID:-dexbench_12tasks}"
if [[ "${OPENPI_TRAIN_CONFIG_NAME}" != pi05_wuji_marvin_54d || "${OPENPI_LEROBOT_REPO_ID}" != dexbench_12tasks ]]; then
    echo 'Training and deploy config must use pi05_wuji_marvin_54d / dexbench_12tasks' >&2
    exit 2
fi
"${PI05_PYTHON_BIN}" "${PI05_DIR}/check_contract.py" --env-cfg-type "${env_cfg_type}" --require-stats
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-ee-${seed}"
ckpt_dir="${PI05_DIR}/checkpoints/${ckpt_setting}"
gpu_count=$(awk -F, '{print NF}' <<<"${gpu_ids}")
default_fsdp=$(( gpu_count < 2 ? 1 : 2 ))
fsdp_devices="${OPENPI_FSDP_DEVICES:-${default_fsdp}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
cache_root="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/dexbench-pi05-$(hostname)}"
mkdir -p "${cache_root}/hf/datasets" "${cache_root}/jax"
export HF_DATASETS_CACHE="${cache_root}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${cache_root}/jax"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
resume_args=()
if [[ "${OPENPI_RESUME:-0}" == 1 ]]; then resume_args+=(--resume); fi
echo "[Pi_05] config=${OPENPI_TRAIN_CONFIG_NAME} dataset=${HF_LEROBOT_HOME}/${OPENPI_LEROBOT_REPO_ID} action=absolute_ee54 fsdp=${fsdp_devices} output=${ckpt_dir}"
cd "${PI05_OPENPI_ROOT}"
exec "${PI05_PYTHON_BIN}" scripts/train.py "${OPENPI_TRAIN_CONFIG_NAME}" \
    --exp-name="${ckpt_setting}" --data.repo-id="${OPENPI_LEROBOT_REPO_ID}" \
    --fsdp-devices="${fsdp_devices}" --checkpoint-dir-override="${ckpt_dir}" \
    --seed="${seed}" --batch-size=256 --num-workers=4 \
    --num-train-steps=60000 --save-interval=1000 --keep-period=5000 \
    --lr-schedule.decay-steps=60000 "${resume_args[@]}" "$@"
