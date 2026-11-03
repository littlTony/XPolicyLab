#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/runtime.sh"
if [[ $# -lt 6 ]]; then echo 'Usage: train.sh <bench> <run> tianji_marvin_wuji ee <seed> <GPU IDs> [Hydra overrides...]' >&2; exit 2; fi
bench=$1; run=$2; robot=$3; action=$4; seed=$5; gpu_ids=$6; shift 6
[[ "${robot}" == tianji_marvin_wuji && "${action}" == ee ]] || { echo 'Expected tianji_marvin_wuji ee' >&2; exit 2; }
"${FASTWAM_PYTHON}" "${FASTWAM_POLICY_DIR}/check_contract.py" --require-text-cache
output="${FASTWAM_POLICY_DIR}/checkpoints/${bench}-${run}-${robot}-ee-${seed}"
[[ ! -e "${output}" ]] || { echo "Output exists: ${output}" >&2; exit 2; }
mkdir -p "${output}"
cp "${FASTWAM_POLICY_DIR}/data/dataset_stats.json" "${output}/dataset_stats.json"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nproc="$(awk -F, '{print NF}' <<<"${gpu_ids}")"
cd "${FASTWAM_ROOT}"
exec bash scripts/train_zero1.sh "${nproc}" task=dexbench_ee58 "seed=${seed}" "output_dir=${output}" "$@"
