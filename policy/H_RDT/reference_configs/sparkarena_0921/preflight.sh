#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/personal/xiangpc/training_hrdt_sparkarena_samplerfix_0921}"
WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/H_RDT
BASE_CODE=/personal/xiangpc/training_0908-real/hrdt_0908-real/code_0908-real
HRDT="$RUN_ROOT/H_RDT"
MANIFEST="$HRDT/data/sparkarena_7task_samplerfix/data_manifest.json"
PRETRAIN_SHA256=e9eaff94ac1c1d039a193f04627b8c27794e826ca1eebd3de5eab60435254be2

export PATH=/personal/miniconda3/envs/hrdt/bin:$PATH
export PYTHONPATH="$HRDT:$BASE_CODE:$BASE_CODE/Xpolicylab"

python -m py_compile \
  "$HRDT/datasets/xpolicylab/index_plan.py" \
  "$HRDT/datasets/xpolicylab/xpolicylab_dataset.py" \
  "$HRDT/datasets/dataset.py" \
  "$HRDT/train/train.py" \
  "$HRDT/tools/prepare_sparkarena_manifest.py" \
  "$HRDT/tools/verify_sparkarena_manifest.py" \
  "$HRDT/tools/audit_index_plan.py"

bash -n "$RUN_ROOT/prepare_data.sh"
bash -n "$RUN_ROOT/preflight.sh"
bash -n "$RUN_ROOT/launch_train.sh"

verify_args=(
  --manifest "$MANIFEST"
  --hash-pretrain
  --expected-pretrain-sha256 "$PRETRAIN_SHA256"
)
if [[ "${DEEP_DATA_CHECK:-0}" == "1" ]]; then
  verify_args+=(--deep)
fi
python "$HRDT/tools/verify_sparkarena_manifest.py" "${verify_args[@]}"

python "$HRDT/tools/audit_index_plan.py" \
  --manifest "$MANIFEST" \
  --sample-count 64000 \
  --global-batch-size 64 \
  --seed 42

echo "H-RDT SparkArena sampler-fix preflight passed; no training was started."
