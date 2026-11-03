#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/personal/xiangpc/training_hrdt_sparkarena_samplerfix_0921}"
WORKSPACE=/personal/xiangpc/0812_Xpolicylab_bench/H_RDT
BASE_CODE=/personal/xiangpc/training_0908-real/hrdt_0908-real/code_0908-real
DATA="$WORKSPACE/data/spark0_bench_7task_0908"
HRDT="$RUN_ROOT/H_RDT"
POLICY_DATA="$HRDT/data/sparkarena_7task_samplerfix"
MANIFEST="$POLICY_DATA/data_manifest.json"
PRETRAIN="$HRDT/pretrain_model/hrdt_pretrain_500k/pytorch_model.bin"
PRETRAIN_SHA256=e9eaff94ac1c1d039a193f04627b8c27794e826ca1eebd3de5eab60435254be2

export PATH=/personal/miniconda3/envs/hrdt/bin:$PATH
export PYTHONPATH="$HRDT:$BASE_CODE:$BASE_CODE/Xpolicylab"

mkdir -p "$POLICY_DATA" "$HRDT/pretrain_model"
for link_path in \
  "$POLICY_DATA/raw" \
  "$POLICY_DATA/stats.json" \
  "$POLICY_DATA/lang_embeddings" \
  "$HRDT/pretrain_model/hrdt_pretrain_500k"; do
  if [[ -e "$link_path" && ! -L "$link_path" ]]; then
    echo "[error] Refusing to replace non-symlink path: $link_path" >&2
    exit 2
  fi
done
ln -sfnT "$DATA/raw_0908_7task" "$POLICY_DATA/raw"
ln -sfnT "$DATA/stats_0908_7task.json" "$POLICY_DATA/stats.json"
ln -sfnT "$DATA/lang_embeddings_0908_7task" "$POLICY_DATA/lang_embeddings"
ln -sfnT "$WORKSPACE/pretrain_model/hrdt_pretrain_500k" "$HRDT/pretrain_model/hrdt_pretrain_500k"
python "$HRDT/tools/prepare_sparkarena_manifest.py" \
  --data-root "$POLICY_DATA/raw" \
  --config "$RUN_ROOT/hrdt_sparkarena_7task.yaml" \
  --stats "$POLICY_DATA/stats.json" \
  --lang-dir "$POLICY_DATA/lang_embeddings" \
  --pretrain "$PRETRAIN" \
  --pretrain-sha256 "$PRETRAIN_SHA256" \
  --expected-episodes-per-task 100 \
  --expected-total-frames 154251 \
  --output "$MANIFEST"

python "$HRDT/tools/verify_sparkarena_manifest.py" \
  --manifest "$MANIFEST" \
  --deep \
  --hash-pretrain \
  --expected-pretrain-sha256 "$PRETRAIN_SHA256"

python "$HRDT/tools/audit_index_plan.py" \
  --manifest "$MANIFEST" \
  --sample-count 5120000 \
  --global-batch-size 64 \
  --seed 42

echo "Prepared read-only data manifest: $MANIFEST"
