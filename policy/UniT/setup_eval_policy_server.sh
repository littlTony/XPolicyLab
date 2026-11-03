#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 || $# -gt 10 ]]; then
  echo 'Usage: setup_eval_policy_server.sh <bench> <task|auto> <ckpt_name|path> <env_cfg> ee <seed> <gpu> <python|env|auto> <port> [host]' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$ROOT/../.." && pwd)"
export EVAL_MAIN_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
if [[ "$4" != tianji_marvin_wuji || "$5" != ee ]]; then
  echo 'UniT adapter supports tianji_marvin_wuji / ee only' >&2; exit 2
fi
if [[ -n "${UNIT_PYTHON:-}" ]]; then
  PYTHON="$UNIT_PYTHON"
elif [[ -x "$8" && -f "$8" ]]; then
  PYTHON="$8"
elif [[ -x "$8/bin/python" ]]; then
  PYTHON="$8/bin/python"
elif [[ "$8" == auto || "$8" == unit ]]; then
  PYTHON=/personal/miniconda3/envs/unit_torch26_recovery/bin/python
else
  PYTHON="/personal/miniconda3/envs/$8/bin/python"
fi
[[ -x "$PYTHON" ]] || { echo "Missing policy Python: $PYTHON" >&2; exit 1; }
STEP="${UNIT_CHECKPOINT_STEP:-60000}"
[[ "$STEP" =~ ^[0-9]+$ ]] && (( STEP > 0 && STEP <= 60000 )) || { echo 'Expected checkpoint step1..60000' >&2; exit 2; }
export PYTHONPATH="$XPL_ROOT/..:$XPL_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
exec env CUDA_VISIBLE_DEVICES="$7" PYTHONUNBUFFERED=1 "$PYTHON" "$XPL_ROOT/setup_policy_server.py" \
  --config_path "${UNIT_DEPLOY_CONFIG:-$ROOT/deploy.yml}" --overrides \
  policy_name=UniT port="$9" host="${10:-localhost}" bench_name="$1" task_name="$2" \
  ckpt_name="$3" env_cfg_type="$4" action_type="$5" seed="$6" gpu_id="$7" checkpoint_step="$STEP"
