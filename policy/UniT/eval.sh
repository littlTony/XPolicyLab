#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 10 ]] || { echo 'Usage: eval.sh bench task ckpt_name env_cfg ee seed policy_gpu env_gpu policy_python_or_env eval_env' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$ROOT/../.." && pwd)"
PORT=$(bash "$XPL_ROOT/utils/get_free_port.sh")
SERVER_PID=
cleanup() { if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
bash "$ROOT/setup_eval_policy_server.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$9" "$PORT" &
SERVER_PID=$!
bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER_PID" 'UniT policy server' 1200
bash "$ROOT/setup_eval_env_client.sh" "$1" "$2" "$3" "$4" "$5" "$6" "$8" "${10}" \
  "ckpt_name=$3,action_type=$5" "$PORT" localhost
