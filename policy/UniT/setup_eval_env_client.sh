#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 10 && $# -le 11 ]] || { echo 'Expected standard XPolicyLab client arguments' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$ROOT/../.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-$(cd "$XPL_ROOT/.." && pwd)}"
exec bash "$XPL_ROOT/utils/setup_env_client.sh" "$XPL_ROOT/utils" "$ROOT/deploy.yml" "$8" "${10}" \
  "$1" "$2" "$4" UniT "$9" "$BENCH_ROOT" "$6" "$7" "${11:-localhost}"
