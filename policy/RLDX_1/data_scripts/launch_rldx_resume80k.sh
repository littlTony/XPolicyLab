#!/usr/bin/env bash
set -euo pipefail

# Private continuation watcher for the xiangpc RLDX-1 run.  The first stage
# was started with a 30k limit; this waits for its complete checkpoint-30000,
# then resumes the same output/W&B run to 80k with 10k save intervals.

ROOT=/mnt/xspark-data/xiangpc/0811_Xpolicylab_bench/RLDX_1
POLICY="$ROOT/policy/RLDX_1"
UPSTREAM="$POLICY/RLDX-1"
RUN_NAME=EgoVLA-benchmark-rldx1_joint38-strict-full_30k-ego_h1_inspire-joint-0
OUT="$POLICY/checkpoints/$RUN_NAME"
CURRENT_PID=2485754
ENTRY="$POLICY/launch_train_egovla.py"

mkdir -p "$POLICY/logs"
echo "[continuation] waiting for existing PID $CURRENT_PID to finish"

cmdline="$(tr '\0' ' ' < "/proc/$CURRENT_PID/cmdline" 2>/dev/null || true)"
[[ "$cmdline" == *"$ENTRY"* && "$cmdline" == *"--max-steps 30000"* ]] || {
  echo "refusing to watch unexpected PID $CURRENT_PID: $cmdline" >&2
  exit 1
}

# The running torchrun inherited the credential from the private launcher.
# Capture it only in memory; never print or write it.
WANDB_API_KEY="$(python3 - "$CURRENT_PID" <<'PY'
import pathlib, sys
items = pathlib.Path(f"/proc/{sys.argv[1]}/environ").read_bytes().split(b"\0")
for item in items:
    if item.startswith(b"WANDB_API_KEY="):
        print(item.split(b"=", 1)[1].decode(), end="")
        break
PY
)"
[[ ${#WANDB_API_KEY} -ge 40 ]] || { echo "could not capture W&B credential from running job" >&2; exit 1; }
export WANDB_API_KEY

while kill -0 "$CURRENT_PID" 2>/dev/null; do
  sleep 30
done
echo "[continuation] original 30k launcher exited; waiting for checkpoint publication"

checkpoint="$OUT/checkpoint-30000"
for attempt in $(seq 1 240); do
  complete=1
  [[ -f "$checkpoint/trainer_state.json" ]] || complete=0
  [[ -f "$checkpoint/latest" ]] || complete=0
  [[ -f "$checkpoint/scheduler.pt" ]] || complete=0
  [[ -f "$checkpoint/model.safetensors.index.json" ]] || complete=0
  [[ "$(find "$checkpoint" -maxdepth 1 -name 'rng_state_*.pth' 2>/dev/null | wc -l)" -eq 8 ]] || complete=0
  [[ "$(find "$checkpoint/global_step30000" -maxdepth 1 -name 'bf16_zero_pp_rank_*_optim_states.pt' 2>/dev/null | wc -l)" -eq 8 ]] || complete=0
  [[ "$(find "$checkpoint/global_step30000" -maxdepth 1 -name 'zero_pp_rank_*_model_states.pt' 2>/dev/null | wc -l)" -eq 8 ]] || complete=0
  if (( complete )); then
    step="$(python3 - "$checkpoint/trainer_state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["global_step"])
PY
)"
    if [[ "$step" == 30000 ]] && grep -q 'global_step30000' "$checkpoint/latest"; then
      echo "[continuation] checkpoint-30000 is complete"
      break
    fi
  fi
  if (( attempt == 240 )); then
    echo "checkpoint-30000 did not become complete" >&2
    exit 1
  fi
  sleep 30
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLDX_RUN_NAME="$RUN_NAME"
export RLDX_LOCAL_CACHE_ROOT=/tmp/rldx-egovla-hpc
export WANDB_RUN_ID=m6697sm5
export WANDB_RESUME=must
export MAX_STEPS=80000
export SAVE_STEPS=10000
export RLDX_SAVE_STEPS_OVERRIDE=10000
export SAVE_TOTAL_LIMIT=12
export DATALOADER_NUM_WORKERS=8
export LEARNING_RATE=2e-5

[[ "${RLDX_SAVE_STEPS_OVERRIDE}" == "${SAVE_STEPS}" ]] || {
  echo "cadence override must match SAVE_STEPS=${SAVE_STEPS}" >&2
  exit 1
}
echo "[continuation] launching resume: max_steps=80000 save_steps=${SAVE_STEPS} override=${RLDX_SAVE_STEPS_OVERRIDE}"
cd "$POLICY"
exec bash "$POLICY/train_egovla_joint38.sh" resume
