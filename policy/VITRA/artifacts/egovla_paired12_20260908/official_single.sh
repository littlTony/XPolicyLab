#!/usr/bin/env bash
set -euo pipefail
POLICY=/personal/xiangpc/0813_Xpolicylab_bench/VITRA/XPolicyLab/policy/VITRA
BENCH='/personal/xiangpc/EgoVLA benchmark'
RUN="$POLICY/checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True"
EVIDENCE="$POLICY/artifacts/egovla_paired12_20260908"
export VITRA_MODEL_PATH="$RUN/checkpoints/epoch=10-step=80000.ckpt"
export VITRA_CONFIG_PATH="$RUN/config.json"
export VITRA_STATISTICS_PATH=/personal/xiangpc/0813_Xpolicylab_bench/VITRA/data/egovla_humanoid_sim_mano_20260902/teledata_statistics.json
export VITRA_STATISTICS_SHA256=8c0170503168fba85850a0842c35081f00e9f9084b06d678381a73742b107003
export VITRA_H1_PAIRED12_PATH="$EVIDENCE/bounded/paired12_0908_v1.json"
export VITRA_H1_PAIRED12_SHA256=b1f734d1e64280f0226d62c3fb12ada8c60946bc23a3939e4b080b91589b1f4b
export EGOVLA_WORKSPACE_ROOT="$BENCH"
export EGOVLA_EVAL_CONDA_ENV="$BENCH/.runtime/conda/egovla-isaaclab-1.2.0"
export XPOLICYLAB_ROOT="$POLICY/../.."
export EVAL_ENV_TYPE=sim
PORT=$(bash "$XPOLICYLAB_ROOT/utils/get_free_port.sh")
ARGS=(env-client --task-name Humanoid-Pour-Balls-v0 --policy-name VITRA
      --env-cfg-type ego_h1_inspire --action-type ee --checkpoint "$VITRA_MODEL_PATH"
      --host localhost --port "$PORT" --protocol ws --benchmark-protocol official
      --episodes 1 --official-split single --official-episodes 1 --official-trials 1
      --num-envs 1 --seed 42 --headless true --device-id 3
      --xpolicy-root "$XPOLICYLAB_ROOT" --output-dir "$EVIDENCE/official_single"
      --additional-info 'mapping=paired12_0908_v1,checkpoint=epoch10_step80000'
      --request-timeout-s 600)
bash "$BENCH/integration/scripts/egovla" "${ARGS[@]}" --dry-run > "$EVIDENCE/official_dry_run.json"
bash "$POLICY/setup_eval_policy_server.sh" EgoVLA Humanoid-Pour-Balls-v0 \
    epoch=10-step=80000.ckpt ego_h1_inspire ee 42 2 \
    /personal/miniconda3/envs/vitra_xpolicylab_0813 "$PORT" > "$EVIDENCE/official_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
bash "$XPOLICYLAB_ROOT/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER_PID" 'paired12 official policy' 1200
bash "$BENCH/integration/scripts/egovla" "${ARGS[@]}" > "$EVIDENCE/official_client.log" 2>&1
echo '[AUDIT] official client exited; inspect episode, result and video'
