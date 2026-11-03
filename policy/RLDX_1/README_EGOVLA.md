# RLDX-1 × EgoVLA joint38

This recipe fine-tunes RLDX-1 on the 12-task EgoVLA H1 + Inspire dataset. It
is isolated from the existing Spark joint54 recipe and does not modify the
vendored `RLDX-1/` upstream checkout.

## Contract

- robot / `env_cfg_type`: `ego_h1_inspire`
- `action_type`: `joint`
- state/action layout: left arm 7, left hand 12, right arm 7, right hand 12
- stored action: source HDF5 `/action`, mapped from native 50D to the policy's
  absolute commanded joint target, 38D; future state is never used as action
- RLDX transform: arms `RELATIVE`, hands `ABSOLUTE`
- cameras: head, left wrist, right wrist; missing raw wrists remain black with
  `observation.camera_mask=0`
- color / resolution: RGB without a channel swap, stored at 384x384 and passed
  through the checkpoint processor at 256x256 in both training and inference
- source frequency: 30 FPS
- action horizon: 16; video context: 4 frames at stride 2

The launcher performs strict full-parameter fine-tuning: the full LLM, vision
tower, cognition tokens, projector, and diffusion action model are trainable;
neither backbone nor action LoRA is enabled. It uses DeepSpeed ZeRO-3 and
gradient checkpointing so this surface can be tested on eight 80 GB GPUs.

## Data

First create the raw-action dataset under this project's `data/` directory.
The command independently clones the already verified videos, replaces the
legacy next-state action column with the retained source command, then compares
every episode against the original HDF5 `/action` and `/observations/qpos`:

```bash
RLDX_PYTHON_BIN="$PWD/policy/RLDX_1/RLDX-1/.venv/bin/python" \
bash data_scripts/prepare_egovla_raw_action_v21.sh
```

The script works in private staging, computes exact absolute and arm-relative
statistics, audits every parquet episode and language label, verifies every
video frame count, and compares all 1,531,638 compressed packets against the
exact v3 source slices before atomically publishing:

```text
data/EgoVLA_benchmark_rldx_v21_raw_action
```

Training refuses data whose audit-bound metadata hashes changed.
Before launching, it also exports `egovla_observation.json` beside the run so
evaluation resizes live 720x1280 RGB cameras to the training resolution and
uses black 384x384 wrist images for tasks that have no wrist cameras.

## Training runtime

The launcher derives the repository root from its own location. Its default
dataset, checkpoint, RLDX-1 environment, and Hugging Face metadata cache live
under this checkout. Override them with `RLDX_DATASET_PATH`,
`RLDX_BASE_MODEL_PATH`, `RLDX_PYTHON_BIN`, `RLDX_SITE_PACKAGES`, or
`RLDX_HF_HOME`. System FFmpeg is required by upstream TorchCodec.

One-step, eight-GPU smoke test:

```bash
cd policy/RLDX_1
bash prepare_egovla_full_base_overlay.sh
bash train_egovla_joint38.sh smoke
```

Formal run target (8 GPUs, global batch 64, per-GPU batch 8, 80,000 optimizer
steps, saving every 10,000 steps, LR 2e-5):

```bash
export WANDB_API_KEY='...'
MAX_STEPS=80000 SAVE_STEPS=10000 \
RLDX_RUN_NAME=EgoVLA-benchmark-rldx1_joint38-strict-full_80k-ego_h1_inspire-joint-0 \
bash train_egovla_joint38.sh start
```

The key must be supplied through the environment and is never stored in the
repository or launcher. The default W&B project is
`XPolicyLab-RLDX1-EgoVLA`. Checkpoints land in:

```text
policy/RLDX_1/checkpoints/
  <EgoVLA-run-directory>/
```

An already-started staged job may retain a historical run-directory suffix for
continuity; that naming detail is not the stopping criterion.  The effective
target remains 80,000 steps with a 10,000-step save cadence.  If an older
launcher state publishes intermediate checkpoints before continuation, those
artifacts are not the final run; continuation must still finish at 80,000 and
publish the requested cadence thereafter.

To continue an interrupted run, supply the same W&B environment and run:

```bash
bash train_egovla_joint38.sh resume
```

## Evaluation

After a complete checkpoint exists, use the standard adapter entry point:

```bash
XPL_ROOT="$(git rev-parse --show-toplevel)"
POLICY_DIR="${XPL_ROOT}/policy/RLDX_1"
EVAL_ENV_PATH=/path/to/egovla-isaaclab-1.2.0
export EVAL_MAIN_ROOT=/path/to/EgoVLA
export RLDX_POLICY_PYTHON_BIN="${POLICY_DIR}/RLDX-1/.venv/bin/python"
export RLDX_POLICY_SITE_PACKAGES="${POLICY_DIR}/RLDX-1/.venv/lib/python3.10/site-packages"
export RLDX_EVAL_PYTHON_BIN="${EVAL_ENV_PATH}/bin/python"
cd "${POLICY_DIR}"
bash eval.sh EgoVLA <task_name> \
  /absolute/path/to/checkpoint-80000 \
  ego_h1_inspire joint 0 0 0 \
  "${POLICY_DIR}/RLDX-1/.venv" "${EVAL_ENV_PATH}"
```

Real numerical inference remains unverified until a trained joint38 checkpoint
is produced. Debug transport alone can be checked with the adapter's reserved
`ckpt_name=debug` flow.
