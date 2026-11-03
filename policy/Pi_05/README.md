# DexBench Pi0.5 — absolute EE + Wuji Hand 2

Code was copied from `/personal/xiangpc/0811_Xpolicylab_bench/pi05/policy/Pi_05`.
The active checkout is `/personal/wenwei/DexBench/XPolicyLab/policy/Pi_05`.
Previous target code is archived under `/personal/wenwei/DexBench/backups/pi05_before_ee54_20261021/Pi_05`.
Source training runs, virtual environments and weights were not copied.

## Data Processing

Use the prepared **LeRobot v3.0** dataset at
`/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks`.
It contains 3,000 episodes, 751,348 rows and three RGB cameras. Keys match
`data_preprocess/export_lerobot.py`; do not run the historical RoboDojo converter
in `openpi/scripts/process_data.py` against this prepared dataset.

Both `observation.state` and `action` are 54D, ordered as:

```text
0:7    left EE:  world x,y,z,qw,qx,qy,qz
7:27   left Wuji Hand 2 absolute joints (20, radians)
27:34  right EE: world x,y,z,qw,qx,qy,qz
34:54  right Wuji Hand 2 absolute joints (20, radians)
```

The exporter stores state[t] and its recorded following-pose target action[t],
dropping the final source frame. No additional time shift belongs in training.
The original SparkArena `pi05_wuji_marvin_54d` config uses `LeRobotWujiDataConfig`,
uses `cam_high`, `cam_left_wrist`, `cam_right_wrist`, and prohibits delta actions.
No quaternion/Rot6D conversion or RGB channel swap is applied. All three cameras
are required at inference. Runtime normalizes the predicted quaternions to unit
length, preserves wxyz order, and leaves absolute xyz/joints unchanged.
Hand joints keep the exporter's Isaac stage-major order: at each of the four
joint stages, index/middle/pinky/ring/thumb. Do not reorder them finger-major.

Prompts match the export: `[{embodiment}/{task}] {instruction}`. Deployment defaults
to `prompt_embodiment: tianji_wujihand2_real`. A fixed task rejects conflicting
prefixes. Use `task_name=auto` to serve multiple tasks with one loaded model;
each observation must contain the full training prompt, or an unprefixed
instruction plus `task_name`. A conflicting embodiment prefix is always rejected.

Prepare full normalization statistics without re-encoding images:

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/Pi_05
bash process_data.sh
```

This reads only parquet numeric columns and writes statistics to
`openpi/assets/pi05_wuji_marvin_54d/dexbench_12tasks/`. State statistics use
all rows; action statistics weight each row exactly as LeRobot's 50-frame forward
chunks with end-of-episode padding. Empirical quantiles replace the official
approximate histogram quantiles. The independent stock
`openpi/scripts/compute_norm_stats.py` remains available as an image-loading
alternative and writes the same action-contract marker. `process_data.sh` is the
canonical preparation entry point: it also binds statistics to the numeric
Parquet/metadata content SHA and records the normalization-file SHA. Training
rejects stale statistics even when the number of episodes and frames is unchanged.
Inference always uses the statistics saved in its selected checkpoint.

## Installation

The copied source uses the already installed shared Python dependencies at
`/personal/xiangpc/0811_Xpolicylab_bench/pi05/policy/Pi_05/openpi/.venv-egovla/bin/python`.
`runtime.sh` prepends this checkout's OpenPI, openpi-client and XPolicyLab paths;
it does not use the original checkout's Python source through editable installs.
Override `PI05_PYTHON_BIN` to use another compatible environment.

For a private environment, run the bundled `install.sh` with `uv` installed,
then set `PI05_PYTHON_BIN` to `openpi/.venv/bin/python`. The shared simulator
defaults to `/personal/wenwei/xspark-0-simeval`; override `EVAL_MAIN_ROOT` if needed.
The shared path/dimension helper reads that simulator's `env_cfg/`.

## Training

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/Pi_05
bash train.sh DexBench pi05_ee54 tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

The active launcher uses full-parameter Pi0.5, 54D, horizon50, global batch256,
60000 steps and FSDP8. Official optimizer/EMA/freezing defaults are preserved,
including peak LR2.5e-5 and final LR2.5e-6; save every1000 and keep every5000.
Local per-host dataset/JAX caches are used. This clone uses the requested absolute EE data
rather than the historical SparkArena joint/delta action representation. Pretrained
parameters come from the existing local `pi05_base` through the partial weight
loader; the action input/output heads are reinitialized for 54D.
No training is launched by installation or tests. Training refuses incompatible
data/config/stats and does not overwrite an existing run. To resume, set
`OPENPI_RESUME=1`; pass further `train.py` CLI flags after the six arguments.

The completed `dq128_60k_pi05_20261027_a01_ga1` run is deployed as
`ckpt_name=pi05_wuji`. It used the explicit override
`--batch-size=128` on A800-11. Its final checkpoint is `59999`, whose internal
optimizer step is **60000**. It finalized on 2026-10-29 at 17:36:39 China time.
The last logged action FM loss is **0.0059 at log step59900**; a separate loss
for the final update was not logged. Do not interpret the directory name as a
missing optimizer update.

Only the canonical pi05_wuji checkpoint directory is retained. The old compatibility
symlink was removed after restarting and validating the real inference server.
No weights or normalization files were modified.

Checkpoints land in:

```text
checkpoints/DexBench-pi05_ee54-tianji_marvin_wuji-ee-42/<step>/
```

Each checkpoint saves its own normalization and `action_contract.json` under
`assets/dexbench_12tasks/`. Inference requires that marker and will not silently
load old joint/delta checkpoints. An explicitly requested missing checkpoint step
raises an error; `checkpoint_num: null` selects the newest finalized numeric step.

## Evaluation

Start the completed model's real WebSocket server on one GPU:

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/Pi_05
PI05_CHECKPOINT_NUM=59999 bash setup_eval_policy_server.sh \
  DexBench auto pi05_wuji \
  tianji_marvin_wuji ee 42 0 uv 6011 0.0.0.0
```

On A800-11 the endpoint is `ws://10.189.16.3:6011`. `auto` serves all trained
tasks with per-observation prompts. For a fixed-task simulator run, replace
`auto` with that task name. The server logs and exposes through `get_metadata`
the exact checkpoint, normalization SHA256, 54D absolute action contract,
three RGB cameras, horizon50, and Euler inference step count (default10).
`PI05_CHECKPOINT_NUM` pins a finalized step; unset it to select the latest.
The startup script binds the copied OpenPI source before the shared dependency
environment, and automatically sizes the JAX pool to the selected GPU.

Run the simulator loop using the same model adapter:

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/Pi_05
bash eval.sh DexBench collect_objects \
  /personal/wenwei/DexBench/XPolicyLab/policy/Pi_05/checkpoints/DexBench-pi05_wuji-tianji_marvin_wuji-ee-42/59999 \
  tianji_marvin_wuji ee 42 0 1 uv RoboDojo
```

The last argument selects the simulator's conda environment. The example uses
the installed `RoboDojo` environment; `Luminis` is another checked environment.
A replacement must provide Isaac Lab and the WebSocket/msgpack client dependencies. Only `action_type=ee` is accepted.
`Model.get_action()` returns a chunk `list[dict]`, each action containing exactly:

```text
left_ee_pose: (7,)        left_ee_joint_state: (20,)
right_ee_pose: (7,)       right_ee_joint_state: (20,)
```

`get_action_batch(env_idx_list)` returns chunks for the requested active env IDs
in that order. The unchanged xspark-sim-eval client infers EE control from these
keys and sends the poses directly to IK; hand joint targets remain absolute.

Extra deploy keys: `train_config_name`/`repo_id` bind the training recipe and
normalization; `policy_uv_env_path`/`policy_python_bin` select source and dependencies;
`prompt_embodiment` sets the exporter-compatible prefix; `checkpoint_num` selects
a finalized step; `inference_steps` selects the flow integration count.
The production adapter always loads real weights. Test-only hold-state fixtures
live under `tests/` and cannot be selected by the server launch configuration.

Replay two recorded Tianji WujiHand2 training episodes per physical task:

```bash
source runtime.sh
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu "$PI05_PYTHON_BIN" \
  tests/evaluate_recorded_episodes.py \
  --url ws://127.0.0.1:6011 \
  --checkpoint checkpoints/DexBench-pi05_wuji-tianji_marvin_wuji-ee-42/59999 \
  --output /personal/wenwei/DexBench/evaluations/pi05_recorded_replay_new
```

The client uses the exact native training dataset loader, verifies full
training/inference preprocessing parity against checkpoint-local normalization,
and sends real RGB/state/prompt observations through the WebSocket server.
It evaluates non-overlapping 50-step chunks covering all frames of each selected
episode, excludes padded tails, and reports position error (cm), quaternion
geodesic error (degrees), hand-joint MAE (radians), and a hold-state baseline.
This is recorded training-data action agreement, not validation FM loss or a
closed-loop task success rate. Episode IDs, per-episode predictions and targets,
server metadata, and per-task metrics are saved for reproduction.

## Verification

```bash
source runtime.sh
JAX_PLATFORMS=cpu "$PI05_PYTHON_BIN" -m pytest -q tests/test_dexbench_contract.py
"$PI05_PYTHON_BIN" check_contract.py --require-stats
```

Tests verify RGB preservation, EE/wxyz packing, absolute action semantics, camera
requirements, Wuji transforms with delta disabled, active-env batch ordering,
checkpoint selection, checkpoint-local statistics, multi-task prompt checks, action-chunk normalization
weights, same-count stale-statistics rejection, and actual three-camera RGB/state
train/runtime preprocessing parity. Physics simulation success must be measured
separately from recorded-episode action agreement.

## Original SparkArena reference

eval_web/backend/app/execution.py selects deploy_sparkarena_54d.yml for
the checkpoint norm asset SparkArena_bench_7task_0909. In the source checkout,
policy/Pi_05/deploy_sparkarena_54d.yml names pi05_wuji_marvin_54d.
The actual launcher is:
`/personal/xiangpc/0811_Xpolicylab_bench/pi05/scripts/train_pi05_sparkarena_7task_8gpu.sh`.

The untouched upstream recipe uses joint54 with delta transforms. This copy
reuses its existing config name/class/model recipe and changes the data/ABI
parameters to cam_high + absolute world EE54 for dexbench_12tasks. EgoVLA
pi05_egovla_joint38_full is not selected. The original source checkout is unchanged.

## Historical A800-6 official-recipe run (2026-10-22; stopped)

Global batch size 256, 60,000 steps, cosine decay 60,000 steps, 8 GPUs,
FSDP8 for the larger batch. Seed42; save every1000 and keep every5000. Action contract,
dataset and normalization statistics remain the verified absolute EE54 recipe.
W&B records offline under logs/wandb. The current train.sh and named config
defaults use batch 256 and 60k steps; the original SparkArena reference above
documents the historical recipe.
