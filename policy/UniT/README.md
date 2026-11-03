# UniT (Spark0 native joint54 adapter)

This directory integrates the official
[`xpeng-robotics/UniT`](https://github.com/xpeng-robotics/UniT) code with
XPolicyLab.  Upstream code remains in `policy/UniT/UniT`; the adapter only maps
XPolicyLab observations and actions to a dedicated native Spark0 joint config.

## Supported contract

- `action_type`: **`joint` only**.  End-effector-pose (`ee`) control is not supported.
- Robot: Spark / `tianji_marvin_wuji`, dual 7-DoF arms plus two 20-DoF hands.
  Dimensions are read at runtime through `get_robot_action_dim_info(env_cfg_type)`.
- Camera: `vision.cam_head.color`, already-decoded RGB.  It is center-cropped
  and resized to 256x256 without a channel swap, then mapped to `video.ego_view`.
- UniT state/action schema: right arm (7), right hand (20), left arm (7), left
  hand (20), exactly 54 values. There is no synthetic waist or zero padding.
- Dedicated data config: `spark0_dual_arm_dexterous_joint`, with mean/std
  normalization on both the original 54D state and 54D action, a 16-step goal
  horizon, DINO processing, and a regular 0.95 crop on the square head view.
- Embodiment: `new_embodiment` (ID 31). Runtime configs require at least 32
  action encoder/decoder embodiment slots and fail fast otherwise.

The benchmark's existing Spark environment config must be available as
`<bench_root>/env_cfg/<env_cfg_type>.yml` and must register
`tianji_marvin_wuji`.  The adapter deliberately fails if that benchmark config
is absent; it does not invent or alter a robot/environment definition.

## Installation

The upstream environment pins Python 3.10, PyTorch 2.5.1, torchvision 0.20.1,
transformers 4.52.0, accelerate 1.3.0, W&B 0.18.0, NumPy `<2`, pyarrow 14.0.1,
and flash-attn 2.7.1.post4.  `install.sh` invokes the official installer and
then installs XPolicyLab editable in the same environment.

```bash
cd /personal/xiangpc/0811_Xpolicylab_bench/UniT/policy/UniT
# Optional: run proxy_on first, or set PIP_INDEX_URL, for faster downloads.
CONDA_ENV_NAME=unit bash install.sh
conda activate unit
```

The official model configs use `facebook/dinov2-large` and
`Qwen/Qwen2.5-VL-3B-Instruct`.  They may resolve from the Hugging Face cache or
from locally adjusted copies of the JSON configs supplied through
`TOKENIZER_BASE_CONFIG` / `PRETRAIN_BASE_CONFIG`.  Do not commit credentials or
machine-specific checkpoint paths.

`pretrain_model/VLA-UniT-3B-fulldata` is retained as the requested reusable
pretrained-asset link, but its published GR1 metadata describes 6-D hands.
Spark has 20-D hands, so that checkpoint is not a valid Spark evaluation
checkpoint and is never selected implicitly by `deploy.yml`.  The adapter also
checks saved state/action modality shapes before loading model weights and
fails fast on incompatible group order/dimensions or any extra waist group.

## Data processing

The converter lives at
`data_scripts/convert_spark0_bench_to_lerobot_v20_joint54.py` in the XPolicyLab
checkout.  It produces six independent LeRobot **v2.0** datasets under:

```text
data/Spark0_bench_lerobotV20_joint54/
├── collect_objects/
├── dual_bottles_pick/
├── hammer_beat/
├── insert_block/
├── retrieve_gap/
└── stack_bowls/
```

Each sample has one RGB ego view, four native arm/hand joint groups, language
annotations, and the corresponding direct HDF5 joint54 action. The converter
never derives formal output from the retired joint57 Parquet or video files.
Run a one-episode-per-task preview first:

```bash
OUTPUT_ROOT=/personal/xiangpc/0811_Xpolicylab_bench/UniT/data/.Spark0_bench_lerobotV20_joint54_preview \
  bash process_data.sh Spark0_bench unit_full <spark_env_cfg> joint 1
```

Then run the complete conversion:

```bash
conda activate unit_torch26_recovery
bash process_data.sh Spark0_bench unit_full <spark_env_cfg> joint
```

Defaults:

- HDF5 source: `/mnt/xspark-data/tjy/spark0_bench`
- output: `/personal/xiangpc/0811_Xpolicylab_bench/UniT/data/Spark0_bench_lerobotV20_joint54`

Override them with `SOURCE_ROOT` and `OUTPUT_ROOT`.  An optional fifth
positional argument limits episodes per task for converter testing only.
`scripts/audit_spark0_joint54.py` performs an independent all-episode HDF5 to
Parquet equality audit, video probing, metadata/language checks, and first/last
reads through `LeRobotSingleDatasetWithGoalImage`. In `--official-loader`
mode it also verifies exact 54D state/action masks, zero padding, and the
mean/std-normalized values actually returned to the model.

The old `data/Spark0_bench_lerobotV20_joint57` data and its checkpoints are
preserved for provenance and marked `RETIRED_WRONG_SCHEMA`; they must not be
used for new training, conversion, or evaluation.

## Training

This is the official `run_gr1_full.sh` two-stage recipe adapted only where the
Spark0 experiment requires it:

1. tokenizer: 40,000 steps;
2. dual-system VLA training: 80,000 steps.

Both stages use eight GPUs, global batch size 64 (`--batch-size 8` per GPU,
accumulation 1), save every 10,000 steps, report to W&B, and train on each
task's official `[:-10]` split (last ten episodes held out).  UniT's trainer
hard-codes seed 42, so the wrapper requires positional seed `42` rather than
putting a misleading seed in the checkpoint name.

Authenticate W&B outside the repository:

```bash
export WANDB_API_KEY='...'
# Or run once: wandb login
```

Then launch:

```bash
conda activate unit_torch26_recovery
cd /personal/xiangpc/0811_Xpolicylab_bench/UniT/policy/UniT
bash train.sh Spark0_bench unit_full <spark_env_cfg> joint 42 0,1,2,3,4,5,6,7
```

Outputs follow the XPolicyLab convention:

```text
checkpoints/Spark0_bench-unit_full-<spark_env_cfg>-joint54-meanstd-42/
├── tokenizer/checkpoint-{10000,20000,30000,40000}/
├── pretrain/checkpoint-{10000,20000,30000,40000,50000,60000,70000,80000}/
└── latest -> pretrain/checkpoint-80000
```

Partial joint54 checkpoints are resumed only when `RESUME_JOINT54=1` is set
explicitly. The separate `joint54-meanstd` output identity prevents either a
retired joint57 checkpoint or the earlier SinCos checkpoint from being
selected. Both upstream training entry points expose `--save-total-limit`;
this wrapper sets it to 4 for Tokenizer and 8 for VLA, so every scheduled
10,000-step checkpoint is retained.
Before Stage 2 can start, `verify_hf_checkpoint.py` requires the tokenizer's
40,000-step `trainer_state.json`, experiment metadata, and either a complete
single-file model weight or every shard named by its weight index.  The same
check protects the final 80,000-step dual-system checkpoint, so a partially
written `config.json` cannot be mistaken for a finished model.

## Evaluation

`ckpt_name` may be an explicit checkpoint path, a full XPolicyLab run folder,
or the short name in the conventional path above.  It must resolve to a
checkpoint produced for the Spark joint54 schema; there is intentionally no
generic pretrained fallback in `deploy.yml`.  A usable inference checkpoint
contains `config.json`, complete model weights, and
`experiment_cfg/metadata.json`.

Real checkpoint evaluation:

```bash
conda activate unit_torch26_recovery
bash eval.sh Spark0_bench stack_bowls \
  /absolute/path/to/pretrain/checkpoint-80000 \
  <spark_env_cfg> joint 42 0 0 unit <eval_env>
```

Offline transport/action-shape smoke test (no model weights or simulator):

```bash
UNIT_DRY_RUN=1 EVAL_ENV_TYPE=debug \
  bash eval.sh Spark0_bench stack_bowls smoke <spark_env_cfg> joint 42 0 0 unit base

UNIT_DRY_RUN=1 DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug \
  bash eval.sh Spark0_bench stack_bowls smoke <spark_env_cfg> joint 42 0 0 unit base
```

`UNIT_DRY_RUN=1` is explicitly a wiring test: it validates the RGB/state input
contract and returns correctly shaped zero joint actions.  It must never be
used for benchmark results.  Real UniT inference has not been validated until
a compatible trained checkpoint and its tokenizer/backbone assets load and a
rollout succeeds.

## EgoVLA benchmark (H1-Inspire, mixed three-view)

The EgoVLA benchmark uses the H1-Inspire 38-D manipulation schema and the
canonical three-view camera ABI: `cam_head`, `cam_left_wrist`, and
`cam_right_wrist`.  The source is
`/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw`; the policy-local
materialization is produced by `data_scripts/prepare_egovla_three_view_v21.py`
under:

```text
data/EgoVLA_benchmark_rldx_v21_three_view
```

The conversion excludes every HDF5 path containing `Deprecated` (case
insensitive): 100 deprecated trajectories are not materialized or trained.
The resulting set has 1,903 active episodes, 510,546 frames, and 12 tasks.
All 5,709 aligned camera videos are retained.  `Insert-And-Unload-Cans` has
900 episodes with real wrist videos and `camera_mask=[1,1,1]`; the other 1,003
active episodes have the converter's explicit black wrist fallback and
`camera_mask=[1,0,0]`.  The mask and lineage remain in parquet/metadata.  UniT
uses the mask when supplied at evaluation, and the DINO view pool automatically
ignores all-black inactive fallback streams during training; this keeps a
head-only episode from diluting its visual signal while preserving true wrist
views for the multiview task.

```bash
conda activate unit_torch26_recovery
bash process_egovla_data.sh --dry-run
bash process_egovla_data.sh
bash process_egovla_data.sh --validate-output
```

The data config is `egovla_h1_joint38_threeview` (the compatibility alias
`egovla_h1_joint38_multiview` is also registered), with video keys
`video.cam_head`, `video.cam_left_wrist`, and `video.cam_right_wrist`.  State
and action groups are left arm 7 + left hand 12 + right arm 7 + right hand 12,
mean/std normalized, with a 16-step action horizon.  The 38-D vector is the
manipulation subset of raw 50-D H1 values; 12 lower-body joints are excluded.
The vision branch restores `[B,V*T,C,H,W]`, masks inactive views, pools aligned
views/time, and returns the existing M-Former patch length.

Launch the two-stage recipe with global batch 64 (8 per card), accumulation 1,
save every 10,000 steps, and an 80,000-step dual-system stage:

```bash
export WANDB_API_KEY='(set in the shell; do not commit it)'
bash train_egovla.sh EgoVLA_benchmark egovla_unit_threeview \
  ego_h1_inspire joint 42 0,1,2,3,4,5,6,7
```

Evaluation automatically selects `deploy_egovla.yml` for
`EgoVLA_benchmark + ego_h1_inspire`, and therefore uses the same three-view
config and camera order.  Set `UNIT_EGOVLA_VIEW_MODE=single_view` or pass an
explicit `DEPLOY_CONFIG` only when evaluating a legacy head-only checkpoint.
The adapter accepts missing wrist payloads on head-only simulator episodes and
creates deterministic black frames; when a `camera_mask` is transported, an
inactive stream is forcibly zeroed before inference.  RGB arrays are consumed
as decoded RGB without channel swapping or model-side video decoding.

```bash
# Real checkpoint (path or conventional run folder)
bash eval.sh EgoVLA_benchmark <task_name> \
  /absolute/path/to/pretrain/checkpoint-80000 \
  ego_h1_inspire joint 42 0 0 unit_torch26_recovery

# Transport/action smoke tests (no checkpoint or simulator)
UNIT_DRY_RUN=1 EVAL_ENV_TYPE=debug \
  bash eval.sh EgoVLA_benchmark smoke smoke ego_h1_inspire joint 42 0 0 \
  unit_torch26_recovery unit_torch26_recovery
UNIT_DRY_RUN=1 DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug \
  bash eval.sh EgoVLA_benchmark smoke smoke ego_h1_inspire joint 42 0 0 \
  unit_torch26_recovery unit_torch26_recovery
```

`UNIT_DRY_RUN=1` validates transport and action shapes only; it must not be
used as a benchmark result.  Real rollout evaluation remains required after a
compatible three-view checkpoint is available.
