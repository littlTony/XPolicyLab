# StarVLA

**Contributor:** RoboDojo Team | **Paper:** StarVLA: A Versatile Vision-Language-Action Model with Efficient Training and Policy Adaptation | **arXiv:** https://arxiv.org/abs/2604.05014 | **Original code:** https://github.com/starVLA/starVLA

`StarVLA` adapts the StarVLA vision-language-action framework to XPolicyLab/RoboDojo, exposing three variants that share the public `Qwen3-VL-4B-Instruct` backbone. Integration scripts live at this directory level; the vendored upstream implementation lives in `source_starvla/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Supported Variants

| Public name | StarVLA framework registry | Action head |
|---|---|---|
| `starVLA-OFT` | `QwenOFT` | MLP regression head over action-token hidden states |
| `starVLA-GR00T` | `QwenGR00T` | Flow-matching DiT action head |
| `starVLA-π` | `QwenPI_v3` | Layer-wise interleaved cross/self-attention DiT flow-matching head |

These names are reporting labels. All three variants use `StarVLA` as the XPolicyLab runtime `policy_name`; the selected checkpoint identifies the framework implementation. The action-policy components are trained from scratch, while Qwen3-VL-4B-Instruct is used as the public backbone initialization. No internal robot data, private demonstrations, hidden VLA pretraining, or unreleased pretrained policy weights are required by this adapter.

## External StarVLA Runtime Contract

The vendored `source_starvla` runtime implements the inference-side contract below. A separate checkout supplied through `starvla_root` must provide the same interface; checkpoints and normalization statistics are user-provided and are not included here.

- Register `QwenOFT`, `QwenGR00T`, and `QwenPI_v3` and reconstruct the framework selected by the checkpoint configuration.
- Accept three RGB observations (`cam_head`, `cam_left_wrist`, and `cam_right_wrist`), a language instruction, and an optional 14-dimensional ARX X5 absolute-joint state ordered as left arm, left gripper, right arm, and right gripper.
- Normalize state with the checkpoint's `arx_x5` training statistics before model inference. The 50-step RoboDojo schema uses q99 normalization for all dimensions, including the continuous grippers.
- Return normalized actions with shape `[batch, horizon, 14]` and expose `action_chunk_size` through websocket server metadata. The runtime must unnormalize actions with the matching `arx_x5` statistics before returning them to XPolicyLab.
- Support a 50-step predicted action chunk. RoboDojo executes the first 16 actions and then requests a new chunk.
- For the released RoboDojo PI-v3 checkpoint, route the layer-wise embeddings through the canonical interleaved DiT forward. The hand-written block loop predating StarVLA fix [`c521dec`](https://github.com/starVLA/starVLA/commit/c521decb7441c7dfea282c61dc758456bcffbb8f) silently treated every block as cross-attention and is incompatible with this checkpoint.

Released checkpoints should retain their run-directory layout so the runtime can find `config.yaml`, `config.full.yaml`, and `dataset_statistics.json` next to the `checkpoints/` directory. The public data-mixture name is `robodojo_arx_x5_h50_q99`; `robodojo_v21_all_h50_q99` remains supported as a compatibility alias.

## Installation

`install.sh` installs PyTorch 2.6, the upstream requirements, flash-attn, and `source_starvla/` in editable mode into the active environment:

```bash
cd XPolicyLab/policy/StarVLA
bash install.sh
conda activate <policy_env>  # e.g. starvla
```

## Data Processing

Converts RoboDojo demos into a LeRobot dataset at `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`. `raw_task_dirs` is the source task dir (or comma-separated list) under `data/<bench_name>/` and defaults to `ckpt_name`; a non-numeric fifth argument is treated as `raw_task_dirs`. Use the same `ckpt_name` when launching `train.sh`, unless you also set `STARVLA_XPOLICY_DATASET_NAME` explicitly.

The output is LeRobot v3.0 with the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)). The bundled converter is used instead of `scripts/transform_lerobot_v30_format.py` because starVLA's GR00T-style loader also needs a `meta/modality.json` written alongside the dataset.

```bash
cd XPolicyLab/policy/StarVLA
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls

# Example: rename the output while using all episodes from the original task data
bash process_data.sh RoboDojo stack_bowls_full arx_x5 joint stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/StarVLA
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Example: train the converted stack_bowls dataset on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. The trainer generates a per-run config from `xpolicy_oft_vla.yaml` under `.generated/`, requires the converted LeRobot dataset (checked via `meta/modality.json`), and infers the process count from a comma-separated `gpu_id`; trailing `extra_args` are forwarded to the upstream trainer. Overrides: `STARVLA_DATA_ROOT` (default `data/`), `STARVLA_DATA_MIX` (default `xpolicylab_runtime`), `STARVLA_XPOLICY_DATASET_NAME` (default `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`).

## DexBench EE58 training with the SparkArena recipe (QwenPI_v3 only)

This route uses the **current DexBench dataset** and the **SparkArena training
hyperparameters**, not the historical 700-episode SparkArena dataset or EgoVLA.
`train.sh` dispatches DexBench / `tianji_marvin_wuji` arguments to
`train_sparkarena.sh`. The supported representation is `action_type=ee`;
obsolete SparkArena/joint calls fail rather than silently training the wrong data.
The existing RoboDojo ARX14 release route remains separate.

The authoritative prepared export is
`/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21`:
3000 episodes, 751348 frames, 12 physical tasks with 250 episodes each, 60
robot/instruction variants, 25 Hz, and `train=0:3000`. It is a prepared LeRobot
**v2.1** export with official `observation.state`, `action` and
`observation.images.*` keys. Its `meta/action_contract.json` identifies
`dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1`.
The RoboDojo-only `process_data.sh` converter is not used here.

Both native **state and action are 58 dimensions**, in this order:

- left EEF: world XYZ3 followed by rotation6D6;
- left Wuji hand:20 joint values;
- right EEF: world XYZ3 followed by rotation6D6;
- right Wuji hand:20 joint values.

Rotation6D contains **matrix row0 followed by row1**. The export already contains
this representation; training applies no second quaternion/rotation conversion,
joint7 substitution, hand remapping, or state-derived target generation.
State is `state[t]`; targets are `action[t:t+16]`, with no extra temporal shift.
Tail chunks repeat the last action. All58 dimensions participate in FM loss,
including physical values that happen to be zero or constant.

The official source clone `upstream/` stays unchanged at
`985c2238627481f25476fbee478083812dd82a21`. Adapter code in `training/` registers
the EE58 dataset and coordinates the trainer. The native loader calls the v2.1
JSONL/episode-parquet schema family `v2.0`; the generated contract records the
actual dataset version as v2.1. `OffsetLeRobotDataset` applies the **per-camera
frame offset** in `meta/video_offsets.json` before decoding each linked packed
MP4. Ignoring these offsets would show the beginning of another episode.

The recipe in `configs/sparkarena_qwen_pi_v3.yaml` uses:

- Qwen3-VL-4B plus the 36-layer LayerwiseFM head; full training, no frozen backbone.
- Three real RGB cameras (`front`, `left_wrist`, `right_wrist`) resized to224x224.
- Independent58-dimensional state/action q99 statistics from the absolute export,
  native clipping to[-2.2,2.2], and float16 sample packing. QwenPI_v3 appends the
  normalized state as discretized state-text tokens.
- Eight GPUs x micro batch4 x GA2 = effective batch64; **60000 optimizer updates**.
- AdamW beta=(0.9,0.95), eps=1e-8, weight decay=1e-8; VLM LR=1e-5,
  action LR=1e-4, 5000 warmup updates, cosine minimum LR=5e-7, clip norm1.
- **16 noise repeats**, explicitly set under `trainer.repeated_diffusion_steps`.
  This is the value the historical forward actually used; the similarly named
  framework field did not control training. Noise repeats do not multiply the
  effective data batch or optimizer-update count.
- bf16/ZeRO-2, explicit Qwen gradient checkpointing and FA2 via Transformers APIs.
  The adapter verifies them: fresh upstream otherwise ignores the checkpointing
  flag and forces SDPA while loading Qwen.
- DeepSpeed owns accumulation, clipping, Adam and zeroing. LR, update counts,
  evaluation and saving advance only on optimizer boundaries, avoiding
  `Accelerate.no_sync` with partitioned ZeRO-2 gradients.
- The configured seed reaches the native mixture sampler. Persistent workers
  read a shared epoch counter, so their samples change when the loader advances
  its epoch. Native sampling remains uniform over episodes with replacement;
  60000x64 is 3840000 draws, about5.11 dataset-size equivalents, not5.11 exact
  passes over every frame. Noise repeats do not count as extra data epochs.

CPU preparation copies metadata into a unique `.generated/<run>_<suffix>/`
directory and links the large parquet/video directories without rewriting the
source. It validates all3000 episode lengths and751348 cached steps, snapshots
source/private metadata hashes and independent statistics, and checks native
loader samples from **two episodes per physical task**. Checks cover EE58
ordering, same-frame targets, q99 normalization, RGB images, packed-video offsets,
and final-action padding. A nonzero-offset frame is checked against its correct
packed-video timestamp and shown to differ from an unshifted read.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/StarVLA

# CPU-only data checks and generated configuration; no training starts.
bash train.sh DexBench tianji_wuji tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7 --dry-run

# Run only after allocating these eight GPUs to this training.
bash train.sh DexBench tianji_wuji tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

Default Python:
`/personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python`.
Default backbone: `/personal/xspark_shared/model_weights/Qwen3-VL-4B-Instruct`.
Override `STARVLA_PYTHON`, `STARVLA_BASE_VLM`, `STARVLA_EE58_DATASET`, or
`EVAL_MAIN_ROOT` as needed. W&B defaults to disabled and model assets are used
offline. The only trailing option is `--dry-run`; arbitrary native overrides are
not forwarded through this fixed recipe.

The shared checkpoint helper selects
`checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/`.
Existing output directories are rejected: choose a fresh `ckpt_name` to preserve
earlier runs. New checkpoint names containing batch-size labels such as `bs64`
or `batch_size64` are rejected. Metadata and adapter hashes are rechecked on all training ranks.
Snapshots include `launch_config.yaml`, `preflight.json`, native
`config.full.yaml`, and `dataset_statistics.json`; final weights are
`final_model/pytorch_model.pt`. This is new EE58 training, not optimizer resume
from a joint54 checkpoint. Old54-dimensional action-head/tokenizer interfaces
and normalization statistics must not be silently reused.

### Native EE58 serving

The EE branch of `model.py` loads the complete native QwenPI_v3 checkpoint
directly into the normal XPolicyLab server. It does not use the legacy ARX
inner websocket server. The existing joint/ARX branch is retained separately.
Each saved checkpoint has an adjacent `<weight>.ee58.json` receipt binding the
weight, `config.full.yaml`, checkpoint-local `ee58_statistics.json`, pinned
upstream revision and runtime/code hashes. Changed or missing sidecars fail
closed. Serving reads no training-dataset files; local Qwen backbone/processor
assets and the pinned source code are still required.

Server wire state is54D XYZ + WXYZ quaternion + Wuji20 per hand. The shared
codec packs it into the same row6D58D layout as training. State then uses the
same native independent q99 transform, clipping and **float16 cast before the
unmodified native state2str**. This preserves all256-bin edge behavior,
including the native -1 token at the lower clipping boundary. Native state and
action key dimensions are explicitly9/20/9/20; no equal split into four parts.
Three current RGB views use the same PIL224 resize and native Qwen processor.
The full model predicts16 steps with4 Euler iterations; action normalization is
inverted with the saved action statistics, followed by row6D SO3 projection and
conversion to the server's WXYZ54D actions. Default execution horizon is16.

```bash
# Requires a newly trained, complete EE58 checkpoint bundle at the run above.
bash setup_eval_policy_server.sh DexBench <task_name> tianji_wuji tianji_marvin_wuji ee 42 0 /personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla 8010 0.0.0.0

# CPU-only alignment oracle; the neural network alone is a GT-output stub.
CUDA_VISIBLE_DEVICES= /personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python training/check_ee58_alignment.py --report /tmp/starvla_ee58_alignment.json
```

The CPU oracle exercises actual `Model` initialization, `update_obs` and
`get_action` on two episodes per physical task. Original54D source quaternions
produce exactly the training58D state, normalized float16 state and native
state tokens. Three-view resized pixels match exactly. Single/batched16-step
GT-normalized actions pass the real inverse normalization and SO3/WXYZ path;
clipping and SO3 projection are included in the expected result. It also checks
statistics tampering and seed/epoch propagation with two persistent workers.
This verifies the adapter pipeline; no large-model neural forward or GPU
training was run for this check, and no new trained EE58 weights are claimed.

The native trainer's `mse_score` uses a training batch and computes L2 norm divided
by element count; it is not standard MSE, a held-out evaluation, or task success.
CPU data checks and GA/update coordination tests do not establish real8-GPU
forward/backward behavior, memory usage, throughput, or checkpoint-server quality.
The old SparkArena joint54 run's timing/loss is historical context, not this
EE58 run's ETA or measured result. Prior joint54 dry-run artifacts remain available.


## Evaluation (RoboDojo releases)

```bash
cd XPolicyLab/policy/StarVLA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained stack_bowls checkpoint
bash eval.sh RoboDojo stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

### Released Hugging Face RoboDojo checkpoints

Release-specific launchers, checkpoint preparation, simulator preflights, and
their reproducibility tests are kept together under `scripts/`. The root-level
`setup_eval_*` files remain the fixed XPolicyLab policy entry points.

Use the dedicated entry point for the released QwenOFT, QwenGR00T, and QwenPI_v3 checkpoints. It pins the exact Hugging Face revision, preserves the complete run-directory layout, verifies the weight and normalization SHA256 values, and starts the vendored StarVLA server with the published RoboDojo runtime contract:

```bash
# From a current RoboDojo checkout, install its official benchmark assets once.
cd RoboDojo
bash scripts/init_assets.sh
cd XPolicyLab/policy/StarVLA

# Exact seed-0 commands used for the 10-layout checks reported below.
STARVLA_ROBODOJO_NUM_ENVS=1 bash scripts/eval_hf_robodojo.sh oft build_tower 0 0 1 \
  <starvla_env> <robodojo_sim_env> 10
STARVLA_ROBODOJO_NUM_ENVS=5 bash scripts/eval_hf_robodojo.sh groot build_tower 0 0 1 \
  <starvla_env> <robodojo_sim_env> 10
STARVLA_ROBODOJO_NUM_ENVS=5 bash scripts/eval_hf_robodojo.sh pi_v3 build_tower 0 0 1 \
  <starvla_env> <robodojo_sim_env> 10

# Use `native` for the task's official episode count instead of a quick check.
bash scripts/eval_hf_robodojo.sh pi_v3 build_tower 0 0 1 <starvla_env> <robodojo_sim_env> native
```

The pinned releases' official 50-episode `build_tower` references are:

| Released checkpoint | Success rate | Score |
|---|---:|---:|
| [QwenOFT](https://huggingface.co/StarVLA/Qwen3vl4b-OFT-RoboDojo) | 36% | 50.60 |
| [QwenGR00T](https://huggingface.co/StarVLA/Qwen3vl4b-GR00T-RoboDojo) | 14% | 20.80 |
| [QwenPI_v3](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo) | 56% | 65.00 |

On 2026-08-09, we ran the three seed-0 commands above on RoboDojo `36bfcb7c` and XPolicyLab `fd420490` plus this change. The downloaded checkpoint and sidecar files matched every SHA256 pinned in the release manifest. The complete 10-layout results were:

| Released checkpoint | Isaac environments | Successes | Score | Unstable |
|---|---:|---:|---:|---:|
| QwenOFT | 1 | 4/10 | 58.00 | 0 |
| QwenGR00T | 5 | 1/10 | 20.00 | 0 |
| QwenPI_v3 | 5 | 5/10 | 57.00 | 0 |

With the same pinned PI-v3 snapshot, seed, layouts, data contract, and simulator, the legacy all-cross-attention forward produced 0/10 successes and score 0; the canonical interleaved forward produced 5/10 and score 57. QwenOFT and QwenGR00T do not execute the PI-v3 action-head path and were not changed by that correction.

The 10-episode command is a quick contract check rather than a replacement for the official 50-episode protocol. Near-zero success together with visibly abnormal arm motion should not be accepted as a small-sample fluctuation: inspect the server handshake and first state-contract log described below.

Do **not** start `deployment/model_server/server_policy.py` from a separate or unpinned upstream StarVLA checkout. `scripts/eval_hf_robodojo.sh` starts the XPolicyLab vendored server itself. The adapter requires the server handshake to confirm all of the following before the simulator can move the robot:

- images are RGB;
- the input is raw 14D ARX X5 state and the server applies the checkpoint's q99 training transform exactly once;
- the server returns unnormalized 14D absolute-joint actions;
- the predicted horizon is 50 and RoboDojo replans after executing 16 actions.

For the pinned RoboDojo PI-v3 release, the dedicated entry point also requires the server handshake to advertise the canonical interleaved forward from upstream StarVLA fix [`c521dec`](https://github.com/starVLA/starVLA/commit/c521decb7441c7dfea282c61dc758456bcffbb8f). The requirement comes from the release manifest rather than the framework name, because older PI-v3 checkpoints may intentionally require the legacy forward. An incompatible runtime stops before opening the simulator instead of loading the weights successfully and emitting invalid actions.

The RGB setting is based on inspection of the actual three-camera LeRobot v2.1 MP4s used by these runs, decoded through the training backend: the stored frames have natural colors. No RGB/BGR swap is needed for any of the three released checkpoints. The first state-bearing request is logged with both raw and normalized values; a missing or incompatible normalization contract now fails before evaluation instead of producing abnormal arm motion.

The dedicated entry point also launches RoboDojo with renderer multi-GPU disabled and an explicit physical simulator device. It defaults to one Isaac environment and runs the requested episodes sequentially as the safest portable 10-layout contract check. The archived author-side PI-v3 reference run used `eval_batch: true` with five parallel environments; reproduce that batching with `STARVLA_ROBODOJO_NUM_ENVS=5` while keeping the requested episode count unchanged. `EVAL_NUM=10` still means ten layouts, processed as two batches of five. The checked-in `deploy.yml` enables the batch-capable adapter, so both one-env and five-env commands use the same model contract. A small launcher shim disables argparse option abbreviation so current RoboDojo does not mistake AppLauncher's `--device cuda:N` for its integer `--device_id`. Set `STARVLA_ROBODOJO_KIT_ARGS` only if the local Isaac setup needs additional Kit arguments. If Isaac Sim prompts for its EULA, accept it according to your organization's process and then set `OMNI_KIT_ACCEPT_EULA=YES` for non-interactive runs.

Before loading the checkpoint, the entry point verifies that RoboDojo's official robot, object, material, and evaluation-layout assets are present; a fresh checkout should install them with `bash scripts/init_assets.sh`. A path-based RoboDojo environment is also checked for the host OpenGL/GLU libraries required by Isaac Sim. Install compatible system runtime packages if that check fails; do not use arbitrary copied shared libraries. A run is valid only after the log reaches `Simulation App Startup Complete` and begins task episodes. `VK_ERROR_DEVICE_LOST`, exit 139, or an NVIDIA Xid before that point is a simulator/GPU failure rather than a zero-score episode; move the run to a healthy GPU (or have the cluster administrator recover the device) and rerun it.

The policy environment argument may be a Conda environment name, a venv directory, a uv project containing `.venv`, or a Python executable. The RoboDojo environment argument may be a Conda environment name or its absolute environment-prefix path; use the prefix path when `conda` is not initialized in non-interactive shells. The pinned release manifest is [`scripts/hf_robodojo_checkpoints.json`](scripts/hf_robodojo_checkpoints.json). Set `STARVLA_HF_ROOT` to place the downloads on a shared filesystem. `STARVLA_HF_LOCAL_FILES_ONLY=1` disables network access. For a pre-materialized directory, set `STARVLA_HF_VERIFY_ONLY=1`. All three releases reference `Qwen/Qwen3-VL-4B-Instruct`; offline installations can set `STARVLA_BASE_VLM=/local/Qwen3-VL-4B-Instruct` without editing the pinned YAML. `STARVLA_HF_SKIP_WEIGHT_HASH=1` skips the full weight hash after checking file size, but should only be used after a successful verified download. The dedicated entry point is single-process and clears inherited torchrun/MPI rank variables before starting either side; otherwise a cluster login shell carrying, for example, `WORLD_SIZE=6` can make PI-v3 wait for five nonexistent evaluation ranks. Policy startup also defaults `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `STARVLA_CPU_THREADS=8`; this avoids oversubscribing high-core-count cluster nodes while still respecting either BLAS variable when it is already set.

## Configuration

`deploy.yml` keys to check before evaluation: `starvla_root`, `checkpoint_path`, `starvla_server_host`, `starvla_server_port`, `unnorm_key`, `use_ddim`, `num_ddim_steps`, `image_size`, `execute_horizon` (number of actions executed before requesting a new predicted chunk), `include_state` (`auto` reads the checkpoint config; an explicit boolean overrides it), `require_runtime_contract` (rejects a server that cannot prove it applies the released checkpoint's RGB/state/action contract), and `required_pi_v3_forward` (set by the dedicated released-checkpoint entry point when the manifest requires a particular PI-v3 forward).

`include_state: auto` reads `datasets.vla_data.include_state` from `config.yaml`, then falls back to `config.full.yaml` and finally `false`. `STARVLA_INCLUDE_STATE` remains the highest-priority explicit override.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `STARVLA_EXECUTE_HORIZON` | Overrides the number of actions executed from each predicted chunk. |
| `STARVLA_IMAGE_SIZE` | Overrides the input image size passed to the adapter. |
| `STARVLA_INCLUDE_STATE` | Explicitly overrides checkpoint-driven proprioceptive state selection. |
| `STARVLA_UNNORM_KEY` | Selects the checkpoint normalization statistics. |
| `STARVLA_CPU_THREADS` | Fallback CPU thread count for policy-process imports when OMP/MKL limits are unset (default: 8). |

The scripts also read `STARVLA_CKPT_PATH`, `STARVLA_ROOT`, and `STARVLA_SERVER_PID`, plus the training overrides listed above.
