# RDT_1B

**Contributor:** RoboDojo Team | **Paper:** RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation | **arXiv:** TBD | **Original code:** https://github.com/thu-ml/RoboticsDiffusionTransformer

`RDT_1B` adapts the RDT-1B diffusion foundation model for bimanual manipulation to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `rdt/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use: RDT_1B has several weight-management modes and external Hugging Face assets. `install.sh` installs dependencies and prepares the pretrained weights under `weights/RDT/`; set `RDT_WEIGHTS_SRC=<dir>` to symlink an existing weights root instead of downloading, or `RDT_SKIP_WEIGHTS=1` to skip weight preparation.

```bash
cd XPolicyLab/policy/RDT_1B
bash install.sh
conda activate <policy_env>  # e.g. rdt_1b (override the name with RDT_CONDA_ENV=<name>)
```

## Data Processing

Links HDF5 data into `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/` and pre-encodes T5 language embeddings into `lang_embeds/` for the same 4-tuple. When `source_path` is omitted, the source is resolved from `RAW_DATA_ROOT`, then `data/<bench_name>/<ckpt_name>`, then `data/<bench_name>_<ckpt_name>`. Optional flags after `source_path`: `--overwrite` (re-encode all `lang_embed.pt` files), `--skip-encode` (only create the data symlink), `--gpu N` (GPU for T5 encoding, default 0).

```bash
cd XPolicyLab/policy/RDT_1B
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [source_path] [--overwrite] [--skip-encode] [--gpu N]

# Example: use default source-path resolution
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: link only the first 50 episodes from a custom HDF5 source, encoding on GPU 1
bash process_data.sh RoboDojo stack_bowls arx_x5 joint 50 /path/to/hdf5 --gpu 1
```

## Training

```bash
cd XPolicyLab/policy/RDT_1B
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. Training expects the pretrained assets prepared by `install.sh` in `weights/RDT/` (`t5-v1_1-xxl`, `siglip-so400m-patch14-384`, `rdt-1b`), overridable through `TEXT_ENCODER_NAME`, `VISION_ENCODER_NAME`, and `RDT_PRETRAINED_MODEL`. The process count is inferred from a comma-separated `gpu_id`.

## Evaluation

```bash
cd XPolicyLab/policy/RDT_1B
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `bench_name`, `task_name`,
`ckpt_name`, `env_cfg_type`, `action_type`, `seed`, `gpu_id`,
`checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `prompt`,
`checkpoint_path`, `model_path`, `config_path`, `text_encoder_path`,
`vision_encoder_path`, `model_root`, `ctrl_freq`, `chunk_size`, `image_size`,
and `camera_mode`.

## EgoVLA benchmark (H1 + Inspire)

The model-root helper `data_scripts/prepare_egovla_rdt.sh` stages the official
canonical EgoVLA conversion under `data/EgoVLA_rdt38/` and links it into the
adapter. It includes all 1,903 active episodes from the 12 tasks and excludes
the 100 paths marked `Deprecated` (and the separate `close_drawer_smoke`
artifact). HDF5 files are hard links/relative links, so the large RGB payload is
not copied. The canonical logical order is
`left_arm7,left_hand12,right_arm7,right_hand12`; the adapter scatters these 38
values into the fixed 128-token RDT state layout using
`rdt/configs/egovla_joint38.py`.

```bash
cd /personal/xiangpc/0812_Xpolicylab_bench/RDT-1B
bash data_scripts/prepare_egovla_rdt.sh --skip-file-validation

# WANDB_API_KEY must be exported only in the launching shell.
bash data_scripts/launch_rdt_egovla_8gpu.sh
```

The launcher uses `egovla_h1_hdf5`, 30 Hz, `RDT_DROP_SHORT_EPISODES=0`,
per-GPU train/sample batch 8 (global batch 64 on 8 GPUs), 80,000 optimizer
steps, and checkpoints every 10,000 steps. A fresh run is written to
`chpt/20260902_egovla_joint38_bs64_s42_80k` and exposed through the standard
checkpoint symlink. For deployment/evaluation, keep `ctrl_freq=30` and supply
the task's canonical HDF5 instruction. SparkArena / Tianji eval uses 25 Hz,
`image_size=[640, 480]` (letterbox, not stretch), and the same HDF5 sentences
or precomputed `lang_embeds_0908_7task` vectors. The fallback is the canonical
sentence, not the raw task slug.

### EgoVLA evaluation contract

The released EgoVLA HDF5 contains one main RGB stream. An older converter
hard-linked that image into `cam_left_wrist` and `cam_right_wrist`
(`provenance/missing_wrist_policy=duplicate-head`). EgoVLA is a single-view
benchmark, so that duplication is not used. The HDF5 loader for
`RDT_DATASET_NAME=egovla*` skips those wrist datasets and fills both wrist
slots with black images of the same shape as `cam_head`. Evaluation of
`ego_h1_inspire` defaults to `RDT_CAMERA_MODE=black_wrist` and does the same
fill at the model ABI boundary. `main_replicated` remains an explicit opt-in
for the checkpoint trained on the duplicated wrist views. `real_wrist` is only
for a checkpoint trained with independent wrist views. The model receives
decoded RGB arrays from the XPolicyLab server and does not decode or reorder
channels. The EgoVLA deployment default is `image_size: [384, 384]`, matching
the staged training data.

Use an absolute, completed checkpoint directory (for example
`checkpoint-10000`) rather than the moving run-directory symlink. The published
benchmark capability manifest intentionally keeps RDT disabled while this local
38D adapter is outside the pinned benchmark checkout. For a reviewed local
run, opt in explicitly with `EGOVLA_RDT_EVAL_FORCE=1`; leaving it unset keeps the
deny-by-default guard.

```bash
MODEL_ROOT=/personal/xiangpc/0812_Xpolicylab_bench/RDT-1B
BENCH_ROOT="/personal/xiangpc/EgoVLA benchmark"
POLICY_ENV=/personal/miniconda3/envs/rdt_1b
EVAL_ENV="${BENCH_ROOT}/.runtime/conda/egovla-isaaclab-1.2.0"
CKPT="${MODEL_ROOT}/chpt/20260902_egovla_joint38_bs64_s42_80k/checkpoint-10000"

export EVAL_MAIN_ROOT="${BENCH_ROOT}"
export EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}"
export RDT_CAMERA_MODE=black_wrist
export RDT_CTRL_FREQ=30
export RDT_EVAL_BATCH=false
export RDT_EVAL_REQUIRE_COMPLETE=1
export EGOVLA_RDT_EVAL_FORCE=1       # local, explicitly unaudited opt-in
export EVAL_ENV_TYPE=sim

cd "${MODEL_ROOT}/Xpolicylab/policy/RDT_1B"
bash eval.sh EgoVLA_benchmark Humanoid-Close-Drawer-v0 "${CKPT}" \
  ego_h1_inspire joint 42 0 0 "${POLICY_ENV}" "${EVAL_ENV}"
```

For a no-simulator wiring check (does not load the checkpoint or consume a
GPU), run the same command with `EVAL_ENV_TYPE=debug` and
`RDT_EVAL_DRY_RUN=1 RDT_EVAL_REQUIRE_COMPLETE=0`, using an existing placeholder
path such as `/tmp` for `CKPT`. This validates the 38D map, environment/Python resolution, camera mode,
prompt path, and benchmark-root selection. A real debug/simulation rollout is
only considered valid after a complete checkpoint has been written and the
policy GPU is idle. The wrapper records the resolved camera mode and frequency
in its startup log; it does not modify the benchmark capability manifest.
