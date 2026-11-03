# H_RDT

> SparkArena source inspection: clean official code is in `upstream/`; exact historical configuration is in `reference_configs/sparkarena_0921/`. The generic launcher below is not the audited SparkArena sampler-fix recipe. Reference scripts retain historical absolute paths. No new training has been launched.

**Contributor:** RoboDojo Team | **Paper:** H-RDT: Human-to-Robotics Diffusion Transformer | **arXiv:** https://arxiv.org/abs/2507.23523 | **Original code:** https://github.com/HongzheBi/H_RDT

`H_RDT` adapts H-RDT (Hybrid Robot Diffusion Transformer) to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `H_RDT/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## DexBench EE58 training and serving

The new `DexBench / tianji_marvin_wuji / ee` training route uses the unified
`/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21` export. This is
**3000 episodes, 751348 frames, 12 physical tasks and 60 exact instruction
variants**. SparkArena supplies the training hyperparameters, not its old
7-task/joint54 data or its cached language embeddings.

Both state and action are compact 58-dimensional vectors:
`[left xyz3, left row6D6, left Wuji20, right xyz3, right row6D6, right Wuji20]`.
The row6D convention is `matrix_rows_0_then_1`. State is `observation.state[t]`;
targets are `action[t:t+16]` with repeated-last-action tail padding and no extra
shift. Independent state/action q01/q99 statistics clip to [-1,1]. All 58
coordinates enter the native FM loss; there is no 128-dimensional padding or
padding mask. Video reads add each camera's own `meta/video_offsets.json` value
before seeking an exact decoded frame and return RGB without channel swapping.

The official `upstream/` runner is used with `state_dim=action_dim=output_size=58`.
Its native finetune loader discards the entire state/action input MLPs and output
decoder, initializes new 58-dimensional interfaces, and retains the human48 DiT
backbone and compatible image/language adapters. Human48 uses column6D and
fingertip positions; our row6D/Wuji joint interfaces are new. This is a valid
shape-compatible transfer path, not a guarantee of equal transfer quality.

The independent training code lives under `training/`. The public `Model`
dispatches `action_type=ee` to `ee58_runtime.py` and keeps the legacy joint
implementation compatible. The original `H_RDT/` source and old weights are
preserved. Train and serve share the clean `upstream/` native runner and the same
native DINO/SigLIP RGB384 letterbox transform factory.

The recipe is 8 GPUs x micro16 x GA1 = effective batch128, 60000 optimizer updates,
H16, bf16/ZeRO-1, AdamW 1e-4, constant LR after 500 warmup updates.
The adapter disables Accelerate's per-rank scheduler multiplication so the
warmup spans 500 optimizer updates on 8 ranks, rather than advancing eight times
per update. This fixes the schedule clock; it does not claim the old historical
run already used that corrected clock. Local human 500k and DINO/SigLIP weights
are reused; no downloads occur. The existing Python
default is `/personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python`,
which already contains AV/Arrow plus native H-RDT dependencies. Override
`HRDT_PYTHON` to another complete environment if needed; the old `hrdt` env is
missing AV and Arrow, so it cannot consume this prepared video/parquet export.
Native corruption augmentation and its eagerly imported legacy Robotwin/EgoDex
backends are guarded when optional `imgaug` is absent. The fixed EE58 Consumer
does not instantiate those backends and rejects image augmentation; attempting
an unsupported path raises an error. No image corruption is silently simulated.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/H_RDT

# CPU data and pretrained-header checks; never launches training.
bash train.sh DexBench tianji_wuji tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7 --dry-run
```

Preflight checks two episodes per physical task (24 episodes), normalized
state/action shapes and finiteness, final-action padding, three real decoded RGB
frames and each camera's timestamp offset. A meta-device native model and a
meta-mapped human checkpoint verify all retained backbone tensor shapes without
materializing the 4GB checkpoint or allocating GPU memory. It saves a unique
`training/generated/<run>_<suffix>/preflight.json`, `pretrained_header.log` and
generated `config.yaml`. These checks do not establish GPU forward/backward,
throughput or model accuracy.

The complete 60-instruction T5 cache is a **separate prerequisite**. A genuine
cache was generated on CPU with the existing `hrdt` interpreter and all 60 text
and embedding records passed validation. It is not
silently substituted with the old SparkArena cache or fake zero embeddings.
The dry-run reports pending language setup when absent; real launch fails before
GPU/model initialization until all text hashes, T5 configuration, source weight
provenance, saved embedding hashes, 4096-dimensional features and sequence lengths
pass. Generating it loads large local T5-XXL weights, so preparation is explicit:

```bash
# Run separately on an explicitly chosen device with enough available memory.
# The existing cache was generated on CPU. To rebuild, use a NEW output directory.
PYTHON=/personal/miniconda3/envs/hrdt/bin/python
"$PYTHON" training/prepare_language.py \
  --encoder /personal/xiangpc/0812_Xpolicylab_bench/H_RDT/pretrain_model/t5-v1_1-xxl \
  --output /personal/wenwei/DexBench/XPolicyLab/policy/H_RDT/data/dexbench_ee58_t5_60 \
  --device cpu

# Only after the cache passes and the 8 GPUs are allocated:
bash train.sh DexBench tianji_wuji tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

The shared checkpoint helper chooses
`checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/`. New checkpoint names
containing batch-size labels such as `bs128` or `batch_size128` are rejected.
Existing runs are never
overwritten; choose a new checkpoint name. This route starts new finetuning from
the human backbone and does not claim full optimizer resume from joint54.
Overrides: `HRDT_EE58_DATASET`, `HRDT_PRETRAIN`, `HRDT_VISION_ROOT`,
`HRDT_LANG_CACHE`, `HRDT_PYTHON`, `EVAL_MAIN_ROOT`, `HRDT_MASTER_PORT`.
The native periodic sampling callback is disabled using a positive period beyond
the 60k budget; the native code also supports `-1` with its `sample_period>0`
guard. Its validation dataset is not a held-out evaluation split.

### Immutable EE58 checkpoints and inference

Before training starts, the run stores `launch_config.yaml`, `ee58_runtime.json`,
independent `ee58_stats.json`, exact `ee58_tasks.json`, and a private copy of all
60 language vectors plus their manifest. Training metadata is frozen into
`ee58_dataset/meta`; large video/parquet directories are linked, not copied.
Inference reads these checkpoint snapshots, never current training-data stats.
The sidecar binds config, statistics, source metadata and text/vector SHA256
values. Every completed native weight save gets an `ee58_weight_binding.json`
containing the full weight digest and its runtime-contract digest; serving skips
unbound/incomplete saves and refuses old/default 14D configuration fallbacks.

The environment wire format remains **54** values (two xyz+wxyz+Wuji20 groups).
`Model.update_obs` converts it to row6D **58**, normalizes with the saved state
statistics, and orders cameras head/left/right. `get_action` predicts H16x58,
uses saved action statistics, projects row6D onto SO(3), converts to wxyz and
returns H16 wire54 action dictionaries. Constant-stat dimensions restore their
saved constant. There are no dummy embeddings and no inferred body-prefix maps.
Use an exact full training instruction; for an environment sending the plain
sentence, set its exact bound training key explicitly. A mismatching sentence or
unknown key raises an error.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/H_RDT
export HRDT_INSTRUCTION_KEY='[tianji_wujihand2_real/collect_objects] Put all the objects into the basket.'
bash setup_eval_policy_server.sh DexBench collect_objects tianji_wuji \
  tianji_marvin_wuji ee 42 0 \
  /personal/xiangpc/0914_Xpolicylab_bench/starvla/.venv-starvla/bin/python \
  6000 localhost
```

This command requires a genuinely trained checkpoint. A real-weight GPU forward
has not been run in this preparation, and micro16 GPU memory use remains
unverified. The CPU oracle runs the actual public `Model.update_obs/get_action`
and batch methods, checks state statistics and row6D/wxyz conversion, and compares
camera preprocessing exactly with training. Its vision/policy outputs and
language vectors are explicit temporary test fixtures, not model-quality
evidence; the production constructor rejects such fixtures. A second CPU check
used the genuine 60-text T5 cache: native training Consumer state/action/language
tensors matched the public serving path, including exact camera preprocessing.
Vision and policy network outputs remained stubs in that check.

## Installation (legacy RoboDojo route)

```bash
cd XPolicyLab/policy/H_RDT
bash install.sh
conda activate <policy_env>  # e.g. h-rdt
```

## Data Processing

No top-level `process_data.sh` and no converted dataset: training reads RoboDojo HDF5 trajectories directly from `HRDT_SOURCE_ROOT`. Before training, generate task instructions, action normalization stats, and task language embeddings:

```bash
cd XPolicyLab/policy/H_RDT/H_RDT

# Point this at the RoboDojo sim_cloud root that contains <bench>/<task>/<env_cfg>/data.
export HRDT_SOURCE_ROOT=<path_to_robodojo_sim_cloud>

python datasets/xpolicylab/extract_task_instructions.py \
  "${HRDT_SOURCE_ROOT}" \
  --env_cfg_type arx_x5

python datasets/xpolicylab/calc_stat.py \
  --data_root "${HRDT_SOURCE_ROOT}" \
  --raw_bench_name RoboDojo \
  --env_cfg_type arx_x5 \
  --action_type joint \
  --tasks all \
  --output_path datasets/xpolicylab/stats.json

# Requires the policy environment and T5 weights used by H-RDT.
export T5_MODEL_PATH=<path_to_t5-v1_1-xxl>
export HRDT_CONFIG_PATH="$(pwd)/configs/hrdt_finetune.yaml"
python datasets/xpolicylab/encode_lang_batch.py
```

Expected outputs are `datasets/xpolicylab/task_instructions.csv`, `datasets/xpolicylab/stats.json`, and `datasets/xpolicylab/lang_embeddings/*.pt`.

## Training

```bash
cd XPolicyLab/policy/H_RDT
export HRDT_SOURCE_ROOT=<path_to_robodojo_sim_cloud>

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [pretrained_backbone_path]

# Example: train a cotrain run on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. The optional 7th argument `pretrained_backbone_path` overrides the pretrained H-RDT backbone and defaults to the vendored pretrain checkpoint `H_RDT/checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin`. By default training uses all episodes found for each task; to cap the number of episodes per task, set `XPOLICY_HRDT_MAX_EPISODES=<num>` before running `train.sh` — this replaces the legacy `expert_data_num` / `total_episode_num` positional argument.

## Evaluation

```bash
cd XPolicyLab/policy/H_RDT
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_path`, `config_path`, `lang_embedding_path`, `lang_embedding_dir`, `stats_path`, `device`, `dtype`, `vision_backbone_id`, `vision_image_size`, `allow_dummy_lang_embedding`.

Policy-specific environment variables: `HRDT_SOURCE_ROOT` (required for training and metadata generation), `T5_MODEL_PATH` and `HRDT_CONFIG_PATH` (language-embedding step), `XPOLICY_HRDT_MAX_EPISODES` (optional per-task episode cap; empty or unset uses all episodes), plus the optional overrides `XPOLICY_HRDT_ACTION_TYPE` and `HRDT_ROOT`.
