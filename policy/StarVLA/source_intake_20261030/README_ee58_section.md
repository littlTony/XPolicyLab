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

