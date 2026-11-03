# DexBench OpenWAM-alpha / SparkArena recipe

Copied from:
`/personal/xiangpc/0912_Xpolicylab_bench/openwam/XPolicylab/policy/OpenWAM`.

Active copy:
`/personal/wenwei/DexBench/XPolicyLab/policy/OpenWAM`.
The previous target is archived under
`/personal/wenwei/DexBench/backups/openwam_before_alpha80_20261021/OpenWAM`.
Training weights, environments, checkpoints and output runs were not copied.

## Data Processing

This copy reads the existing **LeRobot v3** export directly:
`/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks`.
`data/dexbench_12tasks` is a convenience symlink. No HDF5 conversion, image
re-encoding, hand retargeting or dataset modifications are performed here.
Keys match `data_preprocess/export_lerobot.py`: `observation.state`, `action`,
`observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`.

The active Hydra config is `dataloader=dexbench`, with registry type
`dexbench` pointing directly to `DexBenchLeRobotDataset`. It uses OpenWAM's
existing `LeRobotV3Reader` for file/video indexing. `spark0` was the historical
SparkArena loader identifier; its legacy class remains in the copied source.
The dataset directory name `dexbench_12tasks` is separate from the loader name.
The current export has 3,000 episodes and 751,348 row-aligned training windows.

Source vectors are `[left xyz+wxyz(7), left Wuji20, right xyz+wxyz(7), right Wuji20]`.
Both observation and target are absolute world/environment EE poses and absolute
hand joints. Conversion yields raw58:
`[left xyz(3), left rot6d(6), left hand(20), right xyz(3), right rot6d(6), right hand(20)]`.
Rot6D concatenates matrix columns 0 and 1, as in the original SparkArena loader.
Hand order remains Isaac stage-major: index/middle/pinky/ring/thumb at each stage.

Official alpha80 destination slots (zero-based, inclusive ranges):

```text
left  xyz      0–2       left  rot6d   3–8       left  hand20  10–29
right xyz     34–36      right rot6d  37–42      right hand20  44–63
```

The other 22 dimensions are zero and False in action/proprio masks. End padding
also has False time masks. The prepared export already pairs state[t] with its
following-pose target action[t]; the loader reads `action[t:t+32]` directly,
without another +1 shift or crossing episode boundaries. Video samples are
state frames t, t+4, ..., t+32, with last-frame padding and validity masks.

The old SparkArena branch uses base-origin poses and legacy xyzw byte semantics.
Those branches are not used for this dataset. No base subtraction/addition or
quaternion component reinterpretation belongs in this copy's live adapter.

Images remain RGB. Train and inference share the exact multiview composition:
head on top, left/right wrists below, 384x320 canvas. Every camera is required;
a decoding failure is surfaced rather than replaced with another episode or
a black wrist. Color jitter is disabled, matching the official loader default.

Prompts use the exporter's `[{embodiment}/{task}] {instruction}`, wrapped with
OpenWAM's original robot-view video template in both training and inference.
Live defaults to `prompt_embodiment: tianji_wujihand2_real`.

Prepare state/action statistics from the full numeric parquet population:

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/OpenWAM
bash process_data.sh
```

Output: `data/meta/normalization_stats.npy`. Separate raw58 `eef` action and
`eef_state` proprio statistics support the existing directional deploy
normalizer. Rot6D min/max and quantile limits stay [-1,1]. Normalize raw58 first,
scatter to80 second; deployment gathers to58, then unnormalizes once. The stats
file embeds `action_contract`; the trainer copies the whole file to its output.
Inference and strict resume check that contract to reject old base/xyzw runs.
The stats also bind the exact source numeric parquet, episode metadata, task
metadata and info.json with a SHA256 fingerprint (the same definition used by
prepare_native_ee58.py). The training wrapper and actual reader reject stale
stats even when row counts stay unchanged. Deployment reads only the saved
checkpoint-local stats snapshot; it never substitutes live dataset statistics.
The min-max mode and empty binary-action dimensions are enforced in both
training and checkpoint loading to prevent normalization or hand projection drift.

## Pretrained weights: 15w4 = step 154000

Both the wrapper and native config pin the explicit file:

```text
/personal/xiangpc/0912_Xpolicylab_bench/openwam/pretrain_model/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors
```

The foundation config uses the original dual-system joint-self-attention model,
80D action and 80D proprio. Its config/tokenizer stay in the same directory.
The file header/length and action dimension are checked without loading its
24.8 GB of tensors onto a GPU. This is finetuning from154k with a new optimizer
and step counter, not resuming the pretraining run's optimizer.

## Installation / Runtime

The existing dependency interpreter is `/personal/miniconda3/envs/openwam/bin/python`.
`source runtime.sh` prepends this copied OpenWAM and XPolicyLab source so the
external editable install cannot silently select another checkout. Override
`OPENWAM_PYTHON` for another compatible environment; use the bundled `install.sh`
for an independently installed conda environment.
`EVAL_MAIN_ROOT` defaults to `/personal/wenwei/xspark-0-simeval`.
Wan component assets use the existing shared `pretrain_model/Wan2.2-TI2V-5B` tree.

## Training

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/OpenWAM
bash train.sh DexBench openwam_ee80 tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

The official-recipe configuration uses per-GPU batch32, gradient
accumulation1 (effective batch256), lr1e-4, 60k steps, ZeRO2, save every2000, bf16 and gradient
checkpointing. Further Hydra overrides can follow the GPU list. Installation,
statistics preparation and tests do not start training. The wrapper refuses an
existing output run directory; strict resume requires a checkpoint with saved
full states (the native `training.save_full_states_for_resume` option).
Checkpoints go to `checkpoints/DexBench-openwam_ee80-tianji_marvin_wuji-ee-42/`.

## Evaluation

```bash
bash eval.sh DexBench add_food_to_pot /absolute/path/to/finetuned/run \
  tianji_marvin_wuji ee 42 0 1 /personal/miniconda3/envs/openwam dexbench_egopro
```

Replace the final conda environment with the simulator's environment if needed.
`get_action()` returns `list[dict]`; each dictionary contains exactly
`left_ee_pose(7)`, `left_ee_joint_state(20)`, `right_ee_pose(7)`,
`right_ee_joint_state(20)`. Poses are absolute xyz+wxyz, suitable for xspark-sim-eval
IK. `get_action_batch(env_idx_list)` preserves the requested active-env order.
`replan_steps` optionally truncates generated chunks (null uses the checkpoint's
inference horizon). Generation uses the original engine with batch-safe sync
execution, video decoding/compile/DiT cache disabled. The engine returns physical
raw58 after checkpoint unnormalization; the adapter only converts rot6d to wxyz.

Extra deployment keys: `ckpt_dir` selects the checkpoint through the shared
resolver; `device` controls the engine; `prompt_embodiment` selects the exported
task prefix. `allow_dummy_policy` explicitly skips loading for transport tests
and echoes the current pose. It is false for real evaluation. The pretrained
foundation alone has no DexBench normalization/action marker and is not accepted
as a finetuned robot checkpoint.

## Verification

```bash
source runtime.sh
CUDA_VISIBLE_DEVICES= "$OPENWAM_PYTHON" check_contract.py
CUDA_VISIBLE_DEVICES= "$OPENWAM_PYTHON" -m pytest -q tests/test_dexbench_contract.py
```

Tests cover slot masks, wxyz/rot6d roundtrips, raw58/80D transforms, full reader
coverage and final-row masks, absence of an extra action shift, directional
normalization, real RGB/multiview/prompt parity across all five source embodiments, native
checkpoint snapshot and preprocessing, SHA stale-source rejection, active-env
output ordering and
configuration compatibility with the154k foundation. A trained-weight simulator
rollout requires finetuning this new data contract first.


## Completed Tianji checkpoint (60000 updates)

Native execution length: 32. ckpt_name: `tianji_wuji`. Checkpoint: `/mnt/xspark-data/wenwei/DexBench/XPolicyLab/policy/OpenWAM/checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/2026-10-27_14-34-38/checkpoint_step_60000.safetensors`.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/OpenWAM
bash setup_eval_policy_server.sh DexBench auto tianji_wuji tianji_marvin_wuji ee 42 6 /personal/miniconda3/envs/openwam 6022 0.0.0.0
```

For task=auto, observations require the exported [tianji_wujihand2_real/task] instruction. For a standard single-task simulator client, replace auto with the concrete task name; the adapter adds its matching prefix.

Use a GPU with sufficient free memory. Optional `DEXBENCH_EVAL_CUDA_MEMORY_FRACTION` limits only this server allocator. Results: `/personal/wenwei/DexBench/evaluations/four_policy_60k_20261030/OpenWAM/recorded_replay/summary.json`. Recorded replay does not measure closed-loop success. The evaluation server has been stopped.
