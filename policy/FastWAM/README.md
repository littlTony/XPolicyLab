# FastWAM — DexBench absolute EE58

**Original code:** copied from `/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/Xpolicylab/policy/FastWAM`.
This adapter supports `tianji_marvin_wuji / ee`, with absolute world EE poses and absolute Wuji20 hand joints. The original SparkArena recipe described joint54; those checkpoints cannot be used with this EE58 adapter.

Shared conventions: [XPolicyLab README](../../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use the existing `/personal/miniconda3/envs/fastwam` environment and `source runtime.sh` to select this copied source. SciPy 1.15.3 was added for quaternion/rotation conversions. `bash install.sh <new-env-name>` installs a separate environment if needed.
Model assets remain under `/personal/xiangpc/0812_Xpolicylab_bench/FastWAM/pretrain_model`, selected by `DIFFSYNTH_MODEL_BASE_PATH` in `runtime.sh`. The original Wan2.2 video weights and preprocessed ActionDiT backbone are used; this is not OpenWAM's step154000 checkpoint. ActionDiT's learned input/output projections and the state encoder are initialized for 58D; the backbone loader excludes input/output projection weights.

## Data Processing

Directly consumes the existing **LeRobot v3** prepared export `/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks`, produced by `data_preprocess/export_lerobot.py`. Keys match the official LeRobot naming: `observation.state`, `action`, and `observation.images.cam_high/cam_left_wrist/cam_right_wrist`. The data symlink is for convenience. No data rewrite or re-encoding occurs.

Both state and action use the identical contiguous 58D physical layout:

```text
0:3   left absolute world xyz
3:9   left rotation6d (matrix column0, then column1)
9:29  left absolute Wuji20 joint angles
29:32 right absolute world xyz
32:38 right rotation6d
38:58 right absolute Wuji20 joint angles
```

The source EE54 uses xyz+wxyz, converted by one shared `pack54` implementation for offline state, offline action, and live state. Hand order is Isaac stage-major: index/middle/pinky/ring/thumb at each joint stage. No delta transform, base offset, alpha80 scattering, or joint permutation is applied.

State is current measured pose/joints; action is the recorded absolute target. The exporter already aligns `action[t]` to the target following `state[t]`, so actions are read without an additional +1 shift. A sample contains action[t:t+32], state[t:t+32] and video at t,t+4,...,t+32. FastWAM's native `build_inputs` uses only `proprio[:,0,:]`; inference likewise uses the current observed state. End padding repeats the final row/frame and is masked out of loss.

Train and inference share the exact RGB image transform: each camera resized to 240x320, then head256x320 over wrists128x160, followed by the official RoboTwin recipe's resize/crop to **384x320**, then [-1,1] normalization. Missing cameras and decode failures raise errors. MP4s use PyAV.

Train/validation are split by episode with fixed seed42 and a 1% validation fraction. Current episode/frame counts come from source metadata. Training preflight verifies the exact train-episode IDs and SHA-256 of physical EE58 state/action values, so changing numeric data without changing counts also requires regenerating stats. Z-score state and action statistics use FastWAM's unmodified native `BaseLerobotDataset.get_dataset_stats` reducer. Each training episode has equal weight: state statistics use one current frame, action statistics include every 32-step forward chunk with final-row replication, and per-episode variances use PyTorch's sample variance (`correction=1`). Validation episodes are excluded. The complete native global/stepwise mean, std, min, max and quantile envelope are saved; the active z-score normalizer uses global mean/std. `stats_provenance` records the native reducer hash, exact train episode IDs and the EE58 numeric-data hash. LeRobot's automatic EE54 statistics are not used for normalization. FastWAM's original normalizer clamps normalized inputs to [-5,5]. The action contract is embedded in the stats and saved training config; deployment and resume reject old joint/delta contracts.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/FastWAM
bash process_data.sh
# Also generate/cache all 60 instruction embeddings (uses visible GPU):
CUDA_VISIBLE_DEVICES=0 bash process_data.sh --text-cache
```

Prompts retain `[embodiment/task] instruction` from the export, wrapped in FastWAM's original robot-view video prompt. Live default embodiment is `tianji_wujihand2_real`; override `prompt_embodiment` only for a matching exported prefix. Text caches use the original T5 tokenizer, context length128, and prompt hash. `train.sh` verifies every expected cache file before launch.

## Training

```bash
bash train.sh DexBench fastwam_ee58 tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
# Hydra overrides may follow GPU IDs:
bash train.sh DexBench fastwam_ee58_bs256 tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7 batch_size=32 gradient_accumulation_steps=1
```

Active Hydra task: `task=dexbench_ee58`. Defaults: per-GPU batch8, accumulation4 (effective batch256), 60000 optimizer updates, cosine LR1e-4, bf16, ZeRO1, checkpoint every2500, evaluation every500, mot_checkpoint_mixed_attn=false. The wrapper refuses an existing output directory and copies stats into the new run. The batch128/GA1 run completed 60000 optimizer updates on 2026-10-29; its final weights are under `DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/checkpoints/weights/step_060000.pt`.

Output: `checkpoints/<bench>-<run>-tianji_marvin_wuji-ee-<seed>/`, containing `config.yaml`, `dataset_stats.json`, `checkpoints/weights/step_XXXXXX.pt`, and optimizer states under `checkpoints/state/`.

## Evaluation

```bash
bash eval.sh DexBench add_food_to_pot /absolute/path/to/finetuned/run tianji_marvin_wuji ee 42 0 1 /personal/miniconda3/envs/fastwam base
```

Replace the final environment with the simulator environment as needed. Checkpoint resolution uses the shared XPolicyLab helper; a run directory selects its largest numeric step, or pass a weight file inside its saved run. Set `checkpoint_num` for an exact optimizer step; a missing or mismatched step raises an error. Inference reconstructs from saved training config and its matching statistics. The original engine normalizes state and unnormalizes predicted actions exactly once; the adapter only converts physical rot6d back to wxyz.

Each returned action has exactly `left_ee_pose(7)`, `left_ee_joint_state(20)`, `right_ee_pose(7)`, `right_ee_joint_state(20)`. Pose convention is absolute xyz+wxyz, compatible with xspark-sim-eval. Active batch environment order is preserved; native inference runs sequentially per active environment. Production inference requires real checkpoint weights. Echo engines exist only in CPU test fixtures. `task_name=auto` requires the exported `[embodiment/task] instruction`; a fixed task accepts an unprefixed instruction.

## Verification

```bash
source runtime.sh
CUDA_VISIBLE_DEVICES= "$FASTWAM_PYTHON" check_contract.py --require-text-cache
CUDA_VISIBLE_DEVICES= "$FASTWAM_PYTHON" test_stats_native.py
CUDA_VISIBLE_DEVICES= "$FASTWAM_PYTHON" test_dexbench_contract.py
EVAL_ENV_TYPE=debug bash eval.sh DexBench add_food_to_pot tianji_wuji tianji_marvin_wuji ee 42 0 0 /personal/miniconda3/envs/fastwam base
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh DexBench add_food_to_pot tianji_wuji tianji_marvin_wuji ee 42 0 0 /personal/miniconda3/envs/fastwam base
```

Contract tests read actual exported rows/video and cover rotation roundtrips, absolute hand order, no extra target shift, padding, train/validation separation, exact image/prompt/state normalization parity, saved-checkpoint contract rejection, active environment order and raw/encoded WebSocket action keys. The CPU WebSocket tests use an injected test fixture and do not measure trained-model quality. The trained checkpoint is also tested separately through the production server using two recorded Tianji training episodes per task; see `/personal/wenwei/DexBench/evaluations/fastwam_60k_20261029/recorded_replay/summary.json`. Recorded-observation agreement does not measure closed-loop task success.

## Completed Tianji checkpoint server

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/FastWAM
FASTWAM_CHECKPOINT_NUM=60000 bash setup_eval_policy_server.sh \
  DexBench auto tianji_wuji tianji_marvin_wuji ee 42 0 \
  /personal/miniconda3/envs/fastwam 6012 0.0.0.0
```

Run on a node with an unoccupied GPU; replace `0` with that GPU index. This resolves the saved run and stats, pins internal optimizer step60000, predicts32 actions and returns the first24. The raw client action format remains EE54 (wxyz poses and Wuji20 hands); normalized internal actions are EE58. `get_metadata` exposes the actual checkpoint, statistics SHA256, internal step, device and horizons after successful weight loading. No compatibility symlink is required.
For an explicitly authorized temporary shared-GPU run, `FASTWAM_CUDA_MEMORY_FRACTION=0.4` limits this server's PyTorch allocator to40% of the selected GPU. This does not reserve capacity or control other processes; check free memory before launching. Stop the server after the evaluation.

The maintained replay tools are `tests/prepare_replay_references.py` (native FastWAM CPU environment) and `tests/evaluate_recorded_episodes.py` (Pi_05's existing CPU data-reader environment, plus both policy runtime paths). They use the same24 training episodes as the previous Pi replay, compare all24 initial images/states/actions against the native training reader, exercise encoded RGB and raw RGB WebSocket inference, exclude padded tails, and save physical action errors and hold-state baselines.