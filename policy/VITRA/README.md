# DexBench VITRA — pretrained human-space contract

Copied from `/personal/xiangpc/0813_Xpolicylab_bench/VITRA/XPolicyLab/policy/VITRA`.
The original files changed for this integration are backed up under
`/personal/wenwei/DexBench/backups/vitra_before_dexbench_20261022`.
Supports `tianji_marvin_wuji / ee` using the original Wuji20 sparse mapping,
state212 and action192 architecture, and `VITRA-VLA-3B/vitra-vla-3b.pt` weights.

## Installation

`source runtime.sh` selects this copied source and the existing environment
`/personal/miniconda3/envs/vitra_xpolicylab_0813`. VITRA and local PaliGemma
processor assets are read from the original shared pretrain_model directory.
No pretrained model or original dataset is overwritten.

## Data Processing

The active loader reads **LeRobot v3.0** directly from
`/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks`.
It uses the current export's standard `observation.state`, `action` and
`observation.images.cam_high/cam_left_wrist/cam_right_wrist` keys. No numeric
episode HDF5 conversion or video re-encoding is needed. Other representations
and upstream HDF5 readers remain available for their existing configurations;
the EE54 export is explicitly guarded as `wuji20`, not fitted MANO45.

```bash
source runtime.sh
"$VITRA_PYTHON" -m pip install -r requirements_lerobot.txt
bash process_data.sh
# Explicitly rebuild camera metadata and native-space statistics:
VITRA_RECOMPUTE_STATS=1 bash process_data.sh
# Full CPU validation of current data, statistics and live command reconstruction:
CUDA_VISIBLE_DEVICES= "$VITRA_PYTHON" validate_dexbench.py --full
```

Override the source with `VITRA_LEROBOT_ROOT`. Preparation reads camera calibration
from the original HDF5 paths in `meta/preview_report.json` once and writes
`meta/vitra/calibration/episode_XXXXXX.npz`. All three cameras' original intrinsics
and camera-to-env graphics-axis extrinsics are preserved for every valid row.
After preparation, training and the default validator use only LeRobot Parquet,
MP4 and these metadata files; they never open source or intermediate HDF5.
Keep `meta/vitra` with the dataset when relocating it. The loader fails on missing
calibration instead of guessing an identity transform.

`meta/vitra/normalization_stats.json` is independently recomputed over all
751348 current state rows (3000 episodes), and all11661568 valid target slots
sampled by16-step chunks. State moments count each row once; action moments count
every valid anchor/horizon slot, excluding masked terminal padding. It stores
separate native camera-state52 and command-step-action52 Gaussian means/stds.
**LeRobot's generic world-EE54 `meta/stats.json`
is not the VITRA normalization file.** The mapping identity, calibration contents,
numeric Parquet files, source metadata and stats are checked with SHA-256.
Regenerate metadata/stats when changing data or mapping.

The pretrained tensor layout and camera/wrist step representation are retained:

- State: absolute wrist xyz + Euler xyz in the anchor head-camera frame, plus
  absolute Wuji20 joints, normalized in native52 before sparse mapping to state212.
- Action: command trajectory steps in one fixed current-camera anchor. The first
  target is relative to the current measured wrist; each later target is relative
  to the previous command wrist. Translation is `p_target-p_previous`; rotation
  is `Euler_xyz(R_target @ R_previous.T)`. Hand targets remain absolute. Native52
  normalization precedes sparse mapping to action192.
- `action[t]` is already a recorded command target. No extra +1 shift is applied.
  It need not equal the next measured EE pose (A3 has an approximately0.5mm
  difference); using future measured states would accumulate that difference.
  End-of-episode padding is zeroed and masked; the final exported target is valid.
- The original signed `mapping_wuji20_mano45.json` is reused, without another hand
  retargeting pass. It is a sparse codec; canonical MANO wrist-axis equivalence is
  an existing calibration assumption, not a new geometric MANO fit.
- Native code applies the graphics-to-VITRA camera axis transform and EE-to-wrist
  calibration once. Inference inverses mapping/normalization once, integrates step
  deltas, and returns absolute xyz+wxyz and absolute hand joints.
- The guarded codec is `dexbench-vitra-wuji20-sparse-camera-command-step-v2`.
  Earlier DexBench measured-state-difference checkpoints are rejected. The command
  chain equals the original human step convention when consecutive measured
  states equal the preceding targets; tensor heads and coordinate conventions
  remain unchanged.
- Head, left-wrist and right-wrist RGB images use the same current frame and the
  native PaliGemma three-image processor. Temporal image windows are rejected.

## Training

```bash
VITRA_PER_DEVICE_BATCH=32 WANDB_MODE=offline bash train.sh DexBench vitra_wuji20_bs256_ga1_60k tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

The configured global batch is256, max_steps60000 optimizer updates, save5000,
bf16 and original FSDP+gradient-checkpointing recipe. The native strategy derives
gradient accumulation as `256 / (GPU count * per-device batch)`; lower the
per-device batch to16/8 when memory requires it. Use a new run name for a fresh
attempt. The wrapper rejects an existing output directory. Offline W&B does
not require an API credential.

Weights, optimizer and metadata are saved inside
`checkpoints/<bench>-<run>-tianji_marvin_wuji-ee-<seed>/<date>-<bench>-<run>_TB256_B<micro>_bf16True/checkpoints/epoch=E-step=S.ckpt/`.
The saved config has the original212/192 heads. Each new run stores an immutable
`normalization_stats.json` and `data_manifest.json` snapshot, bound by SHA-256, so
inference retains the exact training normalization even if source data is moved.

## Evaluation

The native camera/wrist action codec remains unchanged. Training and inference
provide three separate RGB images through native PaliGemma multi-image support
with three image placeholders; architecture and pretrained tensor shapes remain
unchanged. The original single-image recipe is extended for the requested views. Use the
copied adapter's original `eval.sh` with the new saved checkpoint directory,
`env_cfg_type=tianji_marvin_wuji` and `action_type=ee`. Supply all three RGB images and head-camera
intrinsics/extrinsics; the original runtime rejects missing calibration.
Actions returned to the simulator are `left_ee_pose(7)`,
`left_ee_joint_state(20)`, `right_ee_pose(7)`, `right_ee_joint_state(20)`.
The original README is available in the backup directory above.

## Verification

```bash
source runtime.sh
python validate_dexbench.py
```

Verified exact source EE pose identity across all rows during preparation,
3000-episode/751348-row coverage, native train/runtime state equality, wrist
delta inversion back to recorded absolute targets, Wuji20 target identity,
state212/action192 masks, terminal target preservation and static compilation.
The sparse encode/decode was separately verified to roundtrip exactly.
A trained-weight simulator rollout has not been run.

The `meta/vitra` directory must travel with the LeRobot export: it owns copied
per-frame three-camera intrinsic/extrinsic metadata, source hashes, and VITRA
native52 normalization statistics. Raw camera matrices are copied from source
HDF5 without changing their values; the shared codec applies the documented
camera-axis conversion exactly once at both training and runtime. The
EE-to-wrist fixed mount transform is separate from camera calibration. The
Wuji20 signed sparse map and canonical MANO wrist-axis equivalence remain
geometric assumptions; storage/codec parity does not prove complete human-hand
geometric or pretraining-distribution equivalence.


## Completed Tianji checkpoint (60000 updates)

Native execution length: 16. ckpt_name: `tianji_wuji`. Checkpoint: `/mnt/xspark-data/wenwei/DexBench/XPolicyLab/policy/VITRA/checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/2026-10-28-DexBench-dq128_60k_vitra_20261027_a01_ga1_TB128_B16_bf16True/checkpoints/epoch=10-step=60000.ckpt/weights.pt`.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/VITRA
bash setup_eval_policy_server.sh DexBench auto tianji_wuji tianji_marvin_wuji ee 42 4 /personal/miniconda3/envs/vitra_xpolicylab_0813 6020 0.0.0.0
```

Send the plain simulator instruction plus runtime head-camera intrinsics and camera-to-world extrinsics.

Use a GPU with sufficient free memory. Optional `DEXBENCH_EVAL_CUDA_MEMORY_FRACTION` limits only this server allocator. Results: `/personal/wenwei/DexBench/evaluations/four_policy_60k_20261030/VITRA/recorded_replay/summary.json`. Recorded replay does not measure closed-loop success. The evaluation server has been stopped.
