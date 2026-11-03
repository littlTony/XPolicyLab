# RLDX_1: DexBench absolute EE58

This adapter loads `/personal/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/RLDX-1-PT-1592013a` and reuses the native model architecture and its padded 64D state/action heads. **The DexBench absolute EE58 layout is a new embodiment's action semantics; it is not the pretrained DROID DELTA Cartesian-velocity action space.** The pretraining mixture includes different robot conventions: DROID uses raw Cartesian-velocity commands and an absolute gripper command, whereas G1/H1 use absolute arm/hand joints. Successful weight loading does not make their physical meanings interchangeable.

Both DexBench state and action contain, in order, left absolute world xyz + row-rot6d + absolute Wuji20, then the right side. This is 58 physical values. The exporter already aligns each action to its recorded target; no extra time shift, arm-joint substitution, retargeting, or relative-action transform is applied. Native normalization and clipping remain unchanged.

Training and live inference use RGB head, left-wrist and right-wrist images. These are named `front`, `left_wrist` and `right_wrist` in the native modality configuration. RLDX's interface consumes images and world-frame numeric state; it does not take camera intrinsics/extrinsics or convert poses into camera coordinates. The native video feature expands the training history to frames `[-6,-4,-2,0]`; live inference supplies those same four ages and repeats the first observation when history is short.

## Data preparation and statistics

`/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21` is a numeric view of the current `dexbench_12tasks` LeRobot v3 export. Episode and frame counts come from the current source metadata, rather than a fixed historical total. Videos remain zero-copy links; `meta/video_offsets.json` locates episode frames within the concatenated MP4s.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/RLDX_1
bash process_data.sh
```

Preparation always invokes `/personal/wenwei/DexBench/data_preprocess/prepare_native_ee58.py`: it builds a missing view, or validates an existing view against its source fingerprint. If the original numeric dataset changed, the helper refuses the stale view; rebuild the view in a new directory before proceeding. It calculates independent EE58 absolute state/action statistics after conversion to row-rot6d. LeRobot's original EE54 normalization values are not reused for those numeric fields.

The GR00T relative-statistics sidecar may coexist in this shared view. RLDX's four action groups are all `ABSOLUTE`, so the native processor uses their absolute stats; the GR00T relative stats do not override them. Native percentile normalization and inverse clipping retain their upstream behavior: values outside the chosen percentile interval cannot be expected to roundtrip without clipping.

## Training and deployment

```bash
bash train.sh DexBench <run> tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

The wrapper checks the source fingerprint before creating a run, then copies `source_fingerprint.json` into the run for provenance. Effective batch size is 256 and the limit is 60000 optimizer updates. `NATIVE_ACCUMULATION` defaults to 1; increasing it reduces the microbatch to preserve the effective batch. Other training settings follow the native custom-data `run_scripts/train/examples/finetune.sh` recipe. This preparation update does not start training.

Native training injects dataset statistics into the processor and saves that processor with each checkpoint. **Inference loads its selected checkpoint's processor and statistics**, rather than substituting the latest `data/meta/stats.json`. That is required to keep the trained model and its normalizer paired. The copied source fingerprint records the training dataset; it does not relabel older checkpoints or require old checkpoints to use today's dataset statistics.

Live inference calls the native processor once to unnormalize outputs, then converts physical row-rot6d back to wxyz. It returns exactly `left_ee_pose`, `left_ee_joint_state`, `right_ee_pose`, and `right_ee_joint_state`, in absolute world pose7/hand20 form for xspark-sim-eval.

## CPU verification

```bash
source runtime.sh
CUDA_VISIBLE_DEVICES= "$NATIVE_PYTHON" test_dexbench.py
```

The checks use current metadata and real exported RGB frames, cover row-rot6d, camera offsets, absolute numeric groups, single/batched native normalization against the official clipped expectation, the four simulator keys, active environment order, four-frame history and reset. They do not load GPU model weights or establish learned-policy success in the simulator.


## Completed Tianji checkpoint (60000 updates)

Native execution length: 16. ckpt_name: `tianji_wuji`. Checkpoint: `/mnt/xspark-data/wenwei/DexBench/XPolicyLab/policy/RLDX_1/checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/dq128_60k_rldx_20261027_a01_ga1/checkpoint-60000`.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/RLDX_1
bash setup_eval_policy_server.sh DexBench auto tianji_wuji tianji_marvin_wuji ee 42 5 /personal/xiangpc/0811_Xpolicylab_bench/RLDX_1/policy/RLDX_1/RLDX-1/.venv 6021 0.0.0.0
```

For task=auto, observations require the exported [tianji_wujihand2_real/task] instruction. For a standard single-task simulator client, replace auto with the concrete task name; the adapter adds its matching prefix.

Use a GPU with sufficient free memory. Optional `DEXBENCH_EVAL_CUDA_MEMORY_FRACTION` limits only this server allocator. Results: `/personal/wenwei/DexBench/evaluations/four_policy_60k_20261030/RLDX_1/recorded_replay/summary.json`. Recorded replay does not measure closed-loop success. The evaluation server has been stopped.
