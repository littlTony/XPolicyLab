# GR00T N1.7 — native relative EE targets, absolute Wuji hands

Source state/action storage is absolute world EE54 (`xyz+wxyz7+Wuji20` per
side). `prepare_native_ee58.py` builds an absolute numeric LeRobot2.1 EE58
view using native rotation6d **first two rows**, with zero-copy links to the
current LeRobot3 RGB videos. Counts are derived from source info (currently
3000 episodes / 751348 rows), not assumed from an earlier export.

State stays absolute world EE9 + absolute hand20 per side. For actions only,
`dexbench_config.py` registers EEF groups as RELATIVE, hand groups as ABSOLUTE.
The unmodified native processor uses

`T_relative[t+k] = inverse(T_state[t]) @ T_absolute_target[t+k]`, `k=0..39`.

Thus relative translation is `R_state.T @ (p_target-p_state)` and relative
orientation is `R_state.T @ R_target`. This is expressed in the current EE
local frame, held fixed across the whole40-target chunk. It is neither a
camera-frame command nor the difference between successive chunk targets.
Identity rotation6d is `[1,0,0,0,1,0]`, not a zero vector. Hand20 remains an
absolute joint target. Native inference `get_action` already unnormalizes and
applies `T_state @ T_relative`; `model.py` therefore only packs those resulting
absolute world targets into simulator pose7(wxyz)/hand20 keys.

`prepare_relative_stats.py` vectorizes the native `RelativeActionLoader`
semantics. Per episode of length N, it uses `max(0,N-39)` complete windows; every
window shares its current state anchor. Each stat is reduced over windows
(axis0), retaining shape `(40,9)` for each EEF group. Hands use ordinary
absolute statistics. Every episode is checked against the original native
RelativeActionLoader on a complete40-step window; each short episode is checked
to produce zero windows. This matches native `ShardedSingleStepDataset` with
the official default `allow_padding=False`. Current data have 20 episodes
shorter than40, 2980 eligible episodes and 634433 complete windows. Source
episodes and rows are preserved.

The actual pretrained `GR00T-N1.7-3B/config.json` has `action_horizon=40`,
`max_state_dim=max_action_dim=132`; its saved processor and every pretrained
embodiment also use40 action steps. Current supervision, native prediction,
processor decoding and default simulator execution all retain those40 steps.
`model.py` permits an explicitly shorter execution prefix, but `deploy.yml`
defaults to `replan_steps: 40`; the simulator replans after the returned chunk
or stops earlier if an episode ends. The original finetune recipe is unchanged.

Preparation and CPU verification:

```bash
bash process_data.sh
```

It builds/checks the numeric view, prepares current full-data relative stats
with at most8 CPU workers, and runs `test_dexbench.py`. No training is started.
Changing source numeric data, modality config or statistics invalidates the
preparation guard. `meta/action_contract.json` describes absolute *storage*;
`meta/groot_relative_provenance.json` describes the distinct GR00T relative
*training target*. Each future run stores the latter as its policy contract.

Future training remains explicit:

```bash
bash train.sh DexBench <new-run> tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

Effective batch256,60000 steps, native learning rate/optimizer/freeze/image
settings remain unchanged. `NATIVE_ACCUMULATION` starts at1. Training refuses
missing/stale relative statistics; inference refuses earlier absolute-target
checkpoints, a disabled relative processor, or saved processor statistics
that do not match the run contract. The `h40-v3` marker also rejects the earlier
H16 relative checkpoint/statistics; an H40 marker must never be attached to
old H16 or absolute weights.

Weights stay at the original shared GR00T-N1.7-3B pretrained location. This is
a new robot embodiment within the native configurable representation system;
it does not establish an identical robot morphology or pretraining distribution.


## Completed Tianji checkpoint (60000 updates)

Native execution length: 40. ckpt_name: `tianji_wuji`. Checkpoint: `/mnt/xspark-data/wenwei/DexBench/XPolicyLab/policy/GR00T_N17/checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/dq128_60k_groot_20261029_retrain4_a01_ga1/checkpoint-60000`.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/GR00T_N17
bash setup_eval_policy_server.sh DexBench auto tianji_wuji tianji_marvin_wuji ee 42 7 '/personal/xiangpc/0807_GR00T N1.7/XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv' 6023 0.0.0.0
```

For task=auto, observations require the exported [tianji_wujihand2_real/task] instruction. For a standard single-task simulator client, replace auto with the concrete task name; the adapter adds its matching prefix.

Use a GPU with sufficient free memory. Optional `DEXBENCH_EVAL_CUDA_MEMORY_FRACTION` limits only this server allocator. Results: `/personal/wenwei/DexBench/evaluations/four_policy_60k_20261030/GR00T_N17/recorded_replay/summary.json`. Recorded replay does not measure closed-loop success. The evaluation server has been stopped.
