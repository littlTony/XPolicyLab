# Pi0.5 EgoVLA evaluation notes

This private Pi0.5 adapter is wired to the converted EgoVLA benchmark dataset
and the `ego_h1_inspire` upper-body joint interface.  It does not change the
vendored OpenPI implementation or any other policy adapter.

## Model and data contract

- Action space is **joint**, not end-effector (`action_type=joint`).  The
  38-dimensional order is `left arm 7 + left Inspire hand 12 + right arm 7 +
  right Inspire hand 12`.
- The training config is `pi05_egovla_joint38_full`; its model has
  `action_dim=38` and `action_horizon=50`.
- The official EgoVLA data config uses delta joint actions: arm dimensions are
  delta targets while the 12 hand dimensions on each side remain absolute.
- The converted dataset contains 1,903 episodes and 510,546 frames.  The 100
  source episodes whose names contain `Deprecated` were intentionally excluded
  (`include_deprecated=false`); no Deprecated episode is in the training
  manifest.
- Episodes without wrist-camera recordings carry black wrist frames and a
  false camera-mask bit.  The adapter preserves this mask through eval.

The private dataset and run are:

```text
/personal/xiangpc/0811_Xpolicylab_bench/pi05/data/EgoVLA_benchmark
/personal/xiangpc/0811_Xpolicylab_bench/pi05/policy/Pi_05/checkpoints/EgoVLA-benchmark-ego_h1_inspire-joint-0
```

`deploy.yml` sets `eval_batch: false`, `train_config_name:
pi05_egovla_joint38_full`, `repo_id: EgoVLA_benchmark`, and
`policy_python_bin: .venv-egovla/bin/python`.  `checkpoint_num: null` means
the adapter selects the highest finalized numeric Orbax step and ignores
`.orbax-checkpoint-tmp-*` directories.

## Offline wiring checks

Run from the private clone.  The `base` conda environment is only used by the
bundled debug client; the policy server uses OpenPI's isolated
`.venv-egovla` runtime.  The Pi05 client prepends `/mnt/nfs/miniconda3/bin`
automatically when a clean SSH shell has no `conda` on `PATH`.  `PI05_DEBUG_NO_MODEL=1`
deliberately substitutes a zero-action policy, so these checks do not load a
checkpoint or allocate model weights.

Plain RGB observations (10 synthetic episodes) were verified with exit code 0:

```bash
cd /personal/xiangpc/0811_Xpolicylab_bench/pi05
EVAL_MAIN_ROOT="/personal/xiangpc/EgoVLA benchmark" \
EVAL_ENV_TYPE=debug PI05_DEBUG_NO_MODEL=1 \
bash policy/Pi_05/eval.sh \
  EgoVLA push_box EgoVLA-benchmark-ego_h1_inspire-joint-0 \
  ego_h1_inspire joint 0 7 0 uv base
```

The same command with `DEBUG_OBS_ENCODED=1` was also verified with exit code
0.  It sends JPEG/bytes camera payloads and exercises server-side image
decoding:

```bash
EVAL_MAIN_ROOT="/personal/xiangpc/EgoVLA benchmark" \
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 PI05_DEBUG_NO_MODEL=1 \
bash policy/Pi_05/eval.sh \
  EgoVLA push_box EgoVLA-benchmark-ego_h1_inspire-joint-0 \
  ego_h1_inspire joint 0 7 0 uv base
```

Both checks reported `action_dim=38`, selected
`openpi/.venv-egovla/bin/python`, established the websocket connection, and
ended with `[MAIN] eval finished`.

## Real checkpoint evaluation

Run only when a GPU is available (the training job currently occupies all
eight GPUs).  The same positional argument order is used by `eval.sh`:

```text
<bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed>
<policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>
```

For a completed checkpoint, omit `PI05_DEBUG_NO_MODEL` and use the same command
as above.  With `checkpoint_num: null`, the adapter automatically chooses the
latest finalized step under the run directory.  To force a particular step,
set `checkpoint_num` in a private deploy override or pass a complete step path
as `ckpt_name`; never point eval at an in-progress Orbax temporary directory.

If the policy runtime is elsewhere, set `PI05_PYTHON_BIN` to an executable
Python path.  The server keeps OpenPI source resolution anchored to this
private `policy/Pi_05/openpi` tree.
