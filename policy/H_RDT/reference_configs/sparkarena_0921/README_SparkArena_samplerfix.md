# H-RDT SparkArena sampler-fix run

This directory is an isolated, sampler-only retraining candidate for the
seven-task SparkArena joint-54 benchmark. It does not modify the source HDF5,
the existing 0908 checkpoints, Eval Web, PD settings, action execution horizon,
normalization/clipping, or pretrained-weight loading rules.

The locked data contract is:

```text
input  = RGB image[t] from [head, right_wrist, left_wrist] + state[t]
target = original HDF5 action[t:t+16]
action = joint54 = left_arm7 + left_hand20 + right_arm7 + right_hand20
```

`prepare_data.sh` validates all 700 episodes and creates a read-only manifest
under `H_RDT/data/sparkarena_7task_samplerfix/`. The raw corpus, stats,
language embeddings, and Stage-1 checkpoint are linked into the isolated policy
directory; their source files are never rewritten.
`preflight.sh` verifies the manifest and deterministic sample plan without using
a GPU. `launch_train.sh` always runs preflight, then exits unless
`START_HRDT_TRAIN=1` is explicitly set.

The production virtual dataset has 5,120,000 deterministic draws. With eight
GPUs and batch eight per GPU, the prepared dataloader contains exactly 80,000
global batches. Every batch asserts 64 unique virtual indices and 64 unique raw
source-frame IDs across ranks. A complete checkpoint is recognized only after
`trainer_state.json` and `_SUCCESS` have been written.

Do not resume an old 0908 checkpoint. Resume is limited to a complete
checkpoint inside this run's own output directory and must match the data and
training contract hashes.
