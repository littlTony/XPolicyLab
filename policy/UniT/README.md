# UniT: DexBench EE58 training and serving

Official source: https://github.com/xpeng-robotics/UniT, clean reference checkout
`UniT/` at `800c0a428d431964dad4e7adc9c667d639432249`.
The historical SparkArena scripts/configs remain under
`reference_configs/sparkarena_0908/` for reference and are not executed unchanged.

This entry trains **DexBench's current 12 physical tasks / 3000 episodes**, while
retaining the SparkArena optimizer/update/batch recipe. It does not train the old
SparkArena dataset or use EgoVLA Bench configurations. Source metadata reports
60 tasks because source-robot/instruction variants are distinct task indices;
these map to 12 physical task names.

## Representation and pretrained weights

Source: `/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21`.
Both actual state and action have 58 values in this order:

`left xyz3 + rotation6 + Wuji20, right xyz3 + rotation6 + Wuji20`.

Rotation6 is **the first two matrix rows**, matching
`meta/action_contract.json` (`dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1`).
Values are absolute world-frame EE poses and native Wuji joint angles. The
action target is `action[t:t+16]` (indices t..t+15), state is at t. The tokenizer's
future goal image is t+16; that visual goal never shifts action labels by +1.

Real58 is padded to **128** inside the native UniT transforms. Keep the model
JSON state/action/max dimensions128, `max_num_embodiments=32`, and
`new_embodiment` slot31. The 32 slots identify embodiments, not action channels.
Masks are True for all58 actual values and False for the remaining70 padded
zeros, during both tokenizer and policy training. State and action each use
their own new EE58 **mean/std** statistics, preserving the SparkArena UniT
normalization choice. No old54 statistics or GR00T relative-action statistics
are used. The shared native tensor transform explicitly converts continuous
state/action values to float32 before normalization in training and serving.

Padding128 preserves the architecture and checkpoint tensor shapes. This was
checked against the historical tokenizer's safetensors headers without loading
the tensors: state input `[32,128,1024]`, action convolution
`[32,1024,128,5]`, action output `[32,128,1024,5]`, and two codebooks `[128,32]`.
It does **not** make old joint54 weights semantically correct for EE58: projection
channels, normalization, codebook meanings and policy code-prediction targets
have changed. A checkpoint's slot31 may also never have been trained for Wuji.

Official full UniT policies exist at
https://huggingface.co/xpeng-robotics/VLA-UniT-checkpoints. The baseline here
explicitly follows the SparkArena config-only initialization: public pretrained
DINOv2 for the tokenizer and public pretrained Qwen2.5-VL-3B for the policy;
new action/fusion/bridge modules are initialized from their configs. It does not
claim to load an entire pretrained UniT policy. Changing a full UniT policy's
tokenizer changes discrete code-ID meanings even when all tensor shapes match,
so its old code CE predictors/embeddings also require adaptation.

## Which tokenizer is trained?

This is UniT's **multimodal latent action tokenizer**, not the Qwen language
tokenizer. Stage1 learns a visual/action physical representation:

1. Frozen DINOv2-large extracts current and future-goal features from all three
   real RGB cameras. Per-view features are averaged on the aligned patch grid,
   preserving the256-patch M-Former/decoder parameter shapes used in the
   successful three-view SparkArena implementation.
2. An embodiment-conditioned action encoder processes state_t and the16-step
   action chunk. Fusion trains both, vision-only, and action-only scenarios.
3. Eight latent query tokens pass through **two residual-VQ codebooks**, each
   with128 codes of dimension32.
4. The trained decoders reconstruct future DINO features and the action chunk.
   Tokenizer loss is visual cosine reconstruction + masked SmoothL1 action
   reconstruction (beta0.5) + VQ commitment loss. It is not an FM loss.

Stage2 freezes the completed tokenizer in eval mode. The policy learns CE over
the tokenizer's discrete codes together with its continuous FM action expert.
The Qwen text tokenizer stays the pretrained one, with eight bridge markers.
Internal three-scenario expansion does not change the64 source-example batch.

Default: train a new EE58 tokenizer. An explicit weights-only warm start is
allowed using `UNIT_TOKENIZER_WARMSTART=/path/to/tokenizer/checkpoint`; it must
match the padded128/H16 architecture. It still begins a new optimizer at step0
and trains40000 EE58 updates. An old54 tokenizer is never accepted as a finished
EE58 tokenizer for Stage2. Strict HF checkpoint loading bypasses the upstream
catch-all exception that otherwise silently initializes a random model.

## Entry and recipe

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/UniT

# CPU data/config checks only. Does not construct the large models or use GPUs.
bash train_ee58.sh tianji_wuji 0,1,2,3,4,5,6,7 --dry-run

# Explicit training command; only run when these eight GPUs are allocated.
bash train_ee58.sh tianji_wuji 0,1,2,3,4,5,6,7
```

- Tokenizer:40000 optimizer updates,8 GPUs x micro8 x GA1 = batch64, bf16
  autocast, AdamW LR1e-4.
- Policy:60000 optimizer updates,8 GPUs x micro4 x GA2 = batch64, FP32
  parameters, ZeRO2 with DeepSpeed bf16 torch-autocast, gradient checkpointing,
  AdamW LR5e-5. Qwen language layers/projector/action model/bridge embeddings
  train; its vision tower and the completed tokenizer remain frozen. Action
  loss is detached from the VLM path, while code CE trains that path.
- Both stages: beta=(0.95,0.999), eps1e-8, weight decay1e-5, cosine schedule,
  5% warmup, clip norm1, checkpoint every10000 updates, all3000 episodes,
  seed42. There is no held-out task-success evaluation in this entry.

Each shuffled epoch uses751296 draws (a multiple of64), omitting its random
52-frame tail so epoch-end optimizer updates also keep the full batch64. Every
source frame remains eligible in subsequent reshuffles; the dataset still
contains all751348 frames and all3000 episodes. Empty native placeholder
parameters are excluded from optimizer groups before ZeRO partitioning, as in
the historical working implementation. The adapter forwards epoch changes
through Accelerate1.3's batch-sampler wrapper; real CPU DataLoaderShard tests
confirmed that the random52 omitted frames change between epochs.
There are11739 optimizer updates per complete effective epoch. Tokenizer40000
updates are about3.41 epochs; policy60000 updates are about5.11 epochs.

Configuration lives in `configs/ee58_recipe.json`, `ee58_tokenizer.json`,
`ee58_policy.json`, and `deepspeed_zero2.json`. TrainingArguments are constructed
explicitly because the current native CLI hard-codes bf16/deepspeed/checkpointing
settings; merely forwarding flags would not implement the desired recipe.

Defaults: Python `/personal/miniconda3/envs/unit_torch26_recovery/bin/python`,
DINO `/mnt/xspark-data/xiangpc/_shared_caches/unit/dinov2-large`,
Qwen `/personal/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct`.
Override `UNIT_PYTHON`, `UNIT_DINO_PATH`, `UNIT_VLM_PATH`, `UNIT_EE58_DATASET` or
`EVAL_MAIN_ROOT` when moving the code. Assets are read offline; this entry does
not install packages or download weights. W&B defaults to disabled.

Preparation creates private metadata in a unique `.generated/` directory and
links the existing large parquet/videos. The private metadata view uses the
native UniT v2.0-compatible schema, drops irrelevant statistic fields and adds
dtype annotations; numeric source arrays, source metadata and videos are never
rewritten. Independent per-camera video offsets apply to both current and goal
frames, including episodes that share an underlying video file.

The shared checkpoint helper names the run
`checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/`, with `tokenizer/` and
`policy/` below it. `tianji_wuji` is the default checkpoint name; names containing
batch labels such as `bs64` or `batch128` are rejected for new runs. Existing run
directories are rejected. The pipeline starts
Stage2 only after Stage1 returns successfully, the internal trainer step equals
40000, a complete safetensors checkpoint exists and its saved EE58/statistics
contract matches. This finite two-stage command creates no periodic monitor.
It refuses GPUs with existing compute processes and never stops those jobs.

## Evaluation server

The standard XPolicyLab `Model` adapter consumes decoded RGB and the simulator's
raw EE54 wire state. The shared representation helper converts quaternion EE54
to absolute row-rotation EE58. The same native transforms apply separate
state/action mean/std,58+70 padding/masks,slot31,three RGB views,224 image size,
and the same Qwen processor/eight bridge markers/ignore_lang_prefix setting.
Serving uses the deterministic eval form of the training augmentation: center
crop0.95,resize224,no random color jitter. Native `get_action` explicitly runs
with `compute_bridge_loss=False`: current views and state suffice; future goal
images and the frozen training tokenizer are not used online. Predicted128-wide
actions are sliced to58,denormalized with the saved action statistics,projected
onto SO(3),converted to wxyz EE54 and returned as standard environment action
dicts. The predicted chunk and default execution length are16; `execute_steps`
in `deploy.yml` controls execution without rebuilding the action head.

After the new training has actually finished, the expected final policy is:

`checkpoints/DexBench-tianji_wuji-tianji_marvin_wuji-ee-42/policy/checkpoint-60000/`.

```bash
cd /personal/wenwei/DexBench/XPolicyLab/policy/UniT
bash setup_eval_policy_server.sh DexBench auto tianji_wuji tianji_marvin_wuji ee 42 0 auto 11007
```

An explicit policy checkpoint path may replace `tianji_wuji`. Set
`UNIT_CHECKPOINT_STEP=10000` (or another saved step) to select an earlier
checkpoint under the same run. This is a launch instruction for a future
trained EE58 policy; no such training is asserted by the CPU checks.

Each saved policy checkpoint and final policy directory receives its exact
`inference_contract.json`,`source_stats.json`,`source_action_contract.json`,
native `config.json` and `experiment_cfg/metadata.json`. The sidecar binds full
safetensors checksums,configuration/statistics,processor assets,bridge IDs and
the preprocessing source. A mismatch or incomplete/missing checkpoint fails
closed. Loading bypasses upstream random-initialization fallback and never
rebuilds a horizon-mismatched head. Old54/full-UniT weights do not qualify as a
finished EE58 checkpoint merely because their tensor shapes match.

## Validation and limits

CPU preflight uses the actual native dataset/transforms: two episodes per each
of12 physical tasks, source action/state equality, actual58/padded128/masks,
separate mean/std normalization, current/goal video offsets, three camera
shapes, native HF config serialization and tokenizer checkpoint header shapes.
The exact parameter-free three-view pooling helper is exercised with a tiny CPU
vision stand-in. The policy's native Qwen processor/collator was checked on a
real episode: three current views, three future-goal views and all eight bridge
markers reach the batch. Separate tiny CPU tests verified strict checkpoint
rejection, empty optimizer-parameter filtering and the actual sharded sampler's
epoch propagation/full64-update groups. Combined receipt:
`training/ee58_cpu_validation_20261030.json`; detailed native-data receipt:
`.generated/<run>_*/preflight.json`.

The serving oracle in `training/serving_oracle.py` additionally exercises the
actual adapter init/update_obs/get_action,batch/reset,sidecar validation and
native preprocessing/decoding with a normalized GT-action predictor stand-in.
Its24 real episode cases compare runtime normalized state,masks,RGB processor
pixels/grids,tokens and decoded16-step actions against the native training
data path. This checks data and protocol alignment,not learned prediction
quality; the fixture has deliberately invalid placeholder model weights that
the unpatched production loader rejects. Receipt:
`training/ee58_serving_oracle_20261030.json`.

This integration does not claim a real GPU model forward/backward, measured
memory/throughput,successful40000/60000 training,native full-optimizer resume,
or learned-policy simulator performance. Those require their respective runs.
The official source clone and historical reference checkpoints are unchanged.
