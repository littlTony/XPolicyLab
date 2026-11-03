#!/usr/bin/env bash
set -euo pipefail

# A800_13 launcher for RLDX-1 on the audited EgoVLA joint38 LeRobot v2.1 data.
# Usage:
#   WANDB_API_KEY=... bash train_egovla_joint38.sh start
#   WANDB_API_KEY=... bash train_egovla_joint38.sh resume
#   bash train_egovla_joint38.sh smoke

mode="${1:-start}"
if [[ "${mode}" != "start" && "${mode}" != "resume" && "${mode}" != "smoke" ]]; then
  echo "usage: $0 [start|resume|smoke]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM="${SCRIPT_DIR}/RLDX-1"
DATASET="${RLDX_DATASET_PATH:-${XPL_ROOT}/data/EgoVLA_benchmark_rldx_v21_raw_action}"
MODALITY_CONFIG="${SCRIPT_DIR}/egovla_joint38_config.py"
TRAIN_ENTRYPOINT="${SCRIPT_DIR}/launch_train_egovla.py"
BASE_MODEL="${RLDX_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-egovla-full}"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${UPSTREAM}/.venv/bin/python}"
RLDX_SITE_PACKAGES="${RLDX_SITE_PACKAGES:-${UPSTREAM}/.venv/lib/python3.10/site-packages}"
RLDX_HF_HOME="${RLDX_HF_HOME:-${XPL_ROOT}/pretrain_model/hf_home}"
EGOVLA_INTEGRATION_SRC="${EGOVLA_INTEGRATION_SRC:-/personal/xiangpc/EgoVLA benchmark/integration/src}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "exactly eight GPUs are required: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l)" -ne 8 ]]; then
  echo "GPU ids must be unique: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

MAX_STEPS="${MAX_STEPS:-80000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
GLOBAL_BATCH_SIZE=64
NUM_GPUS=8
GRADIENT_ACCUMULATION_STEPS=1
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
STATE_DROPOUT_PROB=0.0

if [[ "${mode}" == "smoke" ]]; then
  MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
  SAVE_STEPS=1000
  SAVE_TOTAL_LIMIT=1
  default_run_name="EgoVLA-benchmark-rldx1_joint38-strict-full-smoke"
  use_wandb=0
else
  default_run_name="EgoVLA-benchmark-rldx1_joint38-rawaction-relarm-bs64-80k"
  use_wandb=1
  : "${WANDB_API_KEY:?WANDB_API_KEY must be supplied through the environment}"
fi
RUN_NAME="${RLDX_RUN_NAME:-${default_run_name}}"
CHECKPOINTS_ROOT="${SCRIPT_DIR}/checkpoints"
OUTPUT_DIR="${CHECKPOINTS_ROOT}/${RUN_NAME}"
PROFILE_PATH="${OUTPUT_DIR}/egovla_observation.json"
PROFILE_TMP="${CHECKPOINTS_ROOT}/.${RUN_NAME}.egovla_observation-${BASHPID}.json"

for required in \
  "${PYTHON_BIN}" \
  "${RLDX_SITE_PACKAGES}" \
  "${RLDX_HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/refs/main" \
  "${EGOVLA_INTEGRATION_SRC}/egovla_xpolicy/observation_profile.py" \
  "${UPSTREAM}/rldx/experiment/launch_train.py" \
  "${TRAIN_ENTRYPOINT}" \
  "${BASE_MODEL}/config.json" \
  "${BASE_MODEL}/model.safetensors.index.json" \
  "${BASE_MODEL}/experiment_cfg/config.yaml" \
  "${BASE_MODEL}/xpolicylab_revision.json" \
  "${DATASET}/meta/info.json" \
  "${DATASET}/meta/modality.json" \
  "${DATASET}/meta/stats.json" \
  "${DATASET}/meta/relative_stats.json" \
  "${DATASET}/meta/egovla_contract.json" \
  "${DATASET}/meta/rldx_egovla_audit.json" \
  "${MODALITY_CONFIG}"; do
  [[ -e "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 1
  }
done

export PYTHONPATH="${RLDX_SITE_PACKAGES}:${UPSTREAM}:${XPL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${RLDX_HF_HOME}"
export TORCH_HOME="${RLDX_LOCAL_CACHE_ROOT:-/tmp/rldx-egovla-hpc}/torch"
export WANDB__DISABLE_STATS=true
export WANDB_PROJECT="${WANDB_PROJECT:-XPolicyLab-RLDX1-EgoVLA}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"
mkdir -p "${TORCH_HOME}" "${CHECKPOINTS_ROOT}"

if [[ -n "${RLDX_VIDEO_BACKEND:-}" && "${RLDX_VIDEO_BACKEND}" != "torchcodec" ]]; then
  echo "RLDX_VIDEO_BACKEND must be unset or torchcodec to preserve RGB ordering" >&2
  exit 1
fi
export RLDX_VIDEO_BACKEND=torchcodec

exec 9>"${CHECKPOINTS_ROOT}/.${RUN_NAME}.launch.lock"
if ! flock -n 9; then
  echo "another launcher owns this run: ${RUN_NAME}" >&2
  exit 1
fi

hf_revision="$(<"${HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/refs/main")"
if [[ "${hf_revision}" != "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7" ]]; then
  echo "unexpected RLDX-1-VLM revision: ${hf_revision}" >&2
  exit 1
fi
hf_snapshot="${HF_HOME}/hub/models--RLWRLD--RLDX-1-VLM/snapshots/${hf_revision}"
for required in \
  "${hf_snapshot}/config.json" \
  "${hf_snapshot}/tokenizer_config.json" \
  "${hf_snapshot}/preprocessor_config.json"; do
  [[ -e "${required}" ]] || {
    echo "incomplete offline RLDX VLM cache: ${required}" >&2
    exit 1
  }
done

"${PYTHON_BIN}" - "${DATASET}" "${MODALITY_CONFIG}" <<'PY'
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return path, digest.hexdigest(), path.stat().st_size


def tree_sha256(root, paths):
    ordered = sorted(path.resolve() for path in paths)
    digest = hashlib.sha256()
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=min(32, len(ordered))) as pool:
        for path, file_digest, size in pool.map(file_sha256, ordered):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(str(size).encode())
            digest.update(b"\0")
            digest.update(file_digest.encode())
            digest.update(b"\n")
            total_bytes += size
    return digest.hexdigest(), total_bytes

dataset = pathlib.Path(sys.argv[1]).resolve()
config_path = pathlib.Path(sys.argv[2])
info = json.loads((dataset / "meta/info.json").read_text())
marker = json.loads((dataset / "meta/rldx_egovla_audit.json").read_text())
contract = json.loads((dataset / "meta/egovla_contract.json").read_text())
expected = {
    "codebase_version": "v2.1",
    "total_episodes": 1903,
    "total_frames": 510546,
    "total_tasks": 12,
    "fps": 30,
}
for key, value in expected.items():
    if info.get(key) != value:
        raise SystemExit(f"dataset {key}={info.get(key)!r}, expected {value!r}")
marker_expected = {
    "audit_version": "3.0.0",
    "action_type": "joint",
    "action_dim": 38,
    "action_source": "raw_hdf5/action",
    "action_is_next_state": False,
    "action_horizon": 16,
    "fps": 30,
    "color_order": "RGB",
    "channel_swap": False,
    "source_image_resolution": [384, 384],
    "processor_image_resolution": [256, 256],
    "real_wrist_episodes": 900,
    "black_wrist_episodes": 1003,
    "black_wrist_videos_first_frame_verified": 2006,
    "total_episodes": 1903,
    "total_frames": 510546,
    "total_tasks": 12,
    "total_videos": 5709,
    "parquet_files": 1903,
    "video_files": 5709,
    "lossless_packet_slices_verified": 5709,
    "lossless_video_packets_verified": 1531638,
}
if any(marker.get(key) != value for key, value in marker_expected.items()):
    raise SystemExit(f"invalid audit marker: {marker}")
raw_vs_next = marker.get("raw_vs_next_state_mae")
if not isinstance(raw_vs_next, (int, float)) or not math.isfinite(raw_vs_next) or raw_vs_next < 0:
    raise SystemExit("invalid raw-action versus next-state diagnostic")
contract_expected = {
    "contract_version": "1.0.0",
    "action_source": "raw_hdf5/action",
    "stored_action_semantics": "absolute commanded joint target",
    "action_dim": 38,
    "action_horizon": 16,
    "image_resolution": [384, 384],
    "processor_resolution": [256, 256],
    "color_order": "RGB",
    "channel_swap": False,
    "total_episodes": 1903,
    "total_frames": 510546,
}
if any(contract.get(key) != value for key, value in contract_expected.items()):
    raise SystemExit(f"invalid EgoVLA training contract: {contract}")
expected_model_semantics = {
    "left_arm": "relative_to_current_state",
    "left_hand": "absolute",
    "right_arm": "relative_to_current_state",
    "right_hand": "absolute",
}
if contract.get("model_action_semantics") != expected_model_semantics:
    raise SystemExit("dataset/model action semantics changed")
if marker.get("benchmark_schema_sha256") != contract.get("benchmark_schema_sha256"):
    raise SystemExit("audit and dataset contract use different benchmark schemas")
for name, expected_hash in marker["sha256"].items():
    actual = hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise SystemExit(f"audited metadata changed: {name}")

parquet_paths = list(dataset.glob("data/chunk-*/episode_*.parquet"))
video_paths = list(dataset.glob("videos/chunk-*/*/episode_*.mp4"))
if len(parquet_paths) != marker.get("parquet_files"):
    raise SystemExit("audited parquet file count changed")
if len(video_paths) != marker.get("video_files"):
    raise SystemExit("audited video file count changed")
parquet_digest, parquet_bytes = tree_sha256(dataset, parquet_paths)
video_digest, video_bytes = tree_sha256(dataset, video_paths)
if (
    parquet_digest != marker.get("parquet_tree_sha256")
    or parquet_bytes != marker.get("parquet_total_bytes")
):
    raise SystemExit("audited parquet training data changed")
if video_digest != marker.get("video_tree_sha256") or video_bytes != marker.get(
    "video_total_bytes"
):
    raise SystemExit("audited video training data changed")

spec = importlib.util.spec_from_file_location("egovla_joint38_config", config_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mods = module.EGOVLA_JOINT38_MODALITY_CONFIG
if mods["state"].modality_keys != ["left_arm", "left_hand", "right_arm", "right_hand"]:
    raise SystemExit("unexpected state modality order")
representations = [item.rep.name for item in mods["action"].action_configs]
if representations != ["RELATIVE", "ABSOLUTE", "RELATIVE", "ABSOLUTE"]:
    raise SystemExit(f"unexpected action semantics: {representations}")
print("validated audited EgoVLA joint38 dataset and RLDX modality semantics")
PY

"${PYTHON_BIN}" - "${BASE_MODEL}/experiment_cfg/config.yaml" \
  "${BASE_MODEL}/xpolicylab_revision.json" <<'PY'
import json
import pathlib
import sys
import yaml

overlay = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
revision = json.loads(pathlib.Path(sys.argv[2]).read_text())
expected_revision = {
    "base_model_repo": "RLWRLD/RLDX-1-PT",
    "base_model_revision": "1592013ab955b4919facd91fb37449f33a292a70",
    "backbone_config_repo": "RLWRLD/RLDX-1-VLM",
    "backbone_config_revision": "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7",
}
if any(revision.get(key) != value for key, value in expected_revision.items()):
    raise SystemExit(f"unexpected pretrained model provenance: {revision}")
model = overlay.get("model", {})
training = overlay.get("training", {})
if model.get("model_name") != "RLWRLD/RLDX-1-VLM":
    raise SystemExit(f"strict full FT requires the RLDX-1-VLM backbone: {model}")
if model.get("backbone_model_type") != "vtc_qwen3_vl" or model.get("action_horizon") != 16:
    raise SystemExit("full-FT overlay architecture does not match the clean checkpoint")
if model.get("state_dropout_prob") != 0.0:
    raise SystemExit("strict clean-base full FT requires state_dropout_prob=0.0")
if training.get("deepspeed_stage") != 3:
    raise SystemExit(f"strict full FT requires ZeRO-3: {training}")
if training.get("gradient_checkpointing") is not True:
    raise SystemExit(f"strict full FT requires gradient checkpointing: {training}")

from rldx.configs.train_config import TrainConfig

defaults = TrainConfig()
if not defaults.tune_projector or not defaults.tune_diffusion_model:
    raise SystemExit("RLDX projector/diffusion full-tune defaults changed")
if defaults.action_model_use_lora or defaults.backbone_use_lora or defaults.freeze_cog_tokens:
    raise SystemExit("strict full FT refuses LoRA or frozen cognition tokens")
print("validated strict full-parameter flags, ZeRO-3, and gradient checkpointing")
PY

"${PYTHON_BIN}" - <<'PY'
import av
import deepspeed
import flash_attn
import torch
import torchcodec
import wandb

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
print(
    f"runtime torch={torch.__version__} cuda={torch.version.cuda} "
    f"torchcodec={torchcodec.__version__} flash_attn={flash_attn.__version__} "
    f"av={av.__version__} deepspeed={deepspeed.__version__} wandb={wandb.__version__}"
)
PY

if [[ "${mode}" == "start" || "${mode}" == "smoke" ]]; then
  if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
    exit 1
  fi
else
  if [[ ! -d "${OUTPUT_DIR}" ]] || ! compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null; then
    echo "resume requested but no checkpoint exists: ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi

while IFS=, read -r index memory_used; do
  memory_used="${memory_used// /}"
  if (( memory_used > 1024 )); then
    echo "GPU ${index} is not idle: ${memory_used} MiB in use" >&2
    exit 1
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

if [[ "${mode}" == "start" || "${mode}" == "smoke" ]]; then
  profile_target="${PROFILE_TMP}"
else
  profile_target="${PROFILE_PATH}"
fi
PYTHONPATH="${EGOVLA_INTEGRATION_SRC}:${PYTHONPATH}" \
  "${PYTHON_BIN}" -m egovla_xpolicy.observation_profile \
  --dataset "${DATASET}" --output "${profile_target}"
if [[ "${profile_target}" == "${PROFILE_TMP}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  mv -T "${PROFILE_TMP}" "${PROFILE_PATH}"
fi
[[ -s "${PROFILE_PATH}" ]] || {
  echo "failed to create the inference observation profile" >&2
  exit 1
}

echo "[RLDX EgoVLA] mode=${mode} run=${RUN_NAME}"
echo "[RLDX EgoVLA] stored_action=raw_hdf5/action action_type=joint action_dim=38 horizon=16 arms=relative hands=absolute"
echo "[RLDX EgoVLA] images=RGB source=384x384 processor=256x256 missing_wrists=black video_backend=torchcodec"
echo "[RLDX EgoVLA] global_batch=64 gpu_count=8 per_gpu_batch=8 max_steps=${MAX_STEPS}"
echo "[RLDX EgoVLA] strict_full_ft=true zero_stage=3 gradient_checkpointing=true state_dropout_prob=${STATE_DROPOUT_PROB} learning_rate=${LEARNING_RATE}"
echo "[RLDX EgoVLA] dataset=${DATASET} base_model=${BASE_MODEL} output=${OUTPUT_DIR}"

wandb_args=()
if [[ "${use_wandb}" == "1" ]]; then
  wandb_args+=(--use-wandb --wandb-project "${WANDB_PROJECT}")
fi

cd "${UPSTREAM}"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_ENTRYPOINT}" \
  --base-model-path "${BASE_MODEL}" \
  --backbone-path RLWRLD/RLDX-1-VLM \
  --dataset-path "${DATASET}" \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path "${MODALITY_CONFIG}" \
  --tune-llm \
  --tune-visual \
  --tune-top-llm-layers 0 \
  --tune-projector \
  --tune-diffusion-model \
  --no-backbone-use-lora \
  --no-action-model-use-lora \
  --no-freeze-cog-tokens \
  --state-dropout-prob "${STATE_DROPOUT_PROB}" \
  --n-cog-tokens 64 \
  --action-horizon 16 \
  --video-length 4 \
  --video-stride 2 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-gpus "${NUM_GPUS}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --output-dir "${CHECKPOINTS_ROOT}" \
  --experiment-name "${RUN_NAME}" \
  "${wandb_args[@]}"
