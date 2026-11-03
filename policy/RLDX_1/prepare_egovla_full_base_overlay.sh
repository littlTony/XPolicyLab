#!/usr/bin/env bash
set -euo pipefail

# Add only an experiment_cfg training overlay to the clean RLDX checkpoint.
# Model weights stay as symlinks to the immutable xiangpc-owned source.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_BASE="${RLDX_SOURCE_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a}"
OVERLAY="${RLDX_FULL_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-egovla-full}"
TRAINING_OVERLAY="${SCRIPT_DIR}/egovla_full_training_overlay.yaml"
RLDX_UPSTREAM="${SCRIPT_DIR}/RLDX-1"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${RLDX_UPSTREAM}/.venv/bin/python}"
RLDX_SITE_PACKAGES="${RLDX_SITE_PACKAGES:-${RLDX_UPSTREAM}/.venv/lib/python3.10/site-packages}"

for required in \
  "${SOURCE_BASE}/config.json" \
  "${SOURCE_BASE}/model.safetensors.index.json" \
  "${SOURCE_BASE}/processor" \
  "${TRAINING_OVERLAY}" \
  "${PYTHON_BIN}" \
  "${RLDX_SITE_PACKAGES}" \
  "${RLDX_UPSTREAM}/rldx/configs/model/rldx.py"; do
  [[ -e "${required}" ]] || {
    echo "missing required base asset: ${required}" >&2
    exit 1
  }
done

validate_overlay() {
  local root="$1"
  for required in \
    "${root}/config.json" \
    "${root}/model.safetensors.index.json" \
    "${root}/processor" \
    "${root}/experiment_cfg/config.yaml"; do
    [[ -e "${required}" ]] || {
      echo "incomplete full-FT base overlay: ${required}" >&2
      return 1
    }
  done
  PYTHONPATH="${RLDX_SITE_PACKAGES}:${RLDX_UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${root}/experiment_cfg/config.yaml" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
model = config.get("model", {})
training = config.get("training", {})
if model.get("model_name") != "RLWRLD/RLDX-1-VLM":
    raise SystemExit(f"overlay lost the checkpoint backbone: {model.get('model_name')!r}")
if model.get("backbone_model_type") != "vtc_qwen3_vl" or model.get("action_horizon") != 16:
    raise SystemExit("overlay model architecture does not match the clean RLDX checkpoint")
if training.get("deepspeed_stage") != 3 or training.get("gradient_checkpointing") is not True:
    raise SystemExit(f"invalid strict-full training overlay: {training!r}")
PY
}

if [[ -e "${OVERLAY}" ]]; then
  validate_overlay "${OVERLAY}"
  echo "READY existing full-FT base overlay=${OVERLAY}"
  exit 0
fi

mkdir -p "$(dirname "${OVERLAY}")"
staging="${OVERLAY}.incomplete-$$"
[[ ! -e "${staging}" ]] || {
  echo "staging already exists: ${staging}" >&2
  exit 1
}
mkdir "${staging}"
trap 'echo "FAILED retained staging='"${staging}"'" >&2' ERR

shopt -s dotglob nullglob
for source in "${SOURCE_BASE}"/*; do
  name="$(basename "${source}")"
  [[ "${name}" == ".cache" || "${name}" == "experiment_cfg" ]] && continue
  ln -s "${source}" "${staging}/${name}"
done
shopt -u dotglob nullglob
mkdir "${staging}/experiment_cfg"
PYTHONPATH="${RLDX_SITE_PACKAGES}:${RLDX_UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - "${SOURCE_BASE}" "${TRAINING_OVERLAY}" \
  "${staging}/experiment_cfg/config.yaml" <<'PY'
from dataclasses import fields
from pathlib import Path
import sys
import yaml

from rldx.configs.model.rldx import RLDXConfig

source = Path(sys.argv[1])
training_overlay = yaml.safe_load(Path(sys.argv[2]).read_text())
destination = Path(sys.argv[3])
model = RLDXConfig.from_pretrained(source, local_files_only=True)
model_dict = {item.name: getattr(model, item.name) for item in fields(RLDXConfig)}
if model_dict["model_name"] != "RLWRLD/RLDX-1-VLM":
    raise SystemExit(f"unexpected clean checkpoint backbone: {model_dict['model_name']!r}")
payload = {"model": model_dict, "training": training_overlay["training"]}
destination.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
validate_overlay "${staging}"

if [[ -e "${OVERLAY}" ]]; then
  echo "overlay appeared during construction; refusing publish: ${OVERLAY}" >&2
  exit 1
fi
mv -T "${staging}" "${OVERLAY}"
trap - ERR
echo "DONE full-FT base overlay=${OVERLAY} source=${SOURCE_BASE} zero_stage=3 gradient_checkpointing=true"
