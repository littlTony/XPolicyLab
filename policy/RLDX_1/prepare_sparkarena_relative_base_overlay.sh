#!/usr/bin/env bash
set -euo pipefail

# Create a lightweight training overlay.  Model tensors and the processor stay
# symlinked to the source checkpoint; only the model config and experiment YAML
# are copied and changed to enable relative-action processing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_BASE="${RLDX_SOURCE_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a}"
OVERLAY="${RLDX_RELATIVE_BASE_MODEL_PATH:-${XPL_ROOT}/pretrain_model/RLDX-1-PT-1592013a-sparkarena-relarm}"
RLDX_UPSTREAM="${RLDX_UPSTREAM_PATH:-${SCRIPT_DIR}/RLDX-1}"
PYTHON_BIN="${RLDX_PYTHON_BIN:-${RLDX_UPSTREAM}/.venv/bin/python}"

if [[ "$(readlink -m "${SOURCE_BASE}")" == "$(readlink -m "${OVERLAY}")" ]]; then
    echo "Source and overlay must be different paths: ${SOURCE_BASE}" >&2
    exit 1
fi

for required in \
    "${SOURCE_BASE}/config.json" \
    "${SOURCE_BASE}/model.safetensors.index.json" \
    "${SOURCE_BASE}/processor" \
    "${PYTHON_BIN}" \
    "${RLDX_UPSTREAM}/rldx/configs/model/rldx.py"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required base asset: ${required}" >&2
        exit 1
    fi
done

validate_overlay() {
    local root="$1"
    for required in \
        "${root}/config.json" \
        "${root}/model.safetensors.index.json" \
        "${root}/processor" \
        "${root}/experiment_cfg/config.yaml"; do
        if [[ ! -e "${required}" ]]; then
            echo "Incomplete SparkArena relative base overlay: ${required}" >&2
            return 1
        fi
    done

    PYTHONPATH="${RLDX_UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" - "${SOURCE_BASE}" "${root}" <<'PY'
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
overlay = Path(sys.argv[2])

source_hf = json.loads((source / "config.json").read_text())
overlay_hf = json.loads((overlay / "config.json").read_text())
expected_hf = deepcopy(source_hf)
expected_hf["use_relative_action"] = True
if overlay_hf != expected_hf:
    raise SystemExit("overlay config.json differs from source beyond use_relative_action=true")

overlay_yaml = yaml.safe_load((overlay / "experiment_cfg/config.yaml").read_text())
model = overlay_yaml.get("model", {})
training = overlay_yaml.get("training", {})
if model.get("use_relative_action") is not True:
    raise SystemExit("overlay experiment config did not enable use_relative_action")
if model.get("backbone_model_type") != "vtc_qwen3_vl" or model.get("action_horizon") != 16:
    raise SystemExit("overlay changed the RLDX architecture or action horizon")
if training.get("deepspeed_stage") != 3 or training.get("gradient_checkpointing") is not True:
    raise SystemExit("overlay must retain ZeRO-3 and gradient checkpointing")

source_yaml_path = source / "experiment_cfg/config.yaml"
if source_yaml_path.is_file():
    expected_yaml = yaml.safe_load(source_yaml_path.read_text())
    expected_yaml = deepcopy(expected_yaml)
    expected_yaml.setdefault("model", {})["use_relative_action"] = True
    if overlay_yaml != expected_yaml:
        raise SystemExit(
            "overlay experiment config differs from source beyond use_relative_action=true"
        )
PY
}

if [[ -e "${OVERLAY}" ]]; then
    validate_overlay "${OVERLAY}"
    echo "READY SparkArena relative base overlay=${OVERLAY}"
    exit 0
fi

mkdir -p "$(dirname "${OVERLAY}")"
staging="${OVERLAY}.incomplete-$$"
if [[ -e "${staging}" ]]; then
    echo "Staging path already exists: ${staging}" >&2
    exit 1
fi
mkdir "${staging}"
trap 'echo "FAILED retained staging='"${staging}"'" >&2' ERR

shopt -s dotglob nullglob
for source_path in "${SOURCE_BASE}"/*; do
    name="$(basename "${source_path}")"
    case "${name}" in
        .cache|config.json|experiment_cfg) continue ;;
    esac
    ln -s "${source_path}" "${staging}/${name}"
done
shopt -u dotglob nullglob
mkdir "${staging}/experiment_cfg"

PYTHONPATH="${RLDX_UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${SOURCE_BASE}" "${staging}" <<'PY'
from dataclasses import fields
import json
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

hf_config = json.loads((source / "config.json").read_text())
hf_config["use_relative_action"] = True
(destination / "config.json").write_text(
    json.dumps(hf_config, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

source_yaml_path = source / "experiment_cfg/config.yaml"
if source_yaml_path.is_file():
    payload = yaml.safe_load(source_yaml_path.read_text())
else:
    from rldx.configs.model.rldx import RLDXConfig

    model = RLDXConfig.from_pretrained(source, local_files_only=True)
    model_dict = {item.name: getattr(model, item.name) for item in fields(RLDXConfig)}
    payload = {
        "model": model_dict,
        "training": {
            "deepspeed_stage": 3,
            "gradient_checkpointing": True,
        },
    }
payload.setdefault("model", {})["use_relative_action"] = True
(destination / "experiment_cfg/config.yaml").write_text(
    yaml.safe_dump(payload, sort_keys=False),
    encoding="utf-8",
)
PY

validate_overlay "${staging}"
if [[ -e "${OVERLAY}" ]]; then
    echo "Overlay appeared during construction; refusing to replace it: ${OVERLAY}" >&2
    exit 1
fi
mv -T "${staging}" "${OVERLAY}"
trap - ERR
echo "DONE SparkArena relative base overlay=${OVERLAY} source=${SOURCE_BASE}"
