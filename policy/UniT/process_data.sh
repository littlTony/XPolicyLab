#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [max_episodes_per_task]" >&2
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
max_episodes_per_task=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERTER="${XPL_ROOT}/data_scripts/convert_spark0_bench_to_lerobot_v20_joint54.py"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/xspark-data/tjy/spark0_bench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${XPL_ROOT}/data/Spark0_bench_lerobotV20_joint54}"
PYTHON_BIN="${PYTHON_BIN:-/personal/miniconda3/envs/unit_torch26_recovery/bin/python}"

if [[ "${action_type}" != "joint" ]]; then
    echo "error: UniT Spark data adapter supports action_type=joint only" >&2
    exit 1
fi
if [[ ! -f "${XPL_ROOT}/../env_cfg/${env_cfg_type}.yml" ]]; then
    echo "error: missing benchmark env config: ${XPL_ROOT}/../env_cfg/${env_cfg_type}.yml" >&2
    echo "Register/select the existing Spark tianji_marvin_wuji env config; this adapter will not invent one." >&2
    exit 1
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
    echo "error: source HDF5 root does not exist: ${SOURCE_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${CONVERTER}" ]]; then
    echo "error: converter is missing: ${CONVERTER}" >&2
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: ${PYTHON_BIN} is not on PATH; activate the conversion environment first" >&2
    exit 1
fi

converter_args=(
    --source-root "${SOURCE_ROOT}"
    --output-root "${OUTPUT_ROOT}"
)
if [[ -n "${max_episodes_per_task}" ]]; then
    converter_args+=(--max-episodes-per-task "${max_episodes_per_task}")
fi

echo "[UniT] converting Spark0 HDF5 directly to six native LeRobot v2.0 joint54 task datasets"
"${PYTHON_BIN}" "${CONVERTER}" "${converter_args[@]}"

task_names=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)
for task_name in "${task_names[@]}"; do
    if [[ ! -f "${OUTPUT_ROOT}/${task_name}/meta/info.json" ]]; then
        echo "error: conversion did not produce ${OUTPUT_ROOT}/${task_name}/meta/info.json" >&2
        exit 1
    fi
done

echo "[UniT] data ready: ${OUTPUT_ROOT}"
echo "[UniT] XPolicyLab setting: ${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
