#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
    echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_env_or_uv_path> <port> [host]" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env_or_uv_path=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
if [[ "${policy_name}" != "RLDX_1" ]]; then
    echo "[SERVER][ERROR] this entrypoint must resolve to policy/RLDX_1, got ${policy_name}" >&2
    exit 1
fi
RLDX_POLICY_ROOT="${XPL_ROOT}/policy/${policy_name}"
UPSTREAM_ROOT="${RLDX_POLICY_ROOT}/RLDX-1"
yaml_file="${SCRIPT_DIR}/deploy.yml"
BRIDGE_BOOTSTRAP="${RLDX_BRIDGE_BOOTSTRAP:-${SCRIPT_DIR}/egovla_bridge_bootstrap.py}"
PRIVATE_MANIFEST="${RLDX_PRIVATE_MANIFEST:-${SCRIPT_DIR}/rldx_capabilities.json}"
if [[ "${bench_name}" == "EgoVLA" ]]; then
    BENCH_ROOT="${EVAL_MAIN_ROOT:-/mnt/xspark-data/xiangpc/EgoVLA benchmark}"
    BRIDGE_ROOT="${EGOVLA_BRIDGE_ROOT:-${BENCH_ROOT}/integration}"
else
    BENCH_ROOT="${EVAL_MAIN_ROOT:-/personal/wenwei/xspark-0-simeval}"
    BRIDGE_ROOT=""
fi

if [[ ! -f "${XPL_ROOT}/XPolicyLab.py" || ! -d "${RLDX_POLICY_ROOT}" || ! -f "${RLDX_POLICY_ROOT}/model.py" || ! -f "${RLDX_POLICY_ROOT}/deploy.py" ]]; then
    echo "[SERVER][ERROR] private RLDX checkout is incomplete: ${XPL_ROOT}" >&2
    exit 1
fi
if [[ "${bench_name}" == "EgoVLA" ]]; then
    if [[ ! -r "${BRIDGE_BOOTSTRAP}" ]]; then
        echo "[SERVER][ERROR] private bridge bootstrap is missing: ${BRIDGE_BOOTSTRAP}" >&2
        exit 1
    fi
    if [[ ! -r "${PRIVATE_MANIFEST}" ]]; then
        echo "[SERVER][ERROR] private RLDX capability manifest is missing: ${PRIVATE_MANIFEST}" >&2
        exit 1
    fi
    if [[ ! -d "${BRIDGE_ROOT}/src/egovla_xpolicy" ]]; then
        echo "[SERVER][ERROR] bridge root lacks integration/src/egovla_xpolicy: ${BRIDGE_ROOT}" >&2
        exit 1
    fi
fi

# RLDX's checkpoint config names RLWRLD/RLDX-1-VLM as its backbone.  Even an
# inference load that skips the large backbone download needs the Qwen3-VL
# config/tokenizer/video metadata.  Training prepared a metadata-only Hub
# cache; select the first complete cache visible on this host.  An explicit
# RLDX_HF_HOME is fail-closed when it is invalid.
DEFAULT_HF_HOME_MNT="/mnt/xspark-data/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/hf_home"
DEFAULT_HF_HOME_PERSONAL="/personal/xiangpc/old_sim_eval/0807_RLDX-1/policy/RLDX_1/pretrained/hf_home"
PRIVATE_HF_HOME_MNT="${XPL_ROOT}/pretrain_model/hf_home"
PRIVATE_HF_HOME_PERSONAL="${RLDX_POLICY_ROOT}/pretrained/hf_home"

validate_hf_home() {
    local home="$1"
    local hub="${home}/hub"
    local repo="${hub}/models--RLWRLD--RLDX-1-VLM"
    local revision snapshot required

    [[ -d "${home}" && -d "${hub}" && -d "${repo}" ]] || return 1
    [[ -f "${repo}/refs/main" && -r "${repo}/refs/main" ]] || return 1
    revision="$(tr -d '[:space:]' < "${repo}/refs/main")" || return 1
    [[ "${revision}" =~ ^[[:xdigit:]]{40}$ ]] || return 1
    snapshot="${repo}/snapshots/${revision}"
    [[ -d "${snapshot}" ]] || return 1

    # The base model weight blobs are intentionally not required: the local
    # RLDX checkpoint contains the consolidated weights.  These metadata files
    # are consumed by AutoConfig/AutoProcessor and Qwen3-VL video processing.
    for required in \
        config.json \
        model.safetensors.index.json \
        preprocessor_config.json \
        video_preprocessor_config.json \
        tokenizer_config.json \
        special_tokens_map.json \
        added_tokens.json \
        vocab.json \
        merges.txt \
        chat_template.jinja; do
        [[ -f "${snapshot}/${required}" && -r "${snapshot}/${required}" ]] || return 1
    done
    printf '%s\n' "${snapshot}"
}

selected_hf_home=""
selected_hf_snapshot=""
if [[ -n "${RLDX_HF_HOME:-}" ]]; then
    selected_hf_home="${RLDX_HF_HOME}"
    if ! selected_hf_snapshot="$(validate_hf_home "${selected_hf_home}")"; then
        echo "[SERVER][ERROR] RLDX_HF_HOME is not a complete local RLWRLD/RLDX-1-VLM cache: ${selected_hf_home}" >&2
        exit 1
    fi
else
    for candidate_hf_home in \
        "${DEFAULT_HF_HOME_MNT}" \
        "${DEFAULT_HF_HOME_PERSONAL}" \
        "${PRIVATE_HF_HOME_MNT}" \
        "${PRIVATE_HF_HOME_PERSONAL}"; do
        if candidate_hf_snapshot="$(validate_hf_home "${candidate_hf_home}" 2>/dev/null)"; then
            selected_hf_home="${candidate_hf_home}"
            selected_hf_snapshot="${candidate_hf_snapshot}"
            break
        fi
    done
    if [[ -z "${selected_hf_home}" ]]; then
        echo "[SERVER][ERROR] No complete local RLWRLD/RLDX-1-VLM cache is visible." >&2
        echo "[SERVER][ERROR] Set RLDX_HF_HOME to a cache containing hub/models--RLWRLD--RLDX-1-VLM." >&2
        exit 1
    fi
fi

HF_HOME="$(cd "${selected_hf_home}" && pwd -P)"
HF_HUB_CACHE="${HF_HOME}/hub"
# Keep legacy and current Hugging Face cache aliases on the same offline Hub.
HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
HF_ASSETS_CACHE="${HF_HOME}/assets"
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_ASSETS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# A800_13 has no /personal mount and no conda command.  Resolve an explicit
# executable first, then a supplied uv/venv directory, and only lastly use
# conda for compatibility with older hosts.
DEFAULT_POLICY_PYTHON_BIN="/mnt/xspark-data/xiangpc/.uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10"
DEFAULT_POLICY_SITE_PACKAGES="${UPSTREAM_ROOT}/.venv/lib/python3.10/site-packages"

resolve_python_bin() {
    local reference candidate conda_exe
    for reference in "${RLDX_POLICY_PYTHON_BIN:-}" "${policy_env_or_uv_path}"; do
        [[ -z "${reference}" ]] && continue
        for candidate in \
            "${reference}" \
            "${reference}/bin/python" \
            "${reference}/bin/python3" \
            "${reference}/.venv/bin/python" \
            "${reference}/.venv/bin/python3"; do
            if [[ -f "${candidate}" && -x "${candidate}" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done
    done

    if [[ -f "${DEFAULT_POLICY_PYTHON_BIN}" && -x "${DEFAULT_POLICY_PYTHON_BIN}" ]]; then
        printf '%s\n' "${DEFAULT_POLICY_PYTHON_BIN}"
        return 0
    fi

    if command -v conda >/dev/null 2>&1; then
        conda_exe="$(command -v conda)"
    elif [[ -x /root/miniconda3/bin/conda ]]; then
        conda_exe=/root/miniconda3/bin/conda
    elif [[ -x /personal/miniconda3/bin/conda ]]; then
        conda_exe=/personal/miniconda3/bin/conda
    else
        return 1
    fi
    source "$(${conda_exe} info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env_or_uv_path}"
    command -v python
}

if ! POLICY_PYTHON_BIN="$(resolve_python_bin)"; then
    echo "Unable to locate a policy Python executable. Set RLDX_POLICY_PYTHON_BIN or pass a valid environment path." >&2
    exit 1
fi

POLICY_SITE_PACKAGES="${RLDX_POLICY_SITE_PACKAGES:-${RLDX_SITE_PACKAGES:-${DEFAULT_POLICY_SITE_PACKAGES}}}"
export RLDX_POLICY_SITE_PACKAGES="${POLICY_SITE_PACKAGES}"
policy_pythonpath="${XPL_ROOT}:${RLDX_POLICY_ROOT}:${POLICY_SITE_PACKAGES}:${UPSTREAM_ROOT}"
if [[ -n "${BRIDGE_ROOT}" ]]; then
    policy_pythonpath="${policy_pythonpath}:${BRIDGE_ROOT}/src"
fi
if [[ ! -d "${POLICY_SITE_PACKAGES}" ]]; then
    echo "[SERVER][WARN] policy site-packages not found: ${POLICY_SITE_PACKAGES}" >&2
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    policy_pythonpath="${policy_pythonpath}:${PYTHONPATH}"
fi

echo "[SERVER] XPL_ROOT=${XPL_ROOT}"
echo "[SERVER] PYTHONPATH=${policy_pythonpath}"

# EgoVLA keeps the private origin gate.  DexBench starts the policy server
# directly so xspark-0-simeval does not need the EgoVLA bridge package.
if [[ "${bench_name}" == "EgoVLA" ]]; then
    if ! env \
        RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_ROOT}" \
        EGOVLA_POLICY_ADAPTER_ROOT="${XPL_ROOT}" \
        EGOVLA_POLICY_NAME="${policy_name}" \
        RLDX_UPSTREAM_ROOT="${UPSTREAM_ROOT}" \
        EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}" \
        EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}" \
        RLDX_PRIVATE_MANIFEST="${PRIVATE_MANIFEST}" \
        RLDX_BOOTSTRAP_REQUIRE_RLDX=1 \
        PYTHONPATH="${policy_pythonpath}" \
        CUDA_VISIBLE_DEVICES="" \
        JAX_PLATFORMS=cpu \
        "${POLICY_PYTHON_BIN}" "${BRIDGE_BOOTSTRAP}" --check-only \
        </dev/null; then
        echo "[SERVER][ERROR] private Python import/origin preflight failed; refusing to start." >&2
        exit 1
    fi
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"
echo "[SERVER] python=${POLICY_PYTHON_BIN}"
echo "[SERVER] site-packages=${POLICY_SITE_PACKAGES}"
echo "[SERVER] HF_HOME=${HF_HOME}"
echo "[SERVER] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[SERVER] HF snapshot=${selected_hf_snapshot}"
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    HF_HOME="${HF_HOME}" \
    HF_HUB_CACHE="${HF_HUB_CACHE}" \
    HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
    HF_ASSETS_CACHE="${HF_ASSETS_CACHE}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    RLDX_XPOLICYLAB_PACKAGE_ROOT="${XPL_ROOT}" \
    EGOVLA_POLICY_ADAPTER_ROOT="${XPL_ROOT}" \
    EGOVLA_POLICY_NAME="${policy_name}" \
    RLDX_UPSTREAM_ROOT="${UPSTREAM_ROOT}" \
    EGOVLA_WORKSPACE_ROOT="${BENCH_ROOT}" \
    EGOVLA_BRIDGE_ROOT="${BRIDGE_ROOT}" \
    PYTHONPATH="${policy_pythonpath}" \
    "${POLICY_PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${policy_server_port}" \
        host="${policy_server_host}" \
        bench_name="${bench_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}"
