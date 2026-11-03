"""Checkpoint-owned FastWAM inference contract (independent of Web certification).

Export once with: python checkpoint_runtime.py export /path/to/run/config.yaml
The adapter reads this file automatically for ego_h1_inspire checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml

CONTRACT_NAME = "fastwam_inference_contract.json"
SCHEMA = "fastwam-inference-v1"
CAMERA_KEYS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(path):
    value = json.loads(Path(path).read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def canonical_task(name):
    value = str(name).strip()
    if value.startswith("Humanoid-") and value.endswith("-v0"):
        value = value[len("Humanoid-"):-len("-v0")]
    return value.replace("_", "-").lower()


def training_config_path(checkpoint):
    weight = Path(checkpoint).expanduser().resolve(strict=True)
    if not weight.is_file():
        raise ValueError(f"FastWAM needs a checkpoint file, got {weight}")
    for parent in list(weight.parents)[:3]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No checkpoint-owned config.yaml beside {weight}; refusing deployment defaults")


def validate_training_config(cfg, action_dim):
    dim = int(action_dim)
    if dim != 38:
        raise ValueError(f"EgoVLA H1 requires 38 logical joints, got {dim}")
    for split in ("train", "val"):
        data = cfg["data"][split]
        if data["concat_multi_camera"] != "robotwin":
            raise ValueError("FastWAM EgoVLA contract requires robotwin camera mosaic")
        if [v["key"] for v in data["shape_meta"]["images"]] != CAMERA_KEYS:
            raise ValueError("Checkpoint camera ordering does not match the FastWAM adapter")
        for meta in (data["shape_meta"], data["processor"]["shape_meta"]):
            for section in ("action", "state"):
                if len(meta[section]) != 1 or meta[section][0]["key"] != "default":
                    raise ValueError("Expected a single merged state/action vector")
                if any(int(meta[section][0][key]) != dim for key in ("shape", "raw_shape")):
                    raise ValueError("Checkpoint state/action dimensions disagree with the robot")
        for key in ("action_output_dim", "proprio_output_dim"):
            if int(data["processor"][key]) != dim:
                raise ValueError(f"Checkpoint processor {key} disagrees with the robot")
        if int(data["global_sample_stride"]) != 1:
            raise ValueError("FastWAM adapter does not support a strided action timeline")
        if len(data["video_size"]) != 2 or min(map(int, data["video_size"])) <= 0:
            raise ValueError("Invalid checkpoint video_size")
    model = cfg["model"]
    if any(int(v) != dim for v in (model["proprio_dim"], model["action_dit_config"]["action_dim"], model["video_dit_config"]["action_dim"])):
        raise ValueError("Checkpoint model head dimensions disagree with the robot")
    train = cfg["data"]["train"]
    if train["video_size"] != cfg["data"]["val"]["video_size"]:
        raise ValueError("Training and validation image geometry disagree")
    if int(train["num_frames"]) < 2 or int(train["action_video_freq_ratio"]) <= 0:
        raise ValueError("Invalid checkpoint temporal horizon")


def dataset_provenance(cfg):
    """Load the audited conversion manifests and derive per-task camera modes."""
    roots = cfg["data"]["train"]["dataset_dirs"]
    if len(roots) != 1:
        raise ValueError("FastWAM EgoVLA contract requires one audited dataset")
    metadata = Path(roots[0]).expanduser().resolve() / "meta"
    conversion_path = metadata / "fastwam_conversion_manifest.json"
    source_path = metadata / "xpolicylab_source_conversion.json"
    conversion = json.loads(conversion_path.read_text())
    source = json.loads(source_path.read_text())
    if conversion.get("converter") != "convert_egovla_fastwam.py" or int(conversion["action_dim"]) != 38:
        raise ValueError("Dataset is not the audited FastWAM commanded-action conversion")
    if conversion.get("wrist_policy") != "missing wrist videos stay black placeholders; camera_mask remains false":
        raise ValueError("Unknown missing-wrist representation; refusing to guess")
    modes = {}
    for episode in source["episodes"]:
        task = canonical_task(episode["task"])
        real = episode["has_real_wrist_cameras"]
        if type(real) is not bool:
            raise ValueError("Camera provenance must be boolean")
        mode = "real_wrist" if real else "black_wrist"
        if task in modes and modes[task] != mode:
            raise ValueError(f"Mixed camera modes inside task {task}; needs an explicit policy")
        modes[task] = mode
    if set(modes) != {canonical_task(t) for t in source["task_episode_counts"]}:
        raise ValueError("Incomplete task camera provenance")
    return conversion_path, source_path, modes


def build_contract(config_path, checkpoint_names=None):
    path = Path(config_path).expanduser().resolve(strict=True)
    cfg = yaml.safe_load(path.read_text())
    validate_training_config(cfg, 38)
    conversion_path, source_path, modes = dataset_provenance(cfg)
    stats = Path(cfg["data"]["train"]["pretrained_norm_stats"]).expanduser().resolve(strict=True)
    weights_root = path.parent / "checkpoints" / "weights"
    if checkpoint_names:
        weights = [(weights_root / name).resolve(strict=True) for name in checkpoint_names]
        for weight in weights:
            if weight.parent != weights_root.resolve() or not weight.is_file():
                raise ValueError(f"Checkpoint selector must name a direct .pt child: {weight}")
    else:
        weights = sorted(weights_root.glob("*.pt"))
    if not weights:
        raise FileNotFoundError("No checkpoint artifacts found to bind")
    artifacts = {}
    for weight in weights:
        print(f"[FastWAM contract] hashing {weight.name}", flush=True)
        before = weight.stat()
        sha = file_sha256(weight)
        after = weight.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint changed while hashing: {weight}")
        artifacts[weight.relative_to(path.parent).as_posix()] = {"sha256": sha, "size": after.st_size}
    return {
        "schema": SCHEMA, "robot": "ego_h1_inspire", "action_type": "joint", "action_dim": 38,
        "training_config_sha256": file_sha256(path),
        "dataset_stats_json_sha256": json_sha256(stats), "training_stats_path": str(stats),
        "source_manifest_sha256": file_sha256(source_path),
        "conversion_manifest_sha256": file_sha256(conversion_path),
        "camera_modes": modes, "video_size": cfg["data"]["train"]["video_size"],
        "action_horizon": int(cfg["data"]["train"]["num_frames"]) - 1,
        "checkpoints": artifacts,
    }


def resolve_runtime(checkpoint, stats_path, task_name, action_dim):
    config_path = training_config_path(checkpoint)
    contract_path = config_path.parent / CONTRACT_NAME
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing {contract_path}; export the checkpoint-owned contract with checkpoint_runtime.py")
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != SCHEMA or contract.get("robot") != "ego_h1_inspire" or contract.get("action_type") != "joint" or contract.get("action_dim") != int(action_dim):
        raise ValueError("FastWAM checkpoint contract does not match the selected robot/action")
    if file_sha256(config_path) != contract["training_config_sha256"]:
        raise ValueError("FastWAM training config changed after contract export")
    cfg = yaml.safe_load(config_path.read_text())
    validate_training_config(cfg, action_dim)
    conversion_path, source_path, camera_modes = dataset_provenance(cfg)
    if (file_sha256(source_path) != contract["source_manifest_sha256"] or
            file_sha256(conversion_path) != contract["conversion_manifest_sha256"]):
        raise ValueError("FastWAM dataset provenance manifests changed after contract export")
    if camera_modes != contract["camera_modes"]:
        raise ValueError("FastWAM sidecar camera modes disagree with dataset provenance")
    if cfg["data"]["train"]["video_size"] != contract["video_size"] or int(cfg["data"]["train"]["num_frames"]) - 1 != contract["action_horizon"]:
        raise ValueError("FastWAM inference contract geometry/horizon is inconsistent")
    if json_sha256(stats_path) != contract["dataset_stats_json_sha256"]:
        raise ValueError("FastWAM runtime normalization does not match checkpoint training statistics")
    weight = Path(checkpoint).expanduser().resolve(strict=True)
    key = weight.relative_to(config_path.parent).as_posix()
    artifact = contract["checkpoints"].get(key)
    if not artifact or weight.stat().st_size != artifact["size"] or file_sha256(weight) != artifact["sha256"]:
        raise ValueError(f"Checkpoint is missing from, or does not match, its FastWAM contract: {key}")
    task = canonical_task(task_name)
    camera_mode = contract["camera_modes"].get(task)
    if camera_mode not in {"real_wrist", "black_wrist"}:
        raise ValueError(f"Unknown EgoVLA task/camera contract: {task_name!r}")
    summary = {"contract": str(contract_path), "contract_sha256": file_sha256(contract_path),
               "checkpoint": str(weight), "checkpoint_sha256": artifact["sha256"],
               "training_config": str(config_path), "training_config_sha256": contract["training_config_sha256"],
               "stats": str(Path(stats_path).resolve()), "stats_json_sha256": contract["dataset_stats_json_sha256"],
               "task": task, "camera_mode": camera_mode, "video_size": contract["video_size"],
               "action_dim": int(action_dim), "action_horizon": contract["action_horizon"]}
    return cfg, camera_mode, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export"])
    parser.add_argument("config_path")
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="checkpoint filename to bind; repeat as needed (default: every .pt artifact)",
    )
    args = parser.parse_args()
    path = Path(args.config_path).resolve().parent / CONTRACT_NAME
    if path.exists():
        raise FileExistsError(f"Contract already exists; review before replacing: {path}")
    value = build_contract(args.config_path, args.checkpoint)
    fd, temporary = tempfile.mkstemp(prefix=CONTRACT_NAME + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"[FastWAM contract] exported {path}")


if __name__ == "__main__":
    main()
