"""Immutable H-RDT EE58 checkpoint sidecars, independent of live training data."""
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

try:
    from .ee58_data import sha256, text_hash
except ImportError:
    from ee58_data import sha256, text_hash

SCHEMA = "hrdt-ee58-row6d-q99-rgb-h16-v1"
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def write_weight_binding(directory, root):
    """Run after native save_pretrained completes; absent binding means incomplete."""
    directory, root = Path(directory), Path(root)
    files = [path for name in WEIGHTS if (path := directory / name).is_file()]
    if not files:
        raise FileNotFoundError("Native save produced no supported full model weights")
    value = {"runtime_contract_sha256": sha256(root / "ee58_runtime.json"),
             "weights": {path.name: sha256(path) for path in files}}
    target = directory / "ee58_weight_binding.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(target)


def validate_weight_binding(path, root):
    path, root = Path(path), Path(root)
    value = json.loads((path.parent / "ee58_weight_binding.json").read_text())
    if value["runtime_contract_sha256"] != sha256(root / "ee58_runtime.json"):
        raise ValueError("Model weights are bound to a different EE58 runtime contract")
    if value["weights"].get(path.name) != sha256(path):
        raise ValueError("Model weight digest mismatch")


def write_sidecars(output, config, spec, language_cache, *, test_fixture=False):
    """Called only for a fresh run, or an explicit test temporary directory."""
    output, language_cache = Path(output), Path(language_cache)
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "ee58_stats.json"
    tasks_path = output / "ee58_tasks.json"
    stats_path.write_text(json.dumps({key: spec["stats"][key] for key in ("observation.state", "action")}, indent=2)+"\n")
    tasks_path.write_text(json.dumps(spec["tasks"], ensure_ascii=False, indent=2)+"\n")
    copied = output / "ee58_language"
    shutil.copytree(language_cache, copied)
    manifest = json.loads((copied / "manifest.json").read_text())
    # Training workers also use frozen metadata; only large numeric/video data
    # are linked to the source. Inference never needs this dataset directory.
    private = output / "ee58_dataset"
    (private / "meta").mkdir(parents=True)
    for name in ("info.json", "action_contract.json", "stats.json", "tasks.jsonl", "episodes.jsonl",
                 "video_offsets.json", "modality.json", "source_fingerprint.json"):
        source = spec["root"] / "meta" / name
        if source.is_file():
            shutil.copy2(source, private / "meta" / name)
    for name in ("data", "videos"):
        (private / name).symlink_to(spec["root"] / name, target_is_directory=True)
    config["ee58_contract"]["source_dataset_root"] = str(spec["root"])
    config["ee58_contract"]["dataset_root"] = str(private)
    config["ee58_contract"]["language_cache"] = str(copied)
    launch_path = output / "launch_config.yaml"
    launch_path.write_text(yaml.safe_dump(config, sort_keys=False))
    contract = {
        "schema": SCHEMA, "source_contract_id": spec["contract"]["id"],
        "rotation_6d": "matrix_rows_0_then_1", "native_dim": 58, "wire_dim": 54,
        "action_horizon": 16, "inference_steps": 5, "image_size": 384,
        "image_resize": "letterbox", "image_color_order": "rgb",
        "camera_order": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        "normalization": "independent_state_action_q01_q99_clip_minus1_1",
        "stats_sha256": sha256(stats_path), "tasks_sha256": sha256(tasks_path),
        "source_tasks_sha256": manifest["dataset_tasks_sha256"],
        "language_manifest_sha256": sha256(copied / "manifest.json"),
        "launch_config_sha256": sha256(launch_path),
        "source_metadata_sha256": config["ee58_contract"]["metadata_sha256"],
        "optimizer_updates": 60000, "global_batch_size": 128,
        "test_fixture": test_fixture,
    }
    (output / "ee58_runtime.json").write_text(json.dumps(contract, indent=2)+"\n")
    validate_bundle(output, allow_test_fixture=test_fixture)
    return contract


def run_root(path):
    path = Path(path).resolve()
    start = path.parent if path.is_file() else path
    for candidate in (start, *list(start.parents)[:3]):
        if (candidate / "ee58_runtime.json").is_file():
            return candidate
    raise FileNotFoundError(f"Missing immutable EE58 runtime contract near {path}; refusing legacy/default 14D config")


def select_weights(path, root):
    path = Path(path).resolve()
    if path.is_file():
        if path.name not in WEIGHTS:
            raise ValueError("Expected full native policy weights, not optimizer/ZeRO shards")
        return path
    for name in WEIGHTS:
        if (path / name).is_file() and (path / "ee58_weight_binding.json").is_file():
            return path / name
    candidates = sorted((p for p in path.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
                        key=lambda p: int(p.name[11:]), reverse=True)
    for directory in candidates:
        for name in WEIGHTS:
            if (directory / name).is_file() and (directory / "ee58_weight_binding.json").is_file():
                return directory / name
    raise FileNotFoundError(f"No complete native model weights under {path}")


def validate_bound_language(cache, tasks, contract, *, allow_test_fixture=False):
    import torch
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    if sha256(manifest_path) != contract["language_manifest_sha256"]:
        raise ValueError("Checkpoint language manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_fixture") and not allow_test_fixture:
        raise ValueError("CPU test language fixtures are forbidden in production")
    if manifest["dataset_tasks_sha256"] != contract["source_tasks_sha256"]:
        raise ValueError("Bound instruction source digest mismatch")
    if manifest.get("feature_dim") != 4096 or manifest.get("max_length") != 1024:
        raise ValueError("Expected T5-XXL 4096-feature embeddings")
    if set(manifest["texts"]) != {str(task["task_index"]) for task in tasks} or len(tasks) != 60:
        raise ValueError("Every exact instruction variant must be bound")
    embeddings = {}
    for task in tasks:
        record = manifest["texts"][str(task["task_index"])]
        path = cache / record["file"]
        if record["text_sha256"] != text_hash(task["task"]) or sha256(path) != record["sha256"]:
            raise ValueError("Instruction text/embedding digest mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        value = payload["embeddings"]
        if payload["instruction"] != task["task"] or value.ndim != 2 or value.shape[1] != 4096:
            raise ValueError("Invalid bound instruction embedding")
        if not (0 < len(value) <= 1024) or not torch.isfinite(value).all():
            raise ValueError("Invalid T5 sequence or non-finite values")
        embeddings[task["task"]] = value
    return embeddings


def validate_bundle(root, *, allow_test_fixture=False):
    root = Path(root)
    contract = json.loads((root / "ee58_runtime.json").read_text())
    if contract.get("schema") != SCHEMA or contract["rotation_6d"] != "matrix_rows_0_then_1":
        raise ValueError("Wrong EE58 runtime schema/rotation convention")
    if contract.get("test_fixture") and not allow_test_fixture:
        raise ValueError("CPU oracle checkpoint fixtures are forbidden in production")
    expected = {"native_dim": 58, "wire_dim": 54, "action_horizon": 16, "image_size": 384,
                "image_color_order": "rgb", "image_resize": "letterbox"}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint serving dimensions/preprocessing contract mismatch")
    if contract.get("camera_order") != ["cam_head", "cam_left_wrist", "cam_right_wrist"]:
        raise ValueError("Checkpoint camera order must be head, left wrist, right wrist")
    if contract.get("source_contract_id") != "dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1":
        raise ValueError("Checkpoint is not bound to the reviewed absolute-world EE58 source")
    for name, field in (("ee58_stats.json", "stats_sha256"), ("ee58_tasks.json", "tasks_sha256"),
                        ("launch_config.yaml", "launch_config_sha256")):
        if sha256(root / name) != contract[field]:
            raise ValueError(f"Checkpoint sidecar digest mismatch: {name}")
    config = yaml.safe_load((root / "launch_config.yaml").read_text())
    stats = json.loads((root / "ee58_stats.json").read_text())
    tasks = json.loads((root / "ee58_tasks.json").read_text())
    if config["common"]["state_dim"] != config["common"]["action_dim"] or config["common"]["action_dim"] != 58:
        raise ValueError("Checkpoint must retain compact58 model interfaces")
    if config["model"]["hrdt"]["output_size"] != 58 or config["common"]["action_chunk_size"] != 16:
        raise ValueError("Checkpoint must predict H16 / compact58")
    for key in ("observation.state", "action"):
        for stat in ("q01", "q99"):
            value = np.asarray(stats[key][stat])
            if value.shape != (58,) or not np.isfinite(value).all():
                raise ValueError("Invalid checkpoint normalization snapshot")
    embeddings = validate_bound_language(root / "ee58_language", tasks, contract, allow_test_fixture=allow_test_fixture)
    return {"root": root, "contract": contract, "config": config, "stats": stats,
            "tasks": tasks, "embeddings": embeddings}
