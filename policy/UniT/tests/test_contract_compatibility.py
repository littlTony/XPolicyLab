"""Exercise the real metadata validator without loading GPU model packages."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _extract_model_method(name: str):
    source = Path(__file__).resolve().parents[1] / "model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "Path": Path, "json": json, "os": os,
        "_EGOVLA_CAMERA_NAMES": ("cam_head", "cam_left_wrist", "cam_right_wrist"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def validate():
    return _extract_model_method("_validate_egovla_contract")


@pytest.fixture
def validate_schema():
    return _extract_model_method("_validate_checkpoint_schema")


@pytest.fixture
def metadata(tmp_path):
    checkpoint = tmp_path / "checkpoint-60000"
    checkpoint.mkdir()
    contract = {
        "format": "xpolicylab-egovla-contract", "status": "complete",
        "run": {"env_cfg_type": "ego_h1_inspire", "action_type": "joint", "checkpoint": str(checkpoint)},
        "robot": {"robot_key": "ego_h1_inspire", "action_type": "joint", "action_dim": 38,
                  "native_dim": 50, "arm_dim": [7, 7], "hand_dim": [12, 12]},
        "camera": {"mode": "three_view_mixed", "order": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                   "features": ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
                   "mask_feature": "observation.camera_mask"},
        "data": {"active_episodes": 5000, "active_frames": 1000000, "total_videos": 15000, "deprecated_excluded": 0},
    }
    instance = SimpleNamespace(model_cfg={}, env_cfg_type="ego_h1_inspire")
    return instance, checkpoint, contract


def run_validation(validate, metadata):
    instance, checkpoint, contract = metadata
    (checkpoint / "egovla_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    validate(instance, checkpoint)


def test_new_dataset_and_checkpoint_do_not_require_adapter_edits(validate, metadata):
    run_validation(validate, metadata)


def test_inference_does_not_require_dataset_audit_statistics(validate, metadata):
    del metadata[2]["data"]
    run_validation(validate, metadata)


@pytest.mark.parametrize("section,key,value", [
    ("robot", "hand_dim", [20, 20]),
    ("robot", "action_dim", 54),
    ("run", "action_type", "ee"),
    ("camera", "order", ["cam_head", "cam_right_wrist", "cam_left_wrist"]),
    ("camera", "mode", "head_only"),
    ("run", "checkpoint", "/stale/other/checkpoint-30000"),
])
def test_incompatible_runtime_schema_still_fails(validate, metadata, section, key, value):
    metadata[2][section][key] = value
    with pytest.raises(ValueError, match="incompatible"):
        run_validation(validate, metadata)


@pytest.mark.parametrize("step", range(10000, 80001, 10000))
def test_training_run_profile_supports_each_checkpoint_without_rewriting_provenance(validate, metadata, step):
    instance, old_checkpoint, contract = metadata
    run_root = old_checkpoint.parent / "training_run"
    checkpoint = run_root / "pretrain" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    contract["run"].update(run_root=str(run_root), checkpoint=str(checkpoint.parent / "checkpoint-80000"))
    contract["checkpoint_artifacts"] = [{"path": "model.safetensors", "sha256": "80k-artifact-provenance"}]
    sidecar = run_root / "egovla_contract.json"
    before = json.dumps(contract)
    sidecar.write_text(before, encoding="utf-8")
    validate(instance, checkpoint)
    assert sidecar.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("problem", ["different_run", "different_stage", "missing_run_root", "wrong_run_root"])
def test_shared_profile_does_not_apply_to_unrelated_weights(validate, metadata, problem):
    instance, old_checkpoint, contract = metadata
    run_root = old_checkpoint.parent / "training_run"
    checkpoint = run_root / "pretrain" / "checkpoint-10000"
    checkpoint.mkdir(parents=True)
    contract["run"].update(run_root=str(run_root), checkpoint=str(checkpoint.parent / "checkpoint-80000"))
    if problem == "different_run":
        contract["run"]["checkpoint"] = str(run_root.parent / "other_run" / "pretrain" / "checkpoint-80000")
    elif problem == "different_stage":
        contract["run"]["checkpoint"] = str(run_root / "finetune" / "checkpoint-80000")
    elif problem == "missing_run_root":
        del contract["run"]["run_root"]
    else:
        contract["run"]["run_root"] = str(run_root.parent)
    (run_root / "egovla_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="run.checkpoint"):
        validate(instance, checkpoint)


def test_checkpoint_local_contract_remains_bound_to_that_checkpoint(validate, metadata):
    instance, checkpoint, contract = metadata
    contract["run"].update(run_root=str(checkpoint), checkpoint=str(checkpoint.parent / "checkpoint-80000"))
    with pytest.raises(ValueError, match="run.checkpoint"):
        run_validation(validate, metadata)


def test_shared_profile_supports_symlinked_latest(validate, metadata):
    instance, old_checkpoint, contract = metadata
    run_root = old_checkpoint.parent / "training_run"
    checkpoint = run_root / "pretrain" / "checkpoint-10000"
    checkpoint.mkdir(parents=True)
    latest = checkpoint.parent / "latest"
    latest.symlink_to(checkpoint, target_is_directory=True)
    contract["run"].update(run_root=str(run_root), checkpoint=str(checkpoint.parent / "checkpoint-80000"))
    (run_root / "egovla_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    validate(instance, latest)


_SPARK_JOINT54 = {"left_arm": [7], "left_hand": [20], "right_arm": [7], "right_hand": [20]}
_REAL_UNIT_0908_7TASK = (
    Path(__file__).resolve().parents[1]
    / "checkpoints_0908_7task"
    / "policy_0908_7task"
    / "checkpoint-60000"
)


def _spark_schema_instance() -> SimpleNamespace:
    return SimpleNamespace(
        embodiment_tag="new_embodiment",
        is_egovla=False,
        is_egovla_threeview=False,
        arm_dims=(7, 7),
        hand_dims=(20, 20),
    )


def _write_spark_metadata(checkpoint: Path, groups: dict[str, list[int]]) -> None:
    modalities = {
        kind: {
            key: {"shape": shape, "absolute": True, "continuous": True}
            for key, shape in groups.items()
        }
        for kind in ("state", "action")
    }
    metadata = {"new_embodiment": {"modalities": modalities}}
    cfg = checkpoint / "experiment_cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_spark_accepts_left_first_0908_metadata_order(validate_schema, tmp_path):
    checkpoint = tmp_path / "checkpoint-60000"
    _write_spark_metadata(checkpoint, _SPARK_JOINT54)
    validate_schema(_spark_schema_instance(), checkpoint)


def test_spark_accepts_right_first_metadata_order(validate_schema, tmp_path):
    checkpoint = tmp_path / "checkpoint-60000"
    _write_spark_metadata(
        checkpoint,
        {"right_arm": [7], "right_hand": [20], "left_arm": [7], "left_hand": [20]},
    )
    validate_schema(_spark_schema_instance(), checkpoint)


def test_spark_real_0908_7task_checkpoint_passes_schema(validate_schema):
    assert _REAL_UNIT_0908_7TASK.is_dir(), _REAL_UNIT_0908_7TASK
    validate_schema(_spark_schema_instance(), _REAL_UNIT_0908_7TASK)


def test_spark_still_rejects_wrong_hand_dim(validate_schema, tmp_path):
    checkpoint = tmp_path / "checkpoint-60000"
    groups = dict(_SPARK_JOINT54)
    groups["left_hand"] = [12]
    _write_spark_metadata(checkpoint, groups)
    with pytest.raises(ValueError, match="Spark joint54"):
        validate_schema(_spark_schema_instance(), checkpoint)


def test_spark_still_rejects_missing_group(validate_schema, tmp_path):
    checkpoint = tmp_path / "checkpoint-60000"
    groups = dict(_SPARK_JOINT54)
    del groups["left_hand"]
    _write_spark_metadata(checkpoint, groups)
    with pytest.raises(ValueError, match="Spark joint54"):
        validate_schema(_spark_schema_instance(), checkpoint)


def test_0908_7task_path_selects_spark_threeview():
    source = Path(__file__).resolve().parents[1] / "model.py"
    namespace: dict[str, object] = {"Path": Path}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        names = {getattr(node, "name", None)}
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & {"_SPARK_THREEVIEW_PATH_MARKERS", "spark_threeview_from_path"}:
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    detect = namespace["spark_threeview_from_path"]
    assert detect(_REAL_UNIT_0908_7TASK)
    assert detect("/personal/xiangpc/0811_Xpolicylab_bench/UniT/policy/UniT/checkpoints_0908-SparkArena/policy_0908-SparkArena/checkpoint-40000")
    assert not detect("/personal/xiangpc/0811_Xpolicylab_bench/UniT/policy/UniT/checkpoints/Spark0_bench-unit_20260813")
