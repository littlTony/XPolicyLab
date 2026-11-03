"""CPU oracle: actual Model/preprocessing/decoding, only the neural predictor stubbed."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

POLICY = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(POLICY.parents[2]), str(POLICY / "UniT")]
from XPolicyLab.policy.UniT.ee58_runtime import write_inference_bundle, digest
from XPolicyLab.policy.UniT.model import Model
from XPolicyLab.policy.UniT.training.ee58_data import EE58DataConfig, EE58Dataset, PARTS, CAMERAS
from XPolicyLab.utils.dexbench_ee58_contract import DEFAULT_DATASET, CONTRACT_ID, validate_dataset, unpack_ee58
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state
from gr00t.model.transforms import collate


class OracleBackend:
    compute_bridge_loss = False

    def __init__(self, prediction):
        self.set_prediction(prediction)
        self.last_batch = None

    def set_prediction(self, prediction):
        self.prediction = torch.as_tensor(prediction, dtype=torch.float32).detach().clone()

    def get_action(self, batch):
        self.last_batch = batch
        return {"action_pred": self.prediction.expand(batch["state"].shape[0], -1, -1).clone()}


def fixture_dataset(tmpdir):
    source = DEFAULT_DATASET.resolve()
    validated = validate_dataset(source)
    path = Path(tmpdir) / "dataset"
    (path / "meta").mkdir(parents=True)
    for name in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl", "video_offsets.json"):
        shutil.copy2(source / "meta" / name, path / "meta" / name)
    info = deepcopy(validated["info"])
    info["codebase_version"] = "v2.0"
    (path / "meta/info.json").write_text(json.dumps(info))
    modality = deepcopy(validated["modality"])
    for group in ("state", "action"):
        for value in modality[group].values():
            value["dtype"] = "float32"
    (path / "meta/modality.json").write_text(json.dumps(modality))
    stats = {key: {name: validated["stats"][key][name] for name in ("mean", "std", "min", "max", "q01", "q99")}
             for key in ("observation.state", "action")}
    (path / "meta/stats.json").write_text(json.dumps(stats))
    (path / "meta/relative_stats.json").write_text("{}")
    for name in ("data", "videos"):
        (path / name).symlink_to(source / name, target_is_directory=True)
    vlm = "/personal/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct"
    cfg = EE58DataConfig(eagle_path=vlm, use_bridge=True, ignore_lang_prefix=True,
                         num_bridge_tokens=8, tokenizer_only=False)
    transforms = cfg.transform(augment=True)
    dataset = EE58Dataset(dataset_path=path, modality_configs=cfg.modality_config(), transforms=transforms,
                          embodiment_tag="new_embodiment", split="[:]", video_backend="torchvision_av")
    transforms.eval()
    return dataset, validated, vlm


def case(dataset, episode=1, base=7):
    raw = dataset.get_step_data(int(episode), int(base))
    raw_state = np.concatenate([raw[f"state.{key}"] for key in PARTS], axis=1)[0]
    raw_action = np.concatenate([raw[f"action.{key}"] for key in PARTS], axis=1)
    robot = get_robot_action_dim_info("tianji_marvin_wuji")
    obs = {"state": unpack_robot_state(unpack_ee58(raw_state).astype(np.float32), "ee", robot),
           "vision": {wire: {"color": raw[f"video.{key}"][0].copy()} for wire, key in
                      zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), CAMERAS)},
           "instruction": str(raw["annotation.human.task_description"][0])}
    expected = [unpack_robot_state(action.astype(np.float32), "ee", robot) for action in unpack_ee58(raw_action)]
    train_features = dataset.transforms(deepcopy(raw))
    train_batch = collate([train_features], dataset.transforms.transforms[-1].eagle_processor)
    return obs, expected, train_batch


def build_oracle_fixture(tmpdir):
    """Return (real_Model, decoded_RGB_observation, expected_16_action_dicts).

    Only constructor._load_native is patched while constructing this instance;
    the class immediately returns to its production strict loader. Safe for the
    parent to run PolicyServer/WsModelClient plain+encoded RPC smoke afterwards.
    """
    os.environ.setdefault("EVAL_MAIN_ROOT", "/personal/wenwei/xspark-0-simeval")
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    dataset, validated, vlm = fixture_dataset(tmpdir)
    obs, expected, training = case(dataset)
    checkpoint = tmpdir / "oracle_checkpoint"
    (checkpoint / "experiment_cfg").mkdir(parents=True)
    config = json.loads((POLICY / "configs/ee58_policy.json").read_text())
    source = validated["root"]
    config["backbone_cfg"].update(eagle_path=vlm, select_layer=36, tune_llm=True, tune_visual=False)
    config["unit_cfg"].update(bridge_type="vision_lang_obs", compute_bridge_loss=True,
                              groot_tokenizer_path=str(tmpdir / "training_teacher_not_needed_online"))
    config["ee58_contract"] = {"id": CONTRACT_ID, "real_dim": 58, "padded_dim": 128,
                               "rotation": "matrix_rows_0_then_1", "normalization": "mean_std",
                               "source_stats_sha256": digest(source / "meta/stats.json")}
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2))
    metadata = {"new_embodiment": dataset.metadata.model_dump(mode="json")}
    (checkpoint / "experiment_cfg/metadata.json").write_text(json.dumps(metadata, indent=2))
    # Deliberately not a UniT checkpoint. An unpatched production loader will
    # reject this placeholder, never fall back to randomly initialized weights.
    from safetensors.torch import save_file
    save_file({"oracle_predictor_placeholder": torch.zeros(1)}, checkpoint / "model.safetensors")
    processor = dataset.transforms.transforms[-1].eagle_processor
    ids = [processor.tokenizer.convert_tokens_to_ids(f"<|bridge_{index}|>") for index in range(8)]
    prepared = {"source_dataset": str(source), "source_metadata_sha256": {
        name: digest(source / "meta" / name) for name in ("stats.json", "action_contract.json")},
        "vlm_path": vlm, "bridge_ids": ids}
    write_inference_bundle(checkpoint, prepared, 60000)
    backend = OracleBackend(training["action"])
    with patch.object(Model, "_load_native", lambda self, bundle: backend):
        model = Model({"policy_name": "UniT", "bench_name": "DexBench", "ckpt_name": "tianji_wuji",
                       "env_cfg_type": "tianji_marvin_wuji", "action_type": "ee", "seed": 42,
                       "checkpoint_path": str(checkpoint), "device": "cpu", "batch_size": 1,
                       "execute_steps": 16})
    model._oracle_dataset = dataset
    model._oracle_source = validated
    model._oracle_training_batch = training
    return model, obs, expected


def assert_action_close(actual, expected):
    assert len(actual) == len(expected) == 16
    for step, target in zip(actual, expected):
        assert set(step) == set(target)
        for key in target:
            np.testing.assert_allclose(step[key], target[key], atol=3e-6, rtol=3e-6)


def verify_serving_case(model, obs, expected, training):
    model.model.set_prediction(training["action"])
    model.update_obs(obs)
    actual = model.get_action()
    assert_action_close(actual, expected)
    batch = model.model.last_batch
    assert not any(key.startswith("goal_image_") for key in batch)
    assert "imagenet_goal_images" not in batch
    assert bool(batch["state_mask"][..., :58].all()) and not bool(batch["state_mask"][..., 58:].any())
    assert not bool(batch["action_mask"].any())  # No GT action in online input.
    torch.testing.assert_close(batch["state"], training["state"], atol=3e-6, rtol=3e-6)
    for key in ("eagle_pixel_values", "eagle_image_grid_thw", "eagle_input_ids", "eagle_attention_mask"):
        torch.testing.assert_close(batch[key], training[key], atol=0, rtol=0)
    return {"state_shape": list(batch["state"].shape), "action_shape": [1, 16, 128],
            "camera_grid": list(batch["eagle_image_grid_thw"].shape), "teacher_images_used": False}


if __name__ == "__main__":
    import tempfile
    directory = Path(tempfile.mkdtemp(prefix="unit_ee58_serving_oracle_", dir=POLICY / ".generated"))
    model, observation, expected = build_oracle_fixture(directory)
    cases = []
    for task, episodes in model._oracle_source["physical_task_groups"].items():
        for episode in (episodes[0], episodes[-1]):
            observation, expected, training = case(model._oracle_dataset, episode)
            result = verify_serving_case(model, observation, expected, training)
            cases.append({"task": task, "episode": int(episode), **result})
    model.update_obs_batch([observation, observation])
    for actual in model.get_action_batch([4, 9]):
        assert_action_close(actual, expected)
    model.reset()
    assert model._observations == []
    report = {"status": "passed", "sampled_episodes": len(cases), "physical_tasks": 12,
              "predictor": "GT normalized oracle only; no neural weight inference",
              "actual_paths": "Model init/update_obs/get_action/batch/reset + native transforms/collate + sidecar validation + row6D decode",
              "configured_policy_updates": 60000, "configured_tokenizer_updates": 40000, "global_batch": 64,
              "fixture_is_trained_checkpoint": False,
              "GPU_and_large_neural_weights_used": False, "cases": cases}
    (POLICY / "training/ee58_serving_oracle_20261030.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))
