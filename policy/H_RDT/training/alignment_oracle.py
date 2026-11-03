"""CPU serving oracle. Fixture vectors and BOTH networks are explicit test stubs.

build_oracle_fixture(tmpdir) returns (real public Model, obs, expected_actions).
It does not create a real checkpoint or generate genuine T5 embeddings.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from .ee58_data import EE58Backend, POLICY, normalize, sha256, text_hash
from .native_assets import build_vision
from .runtime_bundle import write_sidecars, write_weight_binding
from XPolicyLab.utils.dexbench_ee58_contract import unpack_ee58
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state


class RecordingVision(torch.nn.Module):
    embed_dim, num_patches = 2176, 729

    def forward(self, inputs):
        self.last_inputs = {key: value.detach().cpu().clone() for key, value in inputs.items()}
        value = next(iter(inputs.values()))
        return torch.zeros((len(value), 729, 2176), device=value.device, dtype=value.dtype)


class OraclePolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.register_buffer("oracle_action", torch.as_tensor(action)[None])

    def predict_action(self, **inputs):
        self.last_inputs = {key: value.detach().cpu().clone() for key, value in inputs.items()}
        if inputs["state_tokens"].shape != (1, 1, 58) or inputs["lang_tokens"].shape[-1] != 4096:
            raise AssertionError("Serving native condition shape mismatch")
        return self.oracle_action.clone()


def build_oracle_fixture(tmpdir):
    from ..model import Model
    from ..ee58_runtime import denormalize_action
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    backend = EE58Backend(require_language=False)
    spec = backend.spec
    state, action, task_index = backend.raw_step(0, 7)
    images, _ = backend.rgb_frames(0, 7)
    normalized_action = normalize(action, spec["stats"]["action"])
    language = tmpdir / "language_fixture_NOT_T5"
    language.mkdir()
    records = {}
    for task in spec["tasks"]:
        value = torch.linspace(0, 1, 8192).reshape(2, 4096) + task["task_index"]
        path = language / f"task_{task['task_index']:03d}.pt"
        torch.save({"instruction": task["task"], "embeddings": value}, path)
        records[str(task["task_index"])] = {"file": path.name, "text_sha256": text_hash(task["task"]), "sha256": sha256(path)}
    manifest = {"test_fixture": True, "not_real_T5": True, "dataset_tasks_sha256": sha256(spec["root"] / "meta/tasks.jsonl"),
                "feature_dim": 4096, "max_length": 1024, "texts": records}
    (language / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    config = yaml.safe_load((POLICY / "configs/hrdt_dexbench_ee58.yaml").read_text())
    output = tmpdir / "hrdt_ee58_fixture_NOT_TRAINED"
    vision_root = "/personal/xiangpc/training_hrdt_sparkarena_samplerfix_0921/H_RDT/bak/dino-siglip"
    config["ee58_contract"] = {"dataset_root": str(spec["root"]), "language_cache": str(language),
                               "vision_root": vision_root, "output_dir": str(output), "run_id": "tianji_wuji",
                               "metadata_sha256": {name: sha256(spec["root"] / "meta" / name) for name in
                                                   ("info.json", "action_contract.json", "stats.json", "tasks.jsonl", "episodes.jsonl", "video_offsets.json", "modality.json")},
                               "optimizer_updates": 60000, "global_batch_size": 128, "micro_batch": 16,
                               "gradient_accumulation": 1, "seed": 42}
    write_sidecars(output, config, spec, language, test_fixture=True)
    torch.save({"test_fixture_network_stub": True}, output / "pytorch_model.bin")
    write_weight_binding(output, output)
    transform = build_vision(vision_root, transform_only=True)
    vision, policy = RecordingVision(), OraclePolicy(normalized_action)
    cfg = {"checkpoint_path": str(output), "bench_name": "DexBench", "ckpt_name": "tianji_wuji",
           "env_cfg_type": "tianji_marvin_wuji", "action_type": "ee", "seed": 42,
           "device": "cpu", "dtype": "float32", "allow_dummy_lang_embedding": False}
    model = Model(cfg, _test_components={"vision_encoder": vision, "image_transform": transform, "policy": policy})
    dims = get_robot_action_dim_info("tianji_marvin_wuji")
    task = next(task for task in spec["tasks"] if task["task_index"] == task_index)
    obs = {"state": unpack_robot_state(unpack_ee58(state[0]), "ee", dims, source_type="obs"),
           "vision": {key: {"color": image[0].copy()} for key, image in zip(
               ("cam_head", "cam_left_wrist", "cam_right_wrist"), images)}, "instruction": task["task"]}
    expected = unpack_robot_state(unpack_ee58(denormalize_action(normalized_action, spec["stats"]["action"])),
                                  "ee", dims, source_type="obs")
    model._oracle = {"backend": backend, "raw_state": state, "normalized_state": normalize(state, spec["stats"]["observation.state"]),
                     "normalized_action": normalized_action, "images": images, "cfg": cfg,
                     "fixture_root": output, "network_stubs": True, "language_fixture_not_T5": True}
    return model, obs, expected


def run_oracle(tmpdir):
    from ..model import Model
    from ..ee58_runtime import denormalize_action
    model, obs, expected = build_oracle_fixture(tmpdir)
    model.update_obs(obs)
    np.testing.assert_allclose(model._impl.observations[0]["state"], model._oracle["normalized_state"], rtol=1e-4, atol=1e-4)
    actions = model.get_action()
    for action, reference in zip(actions, expected):
        for key in reference:
            np.testing.assert_allclose(action[key], reference[key], rtol=1e-6, atol=1e-6)
    # The training Consumer and runtime both call this SAME native transform.
    from PIL import Image
    training_pixels = [model._impl.image_transform(Image.fromarray(image[0])) for image in model._oracle["images"]]
    for key in training_pixels[0]:
        expected_pixels = torch.stack([value[key] for value in training_pixels])
        torch.testing.assert_close(model._impl.vision_encoder.last_inputs[key], expected_pixels, rtol=0, atol=0)
    model.update_obs_batch([deepcopy(obs), deepcopy(obs)], [0, 1])
    batch = model.get_action_batch([0, 1])
    assert len(batch) == 2 and all(len(chunk) == 16 for chunk in batch)
    for chunk in batch:
        for action, reference in zip(chunk, expected):
            for key in reference:
                np.testing.assert_allclose(action[key], reference[key], rtol=1e-6, atol=1e-6)
    # The production constructor must reject explicit CPU fixture checkpoints.
    try:
        Model(model._oracle["cfg"])
    except ValueError as error:
        assert "fixture" in str(error)
    else:
        raise AssertionError("Production accepted a CPU oracle fixture")
    bad = deepcopy(obs)
    bad["instruction"] = "unbound text"
    try:
        model.update_obs(bad)
    except ValueError as error:
        assert "instruction" in str(error)
    else:
        raise AssertionError("Unknown instruction was silently guessed")
    return {"status": "passed", "real_public_Model_update_get": True, "state_norm": "independent checkpoint state stats",
            "action_oracle": "normalized GT -> checkpoint action denorm -> row6D SO3 -> wire54 dictionaries",
            "cameras": ["head", "left_wrist", "right_wrist"], "RGB384_letterbox_pixels": "exactly equal to training transform",
            "batch_envs": 2, "chunk": 16, "native_dim": 58, "wire_dim": 54,
            "network_stubs": ["vision encoder outputs", "GT policy action output"], "T5_cache": "explicit synthetic test fixture, not genuine T5",
            "GPU_used": False, "real_weights_inference": "not tested", "production_fixture_guard": "passed"}
