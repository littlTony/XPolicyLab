"""Native QwenPI_v3 EE58 serving through the normal XPolicyLab server.

Checkpoint-local absolute statistics and the saved runtime/code contract are
the only normalization source. No training dataset is opened during serving.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.dexbench_ee58_contract import CONTRACT_ID, pack_ee54, unpack_ee58
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

POLICY = Path(__file__).resolve().parent
UPSTREAM = POLICY / "upstream"
UPSTREAM_COMMIT = "985c2238627481f25476fbee478083812dd82a21"
PARTS = ("left_eef", "left_hand", "right_eef", "right_hand")
WIDTHS = (9, 20, 9, 20)
WIRE_CAMERAS = (("cam_head", "cam_high", "front", "head_camera"),
                ("cam_left_wrist", "left_wrist", "left_camera"),
                ("cam_right_wrist", "right_wrist", "right_camera"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def native_imports():
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(POLICY / "training"))
    from sparkarena_registry import register
    register("dexbench_ee58_v21")


def code_files():
    files = {"policy:" + rel: POLICY / rel for rel in (
        "ee58_runtime.py", "model.py", "training/sparkarena_registry.py", "training/engine_trainer.py", "training/data_pipeline.py")}
    for rel in ("starVLA/model/framework/VLM4A/QwenPI_v3.py", "starVLA/model/modules/vlm/QWen3.py",
                "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py",
                "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py",
                "starVLA/dataloader/gr00t_lerobot/transform/state_action.py"):
        files["upstream:" + rel] = UPSTREAM / rel
    for rel in ("utils/dexbench_ee58_contract.py", "utils/process_data.py"):
        files["shared:" + rel] = POLICY.parents[1] / rel
    return files


def runtime_contract(config, stats_sha, config_sha, step):
    f, t = config["framework"], config["trainer"]
    require(f["name"] == "QwenPI_v3" and f["action_model"]["action_dim"] == 58
            and f["action_model"]["state_dim"] == 58, "Only the full EE58 QwenPI_v3 model is supported")
    ga = t["gradient_accumulation_steps"]
    world = config.get("sparkarena_contract", {}).get("num_processes", 8)
    effective_batch = world * config["datasets"]["vla_data"]["per_device_batch_size"] * ga
    require(t["max_train_steps"] == 60000 and ga in (1, 2) and world == 8
            and config["datasets"]["vla_data"]["per_device_batch_size"] == 4
            and effective_batch in (32, 64), "Expected60000 updates / 8xmicro4xGA1or2")
    declared_batch = config.get("sparkarena_contract", {}).get("effective_batch", effective_batch)
    require(declared_batch == effective_batch, "Saved effective batch disagrees with world x micro x GA")
    require(f["action_model"]["action_horizon"] == 16 and f["action_model"]["num_inference_timesteps"] == 4
            and f["action_model"]["diffusion_model_cfg"]["interleave_self_attention"] is True, "Expected H16 / four-step canonical FM")
    d = config["datasets"]["vla_data"]
    require(d["state_action_dtype"] == "float16" and d["include_state"] is True and d["action_mode"] == "abs"
            and d["image_size"] == d["obs_image_size"] == [224, 224] and t["repeated_diffusion_steps"] == 16, "Wrong training preprocessing")
    return {"version": 1, "policy": "StarVLA", "framework": "QwenPI_v3", "data_contract_id": CONTRACT_ID,
        "wire_state_dim": 54, "wire_action_dim": 54, "native_state_dim": 58, "native_action_dim": 58,
        "rotation": "matrix_rows_0_then_1", "state_input": "absolute_world_xyz_wxyz_hand20",
        "normalization": "separate_native_q99", "clip": [-2.2, 2.2], "packed_state_dtype": "float16",
        "bins": "native_state2str_unmodified_including_minus1_edge", "camera_order": ["front", "left_wrist", "right_wrist"],
        "image_mode": "RGB", "image_size": [224, 224], "image_resize": "PIL.Image.resize_default",
        "action_horizon": 16, "inference_timesteps": 4, "noise_repeats": 16,
        "effective_batch": effective_batch, "gradient_accumulation_steps": ga, "target_optimizer_updates": 60000,
        "optimizer_step": int(step), "statistics": "ee58_statistics.json", "statistics_sha256": stats_sha,
        "model_config": "config.full.yaml", "model_config_sha256": config_sha,
        "upstream_commit": UPSTREAM_COMMIT, "code_sha256": {key: sha256(path) for key, path in code_files().items()}}


def save_sidecars(run_dir, checkpoint, config, statistics, step):
    """Bind a completed weight file to inference metadata; called by the trainer."""
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    stats_path = run_dir / "ee58_statistics.json"
    stats_path.write_text(json.dumps(statistics, indent=2))
    config_path = run_dir / "config.full.yaml"
    require(config_path.is_file() and checkpoint.is_file(), "Checkpoint/config must be completely saved first")
    contract = runtime_contract(config, sha256(stats_path), sha256(config_path), step)
    contract["checkpoint"] = str(checkpoint.relative_to(run_dir))
    contract["checkpoint_bytes"] = checkpoint.stat().st_size
    contract["checkpoint_sha256"] = sha256(checkpoint)
    target = checkpoint.with_name(checkpoint.name + ".ee58.json")
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(contract, indent=2))
    temporary.replace(target)
    (run_dir / "ee58_runtime_contract.json").write_text(json.dumps(contract, indent=2))
    return target


def resolve_weights(config):
    root = resolve_checkpoint_root(config, POLICY / "checkpoints", policy_dir=POLICY)
    if root.is_file():
        return root
    final = root / "final_model/pytorch_model.pt"
    if final.is_file():
        return final
    files = list((root / "checkpoints").glob("steps_*_pytorch_model.pt"))
    require(files, f"No completed native QwenPI_v3 checkpoint under {root}")
    return max(files, key=lambda p: int(re.search(r"steps_(\d+)", p.name).group(1)))


def read_bundle(weight):
    weight = Path(weight).resolve(strict=True)
    manifest = json.loads(weight.with_name(weight.name + ".ee58.json").read_text())
    run = weight.parent.parent if weight.parent.name in ("checkpoints", "final_model") else weight.parent
    require(manifest["data_contract_id"] == CONTRACT_ID and manifest["framework"] == "QwenPI_v3", "Wrong EE58 checkpoint contract")
    require(manifest["rotation"] == "matrix_rows_0_then_1" and manifest["native_state_dim"] == manifest["native_action_dim"] == 58, "Wrong EE58 dimensions/rotation")
    require(manifest["action_horizon"] == 16 and manifest["inference_timesteps"] == 4, "Wrong inference horizon/steps")
    require(manifest["target_optimizer_updates"] == 60000
            and (manifest["gradient_accumulation_steps"], manifest["effective_batch"]) in ((1, 32), (2, 64))
            and 0 < manifest["optimizer_step"] <= 60000, "Invalid training update/batch contract")
    require(weight.stat().st_size == manifest["checkpoint_bytes"] and sha256(weight) == manifest["checkpoint_sha256"], "Checkpoint content changed")
    require((run / manifest["checkpoint"]).resolve() == weight, "Weight path disagrees with receipt")
    for key, path in code_files().items():
        require(manifest["code_sha256"].get(key) == sha256(path), f"Runtime code differs from checkpoint: {key}")
    require(subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() == manifest["upstream_commit"], "Wrong upstream revision")
    for key, digest in (("statistics", "statistics_sha256"), ("model_config", "model_config_sha256")):
        require(sha256(run / manifest[key]) == manifest[digest], f"Checkpoint {key} changed")
    config = yaml.safe_load((run / manifest["model_config"]).read_text())
    stats = json.loads((run / manifest["statistics"]).read_text())
    expected = runtime_contract(config, manifest["statistics_sha256"], manifest["model_config_sha256"], manifest["optimizer_step"])
    for key, value in expected.items():
        require(manifest[key] == value, f"Inconsistent runtime contract field: {key}")
    return manifest, config, stats


class EE58Codec:
    """Both paths use the very same DataConfig.transform/Normalizer as training."""
    def __init__(self, statistics):
        native_imports()
        from sparkarena_registry import SparkArenaWujiConfig
        from deployment.model_server.policy_norm_processor import _build_dataset_metadata
        self.data_config = SparkArenaWujiConfig()
        self.transform = self.data_config.transform()
        dims = {group: dict(zip([f"{group}.{p}" for p in PARTS], WIDTHS)) for group in ("state", "action")}
        self.transform.set_metadata(_build_dataset_metadata(
            stats_for_key={"state": statistics["observation.state"], "action": statistics["action"]},
            embodiment_tag=self.data_config.embodiment_tag,
            action_keys=self.data_config.action_keys, state_keys=self.data_config.state_keys,
            action_key_dims=dims["action"], state_key_dims=dims["state"]))
        self.transform.eval()

    def normalize(self, values, group, packed=True):
        values = np.asarray(values, dtype=np.float32)
        require(values.ndim == 2 and values.shape[-1] == 58 and np.isfinite(values).all(), "Expected finite (T,58)")
        data, cursor = {}, 0
        for part, width in zip(PARTS, WIDTHS):
            data[f"{group}.{part}"] = values[:, cursor:cursor + width].copy()
            cursor += width
        out = self.transform(data)
        result = np.concatenate([out[f"{group}.{p}"].detach().cpu().numpy() for p in PARTS], axis=-1)
        return result.astype(np.float16 if packed else np.float32)

    def decode_actions(self, normalized):
        import torch
        normalized = np.asarray(normalized, dtype=np.float32)
        require(normalized.ndim == 2 and normalized.shape[-1] == 58 and np.isfinite(normalized).all(), "Invalid normalized actions")
        data, cursor = {}, 0
        for part, width in zip(PARTS, WIDTHS):
            data[f"action.{part}"] = torch.as_tensor(normalized[:, cursor:cursor + width])
            cursor += width
        raw = self.transform.unapply(data)
        values = np.concatenate([np.asarray(raw[f"action.{p}"].detach().cpu().numpy()
                    if hasattr(raw[f"action.{p}"], "detach") else raw[f"action.{p}"]) for p in PARTS], axis=-1)
        return unpack_ee58(values)


def rgb_image(observation, candidates):
    for key in candidates:
        if key in observation["vision"]:
            value = observation["vision"][key]
            value = value["color"] if isinstance(value, dict) else value
            image = np.asarray(value)
            require(image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8, "Server must supply decoded RGB uint8")
            return Image.fromarray(image).resize((224, 224))
    raise KeyError(f"Missing current camera: {candidates}")


def load_native(weight, config, device):
    import torch
    from omegaconf import OmegaConf
    native_imports()
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg = apply_config_compat(OmegaConf.create(config))
    cfg.trainer.pretrained_checkpoint = None
    model = build_framework(cfg)
    # This loads every VLM/projector/FM tensor, not only an action head.
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.bfloat16).eval()
    vlm = model.qwen_vl_interface.model
    vlm.set_attn_implementation("flash_attention_2")
    require(all(c._attn_implementation == "flash_attention_2" for c in
                (vlm.config, vlm.config.text_config, vlm.config.vision_config)), "FA2 backend differs from training")
    require(model.action_model.num_inference_timesteps == 4, "Native FM sampler must run four Euler steps")
    return model


class EE58Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        require(self.model_cfg.get("env_cfg_type") == "tianji_marvin_wuji" and self.model_cfg.get("action_type") == "ee", "Expected Wuji EE58 deployment")
        self.dims = get_robot_action_dim_info("tianji_marvin_wuji")
        require(self.dims["ee_dim"] == [20, 20], "Expected Wuji20 hands")
        self.weight = resolve_weights(self.model_cfg)
        self.metadata, self.config, statistics = read_bundle(self.weight)
        self.codec = EE58Codec(statistics)
        self.execute_horizon = int(self.model_cfg.get("execute_horizon", 16))
        require(0 < self.execute_horizon <= 16, "Execution horizon must be1..16")
        self.model = load_native(self.weight, self.config, self.model_cfg.get("device", "cuda"))
        self.reset()

    def prepare_example(self, obs):
        raw54 = pack_robot_state(obs, "ee", self.dims, source_type="obs").astype(np.float32).reshape(1, 54)
        state58 = pack_ee54(raw54)
        return {"image": [rgb_image(obs, cameras) for cameras in WIRE_CAMERAS],
                "lang": str(obs.get("instruction") or obs.get("instructions") or ""),
                "state": self.codec.normalize(state58, "state", packed=True)}

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.latest_ids = []
        for index, obs in enumerate(obs_list):
            key = int(obs.get("env_idx", index))
            self.observations[key] = obs
            self.latest_ids.append(key)

    def get_action_batch(self, env_idx_list=None):
        ids = self.latest_ids if env_idx_list is None else [int(i) for i in env_idx_list]
        examples = [self.prepare_example(self.observations[i]) for i in ids]
        import torch
        with torch.inference_mode():
            normalized = np.asarray(self.model.predict_action(examples=examples)["normalized_actions"])
        require(normalized.shape == (len(ids), 16, 58), f"Wrong full-model action output: {normalized.shape}")
        return [[unpack_robot_state(vector, "ee", self.dims, source_type="obs")
                 for vector in self.codec.decode_actions(chunk)[:self.execute_horizon]] for chunk in normalized]

    def get_action(self):
        return self.get_action_batch([self.latest_ids[0]])[0]

    def reset(self):
        self.observations, self.latest_ids = {}, []
