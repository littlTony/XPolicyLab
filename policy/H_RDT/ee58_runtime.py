"""Checkpoint-bound H-RDT EE58 serving; legacy joint serving stays separate."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.dexbench_ee58_contract import pack_ee54, unpack_ee58
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state
from .training.ee58_data import normalize, sha256
from .training.native_assets import build_vision, native_paths
from .training.runtime_bundle import run_root, select_weights, validate_bundle, validate_weight_binding

POLICY = Path(__file__).resolve().parent


def denormalize_action(values, statistics):
    lo, hi = (np.asarray(statistics[key], dtype=np.float32) for key in ("q01", "q99"))
    span = hi - lo
    values = np.clip(np.asarray(values, dtype=np.float32), -1, 1)
    result = (values + 1) * .5 * np.where(span < 1e-6, 1, span) + lo
    result[..., span < 1e-6] = lo[span < 1e-6]
    return result.astype(np.float32)


def rgb_array(value):
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("Server must decode images to RGB arrays before Model.update_obs")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = array.transpose(1, 2, 0)
    if array.shape[-1] != 3:
        raise ValueError("Expected 3-channel RGB observations")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


class EE58Model(ModelTemplate):
    def __init__(self, model_cfg, *, _test_components=None):
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "ee" or self.model_cfg.get("env_cfg_type") != "tianji_marvin_wuji":
            raise ValueError("EE58 runtime supports tianji_marvin_wuji / ee only")
        if self.model_cfg.get("allow_dummy_lang_embedding", False):
            raise ValueError("Dummy language embeddings are forbidden for EE58")
        candidate = resolve_checkpoint_root(self.model_cfg, POLICY / "checkpoints", policy_dir=POLICY)
        self.checkpoint_root = run_root(candidate)
        self.bundle = validate_bundle(self.checkpoint_root, allow_test_fixture=_test_components is not None)
        self.checkpoint_path = select_weights(candidate, self.checkpoint_root)
        validate_weight_binding(self.checkpoint_path, self.checkpoint_root)
        explicit_config = self.model_cfg.get("config_path")
        if explicit_config not in (None, "", "null", "None"):
            if sha256(explicit_config) != self.bundle["contract"]["launch_config_sha256"]:
                raise ValueError("Explicit config differs from immutable checkpoint launch_config.yaml")
        explicit_stats = self.model_cfg.get("stats_path")
        if explicit_stats not in (None, "", "null", "None"):
            if sha256(explicit_stats) != self.bundle["contract"]["stats_sha256"]:
                raise ValueError("Explicit stats differ from immutable checkpoint EE58 statistics")
        self.config, self.stats = self.bundle["config"], self.bundle["stats"]
        self.language = self.bundle["embeddings"]
        self.robot_dims = get_robot_action_dim_info("tianji_marvin_wuji")
        self.native_action_dim, self.action_dim, self.action_chunk_size = 58, 54, 16
        self.device = torch.device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.float32 if self.model_cfg.get("dtype") == "float32" else torch.bfloat16
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a complete CPU environment or an allocated GPU")
        self._test_stub_networks = _test_components is not None
        if _test_components is None:
            self._load_networks()
        else:
            self.vision_encoder = _test_components["vision_encoder"]
            self.image_transform = _test_components["image_transform"]
            self.policy = _test_components["policy"]
        self.vision_encoder.to(self.device, dtype=self.dtype).eval()
        self.policy.to(self.device, dtype=self.dtype).eval()
        self.runtime_contract = {**self.bundle["contract"], "action_dim": 54, "native_action_dim": 58,
                                 "state_input": "raw_env_ee54", "action_output": "unnormalized_env_ee54",
                                 "test_stub_networks": self._test_stub_networks}
        self.reset()

    def _load_networks(self):
        native_paths()
        from models.hrdt_runner import HRDTRunner
        self.vision_encoder = build_vision(self.config["ee58_contract"]["vision_root"])
        self.image_transform = self.vision_encoder.get_image_transform()
        self.policy = HRDTRunner(state_dim=58, action_dim=58, pred_horizon=16, config=self.config["model"],
                                 act_pos_emb_config=[("state", 1), ("action", 16)],
                                 img_pos_emb_config=[("image", (1, 3, -self.vision_encoder.num_patches))],
                                 lang_pos_emb_config=[("language", -1024)],
                                 max_img_len=3*self.vision_encoder.num_patches, max_lang_len=1024,
                                 training_mode="lang", mode="pretrain", dtype=self.dtype)
        if self.checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            weights = load_file(str(self.checkpoint_path), device="cpu")
        else:
            weights = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        self.policy.load_state_dict(weights, strict=True)

    def _instruction(self, observation):
        text = observation.get("instruction", observation.get("instructions"))
        if isinstance(text, (list, tuple)) and len(text) == 1:
            text = text[0]
        explicit = self.model_cfg.get("ee58_instruction_key")
        if explicit not in (None, "", "null", "None"):
            if explicit not in self.language:
                raise ValueError("Explicit instruction key is absent from checkpoint-bound 60-text cache")
            plain = explicit.split("] ", 1)[-1]
            if text not in (None, "", explicit, plain):
                raise ValueError("Observation instruction disagrees with explicit training instruction key")
            return explicit
        if not isinstance(text, str) or text not in self.language:
            raise ValueError("Use an exact checkpoint-bound instruction string, or set ee58_instruction_key explicitly; no task/body-prefix guessing")
        return text

    def _encode_obs(self, observation):
        packed = np.asarray(pack_robot_state(observation, "ee", self.robot_dims, source_type="obs"), np.float32)
        if packed.shape != (54,):
            raise ValueError(f"Expected one wire EE54 state, got {packed.shape}")
        state = normalize(pack_ee54(packed)[None], self.stats["observation.state"])
        images = []
        vision = observation.get("vision", {})
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            payload = vision.get(camera)
            if payload is None:
                raise KeyError(f"Missing RGB camera: {camera}")
            if isinstance(payload, dict):
                payload = payload.get("color", payload.get("rgb"))
            images.append(rgb_array(payload))
        return {"state": state, "images": images, "instruction": self._instruction(observation)}

    def update_obs(self, obs, **kwargs):
        self.update_obs_batch([obs], [0])

    def update_obs_batch(self, obs_list, env_idx_list=None, **kwargs):
        env_idx_list = list(range(len(obs_list))) if env_idx_list is None else list(env_idx_list)
        if len(env_idx_list) != len(obs_list):
            raise ValueError("Observation count and environment indices differ")
        self._latest = env_idx_list
        for index, observation in zip(env_idx_list, obs_list):
            self.observations[index] = self._encode_obs(observation)

    def _image_tokens(self, images):
        transformed = [self.image_transform(Image.fromarray(image)) for image in images]
        inputs = {key: torch.stack([image[key] for image in transformed]).to(self.device, self.dtype)
                  for key in transformed[0]}
        features = self.vision_encoder(inputs)
        return features.reshape(1, -1, self.vision_encoder.embed_dim)

    @torch.inference_mode()
    def _infer(self, index):
        if index not in self.observations:
            raise RuntimeError("update_obs must precede get_action")
        obs = self.observations[index]
        state = torch.as_tensor(obs["state"], device=self.device, dtype=self.dtype).unsqueeze(0)
        language = self.language[obs["instruction"]].to(self.device, self.dtype).unsqueeze(0)
        mask = torch.ones(language.shape[:2], device=self.device, dtype=torch.bool)
        prediction = self.policy.predict_action(state_tokens=state, image_tokens=self._image_tokens(obs["images"]),
                                                lang_tokens=language, lang_attn_mask=mask)
        values = prediction.detach().float().cpu().numpy()
        if values.shape != (1, 16, 58) or not np.isfinite(values).all():
            raise ValueError("Native H-RDT must predict a finite [1,16,58] chunk")
        wire = unpack_ee58(denormalize_action(values[0], self.stats["action"]))
        return unpack_robot_state(wire, "ee", self.robot_dims, source_type="obs")

    def get_action(self, **kwargs):
        return self._infer(self._latest[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        return [self._infer(index) for index in (self._latest if env_idx_list is None else env_idx_list)]

    def get_metadata(self):
        return {"action_dim": 54, "action_chunk_size": 16, "runtime_contract": self.runtime_contract}

    def reset(self):
        self.observations, self._latest = {}, [0]
