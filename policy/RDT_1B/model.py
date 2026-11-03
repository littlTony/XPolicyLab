from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PImage

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_RDT_ROOT = _CUR_DIR / "rdt"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_CUR_DIR), str(_RDT_ROOT), str(_RDT_ROOT / "models")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    unpack_robot_state,
)

from .rdt.scripts.robodojo_model import create_model
from .rdt.models.multimodal_encoder.t5_encoder import T5Embedder


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CUR_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_hwc_uint8(image):

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


_EGO_IMAGE_SIZE = (384, 384)
_TIANJI_IMAGE_SIZE = (640, 480)

_EGO_TASK_PROMPTS = {
    "close_drawer": "Close the opened drawer",
    "open_drawer": "Open the closed drawer",
    "flip_mug": "Flip the mug",
    "open_laptop": "open the laptop",
    "pour_balls": "pour balls in cup into bowl",
    "push_box": "push box to the marker",
    "stack_can": "put can on the saucer",
    "stack_can_into_drawer": "Open the drawer, and Put can on the saucer",
    "unload_cans": "unload the right cans and then unload the left cans",
    "insert_cans": "Insert cans into the boxes",
    "sort_cans": "Put sprite cans to the left box, and orange cans to the right box",
    "insert_and_unload_cans": "Insert the left can into the slot and insert the right can into the slot, unload the left cans andd then unload the right cans",
}

# Exact HDF5 sentences used to precompute SparkArena 7-task language embeddings.
_SPARK_TASK_PROMPTS = {
    "click_mouse": "Place the mouse on the mouse pad, then click the left button.",
    "collect_objects": "Put all the objects into the basket.",
    "dual_bottles_pick": "Pick up both bottles on the table.",
    "hammer_beat": "Pick up the hammer and beat the cube with the hammer head.",
    "put_food_in_microwave": "Put the bread in the microwave, then close the door.",
    "retrieve_gap": "Move the two cubes aside upright, then place the garage on the cushion.",
    "stack_bowls": "Stack the three bowls together.",
}


def _task_key(value):
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key.startswith("humanoid_"):
        key = key[len("humanoid_"):]
    if key.endswith("_v0"):
        key = key[:-3]
    return key


def _canonical_task_prompt(task_name):
    key = _task_key(task_name)
    return _SPARK_TASK_PROMPTS.get(key) or _EGO_TASK_PROMPTS.get(key)


def _resolve_runtime_prompt(raw_prompt, default_prompt="", task_name=""):
    text = str(raw_prompt or "").strip()
    mapped = _canonical_task_prompt(text)
    if mapped:
        return mapped
    if text:
        return text
    return _canonical_task_prompt(task_name) or str(default_prompt or "").strip()


def _parse_image_size(value):
    if value is None or value == "" or value is False:
        return None
    if isinstance(value, str):
        stripped = value.strip().strip("[]()")
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        if len(parts) != 2:
            return None
        value = parts
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _resolve_image_size(env_cfg, configured=None):
    """Match the training image ABI.

    SparkArena / Tianji HDF5 is 640x480 and is letterboxed, not stretched.
    EgoVLA staged data is already 384x384. The shared deploy.yml historically
    pinned 384 for EgoVLA; do not apply that square size to Tianji.
    """
    default = _EGO_IMAGE_SIZE if env_cfg == "ego_h1_inspire" else _TIANJI_IMAGE_SIZE
    size = _parse_image_size(configured)
    if size is None:
        return default
    if env_cfg != "ego_h1_inspire" and size == _EGO_IMAGE_SIZE:
        return default
    return size


def _normalise_camera_mode(value):
    mode = str(value or "real_wrist").strip().lower().replace("-", "_")
    if mode == "contract":
        mode = "black_wrist"
    if mode not in {"black_wrist", "main_replicated", "real_wrist"}:
        raise ValueError(
            "unsupported RDT camera_mode="
            f"{value!r}; expected 'black_wrist', 'main_replicated', or 'real_wrist'"
        )
    return mode


def _state_part(state_dict, key, expected_dim):
    if key not in state_dict:
        raise KeyError(f"missing observation state field {key!r}")
    value = np.asarray(state_dict[key], dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"state field {key!r} must be 1-D, got shape {value.shape}")
    if expected_dim is not None and value.shape[0] != int(expected_dim):
        raise ValueError(
            f"state field {key!r} has dim {value.shape[0]}, expected {int(expected_dim)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"state field {key!r} contains non-finite values")
    return value


def encode_obs(observation, default_prompt, camera_mode="real_wrist", state_dims=None):
    """Convert one decoded XPolicyLab observation to the RDT ABI.

    EgoVLA is a single-view benchmark. ``black_wrist`` keeps the released main
    RGB stream on ``cam_high`` and fills both wrist slots with black images of
    the same shape. ``main_replicated`` remains an explicit opt-in for the
    earlier checkpoint that mistakenly copied the main view into the wrist
    slots. The server has already decoded any wire JPEG before this function
    is called.
    """
    if not isinstance(observation, dict):
        raise TypeError(f"observation must be a dict, got {type(observation)!r}")
    mode = _normalise_camera_mode(camera_mode)
    head = ensure_hwc_uint8(
        extract_image(observation, ["cam_head", "cam_high", "fixed_rgb"])
    )
    if mode == "main_replicated":
        right = head.copy()
        left = head.copy()
    elif mode == "black_wrist":
        right = np.zeros_like(head)
        left = np.zeros_like(head)
    else:
        right = ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist", "right_hand_rgb"])
        )
        left = ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist", "left_hand_rgb"])
        )
    images = {
        "cam_high": head,
        "cam_right_wrist": right,
        "cam_left_wrist": left,
    }

    if state_dims is None:
        expected_dims = (None, None, None, None)
    else:
        expected_dims = tuple(state_dims)
        if len(expected_dims) != 4:
            raise ValueError(f"state_dims must contain four entries, got {expected_dims!r}")
    state_dict = observation.get("state")
    if not isinstance(state_dict, dict):
        raise TypeError("observation['state'] must be a mapping")
    state = np.concatenate(
        [
            _state_part(state_dict, "left_arm_joint_state", expected_dims[0]),
            _state_part(state_dict, "left_ee_joint_state", expected_dims[1]),
            _state_part(state_dict, "right_arm_joint_state", expected_dims[2]),
            _state_part(state_dict, "right_ee_joint_state", expected_dims[3]),
        ],
        axis=-1,
    )
    prompt = _resolve_runtime_prompt(
        observation.get("instruction")
        or observation.get("instructions")
        or observation.get("prompt"),
        default_prompt,
    )
    if isinstance(prompt, bytes):
        prompt = prompt.decode("utf-8", errors="replace")
    prompt = str(prompt or "").strip()
    if not prompt:
        raise ValueError("observation has no non-empty language instruction")
    return {"images": images, "state": state, "prompt": prompt}


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("RDT-1b in XPolicyLab currently supports only action_type='joint'.")

        self.env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg) if self.env_cfg is not None else None
        if self.robot_action_dim_info is not None:
            arm_dims = tuple(int(v) for v in self.robot_action_dim_info.get("arm_dim", ()))
            ee_dims = tuple(int(v) for v in self.robot_action_dim_info.get("ee_dim", ()))
            self.state_dims = tuple(
                value for pair in zip(arm_dims, ee_dims) for value in pair
            )
        else:
            self.state_dims = None
        requested_prompt = self.model_cfg.get("prompt")
        self.default_prompt = _resolve_runtime_prompt(
            requested_prompt,
            _canonical_task_prompt(self.task_name) or str(self.task_name),
            self.task_name,
        )
        self.image_size = _resolve_image_size(self.env_cfg, self.model_cfg.get("image_size"))
        requested_camera_mode = (
            self.model_cfg.get("camera_mode")
            or os.environ.get("RDT_CAMERA_MODE")
            or ("black_wrist" if self.env_cfg == "ego_h1_inspire" else "real_wrist")
        )
        self.camera_mode = _normalise_camera_mode(requested_camera_mode)

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.dtype = torch.bfloat16 if self.model_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32

        self.config = self._build_runtime_config()
        self.args = self._build_model_args()
        self.policy = create_model(
            args=self.args["config"],
            dtype=self.dtype,
            pretrained=self.args["pretrained_model_name_or_path"],
            pretrained_vision_encoder_name_or_path=self.args["pretrained_vision_encoder_name_or_path"],
            control_frequency=self.args["ctrl_freq"],
        )

        self.tokenizer, self.text_encoder = self._load_text_embedder()
        self.observation_window = None
        self._observation_windows: dict[int, deque] = {}
        self._latest_encoded_obs_list = []
        self.lang_embeddings = None
        self._latest_env_idx_list: list[int] = [0]
        self.model = self.policy

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _build_runtime_config(self):
        if self.robot_action_dim_info is None:
            raise ValueError("RDT-1b requires env_cfg or env_cfg_type so action dimensions can be resolved.")

        arm_dim = self.robot_action_dim_info.get("arm_dim")
        ee_dim = self.robot_action_dim_info.get("ee_dim")
        if arm_dim is None or len(arm_dim) != 2:
            raise ValueError(
                f"RDT-1b expects a dual-arm env_cfg with joint-state dimensions, got env_cfg={self.env_cfg!r}, arm_dim={arm_dim!r}."
            )
        if ee_dim is None or len(ee_dim) != 2:
            raise ValueError(
                f"RDT-1b expects a dual-hand env_cfg with end-effector joint dimensions, got env_cfg={self.env_cfg!r}, ee_dim={ee_dim!r}."
            )

        state_dim = int(sum(arm_dim) + sum(ee_dim))
        configured_action_dim = self.model_cfg.get("action_dim")
        if configured_action_dim is not None and int(configured_action_dim) != state_dim:
            raise ValueError(
                f"RDT raw action_dim={configured_action_dim} does not match the registered "
                f"observation/action dimension {state_dim} for {self.env_cfg!r}."
            )
        if self.env_cfg == "ego_h1_inspire" and state_dim != 38:
            raise ValueError(f"ego_h1_inspire must expose 38 raw state/action values, got {state_dim}")
        # A few lightweight contract tests construct Model with object.__new__
        # and call this method without running __init__. Keep that path
        # compatible while retaining the explicit EgoVLA camera default.
        camera_mode = getattr(
            self,
            "camera_mode",
            "black_wrist" if self.env_cfg == "ego_h1_inspire" else "real_wrist",
        )
        return {
            "episode_len": int(self.model_cfg.get("episode_len", 10000)),
            # This is the raw robot ABI; the RDT backbone still uses 128 state tokens.
            "state_dim": state_dim,
            "chunk_size": int(self.model_cfg.get("chunk_size", 64)),
            "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
            "camera_mode": camera_mode,
            "image_size": list(
                getattr(self, "image_size", _resolve_image_size(self.env_cfg, self.model_cfg.get("image_size")))
            ),
        }

    def _resolve_checkpoint_root(self) -> Path | None:
        # Shared precedence: explicit path keys > ckpt_name-as-path > 5-tuple
        # concat under checkpoints/ > checkpoints/<ckpt_name> verbatim. The
        # within-root step-dir discovery below is preserved.
        explicit_keys = ("checkpoint_path", "model_path", "model_root")
        if not self.model_cfg.get("ckpt_name") and not any(self.model_cfg.get(key) for key in explicit_keys):
            return None

        checkpoint_root = resolve_checkpoint_root(
            self.model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_CUR_DIR,
            explicit_keys=explicit_keys,
            must_exist=False,
        )
        if not checkpoint_root.is_dir():
            return checkpoint_root

        candidate_dirs = []
        if any((checkpoint_root / marker).exists() for marker in ("config.json", "pytorch_model.bin", "pytorch_model")):
            candidate_dirs.append(checkpoint_root)
        candidate_dirs.extend(
            child
            for child in sorted(checkpoint_root.iterdir())
            if child.is_dir() and any((child / marker).exists() for marker in ("config.json", "pytorch_model.bin", "pytorch_model"))
        )
        if not candidate_dirs:
            return checkpoint_root

        checkpoint_num = self.model_cfg.get("checkpoint_num")
        desired_step = _extract_step_number(checkpoint_num)
        if desired_step is not None:
            for candidate in candidate_dirs:
                candidate_step = _extract_step_number(candidate.name)
                if candidate_step is None:
                    continue
                scaled_step = desired_step
                while len(str(scaled_step)) < len(str(candidate_step)):
                    scaled_step *= 10
                if candidate_step in {desired_step, scaled_step}:
                    return candidate

        numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
        if numeric_dirs:
            return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
        return candidate_dirs[0]

    def _resolve_indexed_path(self, base_dir: Path | None, explicit_value: str | None, candidate_relpaths: list[str]) -> str | None:
        explicit_path = _resolve_path(explicit_value)
        if explicit_path is not None:
            return str(explicit_path)
        search_roots = []
        for root in (base_dir, _CHECKPOINTS_DIR):
            if root is not None and root not in search_roots:
                search_roots.append(root)
        if not search_roots:
            return None

        for root in search_roots:
            fallback_path = root
            for relative_path in candidate_relpaths:
                candidate = root / relative_path if relative_path else root
                fallback_path = candidate
                if candidate.exists():
                    return str(candidate)
            if root is base_dir:
                return str(fallback_path)
        return str(fallback_path)

    def _with_weights_fallback(self, explicit_value: str | None, resolved: str | None, weight_dirname: str) -> str | None:
        if explicit_value:
            return resolved
        if resolved is not None and Path(resolved).exists():
            return resolved
        fallback = _CUR_DIR / "weights" / "RDT" / weight_dirname
        if fallback.exists():
            return str(fallback)
        return resolved

    def _default_model_paths(self):
        checkpoint_root = self._resolve_checkpoint_root()
        model_root = _resolve_path(self.model_cfg.get("model_root")) or checkpoint_root or _RDT_ROOT
        default_config_path = model_root / "configs" / "base.yaml"
        if not default_config_path.exists():
            default_config_path = _RDT_ROOT / "configs" / "base.yaml"

        return {
            "config_path": self.model_cfg.get("config_path") or str(default_config_path),
            "text_encoder_path": self._with_weights_fallback(
                self.model_cfg.get("text_encoder_path"),
                self._resolve_indexed_path(
                    model_root,
                    self.model_cfg.get("text_encoder_path"),
                    [
                        "shared/t5-v1_1-xxl",
                        "text_encoder",
                        "weights/RDT/t5-v1_1-xxl",
                        "google/t5-v1_1-xxl",
                        "t5-v1_1-xxl",
                    ],
                ),
                "t5-v1_1-xxl",
            ),
            "vision_encoder_path": self._with_weights_fallback(
                self.model_cfg.get("vision_encoder_path"),
                self._resolve_indexed_path(
                    model_root,
                    self.model_cfg.get("vision_encoder_path"),
                    [
                        "shared/siglip-so400m-patch14-384",
                        "vision_encoder",
                        "weights/RDT/siglip-so400m-patch14-384",
                        "google/siglip-so400m-patch14-384",
                        "siglip-so400m-patch14-384",
                    ],
                ),
                "siglip-so400m-patch14-384",
            ),
            "checkpoint_path": self._resolve_indexed_path(
                checkpoint_root,
                self.model_cfg.get("checkpoint_path") or self.model_cfg.get("model_path"),
                ["", "checkpoint", "model", "pretrained_model"],
            ),
        }

    def _build_model_args(self):
        paths = self._default_model_paths()
        if paths["checkpoint_path"] is None:
            raise ValueError("ckpt_name, checkpoint_path, or model_path is required for RDT-1b.")

        return {
            "max_publish_step": int(self.model_cfg.get("max_publish_step", 10000)),
            "seed": self.model_cfg.get("seed"),
            "ctrl_freq": int(
                self.model_cfg.get(
                    "ctrl_freq", 30 if self.env_cfg == "ego_h1_inspire" else 25
                )
            ),
            "chunk_size": int(self.model_cfg.get("chunk_size", 64)),
            "config_path": paths["config_path"],
            "pretrained_model_name_or_path": paths["checkpoint_path"],
            "pretrained_vision_encoder_name_or_path": paths["vision_encoder_path"],
            "text_encoder_path": paths["text_encoder_path"],
            "config": self._load_yaml(paths["config_path"]),
        }

    def _load_yaml(self, config_path):
        with open(config_path, "r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
        config["arm_dim"] = {
            "left_arm_dim": self.robot_action_dim_info["arm_dim"][0],
            "right_arm_dim": self.robot_action_dim_info["arm_dim"][1],
        }
        return config

    def _load_text_embedder(self):
        # T5-XXL is only used once to encode the task instruction. Keep it on
        # CPU so it cannot OOM the policy GPU after the 1.2B diffusion weights
        # are already resident (the 0908 SparkArena collect/stack/retrieve
        # failures: ~12GB RDT + T5 on a card that already held other jobs).
        embed_device = str(os.environ.get("RDT_T5_DEVICE") or "cpu").strip() or "cpu"
        text_embedder = T5Embedder(
            from_pretrained=self.args["text_encoder_path"],
            model_max_length=self.args["config"]["dataset"]["tokenizer_max_length"],
            device=embed_device,
            use_offload_folder=None,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        text_encoder.eval()
        print(f"[RDT_1B] T5 text encoder on {embed_device}")
        return tokenizer, text_encoder

    def _resolve_precomp_lang_embed(self) -> Path | None:
        explicit = self.model_cfg.get("lang_embed_path") or os.environ.get("RDT_LANG_EMBED_PATH")
        explicit_path = _resolve_path(str(explicit)) if explicit else None
        if explicit_path is not None and explicit_path.is_file():
            return explicit_path

        lang_dir = self.model_cfg.get("lang_embed_dir") or os.environ.get("RDT_LANG_EMBED_DIR")
        search_roots = []
        if lang_dir:
            search_roots.append(Path(lang_dir).expanduser())
        search_roots.append(_CUR_DIR / "lang_embeds_0908_7task")
        search_roots.append(_CUR_DIR / "lang_embeds")

        task_key = _task_key(self.task_name)
        env_name = str(self.env_cfg or "tianji_marvin_wuji")
        rel_candidates = [
            Path("spark0_bench_7task_0908") / task_key / env_name / "lang_embed.pt",
            Path(task_key) / env_name / "lang_embed.pt",
        ]
        for root in search_roots:
            for relative in rel_candidates:
                candidate = root / relative
                if candidate.is_file():
                    return candidate
        return None

    def _set_language_instruction(self, instruction: str):
        instruction = _resolve_runtime_prompt(instruction, self.default_prompt, self.task_name)
        precomp = self._resolve_precomp_lang_embed()
        if precomp is not None:
            embed = torch.load(str(precomp), map_location="cpu")
            if not isinstance(embed, torch.Tensor):
                raise TypeError(f"precomputed lang embed at {precomp} is not a tensor")
            if embed.ndim == 2:
                embed = embed.unsqueeze(0)
            if embed.ndim != 3:
                raise ValueError(f"precomputed lang embed at {precomp} has shape {tuple(embed.shape)}")
            self.lang_embeddings = embed
            return

        device = next(self.text_encoder.parameters()).device
        with torch.no_grad():
            tokens = self.tokenizer(
                instruction,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )["input_ids"].to(device)
            tokens = tokens.view(1, -1)
            output = self.text_encoder(tokens)
            self.lang_embeddings = output.last_hidden_state.detach().cpu()
        torch.cuda.empty_cache()

    def _resize_img(self, img):
        img_size = getattr(self, "image_size", None) or _resolve_image_size(
            self.env_cfg, self.model_cfg.get("image_size")
        )
        if len(img_size) != 2 or any(int(value) <= 0 for value in img_size):
            raise ValueError(f"image_size must be [width,height] with positive values, got {img_size!r}")
        width, height = int(img_size[0]), int(img_size[1])
        if img.shape[1] == width and img.shape[0] == height:
            return img
        return cv2.resize(img, (width, height))

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list")
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self._latest_encoded_obs_list = [
            encode_obs(obs, self.default_prompt, self.camera_mode, self.state_dims)
            for obs in obs_list
        ]
        expected_state_dim = self.config["state_dim"]
        for encoded in self._latest_encoded_obs_list:
            if encoded["state"].shape != (expected_state_dim,):
                raise ValueError(
                    f"RDT expected raw state dim {expected_state_dim}, got {encoded['state'].shape}"
                )
        if self.lang_embeddings is None:
            self._set_language_instruction(self._latest_encoded_obs_list[0]["prompt"])

        for env_idx, encoded_obs in zip(self._latest_env_idx_list, self._latest_encoded_obs_list):
            window = self._observation_windows.get(env_idx)
            if window is None:
                window = deque(maxlen=2)
                window.append(
                    {
                        "qpos": None,
                        "images": {
                            self.config["camera_names"][0]: None,
                            self.config["camera_names"][1]: None,
                            self.config["camera_names"][2]: None,
                        },
                    }
                )
                self._observation_windows[env_idx] = window

            # The shared policy server has already decoded wire images; keep
            # RGB arrays intact apart from the configured spatial resize.
            img_front = self._resize_img(encoded_obs["images"]["cam_high"])
            img_right = self._resize_img(encoded_obs["images"]["cam_right_wrist"])
            img_left = self._resize_img(encoded_obs["images"]["cam_left_wrist"])
            qpos = torch.from_numpy(np.asarray(encoded_obs["state"], dtype=np.float32)).float().to(self.device)

            window.append(
                {
                    "qpos": qpos,
                    "images": {
                        self.config["camera_names"][0]: img_front,
                        self.config["camera_names"][1]: img_right,
                        self.config["camera_names"][2]: img_left,
                    },
                }
            )
        self.observation_window = self._observation_windows[self._latest_env_idx_list[0]]

    @torch.inference_mode()
    def infer(self, observation_window=None):
        observation_window = observation_window or self.observation_window
        if observation_window is None or self.lang_embeddings is None:
            raise AssertionError("update_obs must be called before get_action.")

        image_arrs = [
            observation_window[-2]["images"][self.config["camera_names"][0]],
            observation_window[-2]["images"][self.config["camera_names"][1]],
            observation_window[-2]["images"][self.config["camera_names"][2]],
            observation_window[-1]["images"][self.config["camera_names"][0]],
            observation_window[-1]["images"][self.config["camera_names"][1]],
            observation_window[-1]["images"][self.config["camera_names"][2]],
        ]
        images = [PImage.fromarray(arr) if arr is not None else None for arr in image_arrs]
        proprio = observation_window[-1]["qpos"].unsqueeze(0)
        actions = self.policy.step(proprio=proprio, images=images, text_embeds=self.lang_embeddings)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError(f"single-environment RDT inference returned shape {tuple(actions.shape)}")
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != self.config["state_dim"]:
            raise ValueError(
                f"RDT inference must return [chunk,{self.config['state_dim']}], got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("RDT inference returned non-finite action values")
        return actions.float().cpu().numpy()

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for env_idx in env_idx_list:
            raw_actions = self.infer(self._observation_windows[env_idx])
            action_list.append(unpack_robot_state(raw_actions, self.action_type, self.robot_action_dim_info, source_type="obs"))
        return action_list

    def reset(self):
        self.lang_embeddings = None
        self.observation_window = None
        self._observation_windows = {}
        self._latest_encoded_obs_list = []
        self._latest_env_idx_list = [0]
