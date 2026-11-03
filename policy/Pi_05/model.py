#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import os
import re
import sys
from typing import Any

import numpy as np

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)
from .inference_warmup import warmup_egovla_policy


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

# The Close-Drawer Pi05 checkpoint was trained from the EgoVLA LeRobot
# conversion where no wrist cameras exist for that task.  The converter stores
# black 384x384 wrist frames and an explicit [1, 0, 0] camera mask.  Keep this
# contract local to the Pi05 adapter; the shared bridge must continue exposing
# real wrist cameras to policies that were trained with them.
_EGOVLA_PI05_TRAIN_CONFIG = "pi05_egovla_joint38_full"
_EGOVLA_PI05_IMAGE_SIZE = 384


def _extract_step_number(value: Any) -> int | None:
    """Extract a numeric checkpoint selector without accepting tmp suffixes."""

    text = str(value).strip().rstrip("/\\")
    if not text:
        return None
    match = re.search(r"(\d+)$", text.rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


def _is_complete_checkpoint_dir(path: Path) -> bool:
    """Return true only for an atomically finalized Orbax checkpoint.

    Orbax writes ``<step>.orbax-checkpoint-tmp-*`` directories while a save is
    in progress.  They can already contain ``params`` and ``assets`` and must
    never be selected by an eval server racing a training save.  The metadata
    sentinel is written only for a finalized checkpoint and is therefore part
    of the contract here.
    """

    return (
        path.is_dir()
        and ".orbax-checkpoint-tmp-" not in path.name
        and (path / "_CHECKPOINT_METADATA").is_file()
        and (path / "params").is_dir()
        and (path / "assets").is_dir()
    )


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs: list[Path] = []
    if _is_complete_checkpoint_dir(checkpoint_root):
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.name.isdigit() and _is_complete_checkpoint_dir(child)
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            "No complete Pi_05 checkpoint was found under "
            f"{checkpoint_root}. Eval ignores in-progress Orbax temporary directories; "
            "wait for the next finalized checkpoint or pass a complete step path."
        )

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step == desired_step:
                return candidate

    numeric_dirs = [
        candidate for candidate in candidate_dirs if candidate.name.isdigit()
    ]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: int(candidate.name))
    return candidate_dirs[0]


class _DebugPolicy:
    """Tiny deterministic policy used only for server/client wiring checks."""

    def __init__(self, action_dim: int, action_horizon: int = 50):
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)

    def infer(self, observation: dict[str, Any], **_: Any) -> dict[str, np.ndarray]:
        state = np.asarray(observation.get("state", np.zeros(self.action_dim)))
        if state.ndim > 1:
            actions = np.zeros(
                (state.shape[0], self.action_horizon, self.action_dim),
                dtype=np.float32,
            )
        else:
            actions = np.zeros(
                (self.action_horizon, self.action_dim), dtype=np.float32
            )
        return {"actions": actions}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_repack_transforms(train_config: _config.TrainConfig) -> _transforms.Group:
    """Map XPolicyLab simulator observations for custom LeRobot configs."""
    if isinstance(train_config.data, _config.LeRobotEgoVLADataConfig):
        return _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "images/cam_high",
                        "observation/left_wrist_image": "images/cam_left_wrist",
                        "observation/right_wrist_image": "images/cam_right_wrist",
                        "observation/camera_mask": "camera_mask",
                        "observation/state": "state",
                        "prompt": "prompt",
                    }
                )
            ]
        )

    if not isinstance(train_config.data, _config.LeRobotWujiDataConfig):
        # Preserve the official behavior for Aloha and every other existing
        # Pi0.5 data configuration.
        return _transforms.Group()

    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/image": "images/cam_high",
                    "observation/left_wrist_image": "images/cam_left_wrist",
                    "observation/right_wrist_image": "images/cam_right_wrist",
                    "observation/state": "state",
                    "prompt": "prompt",
                }
            )
        ]
    )


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        self._close_drawer_camera_contract = (
            train_config_name == _EGOVLA_PI05_TRAIN_CONFIG
            and _is_close_drawer_task(self.task_name)
        )
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        if _truthy(model_cfg.get("debug_no_model")) or _truthy(
            os.environ.get("PI05_DEBUG_NO_MODEL")
        ):
            action_dim = (
                sum(self.robot_action_dim_info["arm_dim"])
                + sum(self.robot_action_dim_info["ee_dim"])
                if self.robot_action_dim_info is not None
                else 38
            )
            action_horizon = int(model_cfg.get("debug_action_horizon", 50))
            print(
                "[Pi_05] PI05_DEBUG_NO_MODEL enabled; using zero-action "
                f"wiring policy ({action_horizon}x{action_dim})."
            )
            return _DebugPolicy(action_dim=action_dim, action_horizon=action_horizon)

        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        if self.robot_action_dim_info is not None:
            expected_action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
                self.robot_action_dim_info["ee_dim"]
            )
            configured_action_dim = int(config.model.action_dim)
            if configured_action_dim != expected_action_dim:
                raise ValueError(
                    f"Pi_05 config {train_config_name!r} emits {configured_action_dim}D, "
                    f"but env_cfg_type requires {expected_action_dim}D; check deploy.yml "
                    "train_config_name/repo_id/action_type."
                )
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        policy = _policy_config.create_trained_policy(
            config,
            str(model_root),
            norm_stats=norm_stats,
            repack_transforms=_runtime_repack_transforms(config),
        )
        if isinstance(config.data, _config.LeRobotEgoVLADataConfig):
            # Compile before the websocket listener advertises readiness. The
            # first environment request keeps the normal request-time budget.
            warmup_egovla_policy(policy, action_dim=int(config.model.action_dim))
        return policy

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(
                obs,
                self.action_type,
                self.robot_action_dim_info,
                force_black_wrist=self._close_drawer_camera_contract,
            )
            for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(
    observation,
    action_type,
    robot_action_dim_info,
    *,
    force_black_wrist: bool = False,
):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        high_image = ensure_chw_uint8(observation["images"]["cam_high"])
        if force_black_wrist:
            high_image = _square_crop_resize(high_image, _EGOVLA_PI05_IMAGE_SIZE)
            black_wrist = np.zeros_like(high_image)
            prompt = observation.get("instruction", observation.get("instructions"))
            return {
                "state": state,
                "images": {
                    "cam_high": high_image,
                    "cam_left_wrist": black_wrist.copy(),
                    "cam_right_wrist": black_wrist.copy(),
                },
                "camera_mask": np.asarray([True, False, False], dtype=bool),
                "prompt": prompt,
            }
        left_value = observation["images"].get("cam_left_wrist")
        right_value = observation["images"].get("cam_right_wrist")
        images = {
            "cam_high": high_image,
            "cam_left_wrist": (
                ensure_chw_uint8(left_value) if left_value is not None else np.zeros_like(high_image)
            ),
            "cam_right_wrist": (
                ensure_chw_uint8(right_value) if right_value is not None else np.zeros_like(high_image)
            ),
        }
        inferred_mask = np.asarray([True, left_value is not None, right_value is not None], dtype=bool)
        camera_mask = np.asarray(observation.get("camera_mask", inferred_mask), dtype=bool)
        if camera_mask.shape != (3,):
            raise ValueError(f"camera_mask must have shape (3,), got {camera_mask.shape}")
        camera_mask = camera_mask & inferred_mask
        prompt = observation.get("instruction", observation.get("instructions"))
        return {"state": state, "images": images, "camera_mask": camera_mask, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    high_image = ensure_chw_uint8(
        extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
    )
    if force_black_wrist:
        high_image = _square_crop_resize(high_image, _EGOVLA_PI05_IMAGE_SIZE)
        black_wrist = np.zeros_like(high_image)
        state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
        prompt = observation.get("instruction", observation.get("instructions"))
        return {
            "state": state,
            "images": {
                "cam_high": high_image,
                "cam_left_wrist": black_wrist.copy(),
                "cam_right_wrist": black_wrist.copy(),
            },
            "camera_mask": np.asarray([True, False, False], dtype=bool),
            "prompt": prompt,
        }
    left_value = extract_optional_image(
        observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"]
    )
    right_value = extract_optional_image(
        observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"]
    )
    images = {
        "cam_high": high_image,
        "cam_left_wrist": ensure_chw_uint8(left_value) if left_value is not None else np.zeros_like(high_image),
        "cam_right_wrist": ensure_chw_uint8(right_value) if right_value is not None else np.zeros_like(high_image),
    }
    inferred_mask = np.asarray([True, left_value is not None, right_value is not None], dtype=bool)
    camera_mask = np.asarray(observation.get("camera_mask", inferred_mask), dtype=bool)
    if camera_mask.shape != (3,):
        raise ValueError(f"camera_mask must have shape (3,), got {camera_mask.shape}")
    camera_mask = camera_mask & inferred_mask
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction", observation.get("instructions"))
    return {"state": state, "images": images, "camera_mask": camera_mask, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "camera_mask": np.stack([obs["camera_mask"] for obs in obs_list], axis=0),
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "camera_mask": obs["camera_mask"][batch_index],
        "prompt": obs["prompt"][batch_index],
    }


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


def extract_optional_image(observation, candidate_names):
    try:
        return extract_image(observation, candidate_names)
    except KeyError:
        return None


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))


def _is_close_drawer_task(task_name: Any) -> bool:
    normalized = re.sub(r"[_\s]+", "-", str(task_name).strip().casefold())
    return "close-drawer" in normalized


def _square_crop_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Center-crop a CHW image to square and resize to the training size.

    EgoVLA conversion emits 384x384 RGB frames.  Simulator fixed_rgb is often
    16:9, so padding it would change the image geometry relative to training;
    this helper applies the same square input contract before OpenPI's normal
    224x224 preprocessing.  Existing square inputs are returned without an
    interpolation round-trip.
    """

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got {image.shape}")
    channels, height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Expected non-empty image, got {image.shape}")
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = image[:, top : top + side, left : left + side]
    if side == size:
        return square.copy()

    # PIL is already an OpenPI runtime dependency and provides deterministic
    # bilinear resizing without introducing another native image dependency.
    from PIL import Image

    hwc = np.transpose(square, (1, 2, 0))
    resized = np.asarray(
        Image.fromarray(hwc).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.transpose(resized, (2, 0, 1))
