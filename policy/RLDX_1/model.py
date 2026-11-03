"""XPolicyLab adapter for the upstream RLDX-1 policy.

The adapter only translates XPolicyLab observations to the official
RLDXPolicy API.  Image buffers are decoded by XPolicyLab's policy server
before this module is called.

SparkArena 7-task weights trained on directory slugs (``hammer_beat``)
and OpenCV BGR frames. DexBench sentences are remapped to those slugs,
and server RGB is swapped to BGR. Relarm checkpoints register the GitHub
arm-relative / hand-absolute contract so RLDX can add current qpos back.
EgoVLA / old Spark0 stay as-is.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_batch_size as _shared_get_batch_size,
    get_robot_action_dim_info as _shared_get_robot_action_dim_info,
)

from .instruction_align import (
    action_contract_from_checkpoint,
    align_instruction,
    instruction_style_from_checkpoint,
    swap_rgb_bgr_enabled,
)
from .runtime_support import get_robot_action_dim_info as _private_get_robot_action_dim_info


MODEL_FILE = Path(__file__).resolve()
POLICY_DIR = MODEL_FILE.parent
XPL_ROOT = MODEL_FILE.parents[2]


def _robot_action_dim_info(env_cfg_type: str) -> dict[str, list[int]]:
    """Resolve dimensions without relying on the host's /personal symlink."""

    try:
        return _private_get_robot_action_dim_info(env_cfg_type)
    except (FileNotFoundError, KeyError):
        # Keep compatibility with other XPolicyLab robot entries that may be
        # supplied by a complete shared checkout.
        return _shared_get_robot_action_dim_info(env_cfg_type)


class Model(ModelTemplate):
    """Run an RLDX-1 EgoVLA joint38 checkpoint behind XPolicyLab's RPC API."""

    _STATE_SOURCES = {
        "left_arm": "left_arm_joint_state",
        "left_hand": "left_ee_joint_state",
        "right_arm": "right_arm_joint_state",
        "right_hand": "right_ee_joint_state",
    }
    _ACTION_TARGETS = {
        "left_arm": "left_arm_joint_state",
        "left_hand": "left_ee_joint_state",
        "right_arm": "right_arm_joint_state",
        "right_hand": "right_ee_joint_state",
    }
    _DEFAULT_CAMERA_MAP = {
        "cam_head": "cam_head",
        "cam_left_wrist": "cam_left_wrist",
        "cam_right_wrist": "cam_right_wrist",
    }
    _DEBUG_CKPT_NAMES = {"debug", "debug-no-model", "smoke"}

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg.get("action_type") or "")
        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or "")
        if self.action_type != "joint":
            raise ValueError(
                "RLDX_1's EgoVLA adapter supports action_type='joint' only; "
                f"received {self.action_type!r}."
            )
        if not self.env_cfg_type:
            raise ValueError("env_cfg_type must be provided")
        self._is_egovla = (
            str(model_cfg.get("bench_name") or "").casefold() == "egovla"
            or self.env_cfg_type == "ego_h1_inspire"
        )

        # Prefer the policy-local registry.  The shared env_cfg robot registry
        # is a /personal symlink on the split evaluation host and is unavailable
        # there; keeping this fallback private avoids changing another checkout.
        self.robot_action_dim_info = _robot_action_dim_info(self.env_cfg_type)
        arm_dims = [int(v) for v in self.robot_action_dim_info["arm_dim"]]
        hand_dims = [int(v) for v in self.robot_action_dim_info["ee_dim"]]
        if len(arm_dims) != 2 or len(hand_dims) != 2:
            raise ValueError(
                "RLDX_1 EgoVLA integration requires a dual-arm robot entry; "
                f"got arm_dim={arm_dims}, ee_dim={hand_dims}."
            )
        self._native_dims = {
            "left_arm": arm_dims[0],
            "left_hand": hand_dims[0],
            "right_arm": arm_dims[1],
            "right_hand": hand_dims[1],
        }

        configured_batch = model_cfg.get("batch_size")
        if configured_batch is None:
            try:
                configured_batch = _shared_get_batch_size(self.env_cfg_type)
            except (FileNotFoundError, KeyError):
                configured_batch = 1
        self.batch_size = int(configured_batch)

        self.video_length = int(model_cfg.get("video_length", 4))
        self.video_stride = int(model_cfg.get("video_stride", 2))
        self.execution_horizon = int(model_cfg.get("execution_horizon", 16))
        if self.video_length != 4 or self.video_stride != 2:
            raise ValueError(
                "RLDX-1-PT requires the official 4-frame, stride-2 video recipe; "
                f"got video_length={self.video_length}, video_stride={self.video_stride}."
            )
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")

        camera_map = model_cfg.get("camera_map") or self._DEFAULT_CAMERA_MAP
        if not isinstance(camera_map, dict) or not camera_map:
            raise ValueError("camera_map must map RLDX video keys to XPolicyLab camera keys")
        self.camera_map = {str(k): str(v) for k, v in camera_map.items()}
        self.input_image_size = tuple(
            int(value) for value in model_cfg.get("input_image_size", (384, 384))
        )
        self.processor_image_size = tuple(
            int(value) for value in model_cfg.get("processor_image_size", (256, 256))
        )
        self.missing_wrist_policy = str(
            model_cfg.get("missing_wrist_policy") or "black"
        ).casefold()
        if self._is_egovla and self.input_image_size != (384, 384):
            raise ValueError(
                "EgoVLA RLDX input images must be 384x384 to match training; "
                f"got {self.input_image_size}"
            )
        if self._is_egovla and self.processor_image_size != (256, 256):
            raise ValueError(
                "EgoVLA RLDX processor images must be 256x256 to match training; "
                f"got {self.processor_image_size}"
            )
        if self._is_egovla and self.missing_wrist_policy != "black":
            raise ValueError("EgoVLA missing_wrist_policy must be 'black'")
        self.language_key = str(
            model_cfg.get("language_key") or "annotation.human.action.task_description"
        )

        span = (self.video_length - 1) * self.video_stride + 1
        self._history_span = span
        self._histories: dict[Any, deque[dict[str, Any]]] = {}
        self._last_batch_order: list[Any] = []
        self._sessions_needing_reset: set[Any] = set()
        self._all_sessions_need_reset = True

        ckpt_name = str(model_cfg.get("ckpt_name") or "")
        self.model = None
        self.debug_no_model = bool(model_cfg.get("debug_no_model", False)) or (
            ckpt_name.lower() in self._DEBUG_CKPT_NAMES
        )
        if self.debug_no_model:
            self._configure_alignment(model_cfg, checkpoint_hint="")
            print("[RLDX_1] debug-no-model mode: returning dimension-correct zero actions")
            return

        model_path = self._resolve_checkpoint(model_cfg)
        self._configure_alignment(model_cfg, checkpoint_hint=str(model_path))
        path_text = str(model_path).casefold()
        action_contract = "none"
        if not self._is_egovla:
            action_contract = action_contract_from_checkpoint(
                str(model_path),
                explicit=model_cfg.get("action_contract"),
            )
        if action_contract == "relative":
            # GitHub RLDX: arms are q_target - q_current; hands stay absolute.
            from . import spark_joint54_relative_config as _spark_joint54_config  # noqa: F401
            print("[RLDX_1] SparkArena action contract: arms relative, hands absolute")
        elif action_contract == "absolute" or "sparkarena" in path_text or "joint54" in path_text:
            # Legacy absolute-action SparkArena / joint54 checkpoints.
            from . import spark_joint54_config as _spark_joint54_config  # noqa: F401
            print("[RLDX_1] SparkArena action contract: all joints absolute")
        try:
            import rldx
            from rldx.policy.rldx_policy import RLDXPolicy
        except ImportError as exc:
            raise RuntimeError(
                "RLDX-1 is not installed in the policy environment. Run "
                "policy/RLDX_1/install.sh and pass its .venv to eval.sh."
            ) from exc

        tag_name = str(model_cfg.get("embodiment_tag") or "GENERAL_EMBODIMENT")
        try:
            embodiment_tag = rldx.EmbodimentTag[tag_name]
        except KeyError as exc:
            valid = ", ".join(tag.name for tag in rldx.EmbodimentTag)
            raise ValueError(f"unknown embodiment_tag={tag_name!r}; valid values: {valid}") from exc

        device = str(model_cfg.get("device") or "cuda")
        self.model = RLDXPolicy(
            model_path=str(model_path),
            embodiment_tag=embodiment_tag,
            device=device,
            strict=bool(model_cfg.get("strict", True)),
            verbose=bool(model_cfg.get("verbose", False)),
        )
        self._validate_checkpoint_schema()
        print(f"[RLDX_1] loaded checkpoint: {model_path}")

    def _configure_alignment(self, model_cfg: dict[str, Any], *, checkpoint_hint: str) -> None:
        hint = checkpoint_hint or str(
            model_cfg.get("model_path") or model_cfg.get("ckpt_name") or ""
        )
        self._instruction_style = instruction_style_from_checkpoint(
            hint,
            explicit=model_cfg.get("instruction_style"),
        )
        swap_cfg = model_cfg.get("swap_rgb_bgr")
        self._swap_rgb_bgr = swap_rgb_bgr_enabled(
            None if swap_cfg is None else bool(swap_cfg),
            checkpoint_path=hint,
        )
        if self._is_egovla and self._swap_rgb_bgr:
            raise ValueError(
                "EgoVLA RLDX checkpoints require RGB input with swap_rgb_bgr=false"
            )
        self._task_name = str(model_cfg.get("task_name") or "").strip()
        self._logged_instruction_align = False
        if self._instruction_style == "short_name":
            print(
                "[RLDX_1] SparkArena language align: DexBench sentences -> "
                "training short names"
            )
        if self._swap_rgb_bgr:
            print("[RLDX_1] SparkArena color align: inference RGB -> training BGR")

    @staticmethod
    def _latest_checkpoint(path: Path) -> Path:
        if not path.is_dir():
            return path
        step_dirs = []
        for child in path.glob("checkpoint-*"):
            if child.is_dir():
                try:
                    step_dirs.append((int(child.name.rsplit("-", 1)[1]), child))
                except ValueError:
                    continue
        return max(step_dirs)[1] if step_dirs else path

    def _resolve_checkpoint(self, model_cfg: dict[str, Any]) -> Path:
        checkpoints_dir = POLICY_DIR / "checkpoints"
        attempted = candidate_checkpoint_roots(
            model_cfg,
            checkpoints_dir,
            policy_dir=POLICY_DIR,
            explicit_keys=("model_path",),
        )

        # Keep the adapter's historical lookup locations as lower-priority
        # compatibility fallbacks.  The shared XPolicyLab candidates above
        # always establish the standard precedence first.
        explicit = str(model_cfg.get("model_path") or "").strip()
        ckpt_name = str(model_cfg.get("ckpt_name") or "").strip()
        if explicit:
            explicit_path = Path(explicit).expanduser()
            if not explicit_path.is_absolute():
                attempted.extend([Path.cwd() / explicit_path, XPL_ROOT / explicit_path])
        elif ckpt_name:
            ckpt_path = Path(ckpt_name).expanduser()
            if not ckpt_path.is_absolute():
                attempted.extend(
                    [Path.cwd() / ckpt_path, POLICY_DIR / ckpt_path, XPL_ROOT / ckpt_path]
                )
            attempted.append(XPL_ROOT / "pretrain_model" / ckpt_name)

            if ckpt_name.lower() in {"base", "pretrain", "rldx-1-pt"}:
                default_path = model_cfg.get("pretrained_model_path")
                if default_path:
                    attempted.extend(
                        candidate_checkpoint_roots(
                            {"model_path": default_path},
                            checkpoints_dir,
                            policy_dir=POLICY_DIR,
                            explicit_keys=("model_path",),
                        )
                    )

        seen: set[Path] = set()
        for candidate in attempted:
            candidate = candidate.resolve(strict=False)
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_dir():
                return self._latest_checkpoint(candidate)
        rendered = "\n  - ".join(str(p) for p in attempted) or "(no candidate configured)"
        raise FileNotFoundError(
            "could not resolve an RLDX-1 checkpoint. Checked:\n  - " + rendered
        )

    def _validate_checkpoint_schema(self) -> None:
        configs = self.model.get_modality_config()
        expected = {
            "video": set(self.camera_map),
            "state": set(self._STATE_SOURCES),
            "action": set(self._ACTION_TARGETS),
            "language": {self.language_key},
        }
        for modality, keys in expected.items():
            actual = set(configs[modality].modality_keys)
            if actual != keys:
                raise ValueError(
                    f"checkpoint {modality} keys {sorted(actual)} do not match the "
                    f"EgoVLA joint38 adapter keys {sorted(keys)}. Use a checkpoint "
                    "fine-tuned with egovla_joint38_config.py."
                )
        video_delta = list(configs["video"].delta_indices)
        expected_video_delta = [
            -(self.video_length - 1 - i) * self.video_stride
            for i in range(self.video_length)
        ]
        if video_delta != expected_video_delta:
            raise ValueError(
                f"checkpoint video delta_indices={video_delta}, expected {expected_video_delta}"
            )

    def _snapshot(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError(f"observation must be a dict, got {type(obs).__name__}")
        vision = obs.get("vision")
        state = obs.get("state")
        if not isinstance(vision, dict) or not isinstance(state, dict):
            raise KeyError("observation must contain 'vision' and 'state' dictionaries")

        images: dict[str, np.ndarray] = {}
        for native_key, xpl_key in self.camera_map.items():
            camera = vision.get(xpl_key)
            color = camera.get("color") if isinstance(camera, dict) else None
            if color is None:
                is_wrist = native_key in {"cam_left_wrist", "cam_right_wrist"}
                if self._is_egovla and is_wrist and self.missing_wrist_policy == "black":
                    image = np.zeros((*self.input_image_size, 3), dtype=np.uint8)
                else:
                    raise KeyError(f"missing RGB camera observation vision.{xpl_key}.color")
            else:
                image = np.asarray(color)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"vision.{xpl_key}.color must be HWC RGB, got shape {image.shape}"
                )
            if self._is_egovla and image.shape[:2] != self.input_image_size:
                raise ValueError(
                    f"vision.{xpl_key}.color must match EgoVLA training resolution "
                    f"{self.input_image_size}, got {image.shape[:2]}"
                )
            if image.dtype != np.uint8:
                raise TypeError(
                    f"vision.{xpl_key}.color must be uint8 RGB, got {image.dtype}"
                )
            # Server-decoded arrays can be read-only views; copy before retaining them.
            image = image.copy()
            if getattr(self, "_swap_rgb_bgr", False):
                image = image[..., ::-1].copy()
            images[native_key] = image

        states: dict[str, np.ndarray] = {}
        for native_key, xpl_key in self._STATE_SOURCES.items():
            if xpl_key not in state:
                raise KeyError(f"missing joint observation state.{xpl_key}")
            value = np.asarray(state[xpl_key], dtype=np.float32).reshape(-1).copy()
            expected_dim = self._native_dims[native_key]
            if value.shape != (expected_dim,):
                raise ValueError(
                    f"state.{xpl_key} has shape {value.shape}; expected ({expected_dim},) "
                    f"from the private robot registry for {self.env_cfg_type!r}"
                )
            states[native_key] = value

        instruction = obs.get("instruction", obs.get("instructions", ""))
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("observation instruction must be a non-empty string")
        aligned = align_instruction(
            instruction,
            style=getattr(self, "_instruction_style", "as_is"),
            task_name=getattr(self, "_task_name", ""),
        )
        if aligned != instruction and not getattr(self, "_logged_instruction_align", False):
            print(f"[RLDX_1] remapped instruction {instruction!r} -> {aligned!r}")
            self._logged_instruction_align = True
        return {"images": images, "states": states, "instruction": aligned}

    def _append(self, key: Any, obs: dict[str, Any]) -> None:
        history = self._histories.setdefault(key, deque(maxlen=self._history_span))
        history.append(self._snapshot(obs))
        if self._all_sessions_need_reset:
            self._sessions_needing_reset.add(key)

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._append("__single__", obs)

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if not isinstance(obs_list, list):
            raise TypeError("update_obs_batch expects a list of observations")
        order: list[Any] = []
        for position, obs in enumerate(obs_list):
            key = obs.get("env_idx", position)
            if isinstance(key, np.generic):
                key = key.item()
            if key in order:
                raise ValueError(f"duplicate env_idx in observation batch: {key!r}")
            self._append(key, obs)
            order.append(key)
        self._last_batch_order = order

    def _history_frames(self, history: deque[dict[str, Any]], camera_key: str) -> np.ndarray:
        if not history:
            raise RuntimeError("get_action called before update_obs")
        samples = list(history)
        latest = len(samples) - 1
        frames = []
        for i in range(self.video_length):
            age = (self.video_length - 1 - i) * self.video_stride
            frames.append(samples[max(0, latest - age)]["images"][camera_key])
        return np.stack(frames, axis=0)

    def _native_batch(self, keys: list[Any]) -> dict[str, Any]:
        histories = []
        for key in keys:
            if key not in self._histories or not self._histories[key]:
                raise RuntimeError(f"no observation history for env {key!r}")
            histories.append(self._histories[key])

        video = {
            native_key: np.stack(
                [self._history_frames(history, native_key) for history in histories], axis=0
            )
            for native_key in self.camera_map
        }
        state = {
            native_key: np.stack(
                [history[-1]["states"][native_key][None, :] for history in histories], axis=0
            ).astype(np.float32, copy=False)
            for native_key in self._STATE_SOURCES
        }
        language = {
            self.language_key: [[history[-1]["instruction"]] for history in histories]
        }
        return {"video": video, "state": state, "language": language}

    def _zero_action_batch(self, batch_size: int) -> list[list[dict[str, np.ndarray]]]:
        return [
            [
                {
                    target_key: np.zeros(self._native_dims[native_key], dtype=np.float32)
                    for native_key, target_key in self._ACTION_TARGETS.items()
                }
                for _ in range(self.execution_horizon)
            ]
            for _ in range(batch_size)
        ]

    def _infer(self, keys: list[Any]) -> list[list[dict[str, np.ndarray]]]:
        if self.debug_no_model:
            return self._zero_action_batch(len(keys))

        native_obs = self._native_batch(keys)
        reset_mask = [self._all_sessions_need_reset or key in self._sessions_needing_reset for key in keys]
        options = {
            "session_ids": [f"xpl-{key}" for key in keys],
            "reset_memory": reset_mask,
        }
        native_action, _info = self.model.get_action(native_obs, options=options)
        self._all_sessions_need_reset = False
        self._sessions_needing_reset.difference_update(keys)

        arrays: dict[str, np.ndarray] = {}
        batch_size = len(keys)
        horizon: int | None = None
        for native_key in self._ACTION_TARGETS:
            if native_key not in native_action:
                raise KeyError(f"RLDX-1 output is missing action key {native_key!r}")
            value = np.asarray(native_action[native_key], dtype=np.float32)
            if value.ndim == 2 and batch_size == 1:
                value = value[None, ...]
            expected_dim = self._native_dims[native_key]
            if value.ndim != 3 or value.shape[0] != batch_size or value.shape[-1] != expected_dim:
                raise ValueError(
                    f"RLDX-1 action {native_key!r} has shape {value.shape}; expected "
                    f"({batch_size}, horizon, {expected_dim})"
                )
            horizon = value.shape[1] if horizon is None else min(horizon, value.shape[1])
            arrays[native_key] = value
        assert horizon is not None
        horizon = min(horizon, self.execution_horizon)

        output: list[list[dict[str, np.ndarray]]] = []
        for batch_idx in range(batch_size):
            chunk = []
            for step in range(horizon):
                chunk.append(
                    {
                        target_key: arrays[native_key][batch_idx, step].copy()
                        for native_key, target_key in self._ACTION_TARGETS.items()
                    }
                )
            output.append(chunk)
        return output

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if "__single__" not in self._histories:
            raise RuntimeError("get_action called before update_obs")
        return self._infer(["__single__"])[0]

    def get_action_batch(self, env_idx_list: list[Any] | None = None) -> list[list[dict[str, np.ndarray]]]:
        if env_idx_list is None:
            keys = list(self._last_batch_order)
            if not keys:
                keys = list(self._histories)[: self.batch_size]
        else:
            keys = [value.item() if isinstance(value, np.generic) else value for value in env_idx_list]
        if not keys:
            raise RuntimeError("get_action_batch called before update_obs_batch")
        return self._infer(keys)

    def reset(self) -> None:
        self._histories.clear()
        self._last_batch_order.clear()
        self._sessions_needing_reset.clear()
        self._all_sessions_need_reset = True
        if self.model is not None:
            self.model.reset()
