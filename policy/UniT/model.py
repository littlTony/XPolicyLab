"""XPolicyLab adapter for the official UniT dual-system policy.

Images handed to this module have already been decoded by the XPolicyLab
policy server and are RGB arrays.  This adapter deliberately performs no
image decoding and no channel swapping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info


_DATA_CONFIG = "spark0_dual_arm_dexterous_joint"
_SPARK_THREEVIEW_DATA_CONFIG = "sparkarena_3view_joint54"
_EGOVLA_DATA_CONFIG = "egovla_h1_joint38_singleview"
# Canonical mixed-view contract; retain ``threeview`` as a compatibility alias
# for early checkpoints/configs created during the single-view run.
_EGOVLA_MULTIVIEW_DATA_CONFIG = "egovla_h1_joint38_multiview"
_EGOVLA_THREEVIEW_DATA_CONFIG = "egovla_h1_joint38_threeview"
_EGOVLA_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_SPARK_THREEVIEW_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# Dataset / VideoToTensor contract is width x height = 640 x 480.
_SPARK_THREEVIEW_IMAGE_HW = (480, 640)
_SPARK_THREEVIEW_PATH_MARKERS = (
    "0908_7task",
    "0908-sparkarena",
    "sparkarena_3view",
)
_CAMERA_ALIASES = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera", "ego_view"),
    "cam_left_wrist": ("cam_left_wrist", "left_wrist_camera", "left_hand_camera"),
    "cam_right_wrist": ("cam_right_wrist", "right_wrist_camera", "right_hand_camera"),
}


def spark_threeview_from_path(path: Path | str) -> bool:
    """0908 SparkArena 7-task runs were trained with sparkarena_3view_joint54."""

    text = str(path).replace("\\", "/").lower()
    return any(marker in text for marker in _SPARK_THREEVIEW_PATH_MARKERS)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Model(ModelTemplate):
    """Map XPolicyLab joint observations/actions to the selected UniT schema."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg["action_type"])
        self.env_cfg_type = str(model_cfg["env_cfg_type"])
        self.data_config_name = str(model_cfg.get("data_config", _DATA_CONFIG))
        self._apply_recipe_flags()
        if self.action_type != "joint":
            raise ValueError("UniT joint adapters support action_type=joint only")

        # Robot dimensions always come from XPolicyLab's robot registry.  The
        # adapter never derives them from a checkpoint or hard-coded slices.
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.arm_dims = tuple(int(v) for v in self.robot_action_dim_info["arm_dim"])
        self.hand_dims = tuple(int(v) for v in self.robot_action_dim_info["ee_dim"])
        if len(self.arm_dims) != 2 or len(self.hand_dims) != 2:
            raise ValueError(
                "UniT Spark adapter requires a dual-arm robot with two hand groups; "
                f"got arm_dim={self.arm_dims}, ee_dim={self.hand_dims}"
            )
        if any(v <= 0 for v in (*self.arm_dims, *self.hand_dims)):
            raise ValueError("robot action dimensions must all be positive")
        expected_hand_dims = (12, 12) if self.is_egovla else (20, 20)
        if self.arm_dims != (7, 7) or self.hand_dims != expected_hand_dims:
            schema_name = "EgoVLA joint38" if self.is_egovla else "Spark joint54"
            raise ValueError(
                f"UniT {schema_name} adapter requires arm_dim=[7, 7] and "
                f"ee_dim={list(expected_hand_dims)}, got {self.robot_action_dim_info}"
            )

        # ``camera_name`` is retained for legacy single-view deploy files; the
        # three-view contract accepts either ``camera_names: [...]`` or a
        # comma-separated ``camera_name`` string. Preserve the canonical order
        # because ConcatTransform and the vision encoder use positional views.
        camera_cfg = model_cfg.get("camera_names", model_cfg.get("camera_name", "cam_head"))
        if isinstance(camera_cfg, str):
            parsed_camera_names = tuple(
                item.strip() for item in camera_cfg.split(",") if item.strip()
            )
        elif isinstance(camera_cfg, (list, tuple)):
            parsed_camera_names = tuple(str(item).strip() for item in camera_cfg if str(item).strip())
        else:
            raise TypeError(
                "camera_name(s) must be a string or list/tuple of names; "
                f"got {type(camera_cfg).__name__}"
            )
        if not parsed_camera_names:
            raise ValueError("at least one camera name is required")
        if self.is_egovla_threeview and parsed_camera_names != _EGOVLA_CAMERA_NAMES:
            raise ValueError(
                "EgoVLA three-view camera order must be "
                f"{list(_EGOVLA_CAMERA_NAMES)}, got {list(parsed_camera_names)}"
            )
        self.camera_names = parsed_camera_names
        self.camera_name = self.camera_names[0]
        if self.is_spark_threeview:
            self._use_spark_threeview_cameras()
        self.camera_mask_policy = str(
            model_cfg.get("camera_mask_policy", "zero_inactive")
        ).strip().lower()
        if self.camera_mask_policy not in {"zero_inactive", "ignore"}:
            raise ValueError(
                "camera_mask_policy must be 'zero_inactive' or 'ignore', got "
                f"{self.camera_mask_policy!r}"
            )
        self.allow_missing_wrist_views = _as_bool(
            model_cfg.get("allow_missing_wrist_views", True)
        )
        # EgoVLA's converted videos and the benchmark VideoToTensor contract
        # are 384x384 at the adapter boundary.  The UniT training transform
        # subsequently resizes those frames to 224x224 for DINO; keeping this
        # distinction here prevents real-eval resolution validation failures.
        expected_image_size = 384 if self.is_egovla else 256
        self.image_size = int(model_cfg.get("image_size", expected_image_size))
        self.execute_action_steps = int(model_cfg.get("execute_action_steps", 16))
        self.embodiment_tag = str(model_cfg.get("embodiment_tag", "new_embodiment"))
        self.denoising_steps = int(model_cfg.get("denoising_steps", 4))
        if self.image_size != expected_image_size:
            schema_name = "EgoVLA" if self.is_egovla else "Spark"
            raise ValueError(
                f"official UniT {schema_name} recipe requires "
                f"image_size={expected_image_size}"
            )
        if self.data_config_name not in {
            _DATA_CONFIG,
            _SPARK_THREEVIEW_DATA_CONFIG,
            _EGOVLA_DATA_CONFIG,
            _EGOVLA_MULTIVIEW_DATA_CONFIG,
            _EGOVLA_THREEVIEW_DATA_CONFIG,
        }:
            raise ValueError(
                f"Unsupported UniT data_config={self.data_config_name!r}; "
                f"expected {_DATA_CONFIG!r}, {_SPARK_THREEVIEW_DATA_CONFIG!r} "
                f"or {_EGOVLA_DATA_CONFIG!r}"
            )
        if self.embodiment_tag != "new_embodiment":
            raise ValueError("UniT adapters require embodiment_tag='new_embodiment'")

        self._obs: dict[str, Any] | None = None
        self._obs_batch: list[dict[str, Any]] | None = None
        self.dry_run = _as_bool(model_cfg.get("dry_run", False)) or _as_bool(
            os.environ.get("UNIT_DRY_RUN", "0")
        )
        self.policy = None
        self.model_path: Path | None = None

        if self.dry_run:
            print(
                "[UniT] UNIT_DRY_RUN enabled: validating XPolicyLab wiring with "
                "zero joint actions; no checkpoint is loaded"
            )
        else:
            self.model_path = self._resolve_model_path()
            self._align_spark_recipe(self.model_path)
            self._validate_checkpoint_schema(self.model_path)
            self.policy = self._load_official_policy(self.model_path)
            print(
                f"[UniT] loaded official policy checkpoint: {self.model_path} "
                f"(data_config={self.data_config_name})"
            )

    def _apply_recipe_flags(self) -> None:
        self.is_egovla = self.data_config_name in {
            _EGOVLA_DATA_CONFIG,
            _EGOVLA_MULTIVIEW_DATA_CONFIG,
            _EGOVLA_THREEVIEW_DATA_CONFIG,
        }
        self.is_egovla_threeview = self.data_config_name in {
            _EGOVLA_MULTIVIEW_DATA_CONFIG,
            _EGOVLA_THREEVIEW_DATA_CONFIG,
        }
        self.is_spark_threeview = self.data_config_name == _SPARK_THREEVIEW_DATA_CONFIG

    def _use_spark_threeview_cameras(self) -> None:
        self.camera_names = _SPARK_THREEVIEW_CAMERA_NAMES
        self.camera_name = self.camera_names[0]

    def _align_spark_recipe(self, model_path: Path) -> None:
        """Upgrade Spark single-view deploy.yml when the checkpoint is 3-view."""

        if self.is_egovla or self.is_spark_threeview:
            return
        if self.data_config_name != _DATA_CONFIG:
            return
        if not spark_threeview_from_path(model_path):
            return
        self.data_config_name = _SPARK_THREEVIEW_DATA_CONFIG
        self._apply_recipe_flags()
        self._use_spark_threeview_cameras()
        print(
            "[UniT] checkpoint path selects SparkArena 3-view recipe "
            f"{_SPARK_THREEVIEW_DATA_CONFIG}: {model_path}"
        )

    @property
    def _policy_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _resolve_model_path(self) -> Path:
        checkpoints_dir = self._policy_dir / "checkpoints"
        ckpt_name = str(self.model_cfg.get("ckpt_name") or "").strip()
        configured = str(self.model_cfg.get("model_path") or "").strip()
        bench = str(self.model_cfg.get("bench_name") or "").strip()
        seed = str(self.model_cfg.get("seed") if self.model_cfg.get("seed") is not None else "")

        candidates: list[Path] = []
        seen: set[str] = set()

        def add_candidate(candidate: Path | str) -> None:
            """Add a path once while retaining the caller-supplied order."""

            path = Path(candidate)
            try:
                key = str(path.expanduser().resolve(strict=False))
            except OSError:
                key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

        # The shared resolver owns the common precedence: explicit deploy paths,
        # a path-valued ckpt_name, the canonical run directory, and the
        # checkpoints/<ckpt_name> fallback.  Keep those candidates first so an
        # explicit model_path wins even when a legacy UniT directory also exists.
        for candidate in candidate_checkpoint_roots(
            self.model_cfg,
            checkpoints_dir,
            policy_dir=self._policy_dir,
        ):
            add_candidate(candidate)

        # UniT's historical layouts are adapter-specific fallbacks.  In
        # particular, retain the EgoVLA ``joint38-singleview`` suffix (and the
        # Spark ``joint54-meanstd`` suffix) plus the reusable pretrain asset
        # path, without overriding the shared resolver's precedence.
        if ckpt_name:
            explicit = Path(os.path.expandvars(os.path.expanduser(ckpt_name)))
            add_candidate(explicit)
            add_candidate(checkpoints_dir / ckpt_name)
            add_candidate(self._policy_dir.parent.parent / "pretrain_model" / ckpt_name)
            if bench and seed:
                if self.is_egovla_threeview:
                    schema_suffix = "joint38-multiview"
                elif self.is_egovla:
                    schema_suffix = "joint38-singleview"
                else:
                    schema_suffix = "joint54-meanstd"
                setting = f"{bench}-{ckpt_name}-{self.env_cfg_type}-{schema_suffix}-{seed}"
                add_candidate(checkpoints_dir / setting)
        if configured:
            configured_path = Path(os.path.expandvars(os.path.expanduser(configured)))
            add_candidate(
                configured_path
                if configured_path.is_absolute()
                else (self._policy_dir / configured_path)
            )

        checked: list[str] = []
        for candidate in candidates:
            candidate = candidate.resolve(strict=False)
            checked.append(str(candidate))
            resolved = self._find_inference_checkpoint(candidate)
            if resolved is not None:
                return resolved

        joined = "\n  - ".join(checked) if checked else "<no checkpoint candidate configured>"
        raise FileNotFoundError(
            "Could not resolve a UniT dual-system checkpoint. Checked:\n  - " + joined
        )

    @staticmethod
    def _find_inference_checkpoint(root: Path) -> Path | None:
        """Return a checkpoint containing both model config and experiment metadata."""

        def usable(path: Path) -> bool:
            return (
                path.is_dir()
                and (path / "config.json").is_file()
                and (path / "experiment_cfg" / "metadata.json").is_file()
            )

        if usable(root):
            return root
        if not root.is_dir():
            return None

        search_roots = [root / "pretrain", root / "dual_system", root / "finetune", root]
        found: list[tuple[int, Path]] = []
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            for path in search_root.glob("checkpoint-*"):
                try:
                    step = int(path.name.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    continue
                if usable(path):
                    found.append((step, path))
        return max(found, key=lambda item: item[0])[1] if found else None

    def _validate_checkpoint_schema(self, model_path: Path) -> None:
        """Reject checkpoints whose saved embodiment metadata has the wrong joint schema."""

        metadata_path = model_path / "experiment_cfg" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embodiment = metadata.get(self.embodiment_tag)
        if embodiment is None and "modalities" in metadata:
            # Accept the unwrapped dataset metadata form as well as the form
            # written by UniT's TrainRunner under the embodiment-tag key.
            embodiment = metadata
        if not isinstance(embodiment, dict):
            raise ValueError(
                f"UniT checkpoint {model_path} has no metadata for embodiment "
                f"{self.embodiment_tag!r}"
            )
        modalities = embodiment.get("modalities")
        if not isinstance(modalities, dict):
            raise ValueError(f"UniT checkpoint {model_path} has no modality metadata")

        if self.is_egovla:
            expected = {
                "left_arm": self.arm_dims[0],
                "left_hand": self.hand_dims[0],
                "right_arm": self.arm_dims[1],
                "right_hand": self.hand_dims[1],
            }
        else:
            expected = {
                "right_arm": self.arm_dims[1],
                "right_hand": self.hand_dims[1],
                "left_arm": self.arm_dims[0],
                "left_hand": self.hand_dims[0],
            }
        errors: list[str] = []
        for kind in ("state", "action"):
            group = modalities.get(kind)
            if not isinstance(group, dict):
                errors.append(f"missing {kind}")
                continue
            # 0908 Spark converters write left-first JSON keys; ConcatTransform
            # and this adapter still look up groups by name, so only the key
            # set and per-group dims must match the selected robot.
            if set(group) != set(expected):
                errors.append(
                    f"{kind} keys={list(group)}, expected {list(expected)}"
                )
            for key, dimension in expected.items():
                item = group.get(key)
                shape = item.get("shape") if isinstance(item, dict) else None
                try:
                    actual = int(shape[-1])
                except (IndexError, TypeError, ValueError):
                    errors.append(f"{kind}.{key}=missing")
                    continue
                if actual != dimension:
                    errors.append(f"{kind}.{key}={actual}, expected {dimension}")
        if errors:
            raise ValueError(
                "UniT checkpoint modality metadata is incompatible with the selected "
                f"{'EgoVLA joint38' if self.is_egovla else 'Spark joint54'} robot: "
                + "; ".join(errors)
            )
        if self.is_egovla_threeview:
            self._validate_egovla_contract(model_path)

    def _validate_egovla_contract(self, model_path: Path) -> None:
        """Check three-view inference compatibility without rebinding provenance.

        A joint38 checkpoint can otherwise look structurally valid while being
        trained with the old head-only recipe.  The training wrapper writes
        ``egovla_contract.json`` in the run root may describe the final saved
        checkpoint while its robot/camera schema applies to the training run.
        Checkpoint-local sidecars remain bound to their named checkpoint.
        """
        configured = str(self.model_cfg.get("egovla_contract_path") or "").strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(os.path.expandvars(os.path.expanduser(configured))))
        cursor = model_path
        for _ in range(6):
            candidates.append(cursor / "egovla_contract.json")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        contract_path = next((path for path in candidates if path.is_file()), None)
        if contract_path is None:
            checked = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                "EgoVLA three-view checkpoint is missing required egovla_contract.json; "
                f"checked: {checked}"
            )
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read EgoVLA contract {contract_path}: {exc}") from exc
        if not isinstance(contract, dict):
            raise ValueError(f"EgoVLA contract {contract_path} must be a JSON object")
        if contract.get("format") != "xpolicylab-egovla-contract":
            raise ValueError(
                f"EgoVLA contract {contract_path} has unexpected format "
                f"{contract.get('format')!r}"
            )
        if contract.get("status") != "complete":
            raise ValueError(
                f"EgoVLA contract {contract_path} is not complete: "
                f"{contract.get('status')!r}"
            )
        run = contract.get("run")
        robot = contract.get("robot")
        camera = contract.get("camera")
        checks = {
            "run.env_cfg_type": (run, "env_cfg_type", self.env_cfg_type),
            "run.action_type": (run, "action_type", "joint"),
            "robot.robot_key": (robot, "robot_key", "ego_h1_inspire"),
            "robot.action_type": (robot, "action_type", "joint"),
            "robot.action_dim": (robot, "action_dim", 38),
            "robot.native_dim": (robot, "native_dim", 50),
            "robot.arm_dim": (robot, "arm_dim", [7, 7]),
            "robot.hand_dim": (robot, "hand_dim", [12, 12]),
            "camera.mode": (camera, "mode", "three_view_mixed"),
            "camera.order": (camera, "order", list(_EGOVLA_CAMERA_NAMES)),
            "camera.features": (
                camera,
                "features",
                [
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ],
            ),
            "camera.mask_feature": (camera, "mask_feature", "observation.camera_mask"),
        }
        # Dataset sizes describe the training run; they are not an inference
        # ABI. A new dataset with the same robot/camera/action schema must not
        # require editing this adapter. Preserve those values in the sidecar
        # for provenance and leave dataset auditing to the training tools.
        errors: list[str] = []
        for label, (section, key, expected) in checks.items():
            actual = section.get(key) if isinstance(section, dict) else None
            if actual != expected:
                errors.append(f"{label}={actual!r}, expected {expected!r}")
        run_checkpoint = run.get("checkpoint") if isinstance(run, dict) else None
        actual_checkpoint = model_path.resolve(strict=False)
        contract_scope = "checkpoint"
        if not run_checkpoint:
            errors.append("run.checkpoint is missing")
        else:
            expected_checkpoint = Path(
                os.path.expandvars(os.path.expanduser(str(run_checkpoint)))
            ).resolve(strict=False)
            declared_root = run.get("run_root")
            contract_root = contract_path.resolve(strict=False).parent
            shared_run_profile = bool(
                declared_root
                and Path(os.path.expandvars(os.path.expanduser(str(declared_root)))).resolve(strict=False)
                == contract_root
                and actual_checkpoint != contract_root
                and actual_checkpoint.is_relative_to(contract_root)
                and expected_checkpoint.is_relative_to(contract_root)
                and actual_checkpoint.parent == expected_checkpoint.parent
            )
            if shared_run_profile:
                contract_scope = "training-run"
            # A run-level schema can cover sibling checkpoints in the same
            # training stage. Its artifact hashes still describe only the
            # checkpoint recorded by the sidecar; never rewrite or certify
            # those hashes as belonging to the checkpoint being loaded.
            if expected_checkpoint != actual_checkpoint and not shared_run_profile:
                errors.append(
                    f"run.checkpoint={expected_checkpoint}, expected loaded {actual_checkpoint}"
                )
        if errors:
            raise ValueError(
                f"EgoVLA three-view contract {contract_path} is incompatible: "
                + "; ".join(errors)
            )
        print(
            f"[UniT] compatible EgoVLA three-view schema: {contract_path}; "
            f"scope={contract_scope}; loaded_checkpoint={actual_checkpoint}; "
            f"provenance_checkpoint={expected_checkpoint}"
        )

    def _load_official_policy(self, model_path: Path):
        # These imports stay lazy so UNIT_DRY_RUN can exercise the full XPL
        # transport contract without installing/loading the multi-billion model.
        from transformers import AutoConfig

        from gr00t.experiment.data_config_unit import load_data_config
        from gr00t.model.policy_unit import Gr00tUniTPolicy

        checkpoint_cfg = AutoConfig.from_pretrained(str(model_path))
        ignore_lang_prefix = bool(getattr(checkpoint_cfg, "ignore_lang_prefix", False))
        use_bridge = bool(checkpoint_cfg.unit_cfg["use_bridge"])
        num_bridge_tokens = (
            int(checkpoint_cfg.unit_cfg["num_bridge_tokens"]) if use_bridge else None
        )
        data_config = load_data_config(
            self.data_config_name,
            eagle_path=checkpoint_cfg.backbone_cfg["eagle_path"],
            use_bridge=use_bridge,
            ignore_lang_prefix=ignore_lang_prefix,
            num_bridge_tokens=num_bridge_tokens,
        )
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        tokenizer = modality_transform.transforms[-1].eagle_processor.tokenizer

        dinov2_override = self.model_cfg.get("dinov2_path_override")
        if dinov2_override:
            dinov2_override = os.path.expandvars(os.path.expanduser(str(dinov2_override)))

        return Gr00tUniTPolicy(
            model_path=str(model_path),
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.embodiment_tag,
            denoising_steps=self.denoising_steps,
            device="cuda",
            tokenizer_len=len(tokenizer),
            dinov2_path_override=dinov2_override,
        )

    def update_obs(self, obs: dict[str, Any]) -> None:
        self._obs = obs
        self._obs_batch = None

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if not isinstance(obs_list, list) or not obs_list:
            raise ValueError("update_obs_batch expects a non-empty observation list")
        self._obs_batch = obs_list
        self._obs = None

    def _state_vector(self, obs: dict[str, Any], key: str, expected_dim: int) -> np.ndarray:
        try:
            value = obs["state"][key]
        except KeyError as exc:
            raise KeyError(f"observation is missing state.{key}") from exc
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != expected_dim:
            raise ValueError(
                f"state.{key} has {array.size} values, expected {expected_dim} "
                f"from get_robot_action_dim_info({self.env_cfg_type!r})"
            )
        return array

    @staticmethod
    def _extract_image(obs: dict[str, Any], candidates: tuple[str, ...]) -> Any | None:
        """Find an RGB image in either XPL's vision or Pi05-style images."""
        containers = []
        for container_name in ("vision", "images"):
            container = obs.get(container_name)
            if isinstance(container, dict):
                containers.append(container)
        for container in containers:
            for candidate in candidates:
                if candidate not in container:
                    continue
                value = container[candidate]
                if isinstance(value, dict):
                    for nested in ("color", "rgb", "image"):
                        if nested in value:
                            return value[nested]
                else:
                    return value
        return None

    def _prepare_rgb(
        self,
        obs: dict[str, Any],
        candidates: tuple[str, ...] | None = None,
        *,
        required: bool = True,
    ) -> np.ndarray | None:
        candidates = candidates or (self.camera_name, "cam_head", "cam_high")
        image = self._extract_image(obs, candidates)
        if image is None:
            if required:
                raise KeyError(f"observation is missing an image for candidates={candidates}")
            return None
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"RGB image for candidates={candidates} must be HWC RGB, got {image.shape}"
            )
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.copy()  # decoded transport arrays may be read-only views

        height, width = image.shape[:2]
        if self.is_spark_threeview:
            target_h, target_w = _SPARK_THREEVIEW_IMAGE_HW
            if (height, width) != (target_h, target_w):
                interpolation = (
                    cv2.INTER_AREA
                    if height * width > target_h * target_w
                    else cv2.INTER_LINEAR
                )
                image = cv2.resize(
                    image, (target_w, target_h), interpolation=interpolation
                )
        else:
            # Match converted single-view Spark data: centered square crop, then 256x256.
            side = min(height, width)
            top = (height - side) // 2
            left = (width - side) // 2
            image = image[top : top + side, left : left + side]
            if image.shape[:2] != (self.image_size, self.image_size):
                interpolation = cv2.INTER_AREA if side > self.image_size else cv2.INTER_LINEAR
                image = cv2.resize(
                    image,
                    (self.image_size, self.image_size),
                    interpolation=interpolation,
                )
        # No RGB/BGR conversion: XPolicyLab and UniT both consume RGB here.
        return np.ascontiguousarray(image)

    @staticmethod
    def _camera_candidates(camera_name: str) -> tuple[str, ...]:
        """Return canonical and benchmark aliases for one camera stream."""
        aliases = _CAMERA_ALIASES.get(camera_name, (camera_name,))
        # Keep order stable while allowing a deploy file to use a custom alias.
        return tuple(dict.fromkeys((camera_name, *aliases)))

    @staticmethod
    def _camera_mask(obs: dict[str, Any]) -> np.ndarray | None:
        """Read the optional three-value EgoVLA camera mask from an observation.

        The benchmark's RLDX rows carry ``observation.camera_mask`` as
        ``[head,left_wrist,right_wrist]``.  XPolicyLab evaluation transports
        generally omit this provenance field because all three image payloads
        are present; in that case ``None`` means all supplied views are used.
        """
        raw: Any = None
        for key in ("camera_mask", "observation.camera_mask"):
            if key in obs:
                raw = obs[key]
                break
        if raw is None:
            for container_name in ("additional_info", "metadata", "info"):
                container = obs.get(container_name)
                if isinstance(container, dict):
                    for key in ("camera_mask", "observation.camera_mask"):
                        if key in container:
                            raw = container[key]
                            break
                if raw is not None:
                    break
        if raw is None:
            return None
        if isinstance(raw, dict):
            values = [raw.get(name, 0.0) for name in _EGOVLA_CAMERA_NAMES]
        else:
            try:
                values = np.asarray(raw, dtype=np.float32).reshape(-1).tolist()
            except (TypeError, ValueError):
                return None
        if len(values) < len(_EGOVLA_CAMERA_NAMES):
            return None
        mask = np.asarray(values[: len(_EGOVLA_CAMERA_NAMES)], dtype=np.float32)
        if not np.isfinite(mask).all():
            return None
        return mask

    @staticmethod
    def _instruction(obs: dict[str, Any]) -> str:
        value = obs.get("instruction", obs.get("instructions", ""))
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value)

    def _to_unit_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        if self.is_egovla or self.is_spark_threeview:
            unit_obs: dict[str, np.ndarray] = {}
            camera_mask = self._camera_mask(obs) if self.is_egovla_threeview else None
            camera_names = self.camera_names
            for camera_index, camera_name in enumerate(camera_names):
                active = camera_mask is None or camera_mask[camera_index] > 0.5
                # The head stream is mandatory.  Wrist streams may be absent in
                # real-eval payloads for single-view tasks; when the mask marks
                # one inactive (or the payload omits it), use a deterministic
                # black frame matching the converter's fallback video.
                required = camera_index == 0 or (
                    not self.allow_missing_wrist_views and active
                )
                image = self._prepare_rgb(
                    obs, self._camera_candidates(camera_name), required=required
                )
                if image is None:
                    fallback_hw = (
                        _SPARK_THREEVIEW_IMAGE_HW
                        if self.is_spark_threeview
                        else (self.image_size, self.image_size)
                    )
                    image = np.zeros((*fallback_hw, 3), dtype=np.uint8)
                if (
                    self.is_egovla_threeview
                    and self.camera_mask_policy == "zero_inactive"
                    and not active
                ):
                    image = np.zeros_like(image)
                unit_obs[f"video.{camera_name}"] = image[None, ...]

            unit_obs.update(
                {
                    "state.left_arm": self._state_vector(
                        obs, "left_arm_joint_state", self.arm_dims[0]
                    )[None, ...],
                    "state.left_hand": self._state_vector(
                        obs, "left_ee_joint_state", self.hand_dims[0]
                    )[None, ...],
                    "state.right_arm": self._state_vector(
                        obs, "right_arm_joint_state", self.arm_dims[1]
                    )[None, ...],
                    "state.right_hand": self._state_vector(
                        obs, "right_ee_joint_state", self.hand_dims[1]
                    )[None, ...],
                    "annotation.human.action.task_description": np.asarray(
                        [self._instruction(obs)]
                    ),
                }
            )
            return unit_obs
        return {
            "video.ego_view": self._prepare_rgb(obs)[None, ...],
            "state.right_arm": self._state_vector(
                obs, "right_arm_joint_state", self.arm_dims[1]
            )[None, ...],
            "state.right_hand": self._state_vector(
                obs, "right_ee_joint_state", self.hand_dims[1]
            )[None, ...],
            "state.left_arm": self._state_vector(
                obs, "left_arm_joint_state", self.arm_dims[0]
            )[None, ...],
            "state.left_hand": self._state_vector(
                obs, "left_ee_joint_state", self.hand_dims[0]
            )[None, ...],
            "annotation.human.coarse_action": np.asarray([self._instruction(obs)]),
        }

    @staticmethod
    def _stack_unit_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        keys = obs_list[0].keys()
        return {key: np.stack([obs[key] for obs in obs_list], axis=0) for key in keys}

    def _zero_chunk(self) -> list[dict[str, np.ndarray]]:
        return [
            {
                "left_arm_joint_state": np.zeros(self.arm_dims[0], dtype=np.float32),
                "left_ee_joint_state": np.zeros(self.hand_dims[0], dtype=np.float32),
                "right_arm_joint_state": np.zeros(self.arm_dims[1], dtype=np.float32),
                "right_ee_joint_state": np.zeros(self.hand_dims[1], dtype=np.float32),
            }
            for _ in range(self.execute_action_steps)
        ]

    @staticmethod
    def _action_array(actions: dict[str, Any], key: str, expected_dim: int) -> np.ndarray:
        if key not in actions:
            raise KeyError(f"UniT output is missing {key}")
        array = np.asarray(actions[key], dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(
                f"UniT output {key} must have shape (horizon, {expected_dim}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"UniT output {key} contains non-finite values")
        return array

    def _to_xpl_chunk(self, actions: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        left_arm = self._action_array(actions, "action.left_arm", self.arm_dims[0])
        left_hand = self._action_array(actions, "action.left_hand", self.hand_dims[0])
        right_arm = self._action_array(actions, "action.right_arm", self.arm_dims[1])
        right_hand = self._action_array(actions, "action.right_hand", self.hand_dims[1])

        horizons = {len(left_arm), len(left_hand), len(right_arm), len(right_hand)}
        if len(horizons) != 1:
            raise ValueError(f"UniT action groups have inconsistent horizons: {sorted(horizons)}")
        steps = min(self.execute_action_steps, horizons.pop())
        return [
            {
                "left_arm_joint_state": left_arm[i].copy(),
                "left_ee_joint_state": left_hand[i].copy(),
                "right_arm_joint_state": right_arm[i].copy(),
                "right_ee_joint_state": right_hand[i].copy(),
            }
            for i in range(steps)
        ]

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if self._obs is None:
            raise RuntimeError("get_action called before update_obs")
        if self.dry_run:
            # Still validate the complete Spark observation contract.
            self._to_unit_obs(self._obs)
            return self._zero_chunk()
        assert self.policy is not None
        return self._to_xpl_chunk(self.policy.get_action(self._to_unit_obs(self._obs)))

    def get_action_batch(
        self, env_idx_list: list[int] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        if self._obs_batch is None:
            raise RuntimeError("get_action_batch called before update_obs_batch")
        # XPL's normal batched loop always supplies active environment indices.
        # Only the ModelTemplate fallback path needs a benchmark sim config.
        expected = (
            len(env_idx_list)
            if env_idx_list is not None
            else int(get_batch_size(self.env_cfg_type))
        )
        if len(self._obs_batch) != expected:
            raise ValueError(
                f"stored observation batch has {len(self._obs_batch)} entries, expected {expected}"
            )
        unit_obs_list = [self._to_unit_obs(obs) for obs in self._obs_batch]
        if self.dry_run:
            return [self._zero_chunk() for _ in unit_obs_list]

        assert self.policy is not None
        batched_actions = self.policy.get_action(self._stack_unit_obs(unit_obs_list))
        chunks: list[list[dict[str, np.ndarray]]] = []
        for batch_index in range(len(unit_obs_list)):
            per_env = {key: np.asarray(value)[batch_index] for key, value in batched_actions.items()}
            chunks.append(self._to_xpl_chunk(per_env))
        return chunks

    def reset(self) -> None:
        self._obs = None
        self._obs_batch = None
