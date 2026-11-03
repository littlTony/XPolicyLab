"""EgoVLA H1+Inspire joint38 real-action HDF5 dataloader for OpenWAM.

Layout:
    <dataset_dir>/<task>/episode_*.hdf5

Raw files store independent 50-D qpos and absolute joint-position action
targets. This reader gathers both into the compact 38-D policy ordering used
by XPolicyLab:

    [left_arm7, left_hand12, right_arm7, right_hand12]

The temporal contract is ``proprio=qpos[t]`` and ``target=action[t]``.  The
terminal action is excluded because its following visual observation was not
recorded.  qpos[t+1] is never used as an action target.  State and action use
separate normalization blocks (``joint_state`` and ``joint``), then scatter to
unified 80-D slots 0:38.  The benchmark is single-view:
``observations/images/main`` contains HWC uint8 RGB arrays.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robodojo import _normalize_mode
from openwam.dataloader.transforms.multiview import (
    crop_and_resize,
    format_prompt_for_inference,
)
from openwam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    Normalizer,
)
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import STAT_KEYS
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

DEPLOY_ACTION_MODE = "joint"
EGOVLA_EMBODIMENT = "ego_h1_inspire"
EGOVLA_CONTRACT_ID = "egovla-ego_h1_inspire-joint38-real-action-v2"
JOINT38_DIM = 38
RAW_DIM = 50
POLICY_JOINT_INDICES = np.asarray(
    (
        4, 8, 12, 16, 20, 22, 24,
        26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48,
        5, 9, 13, 17, 21, 23, 25,
        31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49,
    ),
    dtype=np.int64,
)
TASK_PROMPTS = {
    # Canonical EgoVLA labels. Preserve upstream capitalization and typo.
    "Pour-Balls": "pour balls in cup into bowl",
    "Push-Box": "push box to the marker",
    "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
    "Insert-Cans": "Insert cans into the boxes",
    "Close-Drawer": "Close the opened drawer",
    "Open-Drawer": "Open the closed drawer",
    "Insert-And-Unload-Cans": (
        "Insert the left can into the slot and insert the right can into the slot, "
        "unload the left cans andd then unload the right cans"
    ),
    "Flip-Mug": "Flip the mug",
    "Unload-Cans": "unload the right cans and then unload the left cans",
    "Stack-Can": "put can on the saucer",
    "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
    "Open-Laptop": "open the laptop",
}


def _config_get(config, key: str, default=None):
    if isinstance(config, Mapping):
        return config[key] if key in config else default
    if hasattr(config, key):
        return getattr(config, key)
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def natural_episode_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem.rsplit("_", 1)[1]), str(path)
    except (IndexError, ValueError):
        return (10**18, str(path))


def discover_egovla_tasks(dataset_dir: str | Path) -> list[str]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"EgoVLA dataset root does not exist: {root}")
    tasks = sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and child.name != "meta" and any(child.glob("episode_*.hdf5"))
    )
    if not tasks:
        raise FileNotFoundError(f"no EgoVLA tasks under {root}")
    return tasks


def discover_egovla_episodes(dataset_root: str | Path, task: str) -> list[Path]:
    data_dir = Path(dataset_root) / task
    if not data_dir.is_dir():
        raise FileNotFoundError(f"EgoVLA task directory does not exist: {data_dir}")
    episodes = sorted(
        (path for path in data_dir.glob("episode_*.hdf5") if path.is_file()),
        key=natural_episode_key,
    )
    if not episodes:
        raise FileNotFoundError(f"no EgoVLA episodes in {data_dir}")
    return episodes


def validate_conversion_manifest(dataset_root: str | Path) -> dict[str, Any]:
    in_progress = Path(dataset_root) / "meta" / ".conversion_in_progress"
    if in_progress.exists():
        raise RuntimeError(f"EgoVLA conversion is incomplete or still running: {in_progress}")
    path = Path(dataset_root) / "meta" / "conversion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing EgoVLA conversion manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    success_path = Path(dataset_root) / "_SUCCESS"
    if not success_path.is_file():
        raise FileNotFoundError(f"missing atomic publication marker: {success_path}")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if manifest.get("contract_id") != EGOVLA_CONTRACT_ID:
        raise ValueError(
            f"{path}: contract_id={manifest.get('contract_id')!r}, expected {EGOVLA_CONTRACT_ID!r}"
        )
    if manifest.get("state_source") != "observations/qpos" or manifest.get("action_source") != "action":
        raise ValueError(f"{path}: state/action provenance is not the real-action contract")
    if manifest.get("temporal_alignment") != "proprio=qpos[t], target=action[t]":
        raise ValueError(f"{path}: invalid temporal alignment {manifest.get('temporal_alignment')!r}")
    if (
        success.get("contract_id") != EGOVLA_CONTRACT_ID
        or success.get("stats_sha256") != manifest.get("stats_sha256")
        or success.get("joint_stream_sha256") != manifest.get("joint_stream_sha256")
    ):
        raise ValueError(f"{success_path}: publication marker does not match the manifest")
    stats_path = Path(dataset_root) / "meta" / "normalization_stats.npy"
    if not stats_path.is_file():
        raise FileNotFoundError(f"missing normalization stats: {stats_path}")
    digest = hashlib.sha256()
    with stats_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest.get("stats_sha256"):
        raise ValueError(f"{stats_path}: SHA-256 does not match the conversion manifest")
    source_manifest = Path(str(manifest.get("source_manifest", "")))
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source EgoVLA manifest is unavailable: {source_manifest}")
    source_digest = hashlib.sha256()
    with source_manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            source_digest.update(chunk)
    if source_digest.hexdigest() != manifest.get("source_manifest_sha256"):
        raise ValueError(f"{source_manifest}: source dataset manifest hash mismatch")
    prompt_map = {item["task"]: item["instruction"] for item in manifest.get("tasks", [])}
    if prompt_map != TASK_PROMPTS:
        raise ValueError(f"{path}: task prompts do not match the canonical EgoVLA registry")
    prompt_hash = hashlib.sha256(
        json.dumps(
            TASK_PROMPTS,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("prompt_registry_sha256") != prompt_hash:
        raise ValueError(f"{path}: canonical prompt registry hash mismatch")
    vision = manifest.get("vision", {})
    if vision.get("source_color_order") != "RGB" or vision.get("training_canvas_hw") != [384, 320]:
        raise ValueError(f"{path}: vision contract must be RGB with 384x320 training canvas")
    return manifest


def _read_joint38(
    episode: h5py.File | str | Path,
    key: str,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    if isinstance(episode, (str, Path)):
        with h5py.File(episode, "r") as handle:
            return _read_joint38(handle, key, start, end)
    if key not in episode:
        raise KeyError(f"{episode.filename}: missing {key}")
    arr = np.asarray(episode[key][slice(start, end)], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != RAW_DIM:
        raise ValueError(f"{episode.filename}:{key} expected (T, {RAW_DIM}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{episode.filename}:{key} contains non-finite values")
    return arr[:, POLICY_JOINT_INDICES]


def read_joint38(
    episode: h5py.File | str | Path,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    return _read_joint38(episode, "observations/qpos", start, end)


def read_action_joint38(
    episode: h5py.File | str | Path,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    return _read_joint38(episode, "action", start, end)


def validate_egovla_episode(path: str | Path) -> dict[str, Any]:
    episode_path = Path(path)
    with h5py.File(episode_path, "r") as handle:
        required = ("observations/qpos", "action", "observations/images/main")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"{episode_path}: missing {missing}")
        length = int(handle["observations/qpos"].shape[0])
        if length < 2:
            raise ValueError(f"{episode_path}: need T>=2, got {length}")
        if handle["observations/qpos"].shape != handle["action"].shape:
            raise ValueError(f"{episode_path}: qpos/action shape mismatch")
        image = handle["observations/images/main"]
        if int(image.shape[0]) != length:
            raise ValueError(f"{episode_path}: image/qpos T mismatch")
        if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{episode_path}: expected uint8 HWC RGB images, got {image.shape} {image.dtype}")
    task = episode_path.parent.name
    if task not in TASK_PROMPTS:
        raise KeyError(f"{episode_path}: unknown EgoVLA task {task!r}; canonical prompt unavailable")
    return {"length": length, "instruction": TASK_PROMPTS[task]}


def _validated_stats_block(
    path: str | Path,
    block_name: str,
    stats: Mapping,
    raw_dim: int,
) -> dict[str, np.ndarray]:
    validated = {}
    for key in STAT_KEYS:
        if key not in stats:
            raise KeyError(f"{path}:{block_name} is missing {key}")
        value = np.asarray(stats[key], dtype=np.float32)
        if value.shape != (raw_dim,):
            raise ValueError(f"{path}:{block_name}.{key} must have shape ({raw_dim},), got {value.shape}")
        validated[key] = value
    if np.any(validated["std"] <= 0) or np.any(validated["max"] < validated["min"]):
        raise ValueError(f"{path}: invalid {block_name} stats")
    return validated


def _load_action_state_stats(
    path: str | Path,
    action_mode: str,
    raw_dim: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a dict payload")
    state_key = f"{action_mode}_state"
    if action_mode not in payload:
        raise KeyError(f"{path}: missing action stats block {action_mode!r}")
    if state_key not in payload:
        raise KeyError(
            f"{path}: missing state stats block {state_key!r}; refusing to normalize qpos with action stats"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("contract_id") != EGOVLA_CONTRACT_ID:
        raise ValueError(f"{path}: stats are not bound to {EGOVLA_CONTRACT_ID}")
    if metadata.get("action_source") != "action" or metadata.get("forbidden_target") != "observations/qpos[t+1]":
        raise ValueError(f"{path}: stats provenance does not forbid next-state action targets")
    return (
        _validated_stats_block(path, action_mode, payload[action_mode], raw_dim),
        _validated_stats_block(path, state_key, payload[state_key], raw_dim),
    )


def _rgb_frame(handle: h5py.File, frame_index: int) -> Image.Image:
    frame = np.asarray(handle["observations/images/main"][frame_index])
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"{handle.filename}: main image[{frame_index}] shape {frame.shape}")
    frame = np.ascontiguousarray(frame[..., :3])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame, mode="RGB")


class EgoVLADataset(BaseDataset):
    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    def __init__(
        self,
        data_root: str | Path,
        *,
        dataset_root: str | Path,
        task_name: str,
        embodiment: str = EGOVLA_EMBODIMENT,
        action_mode: str = DEPLOY_ACTION_MODE,
        num_frames: int = 33,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        window_stride: int = 1,
        video_stride: int = 4,
        multiview: bool = False,
        camera_layout: Sequence[str] | None = None,
        target_camera: str = "cam_head",
        unify_action: bool = True,
        unify_action_map: Any = None,
        unify_state_map: Any = None,
        normalize_mode: str | None = "min-max",
        normalization_stats_path: str | Path | None = None,
        color_jitter: Any = None,
    ):
        super().__init__()
        if action_mode != "joint":
            raise ValueError(
                "EgoVLA real-action training supports action_mode='joint' only: the source 'action' "
                f"dataset is an absolute joint-position target, got {action_mode!r}"
            )
        if embodiment != EGOVLA_EMBODIMENT:
            raise ValueError(f"EgoVLA embodiment must be {EGOVLA_EMBODIMENT!r}, got {embodiment!r}")
        if num_frames < 2:
            raise ValueError(f"num_frames must be >= 2, got {num_frames}")
        self.embodiment = embodiment
        self.action_mode = action_mode
        self.task_name = str(task_name)
        self.num_frames = int(num_frames)
        self.num_action_steps = self.num_frames - 1
        self.height = int(height)
        self.width = int(width)
        self.split = split
        self.window_stride = int(window_stride)
        self.video_stride = int(video_stride)
        self._video_sample_indices = list(range(0, self.num_frames, self.video_stride))
        self.multiview = bool(multiview)
        self.target_camera = str(target_camera)
        self.contract_id = EGOVLA_CONTRACT_ID
        self.source_frame = "ego_h1_inspire_joint38_real_action"
        self._raw_dim = JOINT38_DIM
        self._default_unify = ["0-37"]
        if self.multiview:
            raise ValueError(
                "EgoVLA has mixed wrist-camera availability. This recipe is pinned to the uniform "
                "head-only contract multiview=false; the saved checkpoint makes inference ignore "
                "wrist inputs as well."
            )

        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            jitter_get = color_jitter.get if hasattr(color_jitter, "get") else lambda key, default: default
            self._color_jitter = VideoColorJitter(
                brightness=float(jitter_get("brightness", 0.2)),
                contrast=float(jitter_get("contrast", 0.2)),
                saturation=float(jitter_get("saturation", 0.2)),
                hue=float(jitter_get("hue", 0.0)),
            )

        episode_paths = discover_egovla_episodes(dataset_root, self.task_name)
        expected = Path(dataset_root) / self.task_name
        if Path(data_root).resolve() != expected.resolve():
            raise ValueError(f"data_root {data_root} != {expected}")
        all_metadata = [validate_egovla_episode(path) for path in episode_paths]
        self.dataset_root = str(Path(dataset_root))
        self.data_root = str(expected)
        self._episode_files = [str(path) for path in episode_paths]
        self._episode_lengths = [int(item["length"]) for item in all_metadata]
        self._instructions = [str(item["instruction"]) for item in all_metadata]
        self._window_index: list[tuple[int, int]] = []
        for episode_index, episode_length in enumerate(self._episode_lengths):
            for start in range(0, episode_length - 1, self.window_stride):
                self._window_index.append((episode_index, start))
        if not self._window_index:
            raise ValueError(f"no EgoVLA windows for task {self.task_name}")

        self._unify_action = bool(unify_action)
        if unify_state_map is not None and list(unify_state_map) != list(self._default_unify):
            raise ValueError(f"EgoVLA unify_state_map must be null or {self._default_unify}")
        self._unify_dst_index = None
        if self._unify_action:
            if unify_action_map is None:
                unify_action_map = list(self._default_unify)
            self._unify_dst_index = parse_unify_spec(unify_action_map, UNIFY_DIM)
            if self._unify_dst_index.shape != (self._raw_dim,):
                raise ValueError(
                    f"EgoVLA unify_action_map must map {self._raw_dim} raw dims, got {self._unify_dst_index.shape[0]}"
                )
            self._action_dim_value = UNIFY_DIM
        else:
            self._action_dim_value = self._raw_dim

        self.normalize_mode = _normalize_mode(normalize_mode)
        self._action_stats = None
        self._state_stats = None
        self._action_normalizer = None
        self._state_normalizer = None
        self.normalization_stats_path = None
        if self.normalize_mode is not None:
            if normalization_stats_path is None:
                raise ValueError("EgoVLA normalize_mode requires normalization_stats_path")
            stats_path = Path(normalization_stats_path)
            if not stats_path.is_file():
                raise FileNotFoundError(f"EgoVLA normalization stats do not exist: {stats_path}")
            self._action_stats, self._state_stats = _load_action_state_stats(
                stats_path, self.action_mode, self._raw_dim
            )
            mode = YAML_TO_NORM_MODE[self.normalize_mode]
            self._action_normalizer = Normalizer(mode=mode, stats=self._action_stats)
            self._state_normalizer = Normalizer(mode=mode, stats=self._state_stats)
            self.normalization_stats_path = str(stats_path)

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self):
        if self._action_stats is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._action_stats.items()}

    def denormalize_action(self, action) -> np.ndarray:
        array = np.asarray(action)
        if self._unify_dst_index is not None:
            array = unmap_from_unify(array, self._unify_dst_index)
        if self._action_normalizer is not None:
            array = self._action_normalizer.unnormalize(array)
        return np.asarray(array).copy()

    def __len__(self) -> int:
        return len(self._window_index)

    def _build_sample(self, episode_index: int, start: int) -> dict[str, Any]:
        path = self._episode_files[episode_index]
        episode_length = self._episode_lengths[episode_index]
        actual_length = min(self.num_frames, episode_length - start)
        if actual_length < 2:
            raise IndexError("EgoVLA window has no action with a following visual observation")
        valid_action_length = actual_length - 1
        with h5py.File(path, "r") as handle:
            # Same-row real action alignment: state[t] -> action[t].  Do not
            # shift either stream and do not derive targets from qpos[t+1].
            state = read_joint38(handle, start, start + 1)
            actions = read_action_joint38(handle, start, start + valid_action_length)
            sampled_video = []
            for relative_index in self._video_sample_indices:
                source_index = start + relative_index if relative_index < actual_length else episode_length - 1
                sampled_video.append(crop_and_resize(_rgb_frame(handle, source_index), self.height, self.width))
        if self._color_jitter is not None:
            sampled_video = self._color_jitter.apply({"video": sampled_video})["video"]

        transformed_state = state
        transformed_action = actions
        if self._state_normalizer is not None:
            transformed_state = self._state_normalizer.normalize(transformed_state)
        if self._action_normalizer is not None:
            transformed_action = self._action_normalizer.normalize(transformed_action)
        state_dim_mask = np.ones(self._raw_dim, dtype=bool)
        action_dim_mask = np.ones(self._raw_dim, dtype=bool)
        if self._unify_dst_index is not None:
            transformed_state, state_dim_mask = map_to_unify(
                np.asarray(transformed_state, dtype=np.float32), self._unify_dst_index, UNIFY_DIM
            )
            transformed_action, action_dim_mask = map_to_unify(
                np.asarray(transformed_action, dtype=np.float32), self._unify_dst_index, UNIFY_DIM
            )
        transformed_state = np.asarray(transformed_state, dtype=np.float32)
        transformed_action = np.asarray(transformed_action, dtype=np.float32)
        if valid_action_length < self.num_action_steps:
            transformed_action = np.concatenate(
                [
                    transformed_action,
                    np.repeat(
                        transformed_action[-1:],
                        self.num_action_steps - valid_action_length,
                        axis=0,
                    ),
                ],
                axis=0,
            )
        proprio = torch.from_numpy(transformed_state)
        action = torch.from_numpy(transformed_action)
        time_validity = torch.tensor(
            [action_index < valid_action_length for action_index in range(self.num_action_steps)],
            dtype=torch.bool,
        )
        action_dimension_validity = torch.from_numpy(action_dim_mask)
        state_dimension_validity = torch.from_numpy(state_dim_mask)
        action_mask = (time_validity[:, None] & action_dimension_validity[None, :]).contiguous()
        proprio_mask = state_dimension_validity[None, :].clone()
        video_mask = torch.tensor(
            [relative_index < actual_length for relative_index in self._video_sample_indices],
            dtype=torch.bool,
        )
        return {
            "video": sampled_video,
            "vace_video": None,
            "first_frame_image": [sampled_video[0]],
            "action": action,
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "prompt": format_prompt_for_inference(self._instructions[episode_index]),
            "episode_index": episode_index,
            "episode_path": path,
            "start_frame": start,
            "end_frame": min(start + self.num_frames, episode_length),
            "episode_length": episode_length,
            "task_name": self.task_name,
            "embodiment": self.embodiment,
            "source_frame": self.source_frame,
            "active_arm": "both",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index, start = self._window_index[index]
        return self._build_sample(episode_index, start)


class MultiTaskEgoVLADataset(BaseDataset):
    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    @classmethod
    def from_config(cls, config, split: str = "train"):
        camera_layout = _config_get(config, "camera_layout", None)
        if camera_layout is not None:
            camera_layout = list(camera_layout)
        return cls(
            dataset_dir=_config_get(config, "dataset_dir", _config_get(config, "dataset_root", None)),
            split=split,
            embodiment=_config_get(config, "embodiment", EGOVLA_EMBODIMENT),
            action_mode=_config_get(config, "action_mode", DEPLOY_ACTION_MODE),
            normalization_stats_path=_config_get(config, "normalization_stats_path", None),
            normalize_mode=_config_get(config, "normalize_mode", "min-max"),
            num_frames=int(_config_get(config, "num_frames", 33)),
            height=int(_config_get(config, "height", 384)),
            width=int(_config_get(config, "width", 320)),
            window_stride=int(_config_get(config, "window_stride", 1)),
            video_stride=int(_config_get(config, "video_stride", 4)),
            multiview=bool(_config_get(config, "multiview", False)),
            camera_layout=camera_layout,
            target_camera=_config_get(config, "target_camera", "cam_head"),
            unify_action=bool(_config_get(config, "unify_action", True)),
            unify_action_map=_config_get(config, "unify_action_map", ["0-37"]),
            unify_state_map=_config_get(config, "unify_state_map", None),
            color_jitter=_config_get(config, "color_jitter", None),
        )

    def __init__(self, dataset_dir: str | Path, *, split: str = "train", **dataset_kwargs):
        super().__init__()
        if dataset_dir is None:
            raise ValueError("EgoVLA dataset_dir is required")
        self.conversion_manifest = validate_conversion_manifest(dataset_dir)
        selected_tasks = discover_egovla_tasks(dataset_dir)
        self.dataset_dir = str(Path(dataset_dir))
        self.tasks = selected_tasks
        self.split = split
        self.embodiment = dataset_kwargs.get("embodiment", EGOVLA_EMBODIMENT)
        self.action_mode = dataset_kwargs.get("action_mode", DEPLOY_ACTION_MODE)
        self._sub_datasets = []
        self._cumulative_lengths = []
        cumulative = 0
        for task in selected_tasks:
            episodes = discover_egovla_episodes(dataset_dir, task)
            dataset = EgoVLADataset(
                data_root=episodes[0].parent,
                dataset_root=dataset_dir,
                task_name=task,
                split=split,
                **dataset_kwargs,
            )
            self._sub_datasets.append(dataset)
            cumulative += len(dataset)
            self._cumulative_lengths.append(cumulative)
        if not self._sub_datasets or cumulative == 0:
            raise ValueError("EgoVLA task discovery produced an empty dataset")
        self._total_length = cumulative
        first = self._sub_datasets[0]
        self._action_dim_value = first.action_dim
        self.contract_id = first.contract_id
        self.source_frame = first.source_frame
        self.normalization_stats_path = first.normalization_stats_path
        self._normalization_stats_shared = first.normalization_stats

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self):
        if self._normalization_stats_shared is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._normalization_stats_shared.items()}

    def denormalize_action(self, action) -> np.ndarray:
        return self._sub_datasets[0].denormalize_action(action)

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous = 0 if dataset_index == 0 else self._cumulative_lengths[dataset_index - 1]
        return self._sub_datasets[dataset_index][index - previous]


__all__ = [
    "DEPLOY_ACTION_MODE",
    "EGOVLA_CONTRACT_ID",
    "EGOVLA_EMBODIMENT",
    "EgoVLADataset",
    "JOINT38_DIM",
    "MultiTaskEgoVLADataset",
    "POLICY_JOINT_INDICES",
    "discover_egovla_episodes",
    "discover_egovla_tasks",
    "read_action_joint38",
    "read_joint38",
]
