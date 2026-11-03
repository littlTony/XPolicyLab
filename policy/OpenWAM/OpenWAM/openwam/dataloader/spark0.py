"""Spark0 tianji_marvin_wuji joint54 HDF5 dataloader for OpenWAM.

Reads the formal RoboDojo-style layout:

    <dataset_dir>/<task>/tianji_marvin_wuji/data/episode_*.hdf5

and packs achieved joint state as 54-D:

    [left_arm7, left_hand20, right_arm7, right_hand20]

Training follows the OpenWAM next-state convention: proprio = state[t],
action target = state[t+1:]. Raw 54-D is min-max normalized, then scattered
into the shared 80-D action space at slots 0:54.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robodojo import (
    DEFAULT_ROBODOJO_CAMERA_LAYOUT,
    _CAMERA_DATASETS,
    _decode_instruction,
    _decode_jpeg,
    _normalize_mode,
)
from openwam.dataloader.transforms.multiview import (
    assemble_multiview_layout,
    crop_and_resize,
    format_prompt_for_inference,
)
from openwam.dataloader.transforms.normalize import (
    YAML_TO_NORM_MODE,
    Normalizer,
    load_mode_stats,
)
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import STAT_KEYS
from openwam.dataloader.alpha_eef import (
    SPARK0_EEF_RAW_DIM,
    SPARK0_UNIFY_MAP,
    pack_spark0_alpha_eef,
)
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

DEPLOY_ACTION_MODE = "joint"
SPARK0_EMBODIMENT = "tianji_marvin_wuji"
SPARK0_CONTRACT_ID = "spark0-tianji-marvin-wuji-joint54-v1"
JOINT54_DIM = 54
STATE_KEYS = (
    ("state/left_arm_joint_states", 7),
    ("state/left_ee_joint_states", 20),
    ("state/right_arm_joint_states", 7),
    ("state/right_ee_joint_states", 20),
)
FORMAL_EPISODE_GLOB = "episode_*.hdf5"


def _config_get(config, key: str, default=None):
    if isinstance(config, Mapping):
        return config[key] if key in config else default
    if hasattr(config, key):
        return getattr(config, key)
    if hasattr(config, "get"):
        return config.get(key, default)
    return default


def discover_spark0_tasks(dataset_dir: str | Path, *, embodiment: str = SPARK0_EMBODIMENT) -> list[str]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Spark0 dataset root does not exist: {root}")
    tasks = sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / embodiment / "data").is_dir()
    )
    if not tasks:
        raise FileNotFoundError(f"no Spark0 tasks under {root} for embodiment {embodiment}")
    return tasks


def discover_spark0_episodes(
    dataset_root: str | Path,
    task: str,
    *,
    embodiment: str = SPARK0_EMBODIMENT,
) -> list[Path]:
    data_dir = Path(dataset_root) / task / embodiment / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Spark0 data directory does not exist: {data_dir}")
    episodes = sorted(path for path in data_dir.glob(FORMAL_EPISODE_GLOB) if path.is_file())
    if not episodes:
        raise FileNotFoundError(f"no Spark0 episodes in {data_dir}")
    return episodes


def read_joint54(
    episode: h5py.File | str | Path,
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    if isinstance(episode, (str, Path)):
        with h5py.File(episode, "r") as handle:
            return read_joint54(handle, start, end)
    selection = slice(start, end)
    parts = []
    for key, dim in STATE_KEYS:
        arr = np.asarray(episode[key][selection], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"{episode.filename}:{key} expected (T, {dim}), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{episode.filename}:{key} contains non-finite values")
        parts.append(arr)
    packed = np.concatenate(parts, axis=-1)
    if packed.shape[1] != JOINT54_DIM:
        raise ValueError(f"packed joint dim {packed.shape[1]} != {JOINT54_DIM}")
    return packed


def validate_spark0_episode(path: str | Path) -> dict[str, Any]:
    episode_path = Path(path)
    with h5py.File(episode_path, "r") as handle:
        for key, dim in STATE_KEYS:
            if key not in handle:
                raise KeyError(f"{episode_path}: missing {key}")
            shape = handle[key].shape
            if len(shape) != 2 or shape[1] != dim:
                raise ValueError(f"{episode_path}:{key} must have shape (T, {dim}), got {shape}")
        for camera in DEFAULT_ROBODOJO_CAMERA_LAYOUT:
            key = _CAMERA_DATASETS[camera]
            if key not in handle:
                raise KeyError(f"{episode_path}: missing {key}")
        length = int(handle[STATE_KEYS[0][0]].shape[0])
        if length < 2:
            raise ValueError(f"{episode_path}: need T>=2, got {length}")
        for camera in DEFAULT_ROBODOJO_CAMERA_LAYOUT:
            key = _CAMERA_DATASETS[camera]
            if int(handle[key].shape[0]) != length:
                raise ValueError(f"{episode_path}:{key} T mismatch")
        instruction = _decode_instruction(handle["instruction"][()], source=str(episode_path))
    return {"length": length, "instruction": instruction}


def _load_mode_stats(path: str | Path, action_mode: str, raw_dim: int) -> dict[str, np.ndarray]:
    stats = load_mode_stats(str(path), action_mode)
    if stats is None:
        raise KeyError(f"{path}: missing nested key {action_mode!r}")
    validated = {}
    for key in STAT_KEYS:
        if key not in stats:
            raise KeyError(f"{path}:{action_mode} is missing {key}")
        value = np.asarray(stats[key], dtype=np.float32)
        if value.shape != (raw_dim,):
            raise ValueError(f"{path}:{action_mode}.{key} must have shape ({raw_dim},), got {value.shape}")
        validated[key] = value
    if np.any(validated["std"] <= 0):
        raise ValueError(f"{path}:{action_mode}.std must be positive")
    if np.any(validated["max"] < validated["min"]):
        raise ValueError(f"{path}:{action_mode}.max must be >= min")
    return validated


class Spark0Dataset(BaseDataset):
    """One Spark0 task rooted at ``<task>/tianji_marvin_wuji/data``."""

    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    def __init__(
        self,
        data_root: str | Path,
        *,
        dataset_root: str | Path,
        task_name: str,
        embodiment: str = SPARK0_EMBODIMENT,
        action_mode: str = DEPLOY_ACTION_MODE,
        num_frames: int = 33,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        window_stride: int = 1,
        video_stride: int = 4,
        multiview: bool = True,
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
        if action_mode not in ("joint", "eef"):
            raise ValueError(f"Spark0 supports action_mode 'joint' or 'eef', got {action_mode!r}")
        if embodiment != SPARK0_EMBODIMENT:
            raise ValueError(f"Spark0 embodiment must be {SPARK0_EMBODIMENT!r}, got {embodiment!r}")
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
        self.num_video_frames = len(self._video_sample_indices)
        self.multiview = bool(multiview)
        self.target_camera = str(target_camera)
        self.contract_id = SPARK0_CONTRACT_ID if action_mode == "joint" else "spark0-tianji-marvin-wuji-alpha-eef-v1"
        self.source_frame = "tianji_marvin_wuji_joint54" if action_mode == "joint" else "tianji_marvin_wuji_alpha_eef"
        self._raw_dim = JOINT54_DIM if action_mode == "joint" else SPARK0_EEF_RAW_DIM
        self._default_unify = ["0-53"] if action_mode == "joint" else list(SPARK0_UNIFY_MAP)

        self._color_jitter = None
        if color_jitter_enabled(color_jitter) and split == "train":
            jitter_get = color_jitter.get if hasattr(color_jitter, "get") else lambda key, default: default
            self._color_jitter = VideoColorJitter(
                brightness=float(jitter_get("brightness", 0.2)),
                contrast=float(jitter_get("contrast", 0.2)),
                saturation=float(jitter_get("saturation", 0.2)),
                hue=float(jitter_get("hue", 0.0)),
            )

        if camera_layout is None:
            camera_layout = DEFAULT_ROBODOJO_CAMERA_LAYOUT
        self.camera_layout = tuple(str(camera) for camera in camera_layout)
        if self.multiview:
            if set(self.camera_layout) != set(DEFAULT_ROBODOJO_CAMERA_LAYOUT):
                raise ValueError(f"Spark0 camera_layout must be {list(DEFAULT_ROBODOJO_CAMERA_LAYOUT)}")
        elif self.target_camera not in _CAMERA_DATASETS:
            raise ValueError(f"target_camera must be one of {list(_CAMERA_DATASETS)}")

        episode_paths = discover_spark0_episodes(dataset_root, self.task_name, embodiment=embodiment)
        expected = Path(dataset_root) / self.task_name / embodiment / "data"
        if Path(data_root).resolve() != expected.resolve():
            raise ValueError(f"data_root {data_root} != {expected}")
        all_metadata = [validate_spark0_episode(path) for path in episode_paths]
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
            raise ValueError(f"no Spark0 windows for task {self.task_name}")

        self._unify_action = bool(unify_action)
        if unify_state_map is not None and list(unify_state_map) != list(self._default_unify):
            raise ValueError(f"Spark0 unify_state_map must be null or {self._default_unify}")
        self._unify_dst_index: np.ndarray | None = None
        if self._unify_action:
            if unify_action_map is None:
                unify_action_map = list(self._default_unify)
            self._unify_dst_index = parse_unify_spec(unify_action_map, UNIFY_DIM)
            if self._unify_dst_index.shape != (self._raw_dim,):
                raise ValueError(
                    f"Spark0 unify_action_map must map {self._raw_dim} raw dims, got {self._unify_dst_index.shape[0]}"
                )
            self._action_dim_value = UNIFY_DIM
        else:
            self._action_dim_value = self._raw_dim

        self.normalize_mode = _normalize_mode(normalize_mode)
        self._mode_stats = None
        self._normalizer = None
        self.normalization_stats_path = None
        if self.normalize_mode is not None:
            if normalization_stats_path is None:
                raise ValueError("Spark0 normalize_mode requires normalization_stats_path")
            stats_path = Path(normalization_stats_path)
            if not stats_path.is_file():
                raise FileNotFoundError(f"Spark0 normalization stats do not exist: {stats_path}")
            self._mode_stats = _load_mode_stats(stats_path, self.action_mode, self._raw_dim)
            self._normalizer = Normalizer(
                mode=YAML_TO_NORM_MODE[self.normalize_mode],
                stats=self._mode_stats,
            )
            self.normalization_stats_path = str(stats_path)

    @property
    def action_dim(self) -> int:
        return self._action_dim_value

    @property
    def normalization_stats(self) -> dict[str, np.ndarray] | None:
        if self._mode_stats is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._mode_stats.items()}

    def denormalize_action(self, action) -> np.ndarray:
        array = np.asarray(action)
        if self._unify_dst_index is not None:
            array = unmap_from_unify(array, self._unify_dst_index)
        if self._normalizer is not None:
            array = self._normalizer.unnormalize(array)
        return np.asarray(array).copy()

    def __len__(self) -> int:
        return len(self._window_index)

    def _frame(self, handle: h5py.File, camera: str, frame_index: int) -> Image.Image:
        dataset_key = _CAMERA_DATASETS[camera]
        return _decode_jpeg(
            handle[dataset_key][frame_index],
            source=f"{handle.filename}:{dataset_key}[{frame_index}]",
        )

    def _video_at(self, handle: h5py.File, frame_index: int) -> Image.Image:
        if self.multiview:
            frames = {camera: self._frame(handle, camera, frame_index) for camera in self.camera_layout}
            return assemble_multiview_layout(frames, list(self.camera_layout), self.height, self.width)
        return crop_and_resize(self._frame(handle, self.target_camera, frame_index), self.height, self.width)

    def _build_sample(self, episode_index: int, start: int) -> dict[str, Any]:
        path = self._episode_files[episode_index]
        episode_length = self._episode_lengths[episode_index]
        actual_length = min(self.num_frames, episode_length - start)
        if actual_length < 2:
            raise IndexError(f"Spark0 window start={start} has no next-state target")
        with h5py.File(path, "r") as handle:
            if self.action_mode == "eef":
                states = pack_spark0_alpha_eef(handle, start, start + actual_length)
            else:
                states = read_joint54(handle, start, start + actual_length)
            sampled_video = []
            for relative_index in self._video_sample_indices:
                source_index = start + relative_index if relative_index < actual_length else episode_length - 1
                sampled_video.append(self._video_at(handle, source_index))
        if actual_length < self.num_frames:
            states = np.concatenate(
                [states, np.repeat(states[-1:], self.num_frames - actual_length, axis=0)],
                axis=0,
            )
        if self._color_jitter is not None:
            sampled_video = self._color_jitter.apply({"video": sampled_video})["video"]
        transformed = states
        if self._normalizer is not None:
            transformed = self._normalizer.normalize(transformed)
        dim_mask = np.ones(self._raw_dim, dtype=bool)
        if self._unify_dst_index is not None:
            transformed, dim_mask = map_to_unify(
                np.asarray(transformed, dtype=np.float32),
                self._unify_dst_index,
                UNIFY_DIM,
            )
        transformed = np.asarray(transformed, dtype=np.float32)
        proprio = torch.from_numpy(transformed[0:1])
        action = torch.from_numpy(transformed[1 : self.num_frames])
        time_validity = torch.tensor(
            [action_index + 1 < actual_length for action_index in range(self.num_action_steps)],
            dtype=torch.bool,
        )
        dimension_validity = torch.from_numpy(dim_mask)
        action_mask = (time_validity[:, None] & dimension_validity[None, :]).contiguous()
        proprio_mask = dimension_validity[None, :].clone()
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


class MultiTaskSpark0Dataset(BaseDataset):
    """Concatenate Spark0 tasks with shared joint54 stats."""

    DEPLOY_ACTION_MODE = DEPLOY_ACTION_MODE

    @classmethod
    def from_config(cls, config, split: str = "train"):
        if _config_get(config, "data_format", None) == "lerobot_v3":
            from openwam.dataloader.dexbench import DexBenchLeRobotDataset
            return DexBenchLeRobotDataset.from_config(config, split=split)
        camera_layout = _config_get(config, "camera_layout", None)
        if camera_layout is not None:
            camera_layout = list(camera_layout)
        return cls(
            dataset_dir=_config_get(config, "dataset_dir", _config_get(config, "dataset_root", None)),
            split=split,
            embodiment=_config_get(config, "embodiment", SPARK0_EMBODIMENT),
            action_mode=_config_get(config, "action_mode", DEPLOY_ACTION_MODE),
            normalization_stats_path=_config_get(config, "normalization_stats_path", None),
            normalize_mode=_config_get(config, "normalize_mode", "min-max"),
            num_frames=int(_config_get(config, "num_frames", 33)),
            height=int(_config_get(config, "height", 384)),
            width=int(_config_get(config, "width", 320)),
            window_stride=int(_config_get(config, "window_stride", 1)),
            video_stride=int(_config_get(config, "video_stride", 4)),
            multiview=bool(_config_get(config, "multiview", True)),
            camera_layout=camera_layout,
            target_camera=_config_get(config, "target_camera", "cam_head"),
            unify_action=bool(_config_get(config, "unify_action", True)),
            unify_action_map=_config_get(config, "unify_action_map", ["0-53"]),
            unify_state_map=_config_get(config, "unify_state_map", None),
            color_jitter=_config_get(config, "color_jitter", None),
        )

    def __init__(self, dataset_dir: str | Path, *, split: str = "train", **dataset_kwargs):
        super().__init__()
        if dataset_dir is None:
            raise ValueError("Spark0 dataset_dir is required")
        embodiment = dataset_kwargs.get("embodiment", SPARK0_EMBODIMENT)
        selected_tasks = discover_spark0_tasks(dataset_dir, embodiment=embodiment)
        self.dataset_dir = str(Path(dataset_dir))
        self.tasks = selected_tasks
        self.split = split
        self.embodiment = embodiment
        self.action_mode = dataset_kwargs.get("action_mode", DEPLOY_ACTION_MODE)
        self._sub_datasets: list[Spark0Dataset] = []
        self._cumulative_lengths: list[int] = []
        cumulative = 0
        for task in selected_tasks:
            episodes = discover_spark0_episodes(dataset_dir, task, embodiment=embodiment)
            dataset = Spark0Dataset(
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
            raise ValueError("Spark0 task discovery produced an empty dataset")
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
    def normalization_stats(self) -> dict[str, np.ndarray] | None:
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
    "JOINT54_DIM",
    "MultiTaskSpark0Dataset",
    "SPARK0_CONTRACT_ID",
    "SPARK0_EMBODIMENT",
    "Spark0Dataset",
    "discover_spark0_episodes",
    "discover_spark0_tasks",
    "read_joint54",
    "validate_spark0_episode",
]
