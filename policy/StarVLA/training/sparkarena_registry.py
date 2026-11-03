"""DexBench EE58 registration and packed-MP4 offsets for native StarVLA."""
import json
from pathlib import Path

import numpy as np
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset, ModalityConfig, get_frames_by_timestamps,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

ROBOT_TYPE = "xpolicylab_dexbench_wuji_ee58"
DATA_MIX = "xpolicylab_dexbench_ee58_qwen_pi_v3"
PARTS = ("left_eef", "left_hand", "right_eef", "right_hand")
CAMERAS = ("front", "left_wrist", "right_wrist")


class OffsetLeRobotDataset(LeRobotSingleDataset):
    """Episode videos are links into packed MP4s; offsets are frame counts."""

    def __init__(self, dataset_path, **kwargs):
        dataset_path = Path(dataset_path)
        self.video_offsets = json.loads((dataset_path / "meta/video_offsets.json").read_text())
        self.source_video_fps = float(json.loads((dataset_path / "meta/info.json").read_text())["fps"])
        super().__init__(dataset_path=dataset_path, **kwargs)

    def get_video(self, trajectory_id, key, base_index):
        indices = self.delta_indices[key] + base_index
        length = self.trajectory_lengths[self.get_trajectory_index(trajectory_id)]
        indices = np.clip(indices, 0, length - 1)
        short_key = key.removeprefix("video.")
        original = self.lerobot_modality_meta.video[short_key].original_key
        offset = self.video_offsets[str(int(trajectory_id))][original]
        timestamps = self.curr_traj_data["timestamp"].to_numpy()[indices]
        timestamps = timestamps + float(offset) / self.source_video_fps
        return get_frames_by_timestamps(str(self.get_video_path(trajectory_id, short_key)), timestamps,
            video_backend=self.video_backend, video_backend_kwargs=self.video_backend_kwargs)


class SparkArenaWujiConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    state_keys = [f"state.{key}" for key in PARTS]
    action_keys = [f"action.{key}" for key in PARTS]
    video_keys = [f"video.{key}" for key in CAMERAS]
    language_keys = ["annotation.human.task_description"]
    state_key_dims = dict(zip(state_keys, (9, 20, 9, 20)))
    action_key_dims = dict(zip(action_keys, (9, 20, 9, 20)))

    def make_dataset(self, dataset_name, **kwargs):
        return OffsetLeRobotDataset(**kwargs)

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = []
        for keys in (self.state_keys, self.action_keys):
            transforms.extend([
                StateActionToTensor(apply_to=keys),
                StateActionTransform(apply_to=keys, normalization_modes={key: "q99" for key in keys}),
            ])
        return ComposedModalityTransform(transforms=transforms)


def register(dataset_name):
    from starVLA.dataloader.gr00t_lerobot import registry

    registry.ROBOT_TYPE_CONFIG_MAP[ROBOT_TYPE] = SparkArenaWujiConfig()
    registry.ROBOT_TYPE_TO_EMBODIMENT_TAG[ROBOT_TYPE] = EmbodimentTag.NEW_EMBODIMENT
    registry.DATASET_NAMED_MIXTURES[DATA_MIX] = [(dataset_name, 1.0, ROBOT_TYPE)]
