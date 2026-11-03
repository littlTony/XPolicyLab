"""DexBench absolute EE58: native UniT transforms, padded128, camera offsets."""
import json
from pathlib import Path

import numpy as np
import torch
from gr00t.data.dataset import LeRobotSingleDatasetWithGoalImage
from bounded_video import get_frames_by_timestamps_bounded
from gr00t.experiment.data_config_unit import BaseDataConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy
from gr00t.model.transforms import GR00TTransformWithGoalImage

PARTS = ("left_eef", "left_hand", "right_eef", "right_hand")
CAMERAS = ("front", "left_wrist", "right_wrist")


class EE58DataConfig(BaseDataConfig):
    video_keys = [f"video.{key}" for key in CAMERAS]
    state_keys = [f"state.{key}" for key in PARTS]
    action_keys = [f"action.{key}" for key in PARTS]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self, augment=True):
        transforms = [VideoToTensor(apply_to=self.video_keys)]
        if augment:
            transforms.append(VideoCrop(apply_to=self.video_keys, scale=0.95))
        transforms.append(VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"))
        if augment:
            transforms.append(VideoColorJitter(apply_to=self.video_keys, brightness=0.3,
                                               contrast=0.4, saturation=0.5, hue=0.08))
        transforms.append(VideoToNumpy(apply_to=self.video_keys))
        for keys in (self.state_keys, self.action_keys):
            transforms.extend([StateActionToTensor(apply_to=keys,
                                                    output_dtypes={key: torch.float32 for key in keys}),
                               StateActionTransform(apply_to=keys,
                                                    normalization_modes={key: "mean_std" for key in keys})])
        transforms.extend([
            ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys,
                            action_concat_order=self.action_keys),
            GR00TTransformWithGoalImage(state_horizon=1, action_horizon=16, max_state_dim=128,
                                       max_action_dim=128, eagle_path=self.eagle_path,
                                       use_bridge=self.use_bridge, ignore_lang_prefix=self.ignore_lang_prefix,
                                       enable_imagenet_preprocessing=True, vision_model_type="dinov2",
                                       num_bridge_tokens=self.num_bridge_tokens,
                                       tokenizer_only=self.tokenizer_only),
        ])
        return ComposedModalityTransform(transforms=transforms)


class EE58Dataset(LeRobotSingleDatasetWithGoalImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_offsets = json.loads((self.dataset_path / "meta/video_offsets.json").read_text())

    def video_query(self, trajectory_id, key, base_index):
        """Current/goal local indices plus the independent offset of each camera."""
        horizon = len(self.modality_configs["action"].delta_indices)
        indices = self.delta_indices[key] + base_index
        indices = np.concatenate((indices, indices[-1:] + horizon))
        length = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
        indices = np.clip(indices, 0, length - 1)
        alias = key.removeprefix("video.")
        original = self.lerobot_modality_meta.video[alias].original_key
        offset = int(self.video_offsets[str(int(trajectory_id))][original])
        timestamps = self.curr_traj_data["timestamp"].to_numpy()[indices] + offset / 25.0
        return indices, offset, timestamps

    def get_video(self, trajectory_id, key, base_index):
        indices, offset, timestamps = self.video_query(trajectory_id, key, base_index)
        path = self.get_video_path(trajectory_id, key.removeprefix("video."))
        return get_frames_by_timestamps_bounded(path.as_posix(), timestamps)

