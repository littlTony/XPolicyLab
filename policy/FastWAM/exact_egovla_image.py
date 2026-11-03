"""EgoVLA-only preprocessing that reproduces the FastWAM training recipe."""
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F


_CAMERAS = (
    ("cam_high", "head_camera"),
    ("cam_left_wrist", "left_camera"),
    ("cam_right_wrist", "right_camera"),
)


def _to_recorded_rgb(image: np.ndarray) -> np.ndarray:
    """Return the official 384x384 uint8 RGB observation representation."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "FastWAM EgoVLA images must be decoded HWC uint8 RGB; "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    if min(array.shape[:2]) < 1:
        raise ValueError("FastWAM EgoVLA images must have nonempty spatial dimensions")
    if array.shape[:2] != (384, 384):
        # This is the same native-observation boundary used by the official
        # EgoVLA 30 Hz evaluator before policy preprocessing.
        array = cv2.resize(array, (384, 384), interpolation=cv2.INTER_LINEAR)
    return np.array(array, dtype=np.uint8, order="C", copy=True)


def build_egovla_image_tensor(
    observation: dict,
    *,
    processor: Any,
    video_size: tuple[int, int],
) -> torch.Tensor:
    """Build CPU float32 [1,3,H,W] through the checkpoint's val transforms."""
    from fastwam.datasets.dataset_utils import (
        CenterCrop,
        Normalize,
        ResizeSmallestSideAspectPreserving,
    )

    image_meta = processor.shape_meta["images"]
    if [str(meta["key"]) for meta in image_meta] != [key for key, _ in _CAMERAS]:
        raise ValueError("EgoVLA checkpoint must order cameras as head, left wrist, right wrist")
    if int(processor.num_output_cameras) != 3:
        raise ValueError("EgoVLA checkpoint must use exactly three image streams")
    if int(processor.action_output_dim) != 38 or int(processor.proprio_output_dim) != 38:
        raise ValueError("Exact EgoVLA image preprocessing requires the 38-D EgoVLA processor")
    for meta in image_meta:
        if list(meta["raw_shape"]) != [3, 384, 384] or list(meta["shape"]) != [3, 240, 320]:
            raise ValueError("EgoVLA checkpoint image contract must be 384x384 -> 240x320 RGB")
    height, width = (int(value) for value in video_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid EgoVLA video size: {video_size}")

    images = []
    obs_data = observation["observation"]
    for meta, (key, obs_key) in zip(image_meta, _CAMERAS):
        recorded = _to_recorded_rgb(obs_data[obs_key]["rgb"])
        image = torch.from_numpy(recorded).permute(2, 0, 1).unsqueeze(0)
        transforms = processor.val_transforms
        if isinstance(transforms, Mapping):
            transforms = transforms[key]
        if transforms is None:
            raise ValueError("EgoVLA checkpoint must declare validation image transforms")
        for transform in transforms:
            image = transform(image)
        expected = (1, *[int(value) for value in meta["shape"]])
        if tuple(image.shape) != expected:
            raise ValueError(f"Unexpected transformed camera shape for {key}: {tuple(image.shape)}")
        if image.dtype != torch.float32 or image.device.type != "cpu":
            raise ValueError("EgoVLA validation image transforms must produce CPU float32")
        images.append(image)

    resize_kwargs = {
        "interpolation": transforms_F.InterpolationMode.BILINEAR,
        "antialias": True,
    }
    head = transforms_F.resize(images[0], [256, 320], **resize_kwargs)
    left = transforms_F.resize(images[1], [128, 160], **resize_kwargs)
    right = transforms_F.resize(images[2], [128, 160], **resize_kwargs)
    image = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
    size_args = {"img_h": height, "img_w": width}
    image = ResizeSmallestSideAspectPreserving(args=size_args)(image)
    image = CenterCrop(args=size_args)(image)
    return Normalize(args={"mean": 0.5, "std": 0.5})(image)
