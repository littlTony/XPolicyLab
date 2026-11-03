"""OpenPI transforms for EgoVLA's H1 + Inspire upper-body joint policy.

Action layout (38D): left arm 7, left hand 12, right arm 7, right hand 12.
The LeRobot dataset always exposes three image features and carries a
``camera_mask`` so episodes without wrist cameras use masked black frames.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

EGOVLA_ACTION_DIM = 38


def _parse_image(image: object) -> np.ndarray:
    """Return one image as uint8 HWC without changing its color order."""

    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"expected one image with ndim=3, got {value.shape}")
    if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        value = einops.rearrange(value, "c h w -> h w c")
    if value.shape[-1] not in (1, 3):
        raise ValueError(f"expected HWC/CHW image with 1 or 3 channels, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        finite_max = float(np.nanmax(value)) if value.size else 0.0
        scale = 255.0 if finite_max <= 1.0 + 1e-6 else 1.0
        value = np.clip(value * scale, 0.0, 255.0).astype(np.uint8)
    elif value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


@dataclasses.dataclass(frozen=True)
class EgoVLAInputs(transforms.DataTransformFn):
    """Convert EgoVLA LeRobot/runtime observations to Pi0.5 model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_value = data.get("observation/left_wrist_image")
        right_value = data.get("observation/right_wrist_image")
        left_image = _parse_image(left_value) if left_value is not None else np.zeros_like(base_image)
        right_image = _parse_image(right_value) if right_value is not None else np.zeros_like(base_image)

        default_mask = np.asarray(
            [True, left_value is not None, right_value is not None], dtype=bool
        )
        camera_mask = np.asarray(data.get("observation/camera_mask", default_mask), dtype=bool)
        if camera_mask.shape != (3,):
            raise ValueError(f"camera mask must have shape (3,), got {camera_mask.shape}")
        camera_mask = camera_mask & default_mask

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != EGOVLA_ACTION_DIM:
            raise ValueError(
                f"EgoVLA state must end in {EGOVLA_ACTION_DIM} dims, got {state.shape}"
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(camera_mask[0]),
                "left_wrist_0_rgb": np.bool_(camera_mask[1]),
                "right_wrist_0_rgb": np.bool_(camera_mask[2]),
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != EGOVLA_ACTION_DIM:
                raise ValueError(
                    f"EgoVLA actions must end in {EGOVLA_ACTION_DIM} dims, got {actions.shape}"
                )
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class EgoVLAOutputs(transforms.DataTransformFn):
    """Keep the complete 38D joint action instead of Aloha's 14D slice."""

    action_dim: int = EGOVLA_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., : self.action_dim]}
