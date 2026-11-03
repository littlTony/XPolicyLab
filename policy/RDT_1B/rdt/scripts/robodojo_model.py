"""RoboDojo deployment model; skips ALOHA gripper rescaling used in agilex_model.py."""

import os

import torch

from configs.tianji_joint54 import (
    TIANJI_JOINT54_DIM,
    TIANJI_JOINT54_STATE_INDICES,
)
from configs.egovla_joint38 import (
    EGOVLA_JOINT38_DIM,
    EGOVLA_JOINT38_STATE_INDICES,
)
from scripts.agilex_model import (
    AGILEX_STATE_INDICES,
    RoboticDiffusionTransformerModel as _BaseRoboticDiffusionTransformerModel,
)


class RoboticDiffusionTransformerModel(_BaseRoboticDiffusionTransformerModel):
    @staticmethod
    def _indices_for_joint_dim(joint_dim):
        if joint_dim == EGOVLA_JOINT38_DIM:
            return EGOVLA_JOINT38_STATE_INDICES
        if joint_dim == TIANJI_JOINT54_DIM:
            return TIANJI_JOINT54_STATE_INDICES
        if joint_dim == len(AGILEX_STATE_INDICES):
            return AGILEX_STATE_INDICES
        raise ValueError(
            f"Unsupported RoboDojo joint dimension {joint_dim}; "
            f"expected {EGOVLA_JOINT38_DIM} (EgoVLA H1 + Inspire), "
            f"expected {TIANJI_JOINT54_DIM} (Tianji + Wuji) or "
            f"{len(AGILEX_STATE_INDICES)} (Mobile ALOHA compatibility)."
        )

    def _format_joint_to_state(self, joints):
        B, N, joint_dim = joints.shape
        active_indices = self._indices_for_joint_dim(joint_dim)
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state[:, :, active_indices] = joints
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, active_indices] = 1
        self._robodojo_active_state_indices = active_indices
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        active_indices = getattr(self, "_robodojo_active_state_indices", None)
        if active_indices is None:
            raise RuntimeError(
                "Joint layout is unknown. Call _format_joint_to_state before "
                "decoding an action so training and deployment use the same map."
            )
        return action[:, :, active_indices]


def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)
    return model
