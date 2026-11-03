"""Legacy absolute-action registration for Spark dual-arm joint54 data.

New SparkArena runs use ``spark_joint54_relative_config.py``.  This module is
kept unchanged at the action-contract level so existing absolute-action
checkpoints remain loadable.
"""

import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


SPARK_JOINT54_MODALITY_CONFIG = {
    "video": ModalityConfig(
        # The official video feature rewrites this to [-6, -4, -2, 0] from
        # --video-length 4 and --video-stride 2.
        delta_indices=[0],
        modality_keys=["cam_head", "cam_left_wrist", "cam_right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_hand", "right_arm", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "left_hand", "right_arm", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in range(4)
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(SPARK_JOINT54_MODALITY_CONFIG, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(
    copy.deepcopy(SPARK_JOINT54_MODALITY_CONFIG), EmbodimentTag.NEW_EMBODIMENT
)
