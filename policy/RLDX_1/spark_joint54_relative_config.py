"""Relative-arm RLDX registration for Spark dual-arm joint54 data.

The LeRobot parquet files continue to store absolute joint targets.  During
training RLDX converts each future arm target to ``q_target - q_current``.
Hand targets remain absolute.  At inference RLDX adds the current arm state
back before the action is returned to SparkArena.
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


SPARK_JOINT54_RELATIVE_MODALITY_CONFIG = {
    "video": ModalityConfig(
        # The launcher rewrites this to [-6, -4, -2, 0] for length=4/stride=2.
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
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(
    SPARK_JOINT54_RELATIVE_MODALITY_CONFIG, EmbodimentTag.GENERAL_EMBODIMENT
)
register_modality_config(
    copy.deepcopy(SPARK_JOINT54_RELATIVE_MODALITY_CONFIG),
    EmbodimentTag.NEW_EMBODIMENT,
)
