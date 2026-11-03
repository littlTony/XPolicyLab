"""RLDX modality registration for EgoVLA H1 + Inspire joint38 data.

The stored action is the absolute joint command recorded in the source HDF5
``/action`` dataset. RLDX converts only the two arm groups to deltas from the
current state at load time; the dexterous hands remain absolute. No future
observation is used as a training target.
"""

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


EGOVLA_JOINT38_MODALITY_CONFIG = {
    "video": ModalityConfig(
        # launch_train rewrites this to [-6, -4, -2, 0] for length=4/stride=2.
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


register_modality_config(EGOVLA_JOINT38_MODALITY_CONFIG, EmbodimentTag.GENERAL_EMBODIMENT)
