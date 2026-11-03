"""Tianji + Wuji joint layout in RDT's 128-dimensional unified space.

The raw RoboDojo vector order is:
    left arm (7), left hand (20), right arm (7), right hand (20)

RDT's pretrained architecture has a fixed 128-dimensional state/action token.
This mapping assigns every raw joint to one unique token slot.  The slots
labelled as velocity below are used as additional learned hand-joint channels;
they do not imply that the source values are velocities.  Training and
deployment must both use this exact mapping.
"""

from configs.state_vec import STATE_VEC_IDX_MAPPING, STATE_VEC_LEN


TIANJI_JOINT54_DIM = 54


def _joint_indices(side):
    return (
        [STATE_VEC_IDX_MAPPING[f"{side}_arm_joint_{i}_pos"] for i in range(7)]
        + [STATE_VEC_IDX_MAPPING[f"{side}_gripper_joint_{i}_pos"] for i in range(5)]
        + [STATE_VEC_IDX_MAPPING[f"{side}_arm_joint_{i}_vel"] for i in range(10)]
        + [STATE_VEC_IDX_MAPPING[f"{side}_gripper_joint_{i}_vel"] for i in range(5)]
    )


# Raw order: left arm + left hand + right arm + right hand.
TIANJI_JOINT54_STATE_INDICES = _joint_indices("left") + _joint_indices("right")

assert len(TIANJI_JOINT54_STATE_INDICES) == TIANJI_JOINT54_DIM
assert len(set(TIANJI_JOINT54_STATE_INDICES)) == TIANJI_JOINT54_DIM
assert min(TIANJI_JOINT54_STATE_INDICES) >= 0
assert max(TIANJI_JOINT54_STATE_INDICES) < STATE_VEC_LEN
