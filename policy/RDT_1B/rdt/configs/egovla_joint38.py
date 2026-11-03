"""EgoVLA H1 + Inspire joint layout in RDT's 128D state space.

The canonical EgoVLA files store the controllable state/action vector in the
following order::

    left arm (7), left Inspire hand (12), right arm (7), right Inspire hand (12)

RDT has a fixed 128-dimensional state token.  The seven arm values use the
usual arm-position slots.  The twelve hand values use the five gripper
position slots followed by seven arm-velocity slots on the same side.  Those
velocity-labelled slots are simply unused channels in this dataset; the
source values are hand positions.  This gives every source value a unique,
stable slot and is shared by the HDF5 loader and deployment model.
"""

from configs.state_vec import STATE_VEC_IDX_MAPPING, STATE_VEC_LEN


EGOVLA_JOINT38_DIM = 38


def _side_indices(side):
    return (
        [STATE_VEC_IDX_MAPPING[f"{side}_arm_joint_{i}_pos"] for i in range(7)]
        + [STATE_VEC_IDX_MAPPING[f"{side}_gripper_joint_{i}_pos"] for i in range(5)]
        + [STATE_VEC_IDX_MAPPING[f"{side}_arm_joint_{i}_vel"] for i in range(7)]
    )


# Raw order: left arm + left hand + right arm + right hand.
EGOVLA_JOINT38_STATE_INDICES = _side_indices("left") + _side_indices("right")

assert len(EGOVLA_JOINT38_STATE_INDICES) == EGOVLA_JOINT38_DIM
assert len(set(EGOVLA_JOINT38_STATE_INDICES)) == EGOVLA_JOINT38_DIM
assert min(EGOVLA_JOINT38_STATE_INDICES) >= 0
assert max(EGOVLA_JOINT38_STATE_INDICES) < STATE_VEC_LEN
