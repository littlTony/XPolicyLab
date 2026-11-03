"""Pack Spark0 / EgoVLA next-state into the OpenWAM-α 80-D EEF slot contract.

Raw vectors omit unused gripper / hand-pad slots; unify_action_map scatters
into the official α layout:

    0:3    left xyz     3:9    left rot6d    9      left gripper (unused)
    10:34  left hand24  34:37  right xyz     37:43  right rot6d
    43     right gripper (unused)            44:68  right hand24
"""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d, quat_xyzw_to_rot6d
from openwam.dataloader.utils.normalization import pin_rot6d_identity

# FastWAM / XPolicyLab compact 38-D joints from raw EgoVLA 50-D qpos.
POLICY_JOINT_INDICES = np.asarray(
    (
        4, 8, 12, 16, 20, 22, 24,
        26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48,
        5, 9, 13, 17, 21, 23, 25,
        31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49,
    ),
    dtype=np.int64,
)
# Compact 38-D layout: left_arm7 + left_hand12 + right_arm7 + right_hand12.
LEFT_HAND_INDICES = POLICY_JOINT_INDICES[7:19]
RIGHT_HAND_INDICES = POLICY_JOINT_INDICES[26:38]

SPARK0_EEF_RAW_DIM = 58
EGOVLA_EEF_RAW_DIM = 42
SPARK0_UNIFY_MAP = ["0-2", "3-8", "10-29", "34-36", "37-42", "44-63"]
EGOVLA_UNIFY_MAP = ["0-2", "3-8", "10-21", "34-36", "37-42", "44-55"]
SPARK0_ROT6D_RAW = (3, 4, 5, 6, 7, 8, 32, 33, 34, 35, 36, 37)
EGOVLA_ROT6D_RAW = (3, 4, 5, 6, 7, 8, 24, 25, 26, 27, 28, 29)


def _pose7_to_xyz_rot6d(pose, *, xyzw: bool) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must be (T, 7), got {pose.shape}")
    xyz = pose[:, :3]
    quat = pose[:, 3:7]
    rot = quat_xyzw_to_rot6d(quat) if xyzw else quat_wxyz_to_rot6d(quat)
    return np.concatenate([xyz, rot.astype(np.float32)], axis=-1)


def pack_spark0_alpha_eef(handle, start: int | None = None, end: int | None = None) -> np.ndarray:
    sl = slice(start, end)
    left = _pose7_to_xyz_rot6d(handle["state/left_base_ee_poses"][sl], xyzw=True)
    right = _pose7_to_xyz_rot6d(handle["state/right_base_ee_poses"][sl], xyzw=True)
    lhand = np.asarray(handle["state/left_ee_joint_states"][sl], dtype=np.float32)
    rhand = np.asarray(handle["state/right_ee_joint_states"][sl], dtype=np.float32)
    if lhand.shape[-1] != 20 or rhand.shape[-1] != 20:
        raise ValueError(f"Wuji hands must be 20-D, got {lhand.shape} {rhand.shape}")
    packed = np.concatenate([left, lhand, right, rhand], axis=-1)
    if packed.shape[1] != SPARK0_EEF_RAW_DIM:
        raise ValueError(f"Spark0 α-EEF raw dim {packed.shape[1]} != {SPARK0_EEF_RAW_DIM}")
    if not np.isfinite(packed).all():
        raise ValueError("Spark0 α-EEF contains non-finite values")
    return packed


def _egovla_pose7(handle, side: str, sl):
    # Official EgoVLA tasks mix two names: left_ee_pose vs left_curr_ee_pose.
    candidates = (
        f"observations/{side}_ee_pose",
        f"observations/{side}_curr_ee_pose",
    )
    for key in candidates:
        if key in handle:
            return _pose7_to_xyz_rot6d(handle[key][sl], xyzw=False)
    raise KeyError(f"EgoVLA episode missing EE pose; tried {candidates}")


def pack_egovla_alpha_eef(handle, start: int | None = None, end: int | None = None) -> np.ndarray:
    sl = slice(start, end)
    left = _egovla_pose7(handle, "left", sl)
    right = _egovla_pose7(handle, "right", sl)
    qpos = np.asarray(handle["observations/qpos"][sl], dtype=np.float32)
    lhand = qpos[:, LEFT_HAND_INDICES]
    rhand = qpos[:, RIGHT_HAND_INDICES]
    packed = np.concatenate([left, lhand, right, rhand], axis=-1)
    if packed.shape[1] != EGOVLA_EEF_RAW_DIM:
        raise ValueError(f"EgoVLA α-EEF raw dim {packed.shape[1]} != {EGOVLA_EEF_RAW_DIM}")
    if not np.isfinite(packed).all():
        raise ValueError("EgoVLA α-EEF contains non-finite values")
    return packed


def finalize_eef_stats(stats: dict, rot6d_dims) -> dict:
    pin_rot6d_identity(stats, rot6d_dims)
    return stats
