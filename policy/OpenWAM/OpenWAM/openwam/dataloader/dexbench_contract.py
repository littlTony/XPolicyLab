"""DexBench absolute world EE/Wuji20 <-> OpenWAM-alpha raw58/unified80."""
import re
import numpy as np
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d
from openwam.dataloader.utils.poses import rot6d_to_quat_wxyz
from openwam.dataloader.utils.unify_action import parse_unify_spec, map_to_unify, unmap_from_unify

CONTRACT_ID='dexbench-world-wxyz-wuji20-alpha80-v1'
UNIFY_MAP=['0-2','3-8','10-29','34-36','37-42','44-63']
SLOTS=parse_unify_spec(UNIFY_MAP,80)
ROT6D_DIMS=[3,4,5,6,7,8,32,33,34,35,36,37]
CONTRACT=dict(id=CONTRACT_ID,source='absolute_world_xyz_wxyz_wuji20',
              model='absolute_xyz_rot6d_wuji20_alpha80',unify_map=UNIFY_MAP,
              hand_order='isaac_stage_major_index_middle_pinky_ring_thumb',
              action_alignment='row_action_t_is_recorded_next_target_no_extra_shift',
              cameras=['cam_high','cam_left_wrist','cam_right_wrist'],color='RGB')


def pack54(values):
    values=np.asarray(values,dtype=np.float32)
    if values.shape[-1]!=54 or not np.isfinite(values).all():
        raise ValueError(f'Expected finite absolute EE54, got {values.shape}')
    parts=[]
    for start in [0,27]:
        quat=values[...,start+3:start+7]
        norm=np.linalg.norm(quat,axis=-1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('Zero source quaternion')
        parts += [values[...,start:start+3],quat_wxyz_to_rot6d(quat/norm),values[...,start+7:start+27]]
    return np.concatenate(parts,axis=-1).astype(np.float32)


def unpack58(values):
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or values.shape[-1]!=58 or not np.isfinite(values).all():
        raise ValueError(f'Expected finite physical raw action [T,58], got {values.shape}')
    poses=[]
    for start in [0,29]:
        poses.append(np.concatenate([values[:,start:start+3],rot6d_to_quat_wxyz(values[:,start+3:start+9])],axis=-1))
    return [dict(left_ee_pose=poses[0][i].copy(),left_ee_joint_state=values[i,9:29].copy(),
                 right_ee_pose=poses[1][i].copy(),right_ee_joint_state=values[i,38:58].copy()) for i in range(len(values))]


def scatter58(values):
    return map_to_unify(np.asarray(values,dtype=np.float32),SLOTS,80)


def gather80(values):
    return unmap_from_unify(np.asarray(values,dtype=np.float32),SLOTS)


def task_prompt(instruction,task,embodiment='tianji_wujihand2_real'):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Instruction required')
    prefix=f'[{embodiment}/{task}] '
    if instruction.startswith(prefix):return instruction
    if re.match(r'^\[[^\]]+/[^\]]+\] ',instruction):raise ValueError('Conflicting dataset task prefix')
    return prefix+instruction
