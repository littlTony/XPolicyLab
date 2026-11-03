"""Absolute world EE54 -> contiguous EE58. No deltas or alpha80 padding."""
import re
import numpy as np
from scipy.spatial.transform import Rotation

CONTRACT={'id':'dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1','source':'absolute_world_xyz_wxyz_wuji20',
          'model':'left_xyz3_rot6d6_hand20_right_xyz3_rot6d6_hand20',
          'hand_order':'isaac_stage_major_index_middle_pinky_ring_thumb',
          'rot6d':'matrix_rows_0_then_1','action_alignment':'action_t_no_extra_shift',
          'cameras':['cam_high','cam_left_wrist','cam_right_wrist'],'color':'RGB'}
CAMERAS=CONTRACT['cameras']
DEFAULT_PROMPT="A video recorded from a robot's point of view executing the following instruction: {task}"

def pack54(values):
    x=np.asarray(values,dtype=np.float32)
    if x.shape[-1]!=54 or not np.isfinite(x).all():raise ValueError('Expected finite EE54')
    parts=[]
    for s in [0,27]:
        q=x[...,s+3:s+7]
        if np.any(np.linalg.norm(q,axis=-1)<1e-8):raise ValueError('Zero wxyz quaternion')
        shape=q.shape[:-1]
        r=Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*shape,3,3)
        parts.extend([x[...,s:s+3],r[...,0,:],r[...,1,:],x[...,s+7:s+27]])
    return np.concatenate(parts,axis=-1).astype(np.float32)

def unpack58(values):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=58 or not np.isfinite(x).all():raise ValueError('Expected physical [T,58]')
    poses=[]
    for s in [0,29]:
        a=x[:,s+3:s+6].astype(np.float64); b=x[:,s+6:s+9].astype(np.float64)
        norm=np.linalg.norm(a,axis=-1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('Degenerate predicted rot6d first row')
        a=a/norm; b=b-np.sum(a*b,axis=-1,keepdims=True)*a
        norm=np.linalg.norm(b,axis=-1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('Degenerate predicted rot6d second row')
        b=b/norm
        q=Rotation.from_matrix(np.stack([a,b,np.cross(a,b)],axis=-2)).as_quat()[:,[3,0,1,2]]
        q=np.where(q[:,:1]<0,-q,q)
        poses.append(np.concatenate([x[:,s:s+3],q],axis=-1).astype(np.float32))
    return [dict(left_ee_pose=poses[0][i].copy(),left_ee_joint_state=x[i,9:29].copy(),
                 right_ee_pose=poses[1][i].copy(),right_ee_joint_state=x[i,38:58].copy()) for i in range(len(x))]

def task_prompt(instruction,task,embodiment='tianji_wujihand2_real'):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Instruction required')
    prefix=f'[{embodiment}/{task}] '
    if instruction.startswith(prefix):return instruction
    if re.match(r'^\[[^\]]+/[^\]]+\] ',instruction):raise ValueError('Conflicting task prefix')
    return prefix+instruction

def image_tensor(cameras,video_size=(544,320)):
    """Identical RGB transforms for decoded train frames and live observations."""
    import torch
    import torchvision.transforms.functional as F
    from fastwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving,CenterCrop,Normalize
    tensors=[]
    for key,slot in zip(CAMERAS,[(256,320),(128,160),(128,160)]):
        x=np.asarray(cameras[key])
        if x.ndim==3:x=x[None]
        if x.ndim!=4 or x.shape[-1]!=3 or x.dtype!=np.uint8:raise ValueError(f'Expected uint8 RGB camera {key}')
        t=torch.from_numpy(np.array(x,copy=True)).permute(0,3,1,2).float()/255
        t=F.resize(t,[240,320],antialias=True)
        tensors.append(F.resize(t,list(slot),antialias=True))
    video=torch.cat([tensors[0],torch.cat(tensors[1:],dim=-1)],dim=-2)
    args={'img_h':int(video_size[0]),'img_w':int(video_size[1])}
    return Normalize({'mean':0.5,'std':0.5})(CenterCrop(args)(ResizeSmallestSideAspectPreserving(args)(video)))

def inference_image(observation,*,processor=None,video_size=(544,320)):
    obs=observation['observation']
    return image_tensor(dict(zip(CAMERAS,[obs[k]['rgb'] for k in ['head_camera','left_camera','right_camera']])),video_size)
