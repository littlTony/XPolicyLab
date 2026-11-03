"""CPU regression for command targets that differ from next measured poses."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'VITRA'),str(P.parents[2])]
from vitra.datasets.lerobot_dataset import native_vectors,native_chunk_actions,native_episode_values,action_slot_count
from vitra.datasets.spark0_dataset import (load_wuji_mapping,matrix_to_pose7,pose7_to_matrix,
    runtime_observation_to_native,native_action_chunk_to_xpolicy,native_action_to_human,human_action_to_native)
from vitra.utils.data_utils import GaussianNormalizer


def main():
    mapping=load_wuji_mapping(P/'mapping_wuji20_mano45.json')
    n=27;rng=np.random.default_rng(71)
    state=np.zeros((n,54),np.float32);action=np.zeros_like(state)
    for side,start in [('left',0),('right',27)]:
        targets=np.broadcast_to(np.eye(4),(n,4,4)).copy()
        targets[:,:3,:3]=Rotation.from_euler('xyz',rng.normal(scale=.35,size=(n,3))).as_matrix()
        targets[:,:3,3]=rng.normal(scale=.2,size=(n,3))
        measured=np.concatenate([targets[:1],targets[:-1]],axis=0).copy()
        measured[:,:3,3]+=[.005,-.003,.002]
        measured[:,:3,:3]=Rotation.from_euler('zyx',[.03,-.02,.04]).as_matrix()@measured[:,:3,:3]
        state[:,start:start+7]=matrix_to_pose7(measured)
        action[:,start:start+7]=matrix_to_pose7(targets)
        state[:,start+7:start+27]=rng.normal(size=(n,20))
        action[:,start+7:start+27]=rng.normal(size=(n,20))
    cameras=np.broadcast_to(np.eye(4),(n,4,4)).copy()
    cameras[:,:3,:3]=Rotation.from_euler('xyz',np.stack([np.linspace(.1,.8,n),np.linspace(-.4,.2,n),np.linspace(.4,-.3,n)],axis=-1)).as_matrix()
    cameras[:,:3,3]=np.stack([np.linspace(.1,.3,n),np.linspace(.2,.5,n),np.linspace(1.,1.2,n)],axis=-1)
    normalizer=GaussianNormalizer({key:(rng.normal(size=26) if key.endswith('mean') else rng.uniform(.2,2.,26))
        for key in ['state_left_mean','state_left_std','state_right_mean','state_right_std',
                    'action_left_mean','action_left_std','action_right_mean','action_right_std']})
    states,slots=native_episode_values(state,action,cameras,mapping)
    assert len(slots)==action_slot_count(n)
    cursor=0;max_pos=0.;max_angle=0.;old_error=0.
    for anchor in range(n):
        targets=action[anchor:anchor+16];valid=len(targets)
        chunk=native_chunk_actions(state[anchor],targets,cameras[anchor],mapping)
        np.testing.assert_allclose(slots[cursor:cursor+valid],chunk,atol=1e-6,rtol=0);cursor+=valid
        observation={'instruction':'regression','state':{},'vision':{'cam_head':dict(color=np.zeros((480,640,3),np.uint8),
            extrinsic_matrix=cameras[anchor],intrinsic_matrix=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]]))}}
        for side,start in [('left',0),('right',27)]:
            observation['state'][side+'_ee_pose']=state[anchor,start:start+7]
            observation['state'][side+'_ee_joint_state']=state[anchor,start+7:start+27]
        runtime=runtime_observation_to_native(observation,mapping)
        np.testing.assert_allclose(runtime['native_state'],states[anchor],atol=1e-6,rtol=0)
        human=native_action_to_human(normalizer.normalize_action(chunk),mapping)
        recovered=normalizer.unnormalize_action(human_action_to_native(human,mapping))
        decoded=native_action_chunk_to_xpolicy(recovered,runtime['context'])
        for k,command in enumerate(decoded):
            for side,start in [('left',0),('right',27)]:
                got=pose7_to_matrix(command[side+'_ee_pose']);expected=pose7_to_matrix(targets[k,start:start+7])
                max_pos=max(max_pos,float(np.max(np.abs(got[:3,3]-expected[:3,3]))))
                max_angle=max(max_angle,float(Rotation.from_matrix(got[:3,:3]@expected[:3,:3].T).magnitude()))
                np.testing.assert_allclose(command[side+'_ee_joint_state'],targets[k,start+7:start+27],atol=5e-7,rtol=0)
        if anchor==0:
            old=native_vectors(state[:valid],targets,cameras[anchor],mapping)[1]
            wrong=native_action_chunk_to_xpolicy(old,runtime['context'])
            old_error=float(np.linalg.norm(wrong[-1]['right_ee_pose'][:3]-targets[-1,27:30]))
    assert max_pos<4e-6 and max_angle<4e-6,(max_pos,max_angle)
    assert old_error>.01,old_error
    # With exact measured_next == previous_target, both encodings coincide.
    contiguous=state.copy();contiguous[1:]=action[:-1]
    old=native_vectors(contiguous[:16],action[:16],cameras[0],mapping)[1]
    new=native_chunk_actions(contiguous[0],action[:16],cameras[0],mapping)
    np.testing.assert_allclose(old,new,atol=1e-7,rtol=0)
    print('PASS command-step-v2: lagged measured states, noncommuting rotations, moving camera anchors, absolute hands, normalized sparse roundtrip, all terminal masks/slot counts')
    print('maximum xyz error',max_pos,'maximum rotation error rad',max_angle,'old encoder drift m',old_error)


if __name__=='__main__':main()
