"""Verify qpos order and wrist wxyz convention against recorded fingertip positions."""
import json
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from paired_pilot import OUT, IDX
from compare_pilot import HandFK

report=[]
for task in ['Close-Drawer','Flip-Mug','Pour-Balls']:
    path=f'/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw/{task}/episode_0.hdf5'
    with h5py.File(path) as f:
        for side in IDX:
            fk=HandFK(side)
            errors=[]
            for i in [0,len(f['observations/qpos'])//2,len(f['observations/qpos'])-2]:
                q=f['observations/qpos'][i][IDX[side]]
                pose=f[f'observations/{side}_ee_pose'][i]
                actual=f[f'observations/{side}_finger_tip_pos'][i]
                predicted=fk(q)@Rotation.from_quat(pose[[4,5,6,3]]).as_matrix().T+pose[:3]
                errors.append(np.linalg.norm(predicted-actual,axis=-1).tolist())
            report.append({'task':task,'side':side,'tip_errors_m':errors})
print(json.dumps(report,indent=2))
(OUT/'raw_fk_check.json').write_text(json.dumps(report,indent=2))
