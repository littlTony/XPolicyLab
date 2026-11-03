"""Known-pose CPU cycle through actual MANO fit/decoder, not a mount-name inference."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from paired_pilot import BASE, OUT

POLICY=BASE/'XPolicyLab/policy/VITRA'
sys.path.insert(0,str(POLICY))
from mano45_runtime import Mano45RuntimeCodec
run=POLICY/'checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True'
codec=Mano45RuntimeCodec('/personal/zijian/Spark_0/Spark-0',inverse_mode='linear',
    linear_inverse_path=run/'mano45_to_wuji20_linear_0902_v1.json',
    linear_inverse_sha256='f0a760427d6fb7fd7aa9c7d63aea8666c67f4f7344c25e93f8d7f1adf05caba5')
results=[]
for side in ['left','right']:
    q=codec.hand_models[side].neutral
    for angles in [[0,0,0],[.4,-.2,.7]]:
        pose=np.eye(4); pose[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix()
        pose[:3,3]=[.25,-.17,.83]
        item={'side':side,'input_euler_xyz':angles}
        try:
            fit=codec.wuji_to_mano(side=side,q_stage_major=q,link7_in_env=pose)
            result=codec.mano_to_wuji(side=side,root_in_env=fit.root_in_env,
                hand_pose_rotvec=fit.hand_pose_rotvec,betas=fit.betas,initial_q_stage_major=q)
            item.update(position_error_m=float(np.linalg.norm(result.link7_in_env[:3,3]-pose[:3,3])),
                orientation_error_rad=float(Rotation.from_matrix(result.link7_in_env[:3,:3]@pose[:3,:3].T).magnitude()),
                q20_rms_rad=float(np.sqrt(np.mean((result.q_stage_major-q)**2))),
                backend=codec.last_inverse_backend[side])
        except Exception as e: item['error']=str(e)
        results.append(item);print(json.dumps(item),flush=True)
(OUT/'wrist_cycle.json').write_text(json.dumps(results,indent=2))
