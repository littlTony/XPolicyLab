"""18 paired samples, with raw affine MANO prediction explicitly identified."""
import base64
import json
import h5py
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from paired_pilot import BASE,OUT,IDX,KEEP,metrics
from compare_pilot import H1InspireWujiAdapter,HandFK,physical

run=BASE/'XPolicyLab/policy/VITRA/checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True'
affine=json.loads((run/'mano45_to_wuji20_linear_0902_v1.json').read_text())
def array(p): return np.frombuffer(base64.b64decode(p['data_b64']),dtype=p['dtype']).reshape(p['shape'])
a=H1InspireWujiAdapter('/personal/zijian/Spark_0/Spark-0')
models={s:joblib.load(OUT/'bounded'/f'paired12_pilot_{s}.joblib') for s in IDX}
rows=[]
for task in ['Close-Drawer','Flip-Mug','Pour-Balls']:
    path=f'/personal/xspark_shared/hand_data/hdf5/egovla_humanoid_sim_processed/{task}/episode_0.hdf5'
    with h5py.File(path) as f:
        prov=json.loads(f['metadata/provenance_json'][()].decode())
        with h5py.File(prov['source_path']) as raw:
            qpos=raw['observations/qpos'][()]
        for side in IDX:
            x=f[f'action/{side}_ee_joint_states'][()]
            mano=f[f'mano/action/{side}_ee_joint_states'][()]
            for i in [0,len(x)//2,len(x)-2]:
                y=qpos[prov['validity_trim']['kept_start']+i+1,IDX[side]]
                coef=affine['sides'][side]
                euler=Rotation.from_rotvec(mano[i].reshape(15,3)).as_euler('xyz').reshape(45)
                z=(euler-array(coef['reverse_center']))/array(coef['reverse_scale'])
                q20=np.r_[1.,z]@array(coef['reverse_weights'])
                a.reset(); old=a.wuji_to_h1(x[i],side)
                pred=models[side].predict(x[i:i+1])[0]
                chain=models[side].predict(q20[None])[0]
                item={'task':task,'side':side,'row':i,
                      'mano_affine_q20_rms_rad':float(np.sqrt(np.mean((q20-x[i])**2))),
                      'affine_scope':'raw affine prediction, not public MANO runtime safety/fallback',
                      'old':metrics(old[None],y[None]),'paired_q20':metrics(pred[None],y[None]),
                      'paired_raw_affine_fullchain':metrics(chain[None],y[None]),
                      'thumb_pitch_truth_old_new':[float(y[9]),float(old[9]),float(pred[9])],
                      'q20_geometry':physical(pred[None],y[None],HandFK(side))}
                rows.append(item)
print(json.dumps({'rows':len(rows),'mean_old_q6_rms_rad':np.mean([r['old']['retained6_rms_rad'] for r in rows]),
                  'mean_new_q6_rms_rad':np.mean([r['paired_q20']['retained6_rms_rad'] for r in rows]),
                  'mean_new_fullchain_q12_rms_rad':np.mean([r['paired_raw_affine_fullchain']['rms_rad'] for r in rows])}))
(OUT/'fullchain_18.json').write_text(json.dumps(rows,indent=2))
