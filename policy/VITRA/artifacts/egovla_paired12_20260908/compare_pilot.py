"""Compare unchanged geometric decoder and candidate using true USD fingertip FK."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from paired_pilot import OUT, metrics, sha
USD_AUDIT=OUT/'h1_usd_joints.json'
OUT=Path(os.environ.get('VITRA_COMPARE_OUT',str(OUT)))

POLICY = '/personal/xiangpc/0813_Xpolicylab_bench/VITRA/XPolicyLab/policy/VITRA'
sys.path.insert(0,POLICY)
from h1_inspire_retarget import H1InspireWujiAdapter

ORDER = ['index_proximal','index_intermediate','middle_proximal','middle_intermediate',
         'pinky_proximal','pinky_intermediate','ring_proximal','ring_intermediate',
         'thumb_proximal_yaw','thumb_proximal_pitch','thumb_intermediate','thumb_distal']

def transform(pos, wxyz):
    t = np.eye(4)
    t[:3,:3] = Rotation.from_quat(np.asarray(wxyz)[[1,2,3,0]]).as_matrix()
    t[:3,3] = pos
    return t

class HandFK:
    def __init__(self,side):
        self.prefix = 'L' if side=='left' else 'R'
        self.joints = [j for j in json.loads(USD_AUDIT.read_text())['joints']
                       if j['name'].startswith(self.prefix+'_')]
    def __call__(self,q):
        assert q.shape == (12,) and np.isfinite(q).all()
        angles = {f'{self.prefix}_{n}_joint':v for n,v in zip(ORDER,q)}
        frames = {f'/h1/{self.prefix}_hand_base_link':np.eye(4)}
        pending = self.joints.copy()
        while pending:
            progress = False
            for j in pending.copy():
                r,a = j['relations'],j['attrs']
                parent,child = r['physics:body0'][0],r['physics:body1'][0]
                if parent not in frames: continue
                t0 = transform(a['physics:localPos0'],a['physics:localRot0'])
                t1 = transform(a['physics:localPos1'],a['physics:localRot1'])
                motion = np.eye(4)
                if j['type']=='PhysicsRevoluteJoint':
                    axis = np.eye(3)['XYZ'.index(a['physics:axis'])]
                    motion[:3,:3] = Rotation.from_rotvec(axis*angles[j['name']]).as_matrix()
                frames[child] = frames[parent]@t0@motion@np.linalg.inv(t1)
                pending.remove(j)
                progress = True
            assert progress, pending
        return np.array([frames[f'/h1/{self.prefix}_{finger}_tip'][:3,3]
                         for finger in ['thumb','index','middle','ring','pinky']])

def physical(pred,y,fk):
    pp=np.array([fk(q) for q in pred]); yy=np.array([fk(q) for q in y])
    distance=np.linalg.norm(pp-yy,axis=-1)
    opposition=np.linalg.norm((pp[:,1:]-pp[:,:1])-(yy[:,1:]-yy[:,:1]),axis=-1)
    return {'fingertip_rms_m':float(np.sqrt(np.mean(distance**2))),
            'fingertip_max_m':float(distance.max()),
            'per_fingertip_rms_m':np.sqrt(np.mean(distance**2,axis=0)).tolist(),
            'thumb_opposition_vector_rms_m':float(np.sqrt(np.mean(opposition**2)))}

def main():
    adapter=H1InspireWujiAdapter('/personal/zijian/Spark_0/Spark-0')
    report={'adapter_sha256':sha(Path(POLICY)/'h1_inspire_retarget.py'),
            'sampling':'96 evenly spaced held-out rows per side and split; decoder reset per row',
            'usd_sha256':json.loads(USD_AUDIT.read_text())['sha256'],
            'results':{}}
    for side in ['left','right']:
        fk=HandFK(side)
        for split in ['episode','task']:
            data=np.load(OUT/f'heldout_{side}_{split}.npz')
            indices=np.linspace(0,len(data['x'])-1,96,dtype=int)
            old=[]; accepted=[]; failures=[]
            for i in indices:
                adapter.reset()
                try:
                    old.append(adapter.wuji_to_h1(data['x'][i],side)); accepted.append(i)
                except Exception as e: failures.append({'row':int(i),'error':str(e)})
            y=data['y'][accepted]; new=data['pred'][accepted]; old=np.asarray(old)
            result={'accepted':len(accepted),'failures':failures,
                    'old':{**metrics(old,y),**physical(old,y,fk)},
                    'candidate':{**metrics(new,y),**physical(new,y,fk)}}
            report['results'][side+'_'+split]=result
            print(side,split,json.dumps(result),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
