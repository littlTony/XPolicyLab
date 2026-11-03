"""CPU-only paired decoder experiment; never changes deployment or checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import joblib

BASE = Path('/personal/xiangpc/0813_Xpolicylab_bench/VITRA')
DATA = BASE / 'data/egovla_humanoid_sim_mano_20260902'
OUT = Path(os.environ.get('VITRA_PILOT_OUT','/tmp/vitra_h1_audit_20260908'))
IDX = {'left': [26,36,27,37,28,38,29,39,30,40,46,48],
       'right': [31,41,32,42,33,43,34,44,35,45,47,49]}
KEEP = [8,0,2,6,4,9]
DROP = [1,3,5,7,10,11]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def alias(p):
    p = Path(p)
    if p.exists(): return p
    return Path(str(p).replace('/mnt/xspark-data/', '/personal/', 1))

def metrics(pred, y):
    e = pred-y
    return {'frames': len(y), 'rms_rad': float(np.sqrt(np.mean(e**2))),
            'retained6_rms_rad': float(np.sqrt(np.mean(e[:,KEEP]**2))),
            'discarded6_rms_rad': float(np.sqrt(np.mean(e[:,DROP]**2))),
            'per_joint_rms_rad': np.sqrt(np.mean(e**2,axis=0)).tolist(),
            'max_abs_rad': float(np.max(abs(e)))}

def main():
    manifest = DATA/'spark0_manifest.json'
    stats = DATA/'teledata_statistics.json'
    assert sha(stats) == '8c0170503168fba85850a0842c35081f00e9f9084b06d678381a73742b107003'
    assert sha(manifest) == 'ba6aa23212581ba1b426853fe4ac13129bb13e9d4c9cf754be4b868f89f26ee2'
    json.loads(stats.read_text())
    m = json.loads(manifest.read_text())
    episodes = m['episodes']
    rows = {s:{k:[] for k in ['train','episode','task']} for s in IDX}
    sources, excluded = [], []
    counts = {}
    for ep in episodes:
        task = ep['task']
        counts[task] = counts.get(task,0)+1
        if counts[task]>40: continue  # explicit fixed-size pilot, not full-dataset claim
        p = alias(ep['source'])
        with h5py.File(p,'r') as f:
            prov = json.loads(f['metadata/provenance_json'][()].decode())
            raw = alias(prov['source_path'])
            if not raw.exists():
                excluded.append({'source':str(p),'reason':'provenance raw source missing'})
                continue
            trim = prov['validity_trim']
            start = trim['kept_start']
            n = ep['training_transitions']
            assert prov['action_semantics'] == 'next_state_truncated:1', prov
            with h5py.File(raw,'r') as rf:
                qpos = rf['observations/qpos'][()]
            assert len(qpos) == trim['source_frames']
            assert n+1 == ep['source_frames']
            assert trim['kept_frames']-1 == ep['source_frames']
            split = ('task' if task in ['Flip-Mug','Pour-Balls'] else
                     'episode' if int(p.stem.split('_')[-1])%5 == 0 else 'train')
            for side,idx in IDX.items():
                a = f[f'action/{side}_ee_joint_states'][()]
                assert a.shape == (ep['source_frames'],20), a.shape
                x = a[:n].astype(np.float64)
                y = qpos[start+1:start+1+n][:,idx].astype(np.float64)
                assert y.shape == (n,12)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                rows[side][split].append((x,y))
            sources.append({'task':task,'processed':str(p),'raw':str(raw),
                            'split':split,'frames':n,'kept_start':start,
                            'provenance':prov})
    report = {'scope':'pilot: first 40 manifest episodes per task; all training transitions',
              'statistics_sha256':sha(stats),'manifest_sha256':sha(manifest),
              'source_sha256':sha(__file__) if __file__ != '<stdin>' else None,
              'action_alignment':'raw qpos[kept_start+t+1]; no clipping/filtering/padding',
              'held_tasks':['Flip-Mug','Pour-Balls'], 'sources':sources,'excluded':excluded,
              'warning':'12D conditional prediction cannot invert discarded independent joints',
              'metrics':{}}
    OUT.mkdir(exist_ok=True)
    for side in IDX:
        arrays = {k:tuple(np.concatenate([v[i] for v in vs]) for i in [0,1])
                  for k,vs in rows[side].items()}
        model = ExtraTreesRegressor(n_estimators=64,min_samples_leaf=8,max_depth=22,
                                   random_state=20260908,n_jobs=4)
        print('FIT',side,{k:len(v[0]) for k,v in arrays.items()},flush=True)
        xtrain,ytrain=arrays['train']
        lower=np.array([0]*8+[-.1,0,0,0])
        upper=np.array([1.7]*8+[1.3,.5,.8,1.2])
        valid=np.all((ytrain>=lower)&(ytrain<=upper),axis=1)
        report.setdefault('training_limits',{})[side]={
            'included':int(valid.sum()),'excluded':int((~valid).sum()),
            'rule':'all 12 raw labels within true H1 joint limits; no labels altered',
            'lower':lower.tolist(),'upper':upper.tolist()}
        model.fit(xtrain[valid],ytrain[valid])
        joblib.dump(model,OUT/f'paired12_pilot_{side}.joblib')
        report['metrics'][side] = {}
        for k in ['episode','task']:
            x,y = arrays[k]
            pred = model.predict(x)
            report['metrics'][side][k] = metrics(pred,y)
            np.savez_compressed(OUT/f'heldout_{side}_{k}.npz',x=x,y=y,pred=pred)
        print(json.dumps(report['metrics'][side]),flush=True)
    (OUT/'paired12_pilot_report.json').write_text(json.dumps(report,indent=2))

if __name__ == '__main__': main()
