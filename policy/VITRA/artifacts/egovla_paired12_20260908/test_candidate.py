"""Deterministic decoder equality, finite/shape rejection and unchanged 80mm gate."""
import json
import joblib
import numpy as np
from paired_pilot import OUT,sha
from compare_pilot import H1InspireWujiAdapter
from h1_paired12_inverse import Paired12Decoder,Paired12Adapter

path=OUT/'bounded/paired12_0908_v1.json'
d=Paired12Decoder(path,sha(path),{'representation':'mano45','action_type':'ee'})
a=Paired12Adapter(H1InspireWujiAdapter('/personal/zijian/Spark_0/Spark-0'),d)
result={}
for side in ['left','right']:
    data=np.load(OUT/'bounded'/f'heldout_{side}_task.npz')
    indices=np.linspace(0,len(data['x'])-1,96,dtype=int)
    forest=joblib.load(OUT/'bounded'/f'paired12_pilot_{side}.joblib')
    delta=[]; failures=[]; rms=[]
    for i in indices:
        x=data['x'][i]
        pred=d.predict(side,x)
        delta.append(float(np.max(abs(pred-forest.predict(x[None])[0]))))
        try:
            a.wuji_to_h1(x,side)
            rms.append(a.last_diagnostics[f'0:{side}:wuji_to_h1']['rms_m'])
        except Exception as e: failures.append({'row':int(i),'error':str(e)})
    assert max(delta)<1e-12,delta
    for bad in [np.zeros(19),np.zeros(21),np.full(20,np.nan)]:
        try: d.predict(side,bad)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    result[side]={'equality_max_abs_rad':max(delta),'frames':len(indices),
                  'geometry_max_rms_m':max(rms) if rms else None,'failures':failures}
print(json.dumps(result,indent=2))
(OUT/'candidate_unit_tests.json').write_text(json.dumps(result,indent=2))
