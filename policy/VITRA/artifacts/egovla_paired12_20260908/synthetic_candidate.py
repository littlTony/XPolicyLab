import json
import joblib
import numpy as np
from paired_pilot import OUT, KEEP
from compare_pilot import H1InspireWujiAdapter, physical, HandFK, metrics
from h1_inspire_retarget import H1_FROM_SPARK

a=H1InspireWujiAdapter('/personal/zijian/Spark_0/Spark-0')
results=[]
for side in ['left','right']:
    candidate=joblib.load(OUT/'bounded'/f'paired12_pilot_{side}.joblib')
    for q6 in [[.2,.3,.5,.7,.8,.2],[1,1.2,1.3,1.4,1.4,.4]]:
        a.reset()
        y=a.inspire_models[side].pack_inspire12(np.array(q6))[list(H1_FROM_SPARK)]
        x=a.h1_to_wuji(y,side)
        old=a.wuji_to_h1(x,side)
        pred=candidate.predict(x[None])[0]
        item={'side':side,'q6':q6,'candidate_q6':pred[KEEP].tolist(),
              'old':metrics(old[None],y[None]),'candidate':metrics(pred[None],y[None]),
              'old_physical':physical(old[None],y[None],HandFK(side)),
              'candidate_physical':physical(pred[None],y[None],HandFK(side))}
        results.append(item); print(json.dumps(item),flush=True)
(OUT/'synthetic_candidate.json').write_text(json.dumps(results,indent=2))
