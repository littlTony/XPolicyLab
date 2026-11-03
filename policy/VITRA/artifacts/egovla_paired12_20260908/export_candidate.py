"""Export auditable numeric forests and run-bound sidecar; no pickle in runtime."""
import json
import joblib
import numpy as np
from pathlib import Path
import sys
from paired_pilot import BASE,DATA,OUT,sha

policy=BASE/'XPolicyLab/policy/VITRA'
run=policy/'checkpoints/egovla_humanoid_mano-20260902-tianji_marvin_wuji-ee-42/2026-09-02-egovla_humanoid_mano-20260902_TB64_B8_bf16True'
sys.path.insert(0,str(policy))
from h1_inspire_retarget import H1InspireWujiAdapter
a=H1InspireWujiAdapter('/personal/zijian/Spark_0/Spark-0')
destination=OUT/'bounded'
report=json.loads((destination/'paired12_pilot_report.json').read_text())
meta={'schema':'vitra_0902_wuji20_h1_independent12_v1',
      'status':'experimental_cpu_candidate_not_production_validated',
      'provenance':{'config_sha256':sha(run/'config.json'),
                    'statistics_sha256':sha(DATA/'teledata_statistics.json'),
                    'manifest_sha256':sha(DATA/'spark0_manifest.json'),
                    'representation':'mano45','action_type':'ee','model_chunk':16,
                    'state_action_per_hand':[61,51]},
      'lower_rad':[0]*8+[-.1,0,0,0],'upper_rad':[1.7]*8+[1.3,.5,.8,1.2],
      'output_order':['index_proximal','index_intermediate','middle_proximal','middle_intermediate',
                      'pinky_proximal','pinky_intermediate','ring_proximal','ring_intermediate',
                      'thumb_proximal_yaw','thumb_proximal_pitch','thumb_intermediate','thumb_distal'],
      'input_order':'wuji20_stage_major_index_middle_pinky_ring_thumb',
      'units':'radians', 'irreversible':'six independent raw joints were discarded; outputs are conditional estimates',
      'split_and_metrics':report,'forests':{},'sources':[]}
sources=[run/'config.json',DATA/'teledata_statistics.json',DATA/'spark0_manifest.json',
         run/'checkpoints/epoch=10-step=80000.ckpt/weights.pt',
         run/'mano45_to_wuji20_linear_0902_v1.json',
         Path(json.loads((OUT/'h1_usd_joints.json').read_text())['path']),
         policy/'h1_inspire_retarget.py',Path(__file__).parent/'paired_pilot_bounded.py',
         Path(__file__).parent/'h1_paired12_inverse.py']
for value in a._source_files.values():
    if isinstance(value,Path): sources.append(value)
for side in ['left','right']:
    model=joblib.load(destination/f'paired12_pilot_{side}.joblib')
    arrays={k:[] for k in ['left','right','feature','threshold','value']};roots=[];offset=0
    for est in model.estimators_:
        tree=est.tree_;roots.append(offset)
        arrays['left'].append(np.where(tree.children_left<0,-1,tree.children_left+offset))
        arrays['right'].append(np.where(tree.children_right<0,-1,tree.children_right+offset))
        arrays['feature'].append(tree.feature);arrays['threshold'].append(tree.threshold)
        assert tree.value.shape==(tree.node_count,12,1)
        arrays['value'].append(tree.value[:,:,0]);offset+=tree.node_count
    filename=f'paired12_{side}_0908_v1.npz'
    np.savez_compressed(destination/filename,roots=np.array(roots),**{k:np.concatenate(v) for k,v in arrays.items()})
    meta['forests'][side]={'file':filename,'sha256':sha(destination/filename)}
for p in dict.fromkeys(sources):
    print('HASH',str(p),flush=True)
    meta['sources'].append({'path':str(p),'sha256':sha(p)})
output=destination/'paired12_0908_v1.json'
output.write_text(json.dumps(meta,indent=2))
print('SIDECAR',str(output),sha(output),flush=True)
