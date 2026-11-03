from pathlib import Path
import sys,json,importlib,h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P.parents[2])]
pkg='gr00t' if P.name=='GR00T_N17' else 'rldx'
from dexbench_config import MODALITIES
from native_contract import pack54,unpack58
Loader=importlib.import_module(pkg+'.data.dataset.lerobot_episode_loader').LeRobotEpisodeLoader
Pose=importlib.import_module(pkg+'.data.state_action.pose').EndEffectorPose
ActionFormat=importlib.import_module(pkg+'.data.types').ActionFormat
L=Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21')
loader=Loader(L,MODALITIES,video_backend='pyav')
source_root=Path(json.loads((L/'meta/source_fingerprint.json').read_text())['source_root'])
source_info=json.loads((source_root/'meta/info.json').read_text())
view_info=json.loads((L/'meta/info.json').read_text())
assert len(loader)==source_info['total_episodes']
assert sum(loader.get_episode_lengths())==source_info['total_frames']==view_info['total_frames']
df=loader._load_parquet_data(0)
assert np.stack(df['state.left_eef']).shape[1]==9
assert np.stack(df['state.left_hand']).shape[1]==20
raw=pq.read_table(L/'data/chunk-000/episode_000000.parquet').to_pandas()
pose=np.asarray(raw['observation.state'].iloc[7])[:9]
native=Pose(translation=pose[:3], rotation=pose[3:], rotation_type="rot6d").homogeneous
physical=unpack58(np.stack(raw['observation.state'].iloc[7:8]))[0]['left_ee_pose']
np.testing.assert_allclose(native[:3,:3],Rotation.from_quat(physical[[4,5,6,3]]).as_matrix(),atol=1e-6)
offsets=json.loads((L/'meta/video_offsets.json').read_text())
episode=next(int(i) for i,v in offsets.items() if v['observation.images.cam_high']>500)
video=loader._load_video_data(episode,np.array([0,1]))
assert list(video)==['front','left_wrist','right_wrist']
for frames in video.values():assert frames.shape==(2,480,640,3) and frames.dtype==np.uint8
stats=loader.get_dataset_statistics()
assert len(stats['state']['left_eef']['mean'])==9
for action in MODALITIES['action'].action_configs:assert action.rep.name=='ABSOLUTE'
Model=importlib.import_module('XPolicyLab.policy.'+P.name+'.model').Model
m=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True))
x=np.zeros(54,np.float32);x[:3]=[.2,.3,.9];x[27:30]=[.1,-.2,.8];x[[3,30]]=1;x[7:27]=np.arange(20)/20;x[34:54]=np.arange(20)/30
state=unpack58(pack54(x[None]))[0]
obs={'state':state,'instruction':'test','vision':{k:{'color':np.zeros((480,640,3),np.uint8)} for k in ['cam_head','cam_left_wrist','cam_right_wrist']}}
m.update_obs(obs);a=m.get_action()[0]
for k,v in state.items():np.testing.assert_allclose(a[k],v,atol=1e-6)
print('PASS',P.name,'native EEF rot6d ROW convention, abs state/action groups, native stats, three real RGB cameras with exact offsets, simulator pose+hand keys;',source_info['total_episodes'],'episodes /',source_info['total_frames'],'rows')

# Exercise the actual adapter native-call path without loading another GPU model.
# The stand-in returns physical targets so this checks packing/history/order only.
class EchoNative:
    def get_action(self, payload, options=None):
        expected=[6] if P.name=='GR00T_N17' else [0,2,4,6]
        for cam in ['front','left_wrist','right_wrist']:
            assert payload['video'][cam].shape==(2,len(expected),480,640,3)
            np.testing.assert_array_equal(payload['video'][cam][0,:,0,0,0],expected)
        assert payload['state']['left_eef'].shape==(2,1,9)
        assert len(payload['language']['annotation.human.task_description'])==2
        return {key:np.repeat(value,16,axis=1) for key,value in payload['state'].items()},{}
m=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True))
for frame in range(7):
    batch=[]
    for idx in [3,1]:
        value=x.copy();value[0]=idx
        batch.append({'env_idx':idx,'state':unpack58(pack54(value[None]))[0],'instruction':'test',
            'vision':{k:{'color':np.full((480,640,3),frame,np.uint8)} for k in ['cam_head','cam_left_wrist','cam_right_wrist']}})
    m.update_obs_batch(batch)
m.dummy=False;m.model=EchoNative()
result=m.get_action_batch([3,1])
assert len(result)==2 and all(len(chunk)==16 for chunk in result)
assert result[0][0]['left_ee_pose'][0]==3 and result[1][0]['left_ee_pose'][0]==1
m.reset();assert not m.histories
print('PASS native inference payload, active batch order, video history stride, reset')


# Verify the actual native train/deploy normalization with the current view.
# Percentile inverse clipping is intentional upstream behavior: compare to
# clipped physical targets rather than requiring an impossible lossless
# roundtrip for percentile outliers.
from rldx.data.dataset.sharded_mixture_dataset import merge_statistics
from rldx.data.state_action.state_action_processor import StateActionProcessor
from rldx.data.embodiment_tags import EmbodimentTag
raw_stats=loader.get_dataset_statistics()
merged={key:merge_statistics([value],[1.],is_relative_stats=key=='relative_action')
        for key,value in raw_stats.items()}
tag=EmbodimentTag.GENERAL_EMBODIMENT.value
processor=StateActionProcessor({tag:MODALITIES},{tag:merged},
    use_relative_action=True,use_percentiles=True,clip_outliers=True)
groups=[('left_eef',0,9),('left_hand',9,29),('right_eef',29,38),('right_hand',38,58)]
assert processor.get_state_dim(tag)==processor.get_action_dim(tag)==58
assert {key:tuple(value['min'].shape) for key,value in processor.norm_params[tag]['action'].items()}=={
    'left_eef':(9,),'left_hand':(20,),'right_eef':(9,),'right_hand':(20,)}
numeric=pq.read_table(L/'data/chunk-000/episode_000000.parquet',
    columns=['observation.state','action']).to_pandas()
states_all=np.stack(numeric['observation.state']);actions_all=np.stack(numeric['action'])
normalized_batches=[];state_batches=[];expected_batches=[]
for anchor in [0,2]:
    current={key:states_all[anchor:anchor+1,start:end] for key,start,end in groups}
    target={key:actions_all[anchor:anchor+16,start:end] for key,start,end in groups}
    normalized=processor.apply_action(target,tag,state=current)
    restored=processor.unapply_action(normalized,tag,state=current)
    expected={}
    for key,value in target.items():
        bounds=processor.norm_params[tag]['action'][key]
        expected[key]=np.clip(value,bounds['min'],bounds['max'])
        np.testing.assert_allclose(restored[key],expected[key],atol=3e-7,rtol=3e-6)
    normalized_batches.append(normalized);state_batches.append(current);expected_batches.append(expected)
restored=processor.unapply_action(
    {key:np.stack([value[key] for value in normalized_batches]) for key,_,_ in groups},tag,
    state={key:np.stack([value[key] for value in state_batches]) for key,_,_ in groups})
errors={}
for key,_,_ in groups:
    expected=np.stack([value[key] for value in expected_batches])
    np.testing.assert_allclose(restored[key],expected,atol=3e-7,rtol=3e-6)
    errors[key]=float(np.max(np.abs(restored[key]-expected)))
raw_absolute_stats=json.loads((L/'meta/stats.json').read_text())
for key in ['observation.state','action']:
    assert raw_absolute_stats[key]['count']==[source_info['total_frames']]
    assert np.asarray(raw_absolute_stats[key]['mean']).shape==(58,)
print('PASS native ABS normalization/unapply single+batch, official percentile-clipped expectation:',errors)
print('PASS shared GR00T relative sidecar does not override ABS state/action stats')


# Inactive histories remain available for future observations, but neither
# default inference nor an explicit inactive ID may produce a stale command.
def active_observation(env_idx):
    value=x.copy();value[0]=env_idx
    return dict(env_idx=env_idx,state=unpack58(pack54(value[None]))[0],instruction='test',
        vision={key:{'color':np.zeros((480,640,3),np.uint8)} for key in ['cam_head','cam_left_wrist','cam_right_wrist']})

active=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True))
active.update_obs_batch([active_observation(3),active_observation(1)])
default=active.get_action_batch()
assert [float(chunk[0]['left_ee_pose'][0]) for chunk in default]==[3.,1.]
active.update_obs_batch([active_observation(3)])
assert len(active.get_action_batch())==1
assert float(active.get_action()[0]['left_ee_pose'][0])==3.
try:active.get_action_batch([1])
except ValueError:pass
else:raise AssertionError('Inactive old environment was allowed to return a stale action')
active.update_obs(active_observation(7))
assert float(active.get_action()[0]['left_ee_pose'][0])==7.
try:active.get_action_batch([3])
except ValueError:pass
else:raise AssertionError('Single-observation update retained an active batch ID')
active.reset()
assert not active.histories and not active._latest_active_ids
try:active.get_action_batch()
except ValueError:pass
else:raise AssertionError('Reset allowed stale default inference')
for invalid in [0,17,-1]:
    try:Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True,replan_steps=invalid))
    except ValueError:pass
    else:raise AssertionError(f'Invalid RLDX horizon accepted: {invalid}')
one=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True,replan_steps=1))
one.update_obs(active_observation(5));assert len(one.get_action())==1
print('PASS latest active default order [3,1] -> [3], inactive ID rejection, single-observation ID, reset and 1..16 execution horizon')
