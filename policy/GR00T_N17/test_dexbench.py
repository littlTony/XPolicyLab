"""CPU integration checks for native GR00T anchor-relative EEF and ABS hands."""
from pathlib import Path
import importlib
import json
import sys
import tempfile
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

P=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P.parents[2]),str(P/'gr00t_n17')]
from dexbench_config import MODALITIES
from native_contract import pack54,unpack58
from relative_contract import (GROUPS,HORIZON,POLICY_CONTRACT,validate_policy_contract,
    validate_saved_processor,relative_values_digest,native_merged_statistics)
from prepare_relative_stats import relative_windows,rotation_matrices,check_ready
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import to_json_serializable

L=Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21')


def native_relative(state,targets):
    reference=EndEffectorPose.from_action_format(state,ActionFormat.XYZ_ROT6D)
    return EndEffectorActionChunk.from_array(targets,ActionFormat.XYZ_ROT6D).relative_chunking(reference).to(ActionFormat.XYZ_ROT6D)


def encode9(position,rotation):
    return np.concatenate([position,rotation[...,:2,:].reshape(position.shape[:-1]+(6,))],axis=-1).astype(np.float32)


def test_rotation_and_reference():
    rng=np.random.default_rng(37)
    n=HORIZON+8
    rotation=Rotation.from_euler('xyz',rng.uniform(-1,1,(n,3))).as_matrix()
    state=encode9(rng.normal(size=(n,3)),rotation)
    target_rotation=Rotation.from_euler('zyx',rng.uniform(-1,1,(n,3))).as_matrix()
    target=encode9(rng.normal(size=(n,3)),target_rotation)
    vector=relative_windows(state,target)
    np.testing.assert_allclose(vector[3],native_relative(state[3],target[3:3+HORIZON]),atol=6e-7,rtol=2e-6)
    assert relative_windows(state[:HORIZON-1],target[:HORIZON-1]).shape==(0,HORIZON,9)
    assert relative_windows(state[:HORIZON],target[:HORIZON]).shape==(1,HORIZON,9)
    # A shared rigid world-frame change cancels in inv(T0) @ Ttarget.
    g=Rotation.from_euler('xyz',[.7,-.4,.2]).as_matrix();translation=np.array([.3,-.8,1.1])
    new_s=encode9((g@state[:,:3].T).T+translation,g@rotation)
    new_a=encode9((g@target[:,:3].T).T+translation,g@target_rotation)
    np.testing.assert_allclose(relative_windows(new_s,new_a),vector,atol=8e-7,rtol=2e-6)
    # Relative rotations retain an orientation; identity is [1,0,0,0,1,0].
    identical=np.repeat(state[:1],HORIZON,axis=0)
    identity=native_relative(state[0],identical)
    np.testing.assert_allclose(identity[:,3:],np.tile([1,0,0,0,1,0],(HORIZON,1)),atol=1e-7)
    print('PASS non-unit/noncommuting rotations, EEF-local anchor, rigid world-frame invariance, identity row-rot6d')


def test_native_processor(loader,marker):
    tag=EmbodimentTag.NEW_EMBODIMENT.value
    statistics=loader.get_dataset_statistics()
    merged=native_merged_statistics(statistics)
    processor=StateActionProcessor({tag:MODALITIES},{tag:merged},use_relative_action=True,clip_outliers=True)
    assert processor.get_state_dim(tag)==58 and processor.get_action_dim(tag)==58
    assert processor.norm_params[tag]['action']['left_eef']['min'].shape==(HORIZON,9)
    assert processor.norm_params[tag]['action']['left_hand']['min'].shape==(20,)
    frame=pq.read_table(L/'data/chunk-000/episode_000000.parquet',columns=['observation.state','action']).to_pandas()
    s=np.stack(frame['observation.state']);a=np.stack(frame['action'])
    batches=[];states=[]
    for anchor in [0,2]:
        raw_state={k:s[anchor:anchor+1,lo:hi] for k,lo,hi in GROUPS}
        raw_action={k:a[anchor:anchor+HORIZON,lo:hi] for k,lo,hi in GROUPS}
        normalized=processor.apply_action(raw_action,tag,state=raw_state)
        recovered=processor.unapply_action(normalized,tag,state=raw_state)
        for k in recovered:np.testing.assert_allclose(recovered[k],raw_action[k],atol=2e-6,rtol=3e-6)
        batches.append(normalized);states.append(raw_state)
    restored=processor.unapply_action({k:np.stack([v[k] for v in batches]) for k,_,_ in GROUPS},tag,
        state={k:np.stack([v[k] for v in states]) for k,_,_ in GROUPS})
    for k,lo,hi in GROUPS:
        np.testing.assert_allclose(restored[k],np.stack([a[j:j+HORIZON,lo:hi] for j in [0,2]]),atol=2e-6,rtol=3e-6)
    # Execute the actual native model-facing action/state packing and decoder.
    # Substitute only VLM image/token work; no model weights are constructed.
    from types import SimpleNamespace
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    native=object.__new__(Gr00tN1d7Processor)
    native.state_action_processor=processor;native.modality_configs={tag:MODALITIES}
    native.max_action_horizon=HORIZON;native.max_state_dim=132;native.max_action_dim=132
    native.exclude_state=False;native.state_dropout_prob=0.;native.training=False
    native.eval_image_transform=None;native.formalize_language=False
    native.embodiment_id_mapping={tag:31};native._get_vlm_inputs=lambda **kwargs:{}
    content=SimpleNamespace(embodiment=EmbodimentTag.NEW_EMBODIMENT,states=raw_state,
                            actions=raw_action,images={},masks=None,text='CPU horizon replay')
    packed=native([{'content':content}])
    assert packed['state'].shape==(1,132) and packed['action'].shape==(HORIZON,132)
    assert packed['action_mask'].shape==(HORIZON,132)
    np.testing.assert_array_equal(packed['action_mask'][:,:58],np.ones((HORIZON,58)))
    assert not packed['action_mask'][:,58:].any()
    recovered=native.decode_action(packed['action'].numpy(),EmbodimentTag.NEW_EMBODIMENT,state=raw_state)
    for key in recovered:
        assert recovered[key].shape[0]==HORIZON
        np.testing.assert_allclose(recovered[key],raw_action[key],atol=2e-6,rtol=3e-6)
    print('PASS actual native Gr00tN1d7Processor state1x132/action40x132, full40-step mask and decode40; VLM token/image work substituted')
    assert relative_values_digest(statistics['relative_action'])==marker['relative_values_sha256']
    # Test actual saved-processor schema, including native checkpoint/processor/.
    config={'processor_kwargs':{'use_relative_action':True,'modality_configs':to_json_serializable({tag:MODALITIES}),
                               'max_action_horizon':HORIZON,'max_state_dim':132,'max_action_dim':132}}
    with tempfile.TemporaryDirectory(prefix='groot-relative-cpu-') as directory:
        root=Path(directory);saved=root/'processor';saved.mkdir()
        (root/'config.json').write_text(json.dumps({'action_horizon':HORIZON,'max_state_dim':132,'max_action_dim':132}))
        (saved/'processor_config.json').write_text(json.dumps(config))
        (saved/'statistics.json').write_text(json.dumps({tag:merged}))
        assert validate_saved_processor(root,marker)==tag
        (root/'config.json').write_text(json.dumps({'action_horizon':16,'max_state_dim':132,'max_action_dim':132}))
        try:validate_saved_processor(root,marker)
        except ValueError:pass
        else:raise AssertionError('H16 model architecture accepted')
        (root/'config.json').write_text(json.dumps({'action_horizon':HORIZON,'max_state_dim':132,'max_action_dim':132}))
        config['processor_kwargs']['max_action_horizon']=16
        (saved/'processor_config.json').write_text(json.dumps(config))
        try:validate_saved_processor(root,marker)
        except ValueError:pass
        else:raise AssertionError('H16 processor architecture accepted')
        config['processor_kwargs']['max_action_horizon']=HORIZON
        config['processor_kwargs']['use_relative_action']=False
        (saved/'processor_config.json').write_text(json.dumps(config))
        try:validate_saved_processor(root,marker)
        except ValueError:pass
        else:raise AssertionError('Disabled relative processor accepted')
        config['processor_kwargs']['use_relative_action']=True
        (saved/'processor_config.json').write_text(json.dumps(config))
        wrong=json.loads(json.dumps({tag:merged}));wrong[tag]['relative_action']['left_eef']['mean'][0][0]+=.01
        (saved/'statistics.json').write_text(json.dumps(wrong))
        try:validate_saved_processor(root,marker)
        except ValueError:pass
        else:raise AssertionError('Unbound relative stats accepted')
        (saved/'statistics.json').write_text(json.dumps({tag:merged}))
        config['processor_kwargs']['modality_configs'][tag]['action']['action_configs'][0]['type']='non_eef'
        (saved/'processor_config.json').write_text(json.dumps(config))
        try:validate_saved_processor(root,marker)
        except ValueError:pass
        else:raise AssertionError('Direct-subtraction EEF representation accepted')
    try:validate_policy_contract(dict(marker,id='dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1'))
    except ValueError:pass
    else:raise AssertionError('Old absolute checkpoint accepted')
    try:validate_policy_contract(dict(marker,id='dexbench-groot-anchor-relative-eef9-absolute-wuji20-rowrot6d-v2',horizon=16))
    except ValueError:pass
    else:raise AssertionError('Old H16 relative checkpoint accepted')
    print('PASS native normalize/unapply single+batch, per-horizon relative stats, unchanged ABS hands, checkpoint/statistics guards')


def test_images_and_adapter(loader):
    offsets=json.loads((L/'meta/video_offsets.json').read_text())
    episode=next(int(i) for i,v in offsets.items() if v['observation.images.cam_high']>500)
    video=loader._load_video_data(episode,np.array([0,1]))
    assert list(video)==['front','left_wrist','right_wrist']
    for frames in video.values():assert frames.shape==(2,480,640,3) and frames.dtype==np.uint8
    Model=importlib.import_module('XPolicyLab.policy.GR00T_N17.model').Model
    m=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True))
    raw=np.zeros(54,np.float32);raw[:3]=[.2,.3,.9];raw[27:30]=[.1,-.2,.8]
    quat=Rotation.from_euler('xyz',[.6,-.2,.4]).as_quat();raw[3:7]=quat[[3,0,1,2]];raw[30:34]=raw[3:7]
    raw[7:27]=np.arange(20)/20;raw[34:54]=np.arange(20)/30
    class EchoNative:
        def get_action(self,payload):
            for cam in ['front','left_wrist','right_wrist']:
                assert payload['video'][cam].shape==(2,1,480,640,3)
                assert np.all(payload['video'][cam][:,:,0,0,0]==6)
            # Native get_action has ALREADY decoded physical absolute targets.
            return {k:np.repeat(value,HORIZON,axis=1) for k,value in payload['state'].items()},{}
    for t in range(7):
        observations=[]
        for idx in [3,1]:
            x=raw.copy();x[0]=idx
            observations.append({'env_idx':idx,'state':unpack58(pack54(x[None]))[0],'instruction':'test',
                'vision':{k:{'color':np.full((480,640,3),t,np.uint8)} for k in ['cam_head','cam_left_wrist','cam_right_wrist']}})
        m.update_obs_batch(observations)
    m.dummy=False;m.model=EchoNative();result=m.get_action_batch([3,1])
    assert len(result)==2 and all(len(c)==HORIZON for c in result)
    for idx,chunk in zip([3,1],result):
        expected=raw.copy();expected[0]=idx;expected=unpack58(pack54(expected[None]))[0]
        assert set(chunk[0])==set(expected)
        for k in expected:np.testing.assert_allclose(chunk[0][k],expected[k],atol=1e-6)
    m.reset();assert not m.histories
    # Default batch RPC must track only the latest active observations and
    # their supplied order, including environment dropout and resumed single IO.
    active=Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True))
    def small_obs(idx):
        x=raw.copy();x[0]=idx
        return {'env_idx':idx,'state':unpack58(pack54(x[None]))[0],'instruction':'test',
                'vision':{k:{'color':np.zeros((4,6,3),np.uint8)} for k in ['cam_head','cam_left_wrist','cam_right_wrist']}}
    active.update_obs_batch([small_obs(3),small_obs(1)])
    default=active.get_action_batch()
    assert [chunk[0]['left_ee_pose'][0] for chunk in default]==[3,1]
    active.update_obs_batch([small_obs(3)])
    assert len(active.get_action_batch())==1
    assert active.get_action()[0]['left_ee_pose'][0]==3
    try:active.get_action_batch([1])
    except ValueError:pass
    else:raise AssertionError('Inactive environment history used for inference')
    active.reset()
    try:active.get_action_batch()
    except ValueError:pass
    else:raise AssertionError('Inference after reset accepted without observations')
    active.update_obs(small_obs(9))
    assert len(active.get_action_batch())==1 and active._latest_env_idx_list==[0]
    for invalid in [0,HORIZON+1]:
        try:Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',allow_dummy_policy=True,replan_steps=invalid))
        except ValueError:pass
        else:raise AssertionError('Invalid replan horizon accepted')
    print('PASS default active batch order/dropout, stale-ID rejection, reset, single-mode update and horizon bounds')

    print('PASS current three RGB camera offsets, xspark four-key ABS ABI, active env order, no double relative composition, reset')


def main():
    marker=check_ready(L);source=json.loads((L/'meta/source_fingerprint.json').read_text())
    info=json.loads((Path(source['source_root'])/'meta/info.json').read_text())
    loader=LeRobotEpisodeLoader(L,MODALITIES,video_backend='pyav')
    assert len(loader)==info['total_episodes'] and sum(loader.get_episode_lengths())==info['total_frames']
    # Native training and statistics both omit short episodes; no repeated-tail
    # labels are introduced to make a 40-step horizon appear to fit.
    from types import SimpleNamespace
    from gr00t.data.stats import RelativeActionLoader
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    lengths=loader.get_episode_lengths()
    native_dataset=SimpleNamespace(episode_loader=loader,action_horizon=HORIZON)
    assert sum(ShardedSingleStepDataset.get_effective_episode_length(native_dataset,i) for i in range(len(loader)))==marker['window_count']
    short=[i for i,n in enumerate(lengths) if n<HORIZON]
    assert len(short)==marker['short_episode_count']
    for i in short:
        assert ShardedSingleStepDataset.get_effective_episode_length(native_dataset,i)==0
    for group in ['left_eef','right_eef']:
        native=RelativeActionLoader(L,EmbodimentTag.NEW_EMBODIMENT,group)
        for i in short:assert native.load_relative_actions(i)==[]
    print('PASS native 40-step complete-window training/statistics coverage; short episodes contribute zero windows',len(short))
    assert [x.rep.name for x in MODALITIES['action'].action_configs]==['RELATIVE','ABSOLUTE','RELATIVE','ABSOLUTE']
    test_rotation_and_reference();test_native_processor(loader,marker);test_images_and_adapter(loader)
    print('PASS GR00T relative EEF / ABS hand integration',info['total_episodes'],'episodes',info['total_frames'],'rows',marker['window_count'],'native complete windows')


if __name__=='__main__':main()
