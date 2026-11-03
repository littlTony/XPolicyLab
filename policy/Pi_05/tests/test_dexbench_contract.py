import dataclasses
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from XPolicyLab.policy.Pi_05 import model
from XPolicyLab.policy.Pi_05.prepare_stats import chunk_weights, statistics
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from openpi import transforms
from openpi.policies.dexbench_contract import ACTION_CONTRACT
from openpi.training import config


def observation(env_idx=0):
    image=np.zeros((16,20,3),dtype=np.uint8)
    image[...,0]=210;image[...,1]=71;image[...,2]=19
    return dict(env_idx=env_idx,instruction='pick the apple',vision={
        k:dict(color=image.copy()) for k in ['cam_head','cam_left_wrist','cam_right_wrist']},
        state=dict(left_ee_pose=np.array([0.1+env_idx,0.2,0.3,0.5,0.5,0.5,0.5],dtype=np.float32),
                   left_ee_joint_state=np.arange(20,dtype=np.float32)/100,
                   right_ee_pose=np.array([-0.4,0.5,0.6,1,0,0,0],dtype=np.float32),
                   right_ee_joint_state=-np.arange(20,dtype=np.float32)/100))


class HoldStatePolicy:
    """A test fixture, deliberately unavailable in the production adapter."""
    def __init__(self, horizon):
        self.horizon = horizon

    def infer(self, observation, **kwargs):
        return {'actions': np.repeat(observation['state'][None], self.horizon, axis=0)}


def adapter():
    with patch.object(model.Model, 'get_model', return_value=HoldStatePolicy(3)):
        return model.Model(dict(task_name='add_food_to_pot',env_cfg_type='tianji_marvin_wuji',
                                action_type='ee'))


def test_absolute_wxyz_pack_and_simulator_output():
    policy=adapter(); obs=observation();policy.update_obs(obs)
    returned=policy.get_action()
    assert len(returned)==3
    for action in returned:
        assert set(action)==set(obs['state'])
        for key in obs['state']:
            np.testing.assert_allclose(action[key],obs['state'][key])
    assert policy.observation_window[0]['prompt']=='[tianji_wujihand2_real/add_food_to_pot] pick the apple'
    np.testing.assert_array_equal(policy.observation_window[0]['images']['cam_high'][:,0,0],[210,71,19])


def test_active_env_subset_order_and_reset():
    policy=adapter();policy.update_obs_batch([observation(3),observation(7)])
    chunks=policy.get_action_batch([7,3])
    assert chunks[0][0]['left_ee_pose'][0]==pytest.approx(7.1)
    assert chunks[1][0]['left_ee_pose'][0]==pytest.approx(3.1)
    assert policy.get_action_batch([])==[]
    policy.reset()
    with pytest.raises(RuntimeError):policy.get_action()


def test_wrong_joint_mode_and_robot_rejected(monkeypatch):
    with pytest.raises(ValueError):model.Model(dict(task_name='x',action_type='joint',debug_no_model=True))
    monkeypatch.setattr(model,'get_robot_action_dim_info',lambda _: {'arm_dim':[6,6],'ee_dim':[1,1]})
    with pytest.raises(ValueError):model.Model(dict(task_name='x',env_cfg_type='arx_x5',debug_no_model=True))


def test_required_rgb_cameras_and_no_decoder():
    policy=adapter();obs=observation();del obs['vision']['cam_left_wrist']
    with pytest.raises(KeyError):policy.update_obs(obs)
    obs=observation();obs['vision']['cam_head']['color']=b'encoded JPEG must be decoded by server'
    with pytest.raises(ValueError):policy.update_obs(obs)


def test_predictions_are_absolute_and_quaternion_order_preserved():
    policy=adapter();obs=observation();policy.update_obs(obs)
    predicted=pack_robot_state(obs,'ee',policy.robot_action_dim_info).astype(np.float32)
    predicted[:3]=[4,5,6];predicted[3:7]=[2,0,0,0];predicted[7:27]=0.7
    class Prediction:
        def infer(self,obs,**kw):return {'actions':predicted[None].copy()}
    policy.policy=Prediction()
    result=policy.get_action()[0]
    np.testing.assert_array_equal(result['left_ee_pose'],[4,5,6,1,0,0,0])
    np.testing.assert_allclose(result['left_ee_joint_state'],0.7)


def test_wuji_pipeline_has_no_delta_and_train_runtime_repack_match():
    cfg=config.get_config(model.TRAIN_CONFIG)
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    assert isinstance(cfg.data,config.LeRobotWujiDataConfig)
    assert cfg.data.extra_delta_transform is False
    assert data.action_contract==ACTION_CONTRACT
    assert not any(isinstance(t,(transforms.DeltaActions,transforms.AbsoluteActions))
                   for t in [*data.data_transforms.inputs,*data.data_transforms.outputs])
    with pytest.raises(ValueError):dataclasses.replace(cfg.data,extra_delta_transform=True).create(cfg.assets_dirs,cfg.model)
    obs=observation(); robot=get_robot_action_dim_info('tianji_marvin_wuji')
    runtime=model.encode_obs(obs,robot,task_name='add_food_to_pot',prompt_embodiment='tianji_wujihand2_real')
    state=pack_robot_state(obs,'ee',robot).astype(np.float32)
    disk={'observation.state':state,
        'observation.images.cam_high':obs['vision']['cam_head']['color'],
        'observation.images.cam_left_wrist':obs['vision']['cam_left_wrist']['color'],
        'observation.images.cam_right_wrist':obs['vision']['cam_right_wrist']['color'],
        'action':state[None], 'prompt':runtime['prompt']}
    training=transforms.compose([*data.repack_transforms.inputs,*data.data_transforms.inputs])(disk)
    inference=transforms.compose([*model._runtime_repack_transforms(cfg).inputs,*data.data_transforms.inputs])(runtime)
    for key in ['state','prompt']:
        np.testing.assert_equal(training[key],inference[key])
    for key in training['image']:
        np.testing.assert_array_equal(training['image'][key],inference['image'][key])
    np.testing.assert_array_equal(training['actions'],state[None])


def test_action_chunk_stats_weights_match_episode_padding():
    for length in [1,2,3,51,60]:
        for horizon in [1,2,5,50]:
            explicit=np.minimum(np.arange(length)[:,None]+np.arange(horizon)[None],length-1).reshape(-1)
            np.testing.assert_array_equal(chunk_weights(length,horizon),np.bincount(explicit,minlength=length))
    values=np.array([[1,4],[2,8],[9,10]],dtype=np.float32)
    weights=chunk_weights(3,5)
    stats=statistics(values,weights)
    expanded=np.repeat(values,weights,axis=0)
    np.testing.assert_allclose(stats.mean,expanded.mean(axis=0),rtol=1e-6)
    np.testing.assert_allclose(stats.std,expanded.std(axis=0),rtol=1e-6)
    np.testing.assert_allclose(stats.q01,np.quantile(expanded,0.01,axis=0))
    np.testing.assert_allclose(stats.q99,np.quantile(expanded,0.99,axis=0))


def test_checkpoint_selection_exact_and_complete_only(tmp_path):
    for name in ['1000','2000','3000.orbax-checkpoint-tmp-1']:
        root=tmp_path/name;root.mkdir();(root/'params').mkdir();(root/'assets').mkdir()
        (root/'_CHECKPOINT_METADATA').touch()
    assert model._resolve_pi05_model_root(dict(model_path=str(tmp_path))).name=='2000'
    assert model._resolve_pi05_model_root(dict(model_path=str(tmp_path),checkpoint_num=1000)).name=='1000'
    with pytest.raises(FileNotFoundError):model._resolve_pi05_model_root(dict(model_path=str(tmp_path),checkpoint_num=9000))


def test_checkpoint_normalization_is_loaded_from_selected_checkpoint(tmp_path, monkeypatch):
    from openpi.shared import normalize
    root=tmp_path/'2000';assets=root/'assets'/'dexbench_12tasks'
    assets.mkdir(parents=True)
    stats={key:normalize.NormStats(mean=np.full(54,value),std=np.ones(54),
        q01=np.full(54,value-2),q99=np.full(54,value+2)) for key,value in [('state',17.),('actions',29.)]}
    normalize.save(assets,stats)
    (assets/'action_contract.json').write_text(json.dumps(ACTION_CONTRACT))
    monkeypatch.setattr(model,'_resolve_pi05_model_root',lambda _:root)
    captured={}
    def factory(cfg,checkpoint,**kwargs):
        captured.update(root=checkpoint,**kwargs)
        return HoldStatePolicy(50)
    monkeypatch.setattr(model.policy_config,'create_trained_policy',factory)
    model.Model(dict(task_name='add_food_to_pot',env_cfg_type='tianji_marvin_wuji',action_type='ee'))
    assert captured['root']==root
    np.testing.assert_array_equal(captured['norm_stats']['state'].mean,np.full(54,17.))
    np.testing.assert_array_equal(captured['norm_stats']['actions'].mean,np.full(54,29.))
    assert captured['sample_kwargs'] == {'num_steps': 10}


def test_multi_task_prompts_are_training_aligned():
    obs = observation()
    dims = get_robot_action_dim_info('tianji_marvin_wuji')
    for task in ['add_food_to_pot', 'collect_objects', 'stack_bowls']:
        obs['instruction'] = f'[tianji_wujihand2_real/{task}] do this task'
        out = model.encode_obs(obs, dims, task_name='auto', prompt_embodiment='tianji_wujihand2_real')
        assert out['prompt'] == obs['instruction']
    with pytest.raises(ValueError):
        model.encode_obs(obs, dims, task_name='collect_objects', prompt_embodiment='tianji_wujihand2_real')
    obs['instruction'] = '[g1_inspire/stack_bowls] do this task'
    with pytest.raises(ValueError):
        model.encode_obs(obs, dims, task_name='auto', prompt_embodiment='tianji_wujihand2_real')
    obs['instruction'] = 'stack the bowls'
    with pytest.raises(ValueError):
        model.encode_obs(obs, dims, task_name='auto', prompt_embodiment='tianji_wujihand2_real')
    obs['task_name'] = 'stack_bowls'
    out = model.encode_obs(obs, dims, task_name='auto', prompt_embodiment='tianji_wujihand2_real')
    assert out['prompt'] == '[tianji_wujihand2_real/stack_bowls] stack the bowls'


def test_current_stats_provenance_and_same_count_stale_rejected(tmp_path):
    import os
    import shutil
    from XPolicyLab.policy.Pi_05.stats_provenance import check_summary
    cfg=config.get_config(model.TRAIN_CONFIG)
    root=Path(os.environ['HF_LEROBOT_HOME'])/'dexbench_12tasks'
    assets=cfg.assets_dirs/'dexbench_12tasks'
    summary=check_summary(root,assets,cfg.model.action_horizon)
    info=json.loads((root/'meta/info.json').read_text())
    assert summary['frames']==info['total_frames']
    assert summary['episodes']==info['total_episodes']
    saved=tmp_path/'assets';saved.mkdir()
    shutil.copyfile(assets/'norm_stats.json',saved/'norm_stats.json')
    stale=dict(summary,source_sha256='0'*64)
    (saved/'dataset_summary.json').write_text(json.dumps(stale))
    with pytest.raises(ValueError,match='another numeric dataset'):
        check_summary(root,saved,cfg.model.action_horizon)
    (saved/'dataset_summary.json').write_text(json.dumps(summary))
    value=json.loads((saved/'norm_stats.json').read_text())
    value['norm_stats']['state']['mean'][0]+=1
    (saved/'norm_stats.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='normalization file changed'):
        check_summary(root,saved,cfg.model.action_horizon)


def test_current_real_rgb_and_state_train_runtime_normalization_match():
    import os
    import av
    import pandas as pd
    import pyarrow.parquet as pq
    root=Path(os.environ['HF_LEROBOT_HOME'])/'dexbench_12tasks'
    info=json.loads((root/'meta/info.json').read_text())
    first=pq.read_table(sorted((root/'data').rglob('*.parquet'))[0],
        columns=['observation.state','action','task_index','episode_index']).slice(0,1).to_pylist()[0]
    episode=int(first['episode_index'])
    cameras=['cam_high','cam_left_wrist','cam_right_wrist']
    columns=['episode_index']+[f'videos/observation.images.{camera}/{field}'
        for camera in cameras for field in ['chunk_index','file_index','from_timestamp']]
    metadata=pd.concat([pq.read_table(path,columns=columns).to_pandas()
        for path in sorted((root/'meta/episodes').rglob('*.parquet'))],ignore_index=True)
    row=metadata[metadata['episode_index']==episode].iloc[0]
    images={}
    for camera in cameras:
        key='observation.images.'+camera
        path=root/info['video_path'].format(video_key=key,chunk_index=int(row[f'videos/{key}/chunk_index']),
            file_index=int(row[f'videos/{key}/file_index']))
        offset=int(round(float(row[f'videos/{key}/from_timestamp'])*info['fps']))
        with av.open(str(path)) as container:
            for index,frame in enumerate(container.decode(video=0)):
                if index==offset:
                    images[camera]=frame.to_ndarray(format='rgb24');break
        assert images[camera].shape==(480,640,3)
    tasks=pq.read_table(root/'meta/tasks.parquet').to_pandas()
    task=str(tasks[tasks['task_index']==int(first['task_index'])].index[0])
    embodiment,task_name=task.split('] ',1)[0][1:].split('/')
    x=np.asarray(first['observation.state'],dtype=np.float32)
    robot=get_robot_action_dim_info('tianji_marvin_wuji')
    physical=unpack_robot_state(x[None],'ee',robot,source_type='obs')[0]
    observation=dict(state=physical,instruction=task,vision={
        live:dict(color=images[camera]) for live,camera in
        [('cam_head','cam_high'),('cam_left_wrist','cam_left_wrist'),('cam_right_wrist','cam_right_wrist')]})
    runtime=model.encode_obs(observation,robot,task_name=task_name,prompt_embodiment=embodiment)
    cfg=config.get_config(model.TRAIN_CONFIG);data=cfg.data.create(cfg.assets_dirs,cfg.model)
    disk={'observation.state':x,'action':np.asarray(first['action'],np.float32)[None],
        'prompt':task,**{'observation.images.'+k:v for k,v in images.items()}}
    normalizer=transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm)
    train=transforms.compose([*data.repack_transforms.inputs,*data.data_transforms.inputs,normalizer])(disk)
    infer=transforms.compose([*model._runtime_repack_transforms(cfg).inputs,*data.data_transforms.inputs,normalizer])(runtime)
    np.testing.assert_allclose(train['state'],infer['state'],atol=1e-6)
    assert train['prompt']==infer['prompt']
    for key in train['image']:
        np.testing.assert_array_equal(train['image'][key],infer['image'][key])
    for key in train['image_mask']:
        np.testing.assert_array_equal(train['image_mask'][key],infer['image_mask'][key])
