from pathlib import Path
import json
import numpy as np
import pytest
from omegaconf import OmegaConf
from XPolicyLab.policy.OpenWAM import model
from openwam.dataloader.dexbench import DexBenchLeRobotDataset
from openwam.dataloader.dexbench_contract import CONTRACT,UNIFY_MAP,SLOTS,pack54,unpack58,scatter58,gather80
from openwam.dataloader.transforms.normalize import Normalizer,load_mode_stats
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.model_loader import _UnifyAwareNormalizer

POLICY=Path(model.__file__).parent


def obs(env_idx=0):
    image=np.zeros((32,40,3),dtype=np.uint8);image[:]=[210,70,20]
    return dict(env_idx=env_idx,instruction='pick up the apple',vision={k:dict(color=image.copy()) for k in
        ['cam_head','cam_left_wrist','cam_right_wrist']},state=dict(
        left_ee_pose=np.array([.1+env_idx,.2,.3,.5,.5,.5,.5],dtype=np.float32),
        left_ee_joint_state=np.arange(20,dtype=np.float32)/100,
        right_ee_pose=np.array([-.1,.5,.6,1,0,0,0],dtype=np.float32),
        right_ee_joint_state=-np.arange(20,dtype=np.float32)/100))


def adapter(**kw):
    return model.Model(dict(task_name='add_food_to_pot',env_cfg_type='tianji_marvin_wuji',
        action_type='ee',allow_dummy_policy=True,replan_steps=3,**kw))


@pytest.fixture(scope='module')
def dataset():
    cfg=OmegaConf.load(POLICY/'OpenWAM/configs/dataloader/dexbench.yaml')
    cfg.color_jitter.enabled=False
    return DexBenchLeRobotDataset.from_config(cfg)


def test_alpha_slots_and_masks():
    raw=np.arange(58,dtype=np.float32)
    unified,mask=scatter58(raw)
    assert mask.sum()==58 and unified.shape==(80,)
    np.testing.assert_array_equal(np.flatnonzero(mask),SLOTS)
    np.testing.assert_array_equal(unified[~mask],0)
    np.testing.assert_array_equal(gather80(unified),raw)
    assert not mask[9] and not mask[43] and not mask[30:34].any() and not mask[64:80].any()


def test_wxyz_rot6d_and_absolute_hand_roundtrip():
    x=obs();p=adapter();packed=p._encode_obs(x)['state']
    chunk=unpack58(np.asarray(packed)[None])[0]
    for key in x['state']:
        np.testing.assert_allclose(chunk[key],x['state'][key],atol=1e-6)
    np.testing.assert_array_equal(np.asarray(packed)[32:38],[1,0,0,0,1,0])


def test_active_env_order_and_strict_cameras():
    p=adapter();p.update_obs_batch([obs(3),obs(7)])
    out=p.get_action_batch([7,3])
    assert out[0][0]['left_ee_pose'][0]==pytest.approx(7.1)
    assert out[1][0]['left_ee_pose'][0]==pytest.approx(3.1)
    assert set(out[0][0])==set(obs()['state'])
    x=obs();del x['vision']['cam_left_wrist']
    with pytest.raises(KeyError):p.update_obs(x)
    p.reset()
    with pytest.raises(RuntimeError):p.get_action()


def test_real_dataset_no_extra_action_shift_and_boundary_masks(dataset):
    sample=dataset[0]
    assert len(dataset)==json.loads((dataset._dataset_dir/'meta/info.json').read_text())['total_frames']
    assert sample['action'].shape==(32,80) and sample['proprio'].shape==(1,80)
    assert sample['action_mask'].sum()==32*58
    row=dataset._eps_df.iloc[0]
    table=dataset._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
    win=table.slice(int(row['_data_row_offset']),2).to_pandas()
    expected=dataset._action_normalizer.normalize(pack54(np.asarray(win['action'].iloc[0])))
    np.testing.assert_allclose(gather80(sample['action'][0].numpy()),expected,atol=1e-6)
    last=dataset[-1]
    assert last['action_mask'].sum()==58  # last recorded action is still a real target
    assert last['video_mask'].sum()==1
    assert not last['action_mask'][1:].any()


def test_train_deploy_directional_normalizer(dataset):
    row=dataset._eps_df.iloc[0]
    table=dataset._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
    win=table.slice(int(row['_data_row_offset']),1).to_pandas()
    stats=load_mode_stats(dataset.normalization_stats_path,'eef')
    deployment=_UnifyAwareNormalizer(Normalizer('min_max',stats),SLOTS,80)
    state=pack54(np.asarray(win['observation.state'].iloc[0]))
    sample=dataset[0]
    np.testing.assert_allclose(deployment.normalize(state),sample['proprio'][0].numpy(),atol=1e-6)
    target=pack54(np.asarray(win['action'].iloc[0]))
    np.testing.assert_allclose(deployment.unnormalize(sample['action'][0].numpy()),target,atol=2e-6)


@pytest.mark.parametrize('embodiment',['g1_inspire','a3_ultra','l7_xhand','tianji_wujihand2_real','vega1_sharpa'])
def test_real_rgb_multiview_prompt_and_packing_match(dataset,embodiment):
    selected=dataset._eps_df['tasks'].apply(lambda texts:any(t.startswith('['+embodiment+'/') for t in texts))
    ep_local=int(np.flatnonzero(selected.to_numpy())[0])
    sample_index=int(dataset._cum_n_starts[ep_local])
    row=dataset._eps_df.iloc[ep_local];cams={}
    for camera,live in [(dataset.HEAD_CAMERA,'cam_head'),(dataset.LEFT_WRIST_CAMERA,'cam_left_wrist'),(dataset.RIGHT_WRIST_CAMERA,'cam_right_wrist')]:
        image=dataset._decode_one_camera(camera,row,ep_local,0,np.array([0]),kind='head')[0]
        cams[live]=dict(color=np.asarray(image))
    table=dataset._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
    win=table.slice(int(row['_data_row_offset']),1).to_pandas()
    state=np.asarray(win['observation.state'].iloc[0])
    text=dataset._task_idx_to_text[int(win['task_index'].iloc[0])]
    embodiment,task=text.split('] ',1)[0][1:].split('/')
    p=model.Model(dict(task_name=task,env_cfg_type='tianji_marvin_wuji',action_type='ee',
                       allow_dummy_policy=True,prompt_embodiment=embodiment))
    raw=dict(vision=cams,instruction=text,state=dict(left_ee_pose=state[:7],left_ee_joint_state=state[7:27],
                    right_ee_pose=state[27:34],right_ee_joint_state=state[34:54]))
    payload=p._encode_obs(raw)
    preprocessor=ObsPreprocessor(multiview=True,camera_layout=[dataset.HEAD_CAMERA,dataset.LEFT_WRIST_CAMERA,dataset.RIGHT_WRIST_CAMERA],
                                img_height=384,img_width=320,requires_proprio=True)
    prepared=preprocessor.preprocess(payload)
    sample=dataset[sample_index]
    np.testing.assert_array_equal(np.asarray(sample['video'][0]),np.asarray(prepared['image']))
    np.testing.assert_allclose(prepared['state'],pack54(state),atol=1e-6)
    assert sample['prompt']==prepared['prompt']


def test_finetune_config_and_154k_checkpoint():
    from hydra import compose,initialize_config_dir
    from openwam.train.utils.ckpt_model_loader import _resolve_ckpt_source,merge_ckpt_model_cfg
    with initialize_config_dir(config_dir=str(POLICY/'OpenWAM/configs'),version_base=None):
        cfg=compose(config_name='train')
    source=cfg.training.finetune_ckpt_path
    root,weights=_resolve_ckpt_source(source)
    assert weights.endswith('checkpoint_step_154000.safetensors')
    original=OmegaConf.load(Path(root)/'config.yaml')
    merged=merge_ckpt_model_cfg(original,cfg,root)
    assert merged.architecture.action_dim==merged.architecture.state_dim==80
    assert cfg.dataloader.type=='dexbench' and list(cfg.dataloader.unify_action_map)==UNIFY_MAP
    assert cfg.dataloader.contract_id==CONTRACT['id']
    from openwam.dataloader.registry import build_dataset, DATASET_REGISTRY
    from openwam.dataloader.dexbench import DexBenchLeRobotDataset
    assert DATASET_REGISTRY['dexbench'] is DexBenchLeRobotDataset
    dataset=build_dataset(cfg.dataloader, split='train')
    assert isinstance(dataset,DexBenchLeRobotDataset) and len(dataset)==json.loads((dataset._dataset_dir/'meta/info.json').read_text())['total_frames']


def test_websocket_rgb_encoded_and_simulator_action_keys():
    import asyncio
    from client_server.ws.model_server import PolicyServer,PolicyServerConfig
    from client_server.ws.model_client import WsModelClient
    from XPolicyLab.utils.process_data import encode_image_bit
    async def run():
        p=adapter()
        server=PolicyServer(p,PolicyServerConfig(host='127.0.0.1',port=0))
        await server.start()
        def client_check():
            with WsModelClient(url=server.url,evaluation_id='openwam-contract',trial_id='offline') as client:
                for encoded in [False,True]:
                    x=obs()
                    if encoded:
                        for camera in x['vision'].values():camera['color']=encode_image_bit(camera['color'])
                    client.call('update_obs',x)
                    result=client.call('get_action')
                    assert len(result)==3
                    for key,value in obs()['state'].items():np.testing.assert_allclose(result[0][key],value,atol=1e-6)
                    pixel=np.asarray(p._batch[0]['images']['head_camera'])[0,0]
                    assert pixel[0]>200 and 60<pixel[1]<80 and pixel[2]<30
                    client.call('reset')
        try:await asyncio.to_thread(client_check)
        finally:await server.stop()
    asyncio.run(run())


def test_current_stats_provenance_and_rot6d_bounds(dataset):
    from openwam.dataloader.dexbench_contract import ROT6D_DIMS
    from openwam.dataloader.dexbench_provenance import validate_stats_source
    info = json.loads((dataset._dataset_dir / 'meta/info.json').read_text())
    stats = np.load(dataset.normalization_stats_path, allow_pickle=True).item()
    current = validate_stats_source(stats, dataset._dataset_dir, info)
    assert stats['num_timesteps'] == info['total_frames']
    assert stats['num_episodes'] == info['total_episodes']
    assert stats['source_fingerprint']['sha256'] == current['sha256']
    for key in ['eef', 'eef_state']:
        for low, high in [('min', 'max'), ('q01', 'q99')]:
            np.testing.assert_array_equal(stats[key][low][ROT6D_DIMS], -1)
            np.testing.assert_array_equal(stats[key][high][ROT6D_DIMS], 1)


def test_same_count_numeric_source_change_is_rejected(tmp_path):
    from openwam.dataloader.dexbench_provenance import source_fingerprint, validate_stats_source
    (tmp_path / 'meta/episodes').mkdir(parents=True)
    (tmp_path / 'data/chunk-000').mkdir(parents=True)
    info = dict(total_frames=2, total_episodes=1)
    (tmp_path / 'meta/info.json').write_text(json.dumps(info))
    (tmp_path / 'meta/tasks.parquet').write_bytes(b'fixed task metadata')
    data = tmp_path / 'data/chunk-000/file-000.parquet'
    data.write_bytes(b'numeric values version 1')
    stats = dict(num_timesteps=2, num_episodes=1, source_fingerprint=source_fingerprint(tmp_path))
    validate_stats_source(stats, tmp_path)
    data.write_bytes(b'numeric values version 2')  # same byte size and metadata counts
    with pytest.raises(ValueError, match='source SHA changed'):
        validate_stats_source(stats, tmp_path)
    with pytest.raises(ValueError, match='source SHA provenance'):
        validate_stats_source(dict(num_timesteps=2, num_episodes=1), tmp_path)


def test_native_checkpoint_snapshot_proprio_action_and_binary_noop(dataset, tmp_path):
    from types import SimpleNamespace
    import copy
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.deploy.policy import WAMPolicy
    from openwam.deploy.server import merge_deploy_cfg, _load_deploy_yaml
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.train.utils.checkpointing import save_normalization_stats

    # Exercise the real training snapshot writer and native deployment loader.
    save_normalization_stats(str(tmp_path), dataset)
    snapshot = np.load(tmp_path / 'normalization_stats.npy', allow_pickle=True).item()
    assert snapshot['source_fingerprint']['sha256']
    poison = copy.deepcopy(snapshot)
    for section in ['eef', 'eef_state']:
        poison[section]['min'] += 10
        poison[section]['max'] += 10
    live_path = tmp_path / 'deliberately_wrong_live_stats.npy'
    np.save(live_path, poison, allow_pickle=True)
    cfg = OmegaConf.create({'dataloader': OmegaConf.load(POLICY / 'OpenWAM/configs/dataloader/dexbench.yaml')})
    cfg.dataloader.normalization_stats_path = str(live_path)
    normalizer = _build_normalizer(cfg, str(tmp_path))

    row = dataset._eps_df.iloc[0]
    table = dataset._load_data_table(int(row['data/chunk_index']), int(row['data/file_index']))
    raw = table.slice(int(row['_data_row_offset']), 1).to_pandas()
    state = pack54(np.asarray(raw['observation.state'].iloc[0]))
    target = pack54(np.asarray(raw['action'].iloc[0]))
    state_model = BaseWAMArchitecture.normalize_deploy_proprio(
        SimpleNamespace(normalizer=normalizer), state[None]
    ).numpy()[0]
    state_expected, _ = scatter58(dataset._state_normalizer.normalize(state))
    action_model, _ = scatter58(dataset._action_normalizer.normalize(target))
    np.testing.assert_allclose(state_model, state_expected, atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(action_model), target, atol=2e-6)
    # DexBench has no binary command channels: the native policy must be noop.
    assert not OmegaConf.select(cfg, 'dataloader.binary_action_dims', default=None)
    action = target[None]
    native_policy = SimpleNamespace(engine=SimpleNamespace(architecture=SimpleNamespace(binary_command_dims=())))
    assert WAMPolicy._project_binary_dims(native_policy, action) is action
    merged = merge_deploy_cfg(cfg, _load_deploy_yaml(str(POLICY / 'OpenWAM/configs/deploy.yaml')))
    preprocess = ObsPreprocessor.from_cfg(merged)
    assert (preprocess.img_height, preprocess.img_width) == (cfg.dataloader.height, cfg.dataloader.width)
    assert preprocess.multiview and len(preprocess.camera_layout) == 3


def test_checkpoint_root_and_native_numeric_step_selection(tmp_path):
    from openwam.deploy.model_loader import _find_latest_checkpoint
    (tmp_path / 'config.yaml').write_text('dataloader: {}\n')
    for name in ['checkpoint_step_2.safetensors', 'checkpoint_step_10.safetensors', 'checkpoint_step_bad.safetensors']:
        (tmp_path / name).touch()
    assert model._resolve_ckpt_dir({'ckpt_dir': str(tmp_path)}) == tmp_path
    assert Path(_find_latest_checkpoint(str(tmp_path))).name == 'checkpoint_step_10.safetensors'


def test_reader_rejects_normalizer_and_binary_contract_drift():
    cfg=OmegaConf.load(POLICY/'OpenWAM/configs/dataloader/dexbench.yaml')
    cfg.normalize_mode='z-score'
    with pytest.raises(ValueError,match='normalize_mode=min-max'):
        DexBenchLeRobotDataset.from_config(cfg)
    cfg.normalize_mode='min-max'
    cfg.binary_action_dims=[9]
    with pytest.raises(ValueError,match='binary_action_dims'):
        DexBenchLeRobotDataset.from_config(cfg)


def test_adapter_checkpoint_guard_is_local_and_rejects_mode_drift(dataset):
    import copy
    cfg=OmegaConf.create(dict(
        dataloader=dict(contract_id=CONTRACT['id'],unify_action_map=UNIFY_MAP,unify_state_map=UNIFY_MAP,
                        normalize_mode='min-max',action_mode='eef'),
        model=dict(architecture=dict(action_dim=80,state_dim=80))))
    stats=np.load(dataset.normalization_stats_path,allow_pickle=True).item()
    archived=copy.deepcopy(stats)
    archived['source_fingerprint']['source_root']='/nonexistent/original-training-source'
    model._validate_checkpoint_contract(cfg,archived)
    cfg.dataloader.normalize_mode='z-score'
    with pytest.raises(ValueError,match='normalize_mode=min-max'):
        model._validate_checkpoint_contract(cfg,stats)
    cfg.dataloader.normalize_mode='min-max'
    cfg.dataloader.binary_action_dims=[9]
    with pytest.raises(ValueError,match='binary_action_dims'):
        model._validate_checkpoint_contract(cfg,stats)
