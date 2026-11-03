"""CPU protocol replay using the real simulator methods and recorded RGB/poses.

Only Isaac rendering/physics and the neural prediction are substituted. Actual
simulator observation assembly, env-origin removal, WS client/server, and policy
encode/decode paths run on CPU against current LeRobot samples.
"""
import ast
import asyncio
from copy import deepcopy
import importlib
import json
import logging
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace
from typing import List

import numpy as np

SIM = Path('/personal/wenwei/xspark-0-simeval')
DEX = Path('/personal/wenwei/DexBench')
VITRA = DEX/'XPolicyLab/policy/VITRA'
SOURCE = DEX/'data_preprocess/data/dexbench_12tasks'
CAMERAS = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']


def child(url, inputs, outputs):
    from client_server.ws import model_client
    from client_server.ws.model_client import WsModelClient
    from XPolicyLab.utils.process_data import images_encoding
    assert str(SIM) in model_client.__file__, model_client.__file__
    observations = pickle.loads(Path(inputs).read_bytes())
    cases = []
    transport_images = []
    with WsModelClient(url=url, evaluation_id='vitra-gr-real-sim-wire', trial_id='cpu-replay') as client:
        for encoded in (False, True):
            client.call(func_name='reset')
            batch = deepcopy(observations)
            if encoded:
                for obs in batch:
                    for camera in obs['vision'].values():
                        camera['color'] = images_encoding([camera['color']])[0][0]
            transport_images.append({obs['env_idx']:[obs['vision'][name]['color'] for name in CAMERAS] for obs in batch})
            client.call(func_name='update_obs_batch', obs=batch)
            actions = client.call(func_name='get_action_batch', obs=[3, 7])
            cases.append(dict(encoded=encoded, order=[3, 7], actions=actions))
            client.call(func_name='update_obs_batch', obs=[batch[0]])
            actions = client.call(func_name='get_action_batch', obs=[7])
            cases.append(dict(encoded=encoded, order=[7], actions=actions))
            client.call(func_name='reset')
            try:
                client.call(func_name='get_action')
            except Exception as exc:
                assert 'update' in str(exc).lower() or 'observation' in str(exc).lower(), str(exc)
            else:
                raise AssertionError('reset retained observations')
    Path(outputs).write_bytes(pickle.dumps(dict(client_file=model_client.__file__, cases=cases, transport_images=transport_images)))


def methods(relative, names):
    path = SIM/relative
    tree = ast.parse(path.read_text())
    selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names), (path, names)
    for node in selected:
        node.decorator_list = []
    import torch
    namespace = dict(np=np, torch=torch, deepcopy=deepcopy, List=List)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def simulator_observations(core, episode, offsets, horizon=16):
    """Replay the real ObsManager on recorded captures/physics with grid offsets."""
    import torch
    from vitra.datasets.spark0_dataset import matrix_to_pose7
    robot_methods = methods('Env/RobotManager/RobotManager.py', ['get_link_pose', 'get_real_endpose'])
    camera_methods = methods('Env/CameraManager/CameraManager.py', ['get_camera_extrinsics_7d', 'get_camera_extrinsics'])
    obs_methods = methods('Env/ObservationManager/obs_manager.py', ['get_obs', '_ee_shift_to_base_origin'])
    eval_methods = methods('src/Maniverse_Eval/eval_env/EvalEnv.py', ['get_obs_batch', 'validate_robot_state_dict', 'get_action_type'])
    state, action = core.index.numeric(episode)
    cal = core.calibration(episode)
    begin = 0 if episode == 0 else int(core.index.cumulative[episode-1])
    samples = {env: core[begin+offset] for env, offset in offsets.items()}
    origins = torch.tensor([[i*2.3, -i*1.7, i*.11] for i in range(8)], dtype=torch.float64)
    robots = [SimpleNamespace(type='target', robot_type='arm', arm_name=side+'_arm',
                            ee_link_name=side+'_link7', robot_name=side, ee_type='hand')
              for side in ['left','right']]
    robot_keys = []
    for side, start in [('left', 0), ('right', 27)]:
        pose = torch.zeros((8, 2, 7), dtype=torch.float64)
        pose[:, :, 3] = 1
        for env, offset in offsets.items():
            pose[env, 0] = torch.from_numpy(state[offset, start:start+7].astype(np.float64))
            pose[env, 0, :3] += origins[env]
            pose[env, 1, :3] = origins[env]+torch.tensor([-.3, -.7, .2])
        robot_keys.append(SimpleNamespace(body_names=[side+'_link7', 'base'], data=SimpleNamespace(body_link_pose_w=pose)))
    rm = SimpleNamespace(robot_list=robots, robot_key=robot_keys, num_envs=8,
                         scene=SimpleNamespace(env_origins=origins), target_arm_nums=2)
    for name, fn in robot_methods.items(): setattr(rm, name, MethodType(fn, rm))
    rm.get_joint = lambda robot, env_idx_list: {env: np.zeros(7) for env in env_idx_list}
    rm._get_curobo_base_link = lambda robot: 'base'
    rm.restore_name = lambda name: name.removesuffix('_joint_state')
    rm.get_robot_by_gripper_name = lambda name: robots[0 if name.startswith('left') else 1]
    rm.process_name = lambda name: name+'_joint_state'
    rm.control_manager = SimpleNamespace(prev_control=[{} for _ in range(8)])
    for env, offset in offsets.items():
        rm.control_manager.prev_control[env] = {
            side+'_ee_joint_state': dict(position=state[offset, start+7:start+27].copy())
            for side, start in [('left', 0), ('right', 27)]}

    cm = SimpleNamespace(num_cams=3, camera_names=[CAMERAS for _ in range(8)],
                         cameras_xform=[[None]*3 for _ in range(8)],
                         sim=SimpleNamespace(scene=SimpleNamespace(env_origins=origins)))
    for name, fn in camera_methods.items(): setattr(cm, name, MethodType(fn, cm))
    def intrinsics(cam_id, env_id):
        k = cal[CAMERAS[cam_id]+'_intrinsics'][offsets[env_id]]
        return np.array([k[0,0], k[1,1], k[0,2], k[1,2]])
    cm.get_camera_intrinsics_vector = intrinsics
    for env, offset in offsets.items():
        for cam_id, name in enumerate(CAMERAS):
            raw = matrix_to_pose7(cal[name+'_extrinsics'][offset]).astype(np.float64)
            pos = torch.from_numpy(raw[:3])+origins[env]
            quat = torch.from_numpy(raw[3:])
            cm.cameras_xform[env][cam_id] = SimpleNamespace(get_world_pose=lambda p=pos,q=quat: (p,q))
            # Prove the actual live matrix and pose7 paths remove grid offset.
            np.testing.assert_allclose(cm.get_camera_extrinsics(cam_id, env), cal[name+'_extrinsics'][offset], atol=1e-7)
    def capture(env_ids):
        return [dict(rgb=[np.concatenate([samples[env]['multi_view_images'][i],
                         np.full((*samples[env]['multi_view_images'][i].shape[:2], 1), 173, np.uint8)], axis=-1)
                          for env in env_ids]) for i in range(3)]
    om = SimpleNamespace(num_envs=8, collect_freq=25, instruction=[core.records[episode]['instruction']]*8,
                         camera_manager=cm, capture_manager=SimpleNamespace(step=capture), robot_manager=rm,
                         env=SimpleNamespace(capture_type='separate'), ANNOTATORS_TO_COLLECT={'rgb':'color'},
                         collect_shape=True, collect_camera_calibration=True, calibration_camera_names=set(CAMERAS),
                         collect_intrinsic_matrix=False, collect_extrinsic_matrix=False,
                         robot_cfg=dict(joint_states=True, world_ee_state=True, base_ee_state=True))
    om.get_obs = MethodType(obs_methods['get_obs'], om)
    om._ee_shift_to_base_origin = obs_methods['_ee_shift_to_base_origin']
    om.render_for_capture = lambda: None
    evaluation = SimpleNamespace(physx_monitor_enabled=False, num_envs=8, end_flag=[False]*8,
                                 obs_manager=om, _stream_vision=lambda *args: None)
    observations = eval_methods['get_obs_batch'](evaluation, list(offsets))
    validator = SimpleNamespace(robot_action_dim_info=dict(arm_dim=[7,7], ee_dim=[20,20]), robot_manager=rm)
    for obs in observations:
        env = obs['env_idx']; offset = offsets[env]
        for camera, image in zip(CAMERAS, samples[env]['multi_view_images']):
            np.testing.assert_array_equal(obs['vision'][camera]['color'], image)
            assert obs['vision'][camera]['shape'] == image.shape
            assert obs['vision'][camera]['intrinsics'].shape == (4,)
            assert obs['vision'][camera]['extrinsics'].shape == (7,)
        for side, start in [('left',0), ('right',27)]:
            np.testing.assert_allclose(obs['state'][side+'_ee_pose'], state[offset,start:start+7], atol=1e-12)
    return observations, samples, {env:action[offset:offset+horizon] for env,offset in offsets.items()}, validator, eval_methods


async def replay(policy):
    os.environ.setdefault('EVAL_MAIN_ROOT', str(SIM))
    # reset-negative tests deliberately make two invalid inference requests.
    logging.getLogger('client_server.ws.model_server').setLevel(logging.CRITICAL)
    sys.path[:0] = [str(DEX), str(DEX/'XPolicyLab'), str(VITRA/'VITRA'), str(VITRA)]
    import torch
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from XPolicyLab.utils.process_data import images_encoding, decode_image_bit
    from vitra.datasets.lerobot_dataset import LeRobotDatasetCore
    from validate_dexbench import adapter as vitra_adapter
    from XPolicyLab.policy.GR00T_N17.native_contract import pack54
    from XPolicyLab.policy.GR00T_N17.relative_contract import GROUPS,HORIZON
    core = LeRobotDatasetCore(str(SOURCE), load_images=True)
    core.set_global_data_statistics(core.data_statistics)
    # Select A3 because its observed EE and next command differ measurably.
    horizon=16 if policy=='VITRA' else HORIZON
    episode = next(i for i,record in enumerate(core.records) if 'a3_ultra' in str(record) and core.episode_lengths[i]>=9+horizon)
    observations, samples, targets, validator, eval_methods = simulator_observations(core, episode, {7:5, 3:9},horizon)
    prepared = {env:core.transform_trajectory(deepcopy(sample)) for env,sample in samples.items()}
    records = []
    if policy == 'VITRA':
        m = vitra_adapter(core); m.env_cfg_type = 'tianji_marvin_wuji'
        m._latest_env_idx_list=[]; m._observations={}; m._raw_observations={}
        m._mano_pending_keyframes={};m._mano_next_tick={};m._mano_warned_odd_query=set();m._h1_adapter=None
        class Oracle:
            def predict_action(self, **kwargs):
                env = next(env for env in [7,3] if np.allclose(kwargs['current_state'][0].numpy(), prepared[env]['current_state'].numpy(), atol=2e-6))
                np.testing.assert_array_equal(kwargs['current_state_mask'][0], prepared[env]['current_state_mask'])
                np.testing.assert_allclose(kwargs['fov'][0], samples[env]['fov'], atol=1e-6)
                assert len(kwargs['image']) == 3
                records.append(dict(env=env, images=[np.asarray(x) for x in kwargs['image']]))
                return prepared[env]['action_list'].numpy()[None]
        m.model = Oracle()
    else:
        module = importlib.import_module('XPolicyLab.policy.GR00T_N17.model')
        import yaml
        deploy_config=yaml.safe_load((DEX/'XPolicyLab/policy/GR00T_N17/deploy.yml').read_text())
        deploy_config.update(env_cfg_type='tianji_marvin_wuji', action_type='ee', task_name='collect_objects', allow_dummy_policy=True)
        m = module.Model(deploy_config)
        assert m.replan_steps==HORIZON
        m.dummy = False
        class Oracle:
            def get_action(self, observation):
                result = {key:[] for key,_,_ in GROUPS}
                for i in range(len(observation['state']['left_eef'])):
                    packed = np.concatenate([observation['state'][key][i,0] for key,_,_ in GROUPS])
                    env = next(env for env in [7,3] if np.allclose(packed, pack54(np.concatenate([
                        observations[[x['env_idx'] for x in observations].index(env)]['state'][key]
                        for key in ['left_ee_pose','left_ee_joint_state','right_ee_pose','right_ee_joint_state']])),atol=1e-6))
                    records.append(dict(env=env, images=[observation['video'][key][i,0].copy() for key in ['front','left_wrist','right_wrist']]))
                    absolute = pack54(targets[env])
                    for key,a,b in GROUPS: result[key].append(absolute[:,a:b])
                return {key:np.stack(value) for key,value in result.items()}, {}
        m.model = Oracle()
    server = PolicyServer(m, PolicyServerConfig(host='127.0.0.1',port=0))
    await server.start()
    try:
        with tempfile.TemporaryDirectory() as td:
            inp=Path(td)/'inputs.pkl';out=Path(td)/'outputs.pkl';inp.write_bytes(pickle.dumps(observations))
            env=os.environ.copy();env['PYTHONPATH']=str(SIM)+':'+str(SIM/'XPolicyLab')
            proc=await asyncio.to_thread(subprocess.run,[sys.executable,str(Path(__file__).resolve()),'--child',server.url,str(inp),str(out)],cwd=SIM,env=env,capture_output=True,text=True,timeout=90)
            if proc.returncode: raise RuntimeError(proc.stdout+'\n'+proc.stderr)
            result=pickle.loads(out.read_bytes())
        max_position=0.;max_rotation=0.;max_hand=0.
        from scipy.spatial.transform import Rotation
        for case in result['cases']:
            for env,chunk in zip(case['order'],case['actions']):
                assert len(chunk)==horizon
                for i,command in enumerate(chunk):
                    eval_methods['validate_robot_state_dict'](validator,command)
                    assert eval_methods['get_action_type'](validator,command)=='ee'
                    assert set(command)=={'left_ee_pose','left_ee_joint_state','right_ee_pose','right_ee_joint_state'}
                    for side,start in [('left',0),('right',27)]:
                        pose=np.asarray(command[side+'_ee_pose']);ref=targets[env][i,start:start+7]
                        max_position=max(max_position,float(np.max(np.abs(pose[:3]-ref[:3]))))
                        delta=Rotation.from_quat(pose[[4,5,6,3]]).inv()*Rotation.from_quat(ref[[4,5,6,3]])
                        max_rotation=max(max_rotation,float(delta.magnitude()))
                        max_hand=max(max_hand,float(np.max(np.abs(command[side+'_ee_joint_state']-targets[env][i,start+7:start+27]))))
        assert max_position<3e-6 and max_rotation<3e-6 and max_hand<3e-6,(max_position,max_rotation,max_hand)
        assert len(records)==6
        for i,record in enumerate(records):
            for observed,reference in zip(record['images'],result['transport_images'][int(i>=3)][record['env']]):
                reference=decode_image_bit(reference)
                np.testing.assert_array_equal(observed,reference)
        report=dict(status='PASS',policy=policy,episode=episode,source_episodes=int(core.index.info['total_episodes']),source_frames=int(core.index.info['total_frames']),
            actual_sim_client=result['client_file'],sim_obs='ObsManager.get_obs + RobotManager.get_link_pose + CameraManager.get_camera_extrinsics_7d',
            wire_cases=len(result['cases']),requests=len(records),steps=horizon,batch_orders=[[3,7],[7]],raw_and_jpeg=True,
            camera_shapes='RGB HWC; intrinsics4; extrinsics7 wxyz',nonzero_env_grid_offsets=True,
            max_xyz_m=max_position,max_rotation_rad=max_rotation,max_hand_rad=max_hand,
            boundary='Known training targets replace neural prediction; learned checkpoints and Isaac rendering/IK not run',
            affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0))
        if policy!='VITRA':
            report['deploy_replan_steps']=deploy_config['replan_steps']
            report['pretrained_model_action_horizon']=json.loads(Path('/personal/xiangpc/0811_Xpolicylab_bench/GrootN17/pretrain_model/GR00T-N1.7-3B/config.json').read_text())['action_horizon']
        report_path=VITRA/'logs'/('simeval_wire_'+policy+('_h40' if policy!='VITRA' else '')+'_20261027.json')
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print('Report:',report_path)
    finally: await server.stop()


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--child': child(*sys.argv[2:])
    else: asyncio.run(replay(sys.argv[1] if len(sys.argv)>1 else 'VITRA'))
