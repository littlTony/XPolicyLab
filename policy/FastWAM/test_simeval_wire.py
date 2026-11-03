"""CPU replay of production simulator observation/dispatch methods, without physics.

Sensors and IK are boundary stand-ins fed a real Tianji source episode; observation
construction, websocket transport, adapters and action dispatch are production code.
"""
import ast
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['EVAL_MAIN_ROOT'] = '/personal/wenwei/xspark-0-simeval'
os.sched_setaffinity(0, set(range(112, 120)))
os.nice(10)
sys.dont_write_bytecode = True
import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SIM = Path(os.environ['EVAL_MAIN_ROOT'])
XPL = Path('/personal/wenwei/DexBench/XPolicyLab')
sys.path[:0] = [str(XPL.parent), str(XPL)]
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit, pack_robot_state
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


def production_method(path, name):
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {'np': np, 'deepcopy': deepcopy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    return ns[name]


cfg = yaml.safe_load((SIM / 'env_cfg/tianji_marvin_wuji.yml').read_text())
dims = json.loads((SIM / 'env_cfg/robot/_robot_info.json').read_text())[cfg['config']['robot']]
reports = json.loads((XPL.parent / 'data_preprocess/data/dexbench_12tasks/meta/preview_report.json').read_text())
record = next(r for r in reports if r['task'].startswith('[tianji_wujihand2_real/'))
task = record['task'].split(']')[0].split('/')[1]
samples = []
with h5py.File(record['source']) as f:
    instruction = f['instruction'][()]
    if isinstance(instruction, bytes):
        instruction = instruction.decode()
    for t in range(7):
        numeric = {g: {k: np.asarray(f[f'{g}/{k}'][t], np.float32)
                        for k in ['left_ee_poses', 'right_ee_poses',
                                  'left_ee_joint_states', 'right_ee_joint_states',
                                  'left_arm_joint_states', 'right_arm_joint_states']}
                   for g in ['state', 'action']}
        images = {cam: decode_image_bit(f[f'vision/{cam}/colors'][t])
                  for cam in ['cam_head', 'cam_left_wrist', 'cam_right_wrist']}
        samples.append((numeric, images))

obs_path = SIM / 'Env/ObservationManager/obs_manager.py'
get_obs = production_method(obs_path, 'get_obs')
camera_names = ['cam_head', 'cam_left_wrist', 'cam_right_wrist']
robots = [NS(type='target', robot_type='arm', arm_name=side + '_arm',
             gripper_name=side + '_ee', ee_type='hand') for side in ['left', 'right']]
current = {'frame': 0}


def numeric(robot, group, suffix, ids):
    side = robot.arm_name.split('_')[0]
    value = samples[current['frame']][0][group][side + '_' + suffix]
    return {i: value.copy() for i in ids}


manager = NS(robot_list=robots, target_arm_nums=2,
             get_joint=lambda robot, env_idx_list: numeric(robot, 'state', 'arm_joint_states', env_idx_list),
             get_real_endpose=lambda robot, env_idx_list, is_relative: numeric(robot, 'state', 'ee_poses', env_idx_list),
             get_link_pose=lambda robot, link_name, env_idx_list, is_relative: {i: np.array([0, 0, 0, 1, 0, 0, 0]) for i in env_idx_list},
             _get_curobo_base_link=lambda robot: 'base',
             restore_name=lambda key: key.replace('_joint_state', ''),
             get_robot_by_gripper_name=lambda name: next(r for r in robots if r.gripper_name == name),
             process_name=lambda name: name + '_joint_state')
obs_manager = NS(num_envs=4, collect_freq=cfg['observation']['collect_freq'],
                 instruction=[str(instruction)] * 4, env=NS(capture_type='tiled'),
                 camera_manager=NS(num_cams=3, camera_names=[camera_names] * 4),
                 capture_manager=NS(step=lambda env_ids: {
                     ith: {'rgb': [{'data': np.concatenate([samples[current['frame']][1][cam],
                                                           np.full((480, 640, 1), 255, np.uint8)], -1)}
                                   for _ in env_ids]} for ith, cam in enumerate(camera_names)}),
                 robot_manager=manager, robot_cfg=cfg['observation']['collect']['robot'],
                 ANNOTATORS_TO_COLLECT={'rgb': 'color'}, collect_shape=True,
                 collect_camera_calibration=True, calibration_camera_names=set(camera_names),
                 collect_intrinsic_matrix=False, collect_extrinsic_matrix=False,
                 collect_approximate_depth=True, collect_depth=False,
                 _ee_shift_to_base_origin=lambda base, ee: np.concatenate([ee[:3] - base[:3], ee[3:]]),
                 render_for_capture=lambda: None)
obs_manager.camera_manager.get_camera_intrinsics_vector = lambda ith, idx: np.array([500, 500, 320, 240])
obs_manager.camera_manager.get_camera_extrinsics_7d = lambda ith, idx: np.array([0, 0, 0, 1, 0, 0, 0])
obs_manager.get_obs = lambda env_idx_list: get_obs(obs_manager, env_idx_list)
env_path = SIM / 'src/Maniverse_Eval/eval_env/EvalEnv.py'
get_batch = production_method(env_path, 'get_obs_batch')
source_env = NS(physx_monitor_enabled=False, obs_manager=obs_manager, end_flag=[False] * 4,
                _stream_vision=lambda idx, data: None)


def real_wire_observations(t, ids):
    current['frame'] = t
    hands = samples[t][0]['state']
    manager.control_manager = NS(prev_control={i: {
        side + '_ee_joint_state': {'position': hands[side + '_ee_joint_states'].copy()}
        for side in ['left', 'right']} for i in ids})
    result = get_batch(source_env, ids)
    for obs in result:
        for cam in camera_names:
            np.testing.assert_array_equal(obs['vision'][cam]['color'], samples[t][1][cam])
        for side in ['left', 'right']:
            np.testing.assert_array_equal(obs['state'][side + '_ee_pose'], hands[side + '_ee_poses'])
            np.testing.assert_array_equal(obs['state'][side + '_ee_joint_state'], hands[side + '_ee_joint_states'])
    return result


def validate_action_dispatch(actions):
    captured = {'ik': [], 'controls': []}
    receiver = NS(physx_monitor_enabled=False, robot_manager=manager, num_envs=4,
                  take_action_cnt=[0] * 4, step_lim=100, end_flag=[False] * 4,
                  robot_action_dim_info=dims, have_empty=lambda ids: True,
                  reward_manager=NS(step=lambda env_idx_list: None),
                  is_episode_end=lambda: False,
                  process_control_info=lambda control, idx: control)
    for name in ['validate_robot_state_dict', 'get_action_type']:
        setattr(receiver, name, production_method(env_path, name).__get__(receiver))
    manager.solve_ik = lambda target_pose, env_idx, robot: (
        captured['ik'].append((robot.arm_name, env_idx, np.asarray(target_pose).copy()))
        or {'status': 'Success', 'joint_value': np.zeros(7)})
    manager.control_manager = NS(push=lambda ids, controls: captured['controls'].extend(controls))
    production_method(env_path, 'take_action_batch')(receiver, actions, [3, 1])
    for j, idx in enumerate([3, 1]):
        for side in ['left', 'right']:
            pose = next(p for arm, i, p in captured['ik'] if i == idx and arm == side + '_arm')
            np.testing.assert_allclose(pose, actions[j][side + '_ee_pose'], atol=1e-7)
            np.testing.assert_array_equal(captured['controls'][j][side + '_ee_joint_state']['position'],
                                          actions[j][side + '_ee_joint_state'])


async def verify_policy(policy):
    module = __import__('XPolicyLab.policy.' + policy + '.model', fromlist=['Model'])
    if policy == 'FastWAM':
        from unittest.mock import patch
        hold = NS(action_horizon=32, _infer_action_chunk=lambda obs, prompt: np.repeat(obs['joint_action']['vector'][None],32,axis=0),reset=lambda:None)
        with patch.object(module.Model,'_load_model',return_value=hold):
            model = module.Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name=task,replan_steps=2))
    else:
        model = module.Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name=task,allow_dummy_policy=True,replan_steps=2))
    # Load the actual simulator client file; its protocol implementation is byte-identical.
    spec = importlib.util.spec_from_file_location('actual_sim_client', SIM / 'XPolicyLab/client_server/ws/model_client.py')
    client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client_module)
    server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=0))
    await server.start()

    def run():
        with client_module.WsModelClient(url=server.url, evaluation_id='simeval-replay', trial_id=policy) as client:
            client.call('reset')
            for t in range(7):
                obs = real_wire_observations(t, [3, 1])
                if t % 2:
                    for item in obs:
                        for camera in item['vision'].values():
                            camera['color'] = encode_image_bit(camera['color'])
                client.call('update_obs_batch', obs)
            result = client.call('get_action_batch', [3, 1])
            for chunk in result:
                for action in chunk:
                    for side in ['left', 'right']:
                        source = samples[6][0]['state']
                        np.testing.assert_allclose(action[side + '_ee_pose'][:3], source[side + '_ee_poses'][:3], atol=1e-7)
                        q = action[side + '_ee_pose'][3:]
                        np.testing.assert_allclose(abs(q @ source[side + '_ee_poses'][3:]), 1, atol=1e-6)
                        np.testing.assert_array_equal(action[side + '_ee_joint_state'], source[side + '_ee_joint_states'])
            # A physical predicted target is intentionally different from the
            # measured state; adapters must not add the state or denormalize it.
            target = samples[6][0]['action']
            target54 = np.concatenate([target['left_ee_poses'], target['left_ee_joint_states'],
                                       target['right_ee_poses'], target['right_ee_joint_states']])
            native_target = module.pack54(target54)
            if policy == 'FastWAM':

                model.model = NS(_infer_action_chunk=lambda obs, prompt: np.repeat(native_target[None], 2, axis=0), reset=lambda: None)
            else:
                def predict(payload, options=None):
                    assert options['session_ids'] == ['dexbench-3', 'dexbench-1']
                    for key, cam in [('front', 'cam_head'), ('left_wrist', 'cam_left_wrist'), ('right_wrist', 'cam_right_wrist')]:
                        np.testing.assert_array_equal(payload['video'][key][0],
                                                      np.stack([samples[t][1][cam] for t in [0, 2, 4, 6]]))
                    return {key: np.tile(native_target[a:b], (2, 2, 1)) for key, a, b in module.GROUPS}, {}
                model.dummy = False
                model.model = NS(get_action=predict, reset=lambda: None)
            predicted = client.call('get_action_batch', [3, 1])
            for chunk in predicted:
                for action in chunk:
                    for side in ['left', 'right']:
                        np.testing.assert_allclose(action[side + '_ee_pose'][:3], target[side + '_ee_poses'][:3], atol=1e-7)
                        np.testing.assert_allclose(abs(action[side + '_ee_pose'][3:] @ target[side + '_ee_poses'][3:]), 1, atol=1e-6)
                        np.testing.assert_array_equal(action[side + '_ee_joint_state'], target[side + '_ee_joint_states'])
            # Restore echo mode for the dropout call below.
            if policy == 'FastWAM': model.model = hold
            else: model.dummy = True
            validate_action_dispatch([chunk[0] for chunk in predicted])
            client.call('update_obs_batch', real_wire_observations(6, [3]))
            assert len(client.call('get_action_batch')) == 1
            client.call('reset')
            if policy == 'RLDX_1':
                assert not model.histories and not model._latest_active_ids
            else:
                assert not model._batch and model.last_obs is None

    try:
        await asyncio.to_thread(run)
    finally:
        await server.stop()
    print('PASS', policy, 'production ObsManager/get_obs_batch -> actual sim WsModelClient -> policy server/model -> production take_action_batch; real Tianji RGB/state, raw+JPEG, env [3,1]->[3], reset')


async def main():
    for policy in ['FastWAM', 'RLDX_1']:
        await verify_policy(policy)


def verify_deploy_guards():
    for policy in ['FastWAM', 'RLDX_1']:
        deploy = __import__('XPolicyLab.policy.' + policy + '.deploy', fromlist=['eval_one_episode_batch'])

        class GuardEnv:
            def __init__(self, initial):
                self.initial = initial
                self.steps = 0
                self.observations = 0
            def is_episode_end(self): return False
            def get_running_env_idx_list(self): return self.initial if not self.steps else []
            def get_obs_batch(self, ids):
                assert ids, 'Post-terminal empty batch was requested'
                self.observations += 1
                return []
            def take_action_batch(self, actions, ids): self.steps += 1

        class Client:
            def __init__(self, chunks): self.chunks = chunks
            def call(self, func_name=None, obs=None):
                if func_name == 'get_action_batch': return self.chunks

        empty = GuardEnv([])
        deploy.eval_one_episode_batch(empty, Client([]))
        assert empty.observations == empty.steps == 0
        for chunks in [[], [[{}]], [[], []], [[{}], [{}, {}]]]:
            try: deploy.eval_one_episode_batch(GuardEnv([3, 1]), Client(chunks))
            except ValueError: pass
            else: raise AssertionError((policy, 'Invalid chunks accepted', chunks))
        drain = GuardEnv([3, 1])
        deploy.eval_one_episode_batch(drain, Client([[{}, {}], [{}, {}]]))
        assert drain.steps == drain.observations == 1
        print('PASS', policy, 'deploy empty-active/invalid-chunk/dropout guards')


if __name__ == '__main__':
    asyncio.run(main())
    verify_deploy_guards()
    print('SOURCE', record['source'], 'frames 0..6; CPU boundary replay only, no Isaac physics/IK model execution')
