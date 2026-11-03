"""DexBench Pi0.5: absolute world EE xyz+wxyz and absolute Wuji20 joints.

54D layout: left EE(7), left hand(20), right EE(7), right hand(20).
Images are already decoded RGB by the XPolicyLab policy server.
"""
from pathlib import Path
import hashlib
import json
import re
from typing import Any

import numpy as np

from openpi import transforms
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

_POLICY_DIR = Path(__file__).resolve().parent
TRAIN_CONFIG = "pi05_wuji_marvin_54d"
REPO_ID = "dexbench_12tasks"


def _is_complete_checkpoint_dir(path: Path) -> bool:
    return (path.is_dir() and '.orbax-checkpoint-tmp-' not in path.name
            and (path / '_CHECKPOINT_METADATA').is_file()
            and (path / 'params').is_dir() and (path / 'assets').is_dir())


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    roots = candidate_checkpoint_roots(model_cfg, _POLICY_DIR / 'checkpoints',
                                      policy_dir=_POLICY_DIR,
                                      explicit_keys=('model_path', 'checkpoint_path'))
    for root in roots:
        if not root.is_dir():
            continue
        candidates = ([root] if _is_complete_checkpoint_dir(root) else [])
        candidates += [p for p in root.iterdir() if p.name.isdigit() and _is_complete_checkpoint_dir(p)]
        requested = model_cfg.get('checkpoint_num')
        if requested is not None:
            # A requested step must never silently load some other epoch.
            text = str(requested).rstrip('/').rsplit('/', 1)[-1]
            match = re.fullmatch(r'(?:steps?_)?(\d+)', text)
            if match is None:
                raise ValueError(f'Invalid checkpoint_num: {requested}')
            candidates = [p for p in candidates if p.name.isdigit() and int(p.name) == int(match[1])]
        if candidates:
            return max(candidates, key=lambda p: int(p.name) if p.name.isdigit() else -1)
    raise FileNotFoundError(f'No finalized Pi0.5 checkpoint matching the requested step: {roots}')


def _runtime_repack_transforms(train_config):
    if not isinstance(train_config.data, config.LeRobotWujiDataConfig):
        raise ValueError('DexBench inference requires LeRobotWujiDataConfig')
    return transforms.Group(inputs=[transforms.RepackTransform({
        'observation/image': 'images/cam_high',
        'observation/left_wrist_image': 'images/cam_left_wrist',
        'observation/right_wrist_image': 'images/cam_right_wrist',
        'observation/state': 'state', 'prompt': 'prompt',
    })])


def _image(image):
    value = np.asarray(image)
    if value.ndim != 3 or value.dtype != np.uint8:
        raise ValueError(f'Expected decoded RGB uint8 image; got {value.shape} {value.dtype}')
    if value.shape[-1] == 3:
        return np.transpose(value, (2, 0, 1))
    if value.shape[0] == 3:
        return value
    raise ValueError(f'Expected HWC/CHW RGB image; got {value.shape}')


def _camera(obs, names):
    vision = obs.get('images', obs.get('vision', {}))
    for name in names:
        if name in vision:
            image = vision[name]
            if isinstance(image, dict):
                image = image.get('color', image.get('rgb'))
            if image is not None:
                return _image(image)
    raise KeyError(f'Required training camera missing: {names}')


def encode_obs(obs, robot_action_dim_info, *, task_name, prompt_embodiment):
    if isinstance(obs.get('state'), dict):
        state = pack_robot_state(obs, 'ee', robot_action_dim_info, source_type='obs')
    else:
        state = np.asarray(obs['state'])
    expected = 7 * len(robot_action_dim_info['arm_dim']) + sum(robot_action_dim_info['ee_dim'])
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (expected,) or not np.isfinite(state).all():
        raise ValueError(f'Expected a finite ({expected},) absolute EE/hand state; got {state.shape}')
    instruction = obs.get('instruction', obs.get('instructions', obs.get('prompt')))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A non-empty instruction is required')
    # A multi-task server accepts the exact training prompt per observation.
    # A fixed-task deployment keeps rejecting accidental cross-task requests.
    match = re.match(r'^\[([^/\]]+)/([^\]]+)\] (.+)$', instruction, re.DOTALL)
    if match:
        embodiment, observed_task, _ = match.groups()
        if embodiment != prompt_embodiment:
            raise ValueError('Observation embodiment prefix differs from deployment')
        if task_name not in (None, 'auto') and observed_task != task_name:
            raise ValueError('Observation task prefix differs from the deployment task')
    else:
        if instruction.startswith('['):
            raise ValueError('Malformed training prompt prefix')
        observed_task = obs.get('task_name') if task_name in (None, 'auto') else task_name
        if not isinstance(observed_task, str) or not observed_task or observed_task == 'auto':
            raise ValueError('Multi-task inference requires a training prompt prefix or task_name in the observation')
        instruction = f'[{prompt_embodiment}/{observed_task}] ' + instruction
    return dict(state=state, prompt=instruction, images={
        'cam_high': _camera(obs, ['cam_high', 'cam_head']),
        'cam_left_wrist': _camera(obs, ['cam_left_wrist']),
        'cam_right_wrist': _camera(obs, ['cam_right_wrist']),
    })


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.task_name = model_cfg.get('task_name') or 'auto'
        self.action_type = model_cfg.get('action_type', 'ee')
        if self.action_type != 'ee' or model_cfg.get('train_config_name', TRAIN_CONFIG) != TRAIN_CONFIG:
            raise ValueError(f'DexBench Pi0.5 requires action_type=ee and train_config_name={TRAIN_CONFIG}')
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg.get('env_cfg_type') or 'tianji_marvin_wuji')
        if len(self.robot_action_dim_info['arm_dim']) != 2 or self.robot_action_dim_info['ee_dim'] != [20, 20]:
            raise ValueError('The prepared dataset requires two arms with Wuji20 joints on each hand')
        self.action_dim = 7 * len(self.robot_action_dim_info['arm_dim']) + sum(self.robot_action_dim_info['ee_dim'])
        self.prompt_embodiment = model_cfg.get('prompt_embodiment', 'tianji_wujihand2_real')
        self.observation_window = None
        self._latest_env_idx_list = []
        self.policy = self.get_model(model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg):
        if model_cfg.get('debug_no_model', False):
            raise ValueError('Production Pi0.5 requires trained weights; fake policies belong in tests')
        train_config = config.get_config(TRAIN_CONFIG)
        if train_config.model.action_dim != self.action_dim or train_config.model.action_horizon != 50:
            raise ValueError('Deployment requires the trained 54D, horizon-50 model')
        root = _resolve_pi05_model_root(model_cfg)
        repo_id = model_cfg.get('repo_id') or REPO_ID
        if repo_id != REPO_ID:
            raise ValueError(f'This deployment requires checkpoint normalization assets for {REPO_ID}')
        # Never fall back to legacy joint/delta normalization statistics.
        from openpi.policies.dexbench_contract import ACTION_CONTRACT
        contract_file = root / 'assets' / repo_id / 'action_contract.json'
        if not contract_file.is_file() or json.loads(contract_file.read_text()) != ACTION_CONTRACT:
            raise ValueError('Checkpoint is missing the DexBench absolute EE54 action contract')
        norm_stats = normalize.load(root / 'assets' / repo_id)
        for key in ('state', 'actions'):
            if np.asarray(norm_stats[key].mean).shape != (self.action_dim,):
                raise ValueError(f'{key} normalization statistics do not match the 54D EE contract')
        inference_steps = int(model_cfg.get('inference_steps', 10))
        if not 1 <= inference_steps <= 100:
            raise ValueError('inference_steps must be between 1 and 100')
        norm_file = root / 'assets' / repo_id / 'norm_stats.json'
        self._deployment_metadata = dict(
            checkpoint_path=str(root), checkpoint_num=int(root.name),
            train_config_name=TRAIN_CONFIG, repo_id=repo_id, action_dim=self.action_dim,
            action_horizon=50, inference_steps=inference_steps,
            action_mode='absolute_world_xyz_wxyz_wuji20', color='RGB',
            cameras=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            prompt_embodiment=self.prompt_embodiment, task_name=self.task_name,
            norm_stats_sha256=hashlib.sha256(norm_file.read_bytes()).hexdigest(),
        )
        print('[Pi_05] deployment=' + json.dumps(self._deployment_metadata), flush=True)
        return policy_config.create_trained_policy(train_config, root, norm_stats=norm_stats,
                                                   repack_transforms=_runtime_repack_transforms(train_config),
                                                   sample_kwargs={'num_steps': inference_steps})

    def get_metadata(self):
        return dict(self._deployment_metadata)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get('env_idx', i) for i, obs in enumerate(obs_list)]
        if len(set(self._latest_env_idx_list)) != len(obs_list):
            raise ValueError('Duplicate env_idx values')
        self.observation_window = [encode_obs(obs, self.robot_action_dim_info,
                                              task_name=self.task_name,
                                              prompt_embodiment=self.prompt_embodiment) for obs in obs_list]

    def get_action(self, **kwargs):
        if not self._latest_env_idx_list:
            raise RuntimeError('Call update_obs first')
        return self.get_action_batch([self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise RuntimeError('Call update_obs or update_obs_batch first')
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        results = []
        for env_idx in env_idx_list:
            # Support the simulator's active-environment subset and ordering.
            index = self._latest_env_idx_list.index(env_idx)
            actions = np.asarray(self.policy.infer(self.observation_window[index], **kwargs)['actions'], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != self.action_dim or not np.isfinite(actions).all():
                raise ValueError(f'Expected finite action chunk [T,{self.action_dim}]; got {actions.shape}')
            chunk = unpack_robot_state(actions, 'ee', self.robot_action_dim_info, source_type='obs')
            for action in chunk:
                for key in ('left_ee_pose', 'right_ee_pose'):
                    pose = action[key].copy()
                    magnitude = np.linalg.norm(pose[3:7])
                    if magnitude < 1e-8:
                        raise ValueError(f'Model emitted a zero quaternion in {key}')
                    pose[3:7] /= magnitude  # preserve wxyz order and absolute xyz
                    action[key] = pose
            results.append(chunk)
        return results

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = []

    def reset_obsrvationwindows(self):
        self.reset()
