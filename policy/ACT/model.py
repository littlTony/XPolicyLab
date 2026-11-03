import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
import os

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self.camera_names = model_cfg.get('camera_names', [])
        model_cfg['camera_names'] = self.camera_names

        # A single shared ACT instance serves all parallel envs. Per-env
        # temporal state lives inside the ACT instance (dicts keyed by env_idx),
        # and get_action_batch loops over envs doing one single-env forward
        # each. This avoids both a batched-forward activation spike and
        # duplicated model weights.
        self.model = self.get_model(model_cfg=model_cfg)
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self.action_type = model_cfg['action_type']
        self._latest_env_idx_list = []

    def get_model(self, model_cfg):
        if not model_cfg.get('ckpt_dir'):
            if not model_cfg.get('ckpt_name'):
                raise ValueError("ACT requires ckpt_name or ckpt_dir during evaluation.")
            # ckpt_name is the full run directory name under checkpoints/.
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(model_cfg['ckpt_name']))
        return ACT(model_cfg, Namespace(**model_cfg))

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [int(obs.get("env_idx", 0)) for obs in obs_list]
        self._latest_env_idx_list = env_idx_list
        for env_idx, obs in zip(env_idx_list, obs_list):
            encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
            self.model.update_obs(encoded_obs, env_idx)

    def get_action(self):
        if not self._latest_env_idx_list:
            raise AssertionError("update_obs must be called before get_action.")
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        if not env_idx_list:
            return []
        raw_actions = self.model.get_action_batch(env_idx_list)
        return [
            unpack_robot_state(
                raw_action,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for raw_action in raw_actions
        ]

    def reset(self):
        self.model.reset()

    def encode_obs(self, observation, action_type, robot_action_dim_info):
        res_dict = dict()

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError(f"Expected camera '{camera_name}' not found in observation['vision']")
            color = cv2.resize(observation["vision"][camera_name]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color

        res_dict["qpos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")

        return res_dict
