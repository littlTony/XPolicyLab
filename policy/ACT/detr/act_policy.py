import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, model_cfg=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, model_cfg)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:

    def __init__(self, args_override=None, model_cfg=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }

        self.camera_names = model_cfg.camera_names

        self.policy = ACTPolicy(args_override, model_cfg)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = model_cfg.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 5000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1

        # Per-env temporal state. DexBench calls update_obs_batch / get_action_batch
        # with num_envs=5; a single shared policy instance serves all envs.
        self.t_by_env = {}
        self.obs_by_env = {}
        self.all_actions_by_env = {}
        self.all_time_actions_by_env = {}
        self.t = 0
        self.obs_cache = None

        # Load statistics and policy weights. Missing files are fatal because
        # running with an unnormalized/randomly initialized ACT policy is invalid.
        ckpt_dir = args_override.get("ckpt_dir", "")
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl") if ckpt_dir else ""
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt") if ckpt_dir else ""
        if not ckpt_dir:
            raise ValueError("ACT requires ckpt_dir during evaluation.")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"ACT checkpoint directory not found: {ckpt_dir}")

        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"ACT dataset stats not found: {stats_path}")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ACT policy checkpoint not found: {ckpt_path}")
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def update_obs(self, obs, env_idx=0):
        env_idx = int(env_idx)
        self.obs_by_env[env_idx] = obs
        self.obs_cache = obs
        if env_idx not in self.t_by_env:
            self.t_by_env[env_idx] = 0

    def update_obs_batch(self, obs_list):
        for obs in obs_list:
            self.update_obs(obs, int(obs.get("env_idx", 0)))

    def _preprocess_qpos(self, qpos_numpy):
        qpos_normalized = self.pre_process(np.array(qpos_numpy))
        return torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

    def _preprocess_image(self, obs):
        curr_images = [obs[cam_name] for cam_name in self.camera_names]
        curr_image = np.stack(curr_images, axis=0)
        return torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

    def _get_raw_actions_batch(self, env_idx_list):
        raw_actions = []
        for e in env_idx_list:
            t = self.t_by_env.get(e, 0)
            qpos = self._preprocess_qpos(self.obs_by_env[e]["qpos"])
            image = self._preprocess_image(self.obs_by_env[e])
            with torch.no_grad():
                if t % self.query_frequency == 0:
                    self.all_actions_by_env[e] = self.policy(qpos, image)
            all_actions = self.all_actions_by_env[e]

            if self.temporal_agg:
                at = self.all_time_actions_by_env.setdefault(
                    e,
                    torch.zeros([
                        self.max_timesteps,
                        self.max_timesteps + self.num_queries,
                        self.state_dim,
                    ]),
                )
                at[[t], t:t + self.num_queries] = all_actions.detach().cpu()
                actions_for_curr_step = at[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % self.query_frequency]

            raw_action = raw_action.cpu().numpy()
            raw_actions.append(self.post_process(raw_action))
            self.t_by_env[e] = t + 1
            self.t = t + 1
        return raw_actions

    def get_action(self, env_idx=0):
        return self.get_action_batch([int(env_idx)])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(self.t_by_env.keys())
        env_idx_list = [int(e) for e in env_idx_list]
        if not env_idx_list:
            return []
        return self._get_raw_actions_batch(env_idx_list)

    def reset(self):
        self.t_by_env.clear()
        self.obs_by_env.clear()
        self.all_actions_by_env.clear()
        self.all_time_actions_by_env.clear()
        self.t = 0
        self.obs_cache = None