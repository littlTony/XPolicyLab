"""XPolicyLab UniT adapter: EE54 wire <-> absolute row-rotation EE58 model."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_batch_size
from .ee58_runtime import POLICY, Runtime, load_bundle, require


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.env_cfg_type = self.model_cfg.get("env_cfg_type", "tianji_marvin_wuji")
        self.action_type = self.model_cfg.get("action_type", "ee")
        require(self.env_cfg_type == "tianji_marvin_wuji" and self.action_type == "ee", "UniT EE58 requires tianji_marvin_wuji / ee")
        self.robot_dims = get_robot_action_dim_info(self.env_cfg_type)
        require(self.robot_dims["arm_dim"] == [7, 7] and self.robot_dims["ee_dim"] == [20, 20], "Expected dual Wuji20 hands")
        self.batch_size = int(self.model_cfg.get("batch_size") or get_batch_size(self.env_cfg_type))
        self.execute_steps = int(self.model_cfg.get("execute_steps", 16))
        require(1 <= self.execute_steps <= 16, "execute_steps must lie within the16-step predicted chunk")
        checkpoint = resolve_checkpoint_root(self.model_cfg, POLICY / "checkpoints", policy_dir=POLICY)
        if checkpoint.is_file():
            checkpoint = checkpoint.parent
        if not (checkpoint / "config.json").exists():
            if (checkpoint / "policy").is_dir():
                checkpoint = checkpoint / "policy"
            step = int(self.model_cfg.get("checkpoint_step") or 60000)
            checkpoint = checkpoint / f"checkpoint-{step}"
        elif not (checkpoint / "inference_contract.json").exists() and (checkpoint / "checkpoint-60000").is_dir():
            checkpoint = checkpoint / "checkpoint-60000"
        self.bundle = load_bundle(checkpoint, self.model_cfg.get("processor_path"))
        self.runtime = Runtime(self.bundle, self.robot_dims)
        self.device = str(self.model_cfg.get("device", "cuda:0"))
        self.model = self._load_native(self.bundle)
        require(getattr(self.model, "compute_bridge_loss", False) is False, "Online UniT must not run the future-image teacher")
        self.reset()

    def _load_native(self, bundle):
        from .training.runtime_compat import strict_checkpoint_load
        from gr00t.model.gr00t_n1_unit import GR00T_N1_5_UniT, GR00T_N1_5_UniT_Config
        config = deepcopy(bundle.config)
        config["backbone_cfg"]["eagle_path"] = bundle.processor_path
        config["unit_cfg"]["compute_bridge_loss"] = False
        # The future-image/DINO tokenizer is only a training teacher. Native
        # get_action never reads it when compute_bridge_loss=False.
        native = strict_checkpoint_load(GR00T_N1_5_UniT, bundle.checkpoint,
            config=GR00T_N1_5_UniT_Config(**config), local_model_path=str(bundle.checkpoint),
            tokenizer_len=len(self.runtime.processor.tokenizer), bridge_type="vision_lang_obs", select_layer=36,
            compute_bridge_loss=False, use_image_type_embedding=True, use_vl_mask=True, use_correct_attn_mask=True,
            groot_tokenizer_path=bundle.sidecar["teacher_tokenizer_path"], detach_vl_for_action=True,
            torch_dtype=torch.float32)
        require(native.action_horizon == 16 and native.action_dim == 128, "Refuse horizon/dimension changes; never rebuild an action head")
        require(native.action_head.num_inference_timesteps == bundle.sidecar["num_inference_timesteps"], "Inference denoising steps differ from training")
        native.backbone.eagle_model.config.use_cache = False
        native.requires_grad_(False).eval().to(self.device)
        return native

    def update_obs(self, obs):
        self._observations = [obs]

    def update_obs_batch(self, obs_list):
        require(isinstance(obs_list, (list, tuple)) and len(obs_list) > 0, "Expected a nonempty observation list")
        self._observations = list(obs_list)

    def _predict(self, observations):
        batch = self.runtime.batch(observations)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            result = self.model.get_action(batch)
        value = result["action_pred"]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return self.runtime.actions(value)

    def get_action(self):
        require(len(self._observations) >= 1, "Call update_obs before get_action")
        return self._predict(self._observations[:1])[0]

    def get_action_batch(self, env_idx_list=None):
        count = len(env_idx_list) if env_idx_list is not None else self.batch_size
        require(len(self._observations) == count, "Batch observations must match the active environment list/default batch size")
        return self._predict(self._observations)

    def reset(self):
        self._observations = []

    def get_metadata(self):
        return {"policy_name": "UniT", "action_chunk_size": 16, "execute_steps": self.execute_steps,
                "state_input": "raw_ee54_wire", "real_state_action_dim": 58, "model_dim": 128,
                "rotation": "matrix_rows_0_then_1", "normalization": "mean_std", "image_color_order": "rgb",
                "compute_bridge_loss": False, "future_images_required": False,
                "checkpoint_step": self.bundle.sidecar["optimizer_step"]}
