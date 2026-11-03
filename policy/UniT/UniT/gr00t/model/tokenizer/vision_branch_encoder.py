# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vision Branch Encoder: vision_model → M-Former → visual query tokens

Extracts visual query tokens from observation and goal images
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from transformers import PretrainedConfig
from transformers.models.vit.modeling_vit import ViTConfig

from .m_former import MFormer


@dataclass
class VisionBranchEncoderConfig(PretrainedConfig):
    """Configuration for Vision Branch Encoder"""
    
    query_num: int = field(default=8, metadata={"help": "Number of query tokens to extract"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    m_former_cfg: dict = field(default=None, metadata={"help": "M-Former configuration dict"})
    pool_views: bool = field(
        default=True,
        metadata={
            "help": (
                "Mean-pool per-view patch features after the vision backbone. "
                "This keeps one DINO patch sequence per image while accepting [B,V,C,H,W]."
            )
        },
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Older tokenizer configs do not contain this field; preserve their
        # single-view behavior and enable view pooling by default for new data.
        self.pool_views = bool(getattr(self, "pool_views", True))


class VisionBranchEncoder(nn.Module):
    """
    Vision Branch Encoder
    
    Architecture:
    1. Vision Model (from backbone): Extract patch-level features from images
    2. M-Former: Extract fixed number of query tokens from patch features
    
    This module handles both observation and goal images through the same pipeline.
    """
    
    config_class = VisionBranchEncoderConfig
    
    def __init__(self, config: VisionBranchEncoderConfig, vision_model: nn.Module):
        """
        Args:
            config: Configuration for the vision branch
            vision_model: Pre-initialized vision backbone (e.g. DINOv2 wrapper).
        """
        super().__init__()
        self.config = config
        self.query_num = config.query_num
        self.hidden_size = config.hidden_size
        # Mean-pool aligned DINO patch grids across camera views.  Keeping the
        # patch count unchanged (normally 256) avoids changing M-Former and
        # vision-decoder positional tables when moving from one to three views.
        self.pool_views = bool(getattr(config, "pool_views", True))
        self.vision_model = vision_model

        if config.m_former_cfg is not None:
            m_former_vit_config = ViTConfig(**config.m_former_cfg)
            # Override critical parameters
            m_former_vit_config.query_num = config.query_num
            m_former_vit_config.hidden_size = config.hidden_size
            m_former_vit_config.input_hidden_size = config.hidden_size
        else:
            # Default M-Former config
            m_former_vit_config = ViTConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=4,
                num_attention_heads=max(1, config.hidden_size // 64),
                intermediate_size=config.hidden_size * 4,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                query_num=config.query_num,
                input_hidden_size=config.hidden_size,
            )
        
        self.m_former = MFormer(m_former_vit_config)
    
    def _encode_dino_images(
        self,
        images: torch.Tensor,
        batch_size: int | None,
        name: str,
        num_views: int | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode DINO images and pool aligned camera/time grids.

        DINOv2 accepts ``[N,C,H,W]`` while the transform collates EgoVLA
        images as ``[B,V*T,C,H,W]`` (view-major flattening).  We restore the
        ``[B,V,T,P,D]`` layout before pooling.  Pooling first across views and
        then across the sampled time frames keeps the existing M-Former patch
        length and avoids treating 48 frames from a three-view episode as 48
        unrelated cameras.

        ``view_mask`` is optional for callers that transport the benchmark's
        ``[head,left_wrist,right_wrist]`` camera mask.  When it is absent, the
        converter's all-black inactive wrist fallback is detected by zero
        spatial variance and excluded from the mean.  This preserves head-only
        episodes without requiring a new model input field.
        """
        if not torch.is_tensor(images):
            raise TypeError(
                f"{name}_input must be a tensor for DINOv2, got {type(images)!r}"
            )

        if images.ndim == 5:
            actual_batch = int(images.shape[0])
            total_images = int(images.shape[1])
            if batch_size not in (None, 1, actual_batch):
                raise ValueError(
                    f"{name}_input batch mismatch: batch_size={batch_size}, "
                    f"tensor shape={tuple(images.shape)}"
                )
            if num_views is None:
                # Legacy callers did not expose a view count. Treat each item
                # as a view and pool them, preserving the historical behavior.
                num_views = total_images
            num_views = int(num_views)
            if num_views <= 0 or total_images % num_views:
                raise ValueError(
                    f"{name}_input shape {tuple(images.shape)} is incompatible "
                    f"with num_views={num_views}"
                )
            frames_per_view = total_images // num_views
            flat_images = images.reshape(actual_batch * total_images, *images.shape[2:])
        elif images.ndim == 4:
            # A collated tensor is [B,C,H,W].  For an uncollated sample,
            # [V,C,H,W] is accepted when batch_size=1.
            if batch_size == 1 and int(images.shape[0]) > 1:
                actual_batch, total_images = 1, int(images.shape[0])
                if num_views is None:
                    num_views = total_images
                num_views = int(num_views)
                if num_views <= 0 or total_images % num_views:
                    raise ValueError(
                        f"{name}_input shape {tuple(images.shape)} is incompatible "
                        f"with num_views={num_views}"
                    )
                frames_per_view = total_images // num_views
                flat_images = images.reshape(actual_batch * total_images, *images.shape[1:])
            else:
                actual_batch, total_images = int(images.shape[0]), 1
                if batch_size not in (None, 1, actual_batch):
                    raise ValueError(
                        f"{name}_input batch mismatch: batch_size={batch_size}, "
                        f"tensor shape={tuple(images.shape)}"
                    )
                num_views = 1 if num_views is None else int(num_views)
                if num_views != 1:
                    raise ValueError(
                        f"{name}_input 4-D batch cannot represent num_views={num_views}"
                    )
                frames_per_view = 1
                flat_images = images.reshape(actual_batch, *images.shape[1:])
        else:
            raise ValueError(
                f"{name}_input must have shape [B,C,H,W] or [B,V*T,C,H,W] for DINOv2; "
                f"got {tuple(images.shape)}"
            )

        encoded = self.vision_model(flat_images)
        if not torch.is_tensor(encoded) or encoded.ndim != 3:
            raise RuntimeError(
                f"DINO vision model must return [N,P,D] patch features; "
                f"got {type(encoded)!r} with shape {getattr(encoded, 'shape', None)}"
            )
        flat_count, patch_count, hidden_size = (int(v) for v in encoded.shape)
        expected_count = actual_batch * total_images
        if flat_count != expected_count:
            raise RuntimeError(
                f"{name}_input flatten mismatch: DINO returned N={flat_count}, "
                f"expected B*V*T={expected_count}"
            )
        if hidden_size != self.hidden_size:
            raise RuntimeError(
                f"DINO hidden size mismatch: returned {hidden_size}, "
                f"configured {self.hidden_size}"
            )

        # Restore view and sampled-time axes.  The data transform is view-major
        # (all times for head, then all times for left wrist, then right wrist).
        encoded = encoded.reshape(
            actual_batch, num_views, frames_per_view, patch_count, hidden_size
        )
        if self.pool_views:
            if num_views > 1:
                # Inactive RLDX wrist fallback videos are exactly black.
                # Spatial variance is therefore zero even after ImageNet
                # normalization; use it as a conservative fallback mask when
                # no explicit mask is transported.
                spatial_std = (
                    flat_images.float()
                    .flatten(2)
                    .std(dim=2, unbiased=False)
                    .mean(dim=1)
                    .reshape(actual_batch, num_views, frames_per_view)
                )
                valid = spatial_std > 1e-6
                if view_mask is not None:
                    mask = view_mask
                    if not torch.is_tensor(mask):
                        mask = torch.as_tensor(mask, device=encoded.device)
                    mask = mask.to(device=encoded.device)
                    if mask.ndim == 1:
                        mask = mask.reshape(1, -1)
                    if mask.ndim == 2:
                        if int(mask.shape[0]) not in (1, actual_batch) or int(mask.shape[1]) != num_views:
                            raise ValueError(
                                f"{name}_view_mask shape {tuple(mask.shape)} does not match "
                                f"[B,V]=[{actual_batch},{num_views}]"
                            )
                        if int(mask.shape[0]) == 1 and actual_batch > 1:
                            mask = mask.expand(actual_batch, -1)
                        mask = mask[:, :, None].expand(-1, -1, frames_per_view)
                    elif mask.ndim == 3:
                        if tuple(mask.shape) != (actual_batch, num_views, frames_per_view):
                            raise ValueError(
                                f"{name}_view_mask shape {tuple(mask.shape)} does not match "
                                f"[B,V,T]=[{actual_batch},{num_views},{frames_per_view}]"
                            )
                    else:
                        raise ValueError(
                            f"{name}_view_mask must have [V], [B,V], or [B,V,T] shape; "
                            f"got {tuple(mask.shape)}"
                        )
                    valid = valid & (mask > 0.5)

                # If a real image happens to be spatially constant, do not
                # drop the entire sample; fall back to all views for that
                # sample/time.
                no_valid = ~valid.any(dim=1, keepdim=True)
                valid = torch.where(no_valid, torch.ones_like(valid), valid)
                weights = valid.to(dtype=encoded.dtype)
                encoded = (encoded * weights[..., None, None]).sum(dim=1)
                encoded = encoded / weights.sum(dim=1).clamp_min(1.0)[..., None, None]
            else:
                # Keep the view axis explicit for the single-view path, then
                # reduce it below together with any sampled-time axis.
                encoded = encoded.mean(dim=1)

        # Keep M-Former's historical [B,P,D] contract.  For V*T input, average
        # the sampled time axis after view pooling; with T=1 this is a no-op.
        if self.pool_views:
            if encoded.ndim == 4:
                encoded = encoded.mean(dim=1)  # [B,P,D] after [B,T,P,D]
            elif encoded.ndim == 5:
                encoded = encoded.mean(dim=1).mean(dim=1)  # [B,P,D]
            elif encoded.ndim != 3:
                raise RuntimeError(f"unexpected pooled {name} shape {tuple(encoded.shape)}")
        else:
            encoded = encoded.reshape(actual_batch, num_views * frames_per_view * patch_count, hidden_size)
        return encoded

    def forward(
        self,
        obs_input,
        goal_input,
        batch_size: int = 1,
        obs_view_mask: torch.Tensor | None = None,
        goal_view_mask: torch.Tensor | None = None,
    ):
        """
        Extract visual query tokens from observation and goal images.

        Args:
            obs_input/goal_input: either a dict of vision-model kwargs (eagle-style)
                or a tensor of pixel values (DINOv2-style). DINO tensors may be
                ``[B,C,H,W]`` or multiview ``[B,V,C,H,W]``.
            batch_size: outer batch size used to reshape patch features.

        Returns:
            visual_tokens: (B, query_num, hidden_size) M-Former query outputs.
            obs_features:  (B, N_patches, hidden_size) patch features for obs.
            goal_features: (B, N_patches, hidden_size) patch features for goal.
        """
        if goal_input is None:
            raise ValueError("Goal input is required for vision branch encoder")

        with torch.no_grad():
            if isinstance(obs_input, dict):
                obs_features = self.vision_model(*obs_input.values()).reshape(batch_size, -1, self.hidden_size)
                goal_features = self.vision_model(*goal_input.values()).reshape(batch_size, -1, self.hidden_size)
            else:
                # Goal images are collated as [B,V,C,H,W], so their second
                # dimension provides the canonical camera count for the
                # flattened observation stack [B,V*T,C,H,W].
                view_count = None
                if torch.is_tensor(goal_input):
                    if goal_input.ndim == 5:
                        view_count = int(goal_input.shape[1])
                    elif goal_input.ndim == 4 and batch_size == 1:
                        view_count = int(goal_input.shape[0])
                obs_features = self._encode_dino_images(
                    obs_input, batch_size, "obs", view_count, obs_view_mask
                )
                goal_features = self._encode_dino_images(
                    goal_input, batch_size, "goal", view_count, goal_view_mask
                )

        # M-Former layout: [query_tokens, cond, sep, target] -- take the first query_num tokens.
        m_former_output = self.m_former(
            cond_hidden_states=obs_features,
            target_hidden_states=goal_features,
        )
        visual_tokens = m_former_output.last_hidden_state[:, :self.query_num, :]
        return visual_tokens, obs_features, goal_features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    def get_encoder_info(self) -> dict:
        """Get encoder module information"""
        return {
            "type": "VisionBranchEncoder",
            "query_num": self.query_num,
            "hidden_size": self.hidden_size,
            "pool_views": self.pool_views,
            "m_former_num_layers": self.m_former.config.num_hidden_layers,
            "architecture": (
                "vision_model -> restore view/time/patch grid -> masked mean-pool "
                "views/time -> M-Former(obs_cond, goal_target) -> query_tokens"
            ),
        }

