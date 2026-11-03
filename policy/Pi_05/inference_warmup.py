"""Compile the EgoVLA Pi policy before serving without advancing its RNG."""
from __future__ import annotations

import time
import numpy as np


def warmup_egovla_policy(policy, *, action_dim: int) -> None:
    # OpenPI's JAX Policy.infer advances _rng, then returns synchronized NumPy
    # actions. Restore the immutable key even if compilation raises. No model
    # observation window, environment, or real rollout is touched here.
    if not hasattr(policy, '_rng') or getattr(policy, '_is_pytorch_model', False):
        raise TypeError('EgoVLA Pi warmup requires the audited OpenPI JAX Policy')
    rng = policy._rng
    observation = {
        'state': np.zeros(action_dim, dtype=np.float32),
        'images': {key: np.zeros((3, 384, 384), dtype=np.uint8)
                   for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')},
        'camera_mask': np.ones(3, dtype=bool),
        'prompt': 'Move the robot to complete the task.',
    }
    started = time.monotonic()
    print('[Pi_05] inference warmup begin (before websocket readiness)', flush=True)
    try:
        actions = np.asarray(policy.infer(observation)['actions'])
        if actions.ndim != 2 or actions.shape[-1] != action_dim or not np.isfinite(actions).all():
            raise RuntimeError(f'Pi warmup returned invalid actions: {actions.shape}')
    finally:
        policy._rng = rng
    print(f'[Pi_05] inference warmup complete seconds={time.monotonic()-started:.3f}; RNG restored', flush=True)
