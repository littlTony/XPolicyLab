import importlib.util
from pathlib import Path

import numpy as np
import torch

from configs.state_vec import STATE_VEC_LEN
from configs.tianji_joint54 import (
    TIANJI_JOINT54_DIM,
    TIANJI_JOINT54_STATE_INDICES,
)


def _load_robodojo_class_without_constructing_model():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "robodojo_model.py"
    spec = importlib.util.spec_from_file_location("robodojo_model_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoboticDiffusionTransformerModel


def test_mapping_is_complete_unique_and_bounded():
    assert len(TIANJI_JOINT54_STATE_INDICES) == TIANJI_JOINT54_DIM
    assert len(set(TIANJI_JOINT54_STATE_INDICES)) == TIANJI_JOINT54_DIM
    assert min(TIANJI_JOINT54_STATE_INDICES) >= 0
    assert max(TIANJI_JOINT54_STATE_INDICES) < STATE_VEC_LEN


def test_training_pack_preserves_all_54_values_and_mask():
    raw = np.arange(TIANJI_JOINT54_DIM, dtype=np.float32)[None, :]
    unified = np.zeros((1, STATE_VEC_LEN), dtype=np.float32)
    unified[..., TIANJI_JOINT54_STATE_INDICES] = raw
    indicator = np.zeros((STATE_VEC_LEN,), dtype=np.float32)
    indicator[TIANJI_JOINT54_STATE_INDICES] = 1

    np.testing.assert_array_equal(
        unified[..., TIANJI_JOINT54_STATE_INDICES], raw
    )
    assert int(indicator.sum()) == TIANJI_JOINT54_DIM


def test_deployment_round_trip_preserves_all_54_values():
    model_cls = _load_robodojo_class_without_constructing_model()
    model = object.__new__(model_cls)
    model.args = {"model": {"state_token_dim": STATE_VEC_LEN}}
    raw = torch.arange(TIANJI_JOINT54_DIM, dtype=torch.float32).reshape(1, 1, -1)

    unified, mask = model._format_joint_to_state(raw)
    restored = model._unformat_action_to_joint(unified)

    torch.testing.assert_close(restored, raw)
    assert int(mask.sum().item()) == TIANJI_JOINT54_DIM


def test_mobile_aloha_14d_compatibility_round_trip():
    model_cls = _load_robodojo_class_without_constructing_model()
    model = object.__new__(model_cls)
    model.args = {"model": {"state_token_dim": STATE_VEC_LEN}}
    raw = torch.arange(14, dtype=torch.float32).reshape(1, 1, -1)

    unified, mask = model._format_joint_to_state(raw)
    restored = model._unformat_action_to_joint(unified)

    torch.testing.assert_close(restored, raw)
    assert int(mask.sum().item()) == 14
