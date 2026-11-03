import numpy as np
import torch

from policy.RDT_1B.model import (
    Model,
    _resolve_image_size,
    _resolve_runtime_prompt,
    encode_obs,
)
from policy.RDT_1B.rdt.scripts.robodojo_model import (
    RoboticDiffusionTransformerModel,
)
from XPolicyLab.utils.process_data import unpack_robot_state


def test_runtime_config_uses_all_arm_and_hand_joints():
    model = object.__new__(Model)
    model.model_cfg = {}
    model.env_cfg = "tianji_marvin_wuji"
    model.robot_action_dim_info = {"arm_dim": [7, 7], "ee_dim": [20, 20]}

    config = model._build_runtime_config()

    assert config["state_dim"] == 54


def test_encode_obs_preserves_tianji_wuji_raw_order_and_width():
    fields = {
        "left_arm_joint_state": np.arange(0, 7, dtype=np.float32),
        "left_ee_joint_state": np.arange(7, 27, dtype=np.float32),
        "right_arm_joint_state": np.arange(27, 34, dtype=np.float32),
        "right_ee_joint_state": np.arange(34, 54, dtype=np.float32),
    }
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "vision": {
            "cam_head": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "state": fields,
    }

    encoded = encode_obs(observation, "test")

    np.testing.assert_array_equal(
        encoded["state"], np.arange(54, dtype=np.float32)
    )


def test_full_deploy_pack_decode_and_robot_unpack_preserves_54_values():
    model = object.__new__(RoboticDiffusionTransformerModel)
    model.args = {"model": {"state_token_dim": 128}}
    raw = torch.arange(54, dtype=torch.float32).reshape(1, 1, 54)

    unified, mask = model._format_joint_to_state(raw)
    decoded = model._unformat_action_to_joint(unified).squeeze(0).numpy()
    unpacked = unpack_robot_state(
        decoded,
        "joint",
        {"arm_dim": [7, 7], "ee_dim": [20, 20]},
        source_type="obs",
    )

    assert int(mask.sum().item()) == 54
    assert len(unpacked) == 1
    np.testing.assert_array_equal(
        unpacked[0]["left_arm_joint_state"], np.arange(0, 7, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        unpacked[0]["left_ee_joint_state"], np.arange(7, 27, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        unpacked[0]["right_arm_joint_state"], np.arange(27, 34, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        unpacked[0]["right_ee_joint_state"], np.arange(34, 54, dtype=np.float32)
    )


def test_egovla_single_view_fills_wrist_slots_with_black():
    head = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    observation = {
        "vision": {
            "cam_head": head,
            "cam_left_wrist": head,
            "cam_right_wrist": head.copy(),
        },
        "state": {
            "left_arm_joint_state": np.zeros(7, dtype=np.float32),
            "left_ee_joint_state": np.zeros(12, dtype=np.float32),
            "right_arm_joint_state": np.zeros(7, dtype=np.float32),
            "right_ee_joint_state": np.zeros(12, dtype=np.float32),
        },
        "instruction": "Close the opened drawer",
    }

    encoded = encode_obs(observation, "fallback", camera_mode="black_wrist")

    np.testing.assert_array_equal(encoded["images"]["cam_high"], head)
    assert encoded["images"]["cam_left_wrist"].shape == head.shape
    assert encoded["images"]["cam_right_wrist"].shape == head.shape
    assert int(encoded["images"]["cam_left_wrist"].sum()) == 0
    assert int(encoded["images"]["cam_right_wrist"].sum()) == 0
    assert not np.shares_memory(encoded["images"]["cam_left_wrist"], encoded["images"]["cam_right_wrist"])


def test_tianji_image_size_matches_training_640x480_even_if_yaml_pins_egovla_384():
    assert _resolve_image_size("tianji_marvin_wuji", None) == (640, 480)
    assert _resolve_image_size("tianji_marvin_wuji", [384, 384]) == (640, 480)
    assert _resolve_image_size("tianji_marvin_wuji", [640, 480]) == (640, 480)
    assert _resolve_image_size("ego_h1_inspire", None) == (384, 384)
    assert _resolve_image_size("ego_h1_inspire", [384, 384]) == (384, 384)


def test_spark_task_slug_is_rewritten_to_hdf5_instruction():
    assert (
        _resolve_runtime_prompt("click_mouse")
        == "Place the mouse on the mouse pad, then click the left button."
    )
    assert (
        _resolve_runtime_prompt("Stack the three bowls together.")
        == "Stack the three bowls together."
    )
    observation = {
        "vision": {
            "cam_head": np.zeros((8, 8, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        "state": {
            "left_arm_joint_state": np.zeros(7, dtype=np.float32),
            "left_ee_joint_state": np.zeros(20, dtype=np.float32),
            "right_arm_joint_state": np.zeros(7, dtype=np.float32),
            "right_ee_joint_state": np.zeros(20, dtype=np.float32),
        },
        "instruction": "stack_bowls",
    }
    encoded = encode_obs(observation, "fallback")
    assert encoded["prompt"] == "Stack the three bowls together."


def test_precomp_lang_embed_resolves_spark_7task_file():
    model = object.__new__(Model)
    model.model_cfg = {}
    model.task_name = "click_mouse"
    model.env_cfg = "tianji_marvin_wuji"
    resolved = model._resolve_precomp_lang_embed()
    assert resolved is not None
    assert resolved.name == "lang_embed.pt"
    assert resolved.parent.name == "tianji_marvin_wuji"
    assert resolved.parent.parent.name == "click_mouse"
