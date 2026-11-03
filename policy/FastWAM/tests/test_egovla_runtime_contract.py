import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from XPolicyLab.policy.FastWAM.checkpoint_runtime import build_contract, resolve_runtime
from XPolicyLab.policy.FastWAM.model import Model


def split(dataset, stats):
    images = [
        {"key": key, "raw_shape": [3, 384, 384], "shape": [3, 240, 320]}
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    ]
    vector = [{"key": "default", "raw_shape": 38, "shape": 38}]
    return {
        "dataset_dirs": [str(dataset)], "shape_meta": {"images": images, "action": vector, "state": vector},
        "num_frames": 33, "global_sample_stride": 1, "action_video_freq_ratio": 4,
        "video_size": [384, 320], "concat_multi_camera": "robotwin",
        "pretrained_norm_stats": str(stats),
        "processor": {"shape_meta": {"images": images, "action": vector, "state": vector},
                      "action_output_dim": 38, "proprio_output_dim": 38},
    }


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.run = root / "run"
        self.dataset = root / "dataset"
        (self.run / "checkpoints/weights").mkdir(parents=True)
        (self.dataset / "meta").mkdir(parents=True)
        self.weight = self.run / "checkpoints/weights/step_000001.pt"
        self.weight.write_bytes(b"tiny checkpoint")
        self.stats = root / "stats.json"
        self.stats.write_text(json.dumps({"state": [1, 2, 3]}))
        manifest = {
            "converter": "convert_egovla_fastwam.py", "action_dim": 38,
            "wrist_policy": "missing wrist videos stay black placeholders; camera_mask remains false",
        }
        (self.dataset / "meta/fastwam_conversion_manifest.json").write_text(json.dumps(manifest))
        source = {
            "episodes": [
                {"task": "Push-Box", "has_real_wrist_cameras": False},
                {"task": "Insert-And-Unload-Cans", "has_real_wrist_cameras": True},
            ],
            "task_episode_counts": {"Push-Box": 1, "Insert-And-Unload-Cans": 1},
        }
        (self.dataset / "meta/xpolicylab_source_conversion.json").write_text(json.dumps(source))
        data = split(self.dataset, self.stats)
        cfg = {
            "eval_num_inference_steps": 10,
            "data": {"train": data, "val": split(self.dataset, self.stats)},
            "model": {"proprio_dim": 38, "action_dit_config": {"action_dim": 38},
                      "video_dit_config": {"action_dim": 38}},
        }
        self.config = self.run / "config.yaml"
        self.config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        contract = build_contract(self.config, [self.weight.name])
        (self.run / "fastwam_inference_contract.json").write_text(json.dumps(contract))

    def tearDown(self):
        self.temp.cleanup()

    def test_task_camera_modes_and_checkpoint_config(self):
        cfg, mode, summary = resolve_runtime(self.weight, self.stats, "Humanoid-Push-Box-v0", 38)
        self.assertEqual(mode, "black_wrist")
        self.assertEqual(cfg["data"]["train"]["video_size"], [384, 320])
        self.assertEqual(summary["action_horizon"], 32)
        _, mode, _ = resolve_runtime(
            self.weight, self.stats, "Humanoid-Insert-And-Unload-Cans-v0", 38
        )
        self.assertEqual(mode, "real_wrist")

    def test_stats_and_weight_changes_fail_closed(self):
        self.stats.write_text(json.dumps({"state": [9]}))
        with self.assertRaisesRegex(ValueError, "normalization"):
            resolve_runtime(self.weight, self.stats, "Humanoid-Push-Box-v0", 38)
        self.stats.write_text(json.dumps({"state": [1, 2, 3]}))
        self.weight.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_runtime(self.weight, self.stats, "Humanoid-Push-Box-v0", 38)

    def test_dataset_provenance_changes_fail_closed(self):
        source_path = self.dataset / "meta/xpolicylab_source_conversion.json"
        source = json.loads(source_path.read_text())
        source["episodes"][0]["has_real_wrist_cameras"] = True
        source_path.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, "provenance manifests changed"):
            resolve_runtime(self.weight, self.stats, "Humanoid-Push-Box-v0", 38)

    def test_unknown_task_and_wrong_dimension_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "robot/action"):
            resolve_runtime(self.weight, self.stats, "Humanoid-Push-Box-v0", 54)
        with self.assertRaisesRegex(ValueError, "Unknown EgoVLA task"):
            resolve_runtime(self.weight, self.stats, "Humanoid-Unknown-v0", 38)


class AdapterCameraTests(unittest.TestCase):
    def observation(self):
        images = {
            "cam_head": {"color": np.full((4, 5, 3), 11, np.uint8)},
            "cam_left_wrist": {"color": np.full((4, 5, 3), 22, np.uint8)},
            "cam_right_wrist": {"color": np.full((4, 5, 3), 33, np.uint8)},
        }
        state = {
            "left_arm_joint_state": np.zeros(7), "left_ee_joint_state": np.zeros(12),
            "right_arm_joint_state": np.zeros(7), "right_ee_joint_state": np.zeros(12),
        }
        return {"vision": images, "state": state}

    def model(self, mode):
        model = Model.__new__(Model)
        model.ego_camera_mode = mode
        model.action_type = "joint"
        model.robot_action_dim_info = {"arm_dim": [7, 7], "ee_dim": [12, 12]}
        model.action_dim = 38
        return model

    def test_black_wrists_and_real_wrists(self):
        filled = self.model("black_wrist")._encode_obs_for_fastwam(self.observation())
        images = filled["observation"]
        self.assertEqual(int(images["head_camera"]["rgb"][0, 0, 0]), 11)
        np.testing.assert_array_equal(images["left_camera"]["rgb"], 0)
        np.testing.assert_array_equal(images["right_camera"]["rgb"], 0)
        real = self.model("real_wrist")._encode_obs_for_fastwam(self.observation())["observation"]
        self.assertEqual(int(real["left_camera"]["rgb"][0, 0, 0]), 22)
        self.assertEqual(int(real["right_camera"]["rgb"][0, 0, 0]), 33)


if __name__ == "__main__":
    unittest.main(verbosity=2)
