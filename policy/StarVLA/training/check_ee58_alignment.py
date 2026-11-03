"""Real ModelTemplate pipeline oracle; only the bottom neural network is stubbed.

No DINO/Qwen pretrained tensors or GPU are loaded. Fixtures live in caller-owned
temporary directories. JPEG transport tests may change pixels and are tested
separately from the exact raw-pixel training/runtime comparison below.
"""
from contextlib import contextmanager
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

POLICY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POLICY.parents[2]))
sys.path.insert(0, str(POLICY / "training"))
from XPolicyLab.utils.dexbench_ee58_contract import DEFAULT_DATASET, pack_ee54, validate_dataset
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state
from XPolicyLab.policy.StarVLA import ee58_runtime as runtime


@lru_cache(maxsize=4)
def source_parquet(path):
    return pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state", "action"])


def original_wire(spec, episode):
    source = Path(json.loads((spec["root"] / "meta/source_fingerprint.json").read_text())["source_root"])
    info = json.loads((source / "meta/info.json").read_text())
    rows = pd.concat([pd.read_parquet(p, columns=["episode_index", "data/chunk_index", "data/file_index"])
                      for p in sorted((source / "meta/episodes").rglob("*.parquet"))]).set_index("episode_index")
    row = rows.loc[episode]
    frame = source_parquet(str(source / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))))
    return frame[frame.episode_index == episode].sort_values("frame_index")


@lru_cache(maxsize=1)
def native_fixture_data():
    from prepare_sparkarena import inspect_source, prepare_native_metadata
    from omegaconf import OmegaConf
    runtime.native_imports()
    from sparkarena_registry import SparkArenaWujiConfig, OffsetLeRobotDataset
    spec = validate_dataset(DEFAULT_DATASET)
    # All source reads are test-only. Production runtime reads only its bundle.
    temporary = tempfile.TemporaryDirectory(prefix="starvla-alignment-data-")
    shadow = Path(temporary.name) / "dexbench_ee58_v21"
    (shadow / "meta").mkdir(parents=True)
    import shutil
    for file in (spec["root"] / "meta").iterdir():
        if file.is_file():
            shutil.copy2(file, shadow / "meta" / file.name)
    prepare_native_metadata(shadow, spec["stats"], spec["episodes"])
    for name in ("data", "videos"):
        (shadow / name).symlink_to(spec["root"] / name, target_is_directory=True)
    data_config = SparkArenaWujiConfig()
    dataset = OffsetLeRobotDataset(dataset_path=shadow, modality_configs=data_config.modality_config(),
        transforms=data_config.transform(), embodiment_tag=data_config.embodiment_tag,
        video_backend="torchvision_av", data_cfg=OmegaConf.create({"lerobot_version": "v2.0", "include_state": True,
            "state_action_dtype": "float16", "image_size": [224, 224], "action_mode": "abs"}))
    return spec, dataset, temporary


class GroundTruthNetwork:
    """Only network outputs are substituted; production Model.__init__ is real."""
    def __init__(self, normalized_actions, state, language):
        self.normalized_actions = normalized_actions
        self.expected_state = state
        self.expected_language = language
        self.examples = None

    def predict_action(self, examples):
        self.examples = examples
        for example in examples:
            np.testing.assert_array_equal(example["state"], self.expected_state)
            assert example["lang"] == self.expected_language
            assert all(image.mode == "RGB" and image.size == (224, 224) for image in example["image"])
        return {"normalized_actions": np.repeat(self.normalized_actions[None], len(examples), axis=0)}


def build_oracle_fixture(tmpdir):
    """Return (actual Model, decoded wire54 observation, expected action dicts).

    Root can pass this Model to the real WS server for plain/encoded/batch tests.
    The stub checks state/language and image contract, not JPEG pixel equality.
    """
    from XPolicyLab.policy.StarVLA.model import Model
    spec, dataset, _lifetime = native_fixture_data()
    episode, base = 2884, 7  # Packed video offset is nonzero, unlike episode0.
    source = original_wire(spec, episode)
    wire = np.asarray(source["observation.state"].iloc[base], np.float32)
    raw = dataset.get_step_data(episode, base)
    sample = dataset._pack_sample(dataset.transforms(copy.deepcopy(raw)))
    dims = get_robot_action_dim_info("tianji_marvin_wuji")
    obs = {"state": unpack_robot_state(wire, "ee", dims, source_type="obs")}
    obs["vision"] = {key: {"color": raw[f"video.{camera}"][0].copy()}
                     for key, camera in zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), ("front", "left_wrist", "right_wrist"))}
    obs["instruction"] = spec["episodes"][episode]["tasks"][0]
    run = Path(tmpdir) / "DexBench-tianji_wuji-tianji_marvin_wuji-ee-42"
    weight = run / "final_model/pytorch_model.pt"
    weight.parent.mkdir(parents=True)
    # Deliberately not a usable neural checkpoint; a real loader would reject it.
    weight.write_bytes(b"CPU ORACLE ONLY: no model weights")
    cfg = yaml.safe_load((POLICY / "configs/sparkarena_qwen_pi_v3.yaml").read_text())
    cfg.update(run_id=run.name, run_root_dir=str(run.parent), seed=42)
    cfg["framework"]["action_model"].update(state_dim=58, action_dim=58)
    cfg["datasets"]["vla_data"]["data_root_dir"] = "/deliberately_missing_training_dataset"
    (run / "config.full.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    runtime.save_sidecars(run, weight, cfg, spec["stats"], 20)
    stub = GroundTruthNetwork(sample["action"].astype(np.float32), sample["state"], obs["instruction"])
    with patch.object(runtime, "load_native", return_value=stub):
        model = Model({"bench_name": "DexBench", "ckpt_name": "tianji_wuji", "env_cfg_type": "tianji_marvin_wuji",
            "action_type": "ee", "seed": 42, "checkpoint_path": str(weight), "device": "cpu", "execute_horizon": 16})
    expected = [[unpack_robot_state(vector, "ee", dims, source_type="obs")
                 for vector in model.backend.codec.decode_actions(sample["action"])]]
    return model, obs, expected[0]


def check_worker_epochs():
    import torch
    from torch.utils.data import DataLoader, Dataset
    from omegaconf import OmegaConf
    from data_pipeline import install_dataset_pipeline
    spec, dataset, _lifetime = native_fixture_data()
    install_dataset_pipeline(19)  # Catch accidental use of the native default42.
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    mixture = get_vla_dataset(data_cfg=OmegaConf.create({"data_root_dir": str(dataset.dataset_path.parent),
        "data_mix": "xpolicylab_dexbench_ee58_qwen_pi_v3", "lerobot_version": "v2.0",
        "action_mode": "abs", "include_state": True, "video_backend": "torchvision_av"}))
    assert mixture.seed == 19 and mixture.mode == "train"

    class SampleKeys(Dataset):
        def __len__(self):
            return 128
        def __getitem__(self, index):
            _, episode, frame = mixture.sample_step(index)
            return np.asarray([mixture.epoch, episode, frame], dtype=np.int64)

    loader = DataLoader(SampleKeys(), batch_size=4, num_workers=2, persistent_workers=True)
    mixture.set_epoch(0)
    first = torch.cat(list(loader)).numpy()
    mixture.set_epoch(1)
    second = torch.cat(list(loader)).numpy()
    assert (first[:, 0] == 0).all() and (second[:, 0] == 1).all()
    changed = int(np.any(first[:, 1:] != second[:, 1:], axis=1).sum())
    assert changed > 0
    del loader
    return {"seed": 19, "persistent_workers": 2, "draws_checked": 128,
        "worker_epoch0_and1": "all observed correctly", "changed_draws": changed,
        "mode": "native uniform-episode replacement, not shuffled no-replacement"}


def check(tmpdir):
    model, obs, expected = build_oracle_fixture(tmpdir)
    spec, dataset, _lifetime = native_fixture_data()
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
    records = []
    for task, ids in spec["physical_task_groups"].items():
        for episode in (ids[0], ids[-1]):
            source = original_wire(spec, episode)
            base = min(7, len(source) - 1)
            wire = np.asarray(source["observation.state"].iloc[base], np.float32)
            raw = dataset.get_step_data(episode, base)
            sample = dataset._pack_sample(dataset.transforms(copy.deepcopy(raw)))
            state = model.backend.codec.normalize(pack_ee54(wire[None]), "state", packed=True)
            np.testing.assert_array_equal(state, sample["state"])
            train_bins = Qwen_PI_v3.state2str_transform(None, sample["state"][0])
            runtime_bins = Qwen_PI_v3.state2str_transform(None, state[0])
            assert train_bins == runtime_bins
            for camera in ("front", "left_wrist", "right_wrist"):
                im = runtime.rgb_image({"vision": {camera: {"color": raw[f"video.{camera}"][0]}}}, (camera,))
                expected_pixel = sample["image"][("front", "left_wrist", "right_wrist").index(camera)]
                np.testing.assert_array_equal(np.asarray(im), np.asarray(expected_pixel))
            records.append({"task": task, "episode": episode, "state_and_bin_exact": True, "pixels_exact": True})
    model.update_obs(obs)
    result = model.get_action()
    for a, b in zip(result, expected):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    for batch in (1, 2):
        model.update_obs_batch([dict(obs, env_idx=i) for i in range(batch)])
        returned = model.get_action_batch(list(range(batch)))
        assert len(returned) == batch and all(len(chunk) == 16 for chunk in returned)
        for chunk in returned:
            for action, oracle in zip(chunk, expected):
                for key in action:
                    np.testing.assert_array_equal(action[key], oracle[key])
    # A changed metadata/checkpoint receipt must stop before the network loader.
    stats = model.backend.weight.parent.parent / "ee58_statistics.json"
    original = stats.read_bytes()
    stats.write_bytes(original + b" ")
    try:
        runtime.read_bundle(model.backend.weight)
    except ValueError:
        tamper_guard = True
    else:
        raise AssertionError("Statistics tampering was accepted")
    stats.write_bytes(original)
    return {"status": "passed", "source_frames": records, "frames_checked": len(records),
        "state_bins": "exact, native q99->float16->native state2str", "pixels": "exact native PIL resize",
        "wire54_native58": "actual Model init/update/get_action; GT normalized actions inverse->SO3->wxyz54",
        "single_and_batch": [1, 2], "statistics_tamper_rejected": tamper_guard,
        "training_dataset_required_at_runtime": False, "GPU_or_large_model_forward": False,
        "network": "explicit GT oracle stub only", "seed_and_epoch_workers": check_worker_epochs()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="starvla-serving-oracle-") as temp:
        report = check(temp)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "source_frames"}, indent=2))
