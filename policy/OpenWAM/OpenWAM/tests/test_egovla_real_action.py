"""Regression tests for EgoVLA raw-action/state/prompt/image alignment."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from openwam.dataloader.egovla import (
    EGOVLA_CONTRACT_ID,
    EgoVLADataset,
    POLICY_JOINT_INDICES,
)


def _stats(lo: float, hi: float) -> dict[str, np.ndarray]:
    lo_arr = np.full(38, lo, dtype=np.float32)
    hi_arr = np.full(38, hi, dtype=np.float32)
    return {
        "mean": (lo_arr + hi_arr) / 2,
        "std": np.full(38, max((hi - lo) / 4, 1e-6), dtype=np.float32),
        "min": lo_arr,
        "max": hi_arr,
        "q01": lo_arr,
        "q99": hi_arr,
    }


def _fixture(root: Path) -> tuple[Path, Path]:
    task = root / "Close-Drawer"
    task.mkdir(parents=True)
    episode = task / "episode_0.hdf5"
    qpos = np.zeros((4, 50), dtype=np.float32)
    action = np.zeros((4, 50), dtype=np.float32)
    for t in range(4):
        qpos[t, POLICY_JOINT_INDICES] = float(t + 1)
        action[t, POLICY_JOINT_INDICES] = float(100 + 10 * t)
    images = np.empty((4, 8, 8, 3), dtype=np.uint8)
    images[...] = np.asarray([250, 10, 20], dtype=np.uint8)  # RGB, deliberately not BGR-symmetric.
    with h5py.File(episode, "w") as handle:
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("action", data=action)
        handle.create_dataset("observations/images/main", data=images)

    stats_path = root / "normalization_stats.npy"
    np.save(
        stats_path,
        {
            "joint": _stats(100.0, 140.0),
            "joint_state": _stats(0.0, 10.0),
            "metadata": {
                "contract_id": EGOVLA_CONTRACT_ID,
                "action_source": "action",
                "forbidden_target": "observations/qpos[t+1]",
            },
        },
        allow_pickle=True,
    )
    return task, stats_path


def test_same_row_raw_action_separate_state_stats_rgb_and_prompt(tmp_path: Path):
    task, stats_path = _fixture(tmp_path)
    dataset = EgoVLADataset(
        data_root=task,
        dataset_root=tmp_path,
        task_name="Close-Drawer",
        num_frames=3,
        video_stride=1,
        height=32,
        width=32,
        normalize_mode="min-max",
        normalization_stats_path=stats_path,
        unify_action=True,
        unify_action_map=["0-37"],
        unify_state_map=["0-37"],
    )

    # start=1: qpos[1]=2 uses state stats [0,10] -> -0.6.
    # action[1]=110 uses action stats [100,140] -> -0.5. qpos[2] would be -0.4.
    sample = dataset._build_sample(0, 1)
    np.testing.assert_allclose(sample["proprio"][0, :38], -0.6, atol=1e-6)
    np.testing.assert_allclose(sample["action"][0, :38], -0.5, atol=1e-6)
    np.testing.assert_allclose(sample["action"][1, :38], 0.0, atol=1e-6)
    assert sample["action"].shape == (2, 80)
    assert sample["action_mask"][:, :38].all()
    assert not sample["action_mask"][:, 38:].any()

    # HWC bytes are interpreted as RGB in training, with no BGR swap.
    pixel = np.asarray(sample["video"][0], dtype=np.uint8)[0, 0]
    np.testing.assert_array_equal(pixel, [250, 10, 20])
    assert sample["video"][0].size == (32, 32)
    assert sample["prompt"] == (
        "A video recorded from a robot's point of view executing the following instruction: "
        "Close the opened drawer"
    )


def test_tail_padding_is_masked_and_eef_or_multiview_fail_closed(tmp_path: Path):
    task, stats_path = _fixture(tmp_path)
    dataset = EgoVLADataset(
        data_root=task,
        dataset_root=tmp_path,
        task_name="Close-Drawer",
        num_frames=3,
        video_stride=1,
        normalize_mode="min-max",
        normalization_stats_path=stats_path,
    )
    tail = dataset._build_sample(0, 2)
    assert tail["action_mask"][0, :38].all()
    assert not tail["action_mask"][1].any()
    np.testing.assert_allclose(tail["action"][0], tail["action"][1])

    with pytest.raises(ValueError, match="joint.*only"):
        EgoVLADataset(
            data_root=task,
            dataset_root=tmp_path,
            task_name="Close-Drawer",
            action_mode="eef",
            normalize_mode=None,
        )
    with pytest.raises(ValueError, match="multiview=false"):
        EgoVLADataset(
            data_root=task,
            dataset_root=tmp_path,
            task_name="Close-Drawer",
            multiview=True,
            normalize_mode=None,
        )
