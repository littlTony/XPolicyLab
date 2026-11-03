"""CPU-only sampler/Accelerate contract tests; no policy checkpoint is loaded."""
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate.data_loader import prepare_data_loader
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FastWAM" / "src"))
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.trainer import Wan22Trainer
from fastwam.utils.samplers import ResumableEpochSampler


class IndexDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks
        self.skip_padding_as_possible = False
    def __len__(self):
        return len(self.tasks)
    def __getitem__(self, index):
        return index
    def get_sample_task_ids(self):
        return self.tasks


def sampler(dataset, mode="task_uniform", world=8):
    return ResumableEpochSampler(dataset, 42, 4, world, mode,
                                 dataset.tasks if mode == "task_uniform" else None)


def sharded_loader(dataset, selected_sampler, rank, world=8):
    loader = DataLoader(dataset, batch_size=4, sampler=selected_sampler)
    return prepare_data_loader(loader, device=torch.device("cpu"), num_processes=world,
                               process_index=rank, split_batches=False, put_on_device=False,
                               rng_types=[])


def batches(loader):
    return [batch.tolist() for batch in loader]


def global_batches(selected):
    sequence = list(selected)
    return [sequence[i:i + 4] for i in range(0, len(sequence), 4)]


class SamplingTests(unittest.TestCase):
    def test_default_preserves_original_shuffle_and_legacy_resume(self):
        dataset = IndexDataset(["task"] * 103)
        selected = sampler(dataset, "frame_uniform")
        expected = torch.randperm(103, generator=torch.Generator().manual_seed(42)).tolist()
        self.assertEqual(list(selected), expected)
        selected.set_epoch_offset(3)
        selected.set_resume_batch_offset(2)
        expected = torch.randperm(103, generator=torch.Generator().manual_seed(45)).tolist()
        self.assertEqual(list(selected), expected[64:])
        self.assertEqual(len(selected), 103)
        selected.validate_resume_state(None)

    def test_task_probability_balanced_despite_frame_imbalance(self):
        dataset = IndexDataset(["long"] * 90000 + ["short"] * 10000)
        selected = sampler(dataset)
        counts = Counter(dataset.tasks[i] for i in selected)
        self.assertAlmostEqual(counts["long"] / sum(counts.values()), 0.5, delta=0.01)
        self.assertGreater(len(set(selected)), 1000)
        self.assertEqual(list(selected), list(sampler(dataset)))
        selected.set_epoch(1)
        self.assertNotEqual(list(selected), list(sampler(dataset)))

    def test_actual_accelerate_eight_rank_sizes_order_and_resume(self):
        dataset = IndexDataset(["long"] * 91 + ["short"] * 12)
        complete = sampler(dataset)
        complete.set_epoch(3)
        expected_batches = global_batches(complete)
        for rank in range(8):
            selected = sampler(dataset)
            selected.set_epoch_offset(3)
            loader = sharded_loader(dataset, selected, rank)
            observed = batches(loader)
            self.assertEqual(len(loader), 4)
            self.assertEqual(observed, expected_batches[rank::8])
            selected = sampler(dataset)
            selected.set_epoch_offset(3)
            selected.set_resume_batch_offset(2)
            loader = sharded_loader(dataset, selected, rank)
            self.assertEqual(len(loader), 2)
            self.assertEqual(batches(loader), observed[2:])
            selected.clear_resume_batch_offset()
            next_epoch = sampler(dataset)
            next_epoch.set_epoch(4)
            self.assertEqual(batches(loader), global_batches(next_epoch)[rank::8])
        self.assertEqual(complete.normalize_resume_position(3, 4), (4, 0))

    def test_selected_hf_row_order_and_cross_shard_task_names(self):
        def fail_decode(_):
            raise AssertionError("sample transform/decode was called")
        def shard(labels, names):
            rows = HFDataset.from_dict({"task_index": labels, "unused": [0] * len(labels)})
            rows.set_transform(fail_decode)
            return SimpleNamespace(hf_dataset=rows, meta=SimpleNamespace(tasks=names), root="test")
        dataset = RobotVideoDataset.__new__(RobotVideoDataset)
        selected = [shard([1, 1, 0], {0: "A", 1: "B"}), shard([0, 1], {0: "B", 1: "C"})]
        class SelectedBase:
            multi_dataset = SimpleNamespace(_datasets=selected)
            def __len__(self):
                return 5
        dataset.lerobot_dataset = SelectedBase()
        self.assertEqual(dataset.get_sample_task_ids(), ["B", "B", "A", "B", "C"])

    def test_trainer_loader_and_resume_contract(self):
        dataset = IndexDataset(["A"] * 91 + ["B"] * 12)
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.cfg = {"train_sampling": "task_uniform"}
        trainer.seed, trainer.batch_size, trainer.num_workers = 42, 4, 0
        loads = []
        trainer.accelerator = SimpleNamespace(num_processes=8, load_state=lambda **k: loads.append(k),
                                              wait_for_everyone=lambda: None)
        loader = trainer._build_loader(dataset)
        self.assertEqual(loader.sampler.sampling_mode, "task_uniform")
        trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 20, 3, 2
        with tempfile.TemporaryDirectory() as temp:
            trainer._save_trainer_state(temp)
            trainer.epoch, trainer.batch_in_epoch = 0, 0
            trainer.load_training_state(temp)
            self.assertEqual((trainer.epoch, trainer.batch_in_epoch), (3, 2))
            self.assertEqual(len(loads), 1)
            full = sampler(dataset)
            full.set_epoch(3)
            self.assertEqual(list(trainer.train_sampler), list(full)[64:])
            state = json.loads((Path(temp) / "trainer_state.json").read_text())
            state["sampling"]["task_fingerprint"] = "changed"
            (Path(temp) / "trainer_state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "sampling contract"):
                trainer.load_training_state(temp)
            self.assertEqual(len(loads), 1)
            state.pop("sampling")
            (Path(temp) / "trainer_state.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "sampling contract"):
                trainer.load_training_state(temp)
            self.assertEqual(len(loads), 1)
        trainer.cfg = {}
        loader = trainer._build_loader(dataset)
        self.assertEqual(loader.sampler.sampling_mode, "frame_uniform")

    def test_contract_validation_and_completed_epoch_resume(self):
        dataset = IndexDataset(["A"] * 91 + ["B"] * 12)
        selected = sampler(dataset)
        selected.validate_resume_state(selected.sampling_state_dict())
        with self.assertRaisesRegex(ValueError, "sampling contract"):
            selected.validate_resume_state(None)
        changed_world = sampler(dataset, world=4)
        with self.assertRaisesRegex(ValueError, "sampling contract"):
            changed_world.validate_resume_state(selected.sampling_state_dict())
        epoch, offset = selected.normalize_resume_position(3, 4)
        selected.set_epoch_offset(epoch)
        selected.set_resume_batch_offset(offset)
        loader = sharded_loader(dataset, selected, 0)
        expected = sampler(dataset)
        expected.set_epoch_offset(4)
        self.assertEqual(batches(loader), batches(sharded_loader(dataset, expected, 0)))
        with self.assertRaisesRegex(ValueError, "one canonical"):
            ResumableEpochSampler(dataset, 42, 4, 8, "task_uniform", ["A"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
