from collections import Counter
import hashlib
import json
from math import ceil
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    """One deterministic global stream; Accelerate shards its batches across ranks.

    ``frame_uniform`` preserves the original shuffled, without-replacement stream.
    ``task_uniform`` samples with replacement with inverse task-frequency weights.
    Only the selected training split contributes weights. Balanced epochs are padded
    to whole global batches so resume also reproduces the last batch on every rank.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int,
                 sampling_mode: str = "frame_uniform", task_ids=None):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in {"frame_uniform", "task_uniform"}:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self.num_samples = len(dataset)
        self.weights = None
        self.task_counts = None
        self.task_fingerprint = None
        if self.sampling_mode == "task_uniform":
            if self.batch_size <= 0 or self.num_processes <= 0 or len(dataset) == 0:
                raise ValueError("Task-balanced sampling requires a nonempty dataset and positive batch/world sizes.")
            if task_ids is None or len(task_ids) != len(dataset):
                raise ValueError("task_ids must contain one canonical task name per training sample.")
            if any(not isinstance(task, str) or not task for task in task_ids):
                raise ValueError("Task names must be nonempty strings.")
            counts = Counter(task_ids)
            self.task_counts = dict(sorted(counts.items()))
            names = list(self.task_counts)
            name_to_id = {name: index for index, name in enumerate(names)}
            labels = torch.tensor([name_to_id[name] for name in task_ids], dtype=torch.int64)
            task_weights = torch.tensor([1.0 / counts[name] for name in names], dtype=torch.float64)
            self.weights = task_weights[labels]
            digest = hashlib.sha256(json.dumps(names, ensure_ascii=True).encode("utf-8"))
            digest.update(labels.numpy().tobytes())
            self.task_fingerprint = digest.hexdigest()
            global_batch = self.batch_size * self.num_processes
            self.num_samples = ceil(len(dataset) / global_batch) * global_batch

    def sampling_state_dict(self):
        """Sampling contract, not the prefetched iterator position (trainer owns it)."""
        return {
            "mode": self.sampling_mode,
            "seed": self.seed,
            "dataset_length": len(self.dataset),
            "batch_size": self.batch_size,
            "num_processes": self.num_processes,
            "task_fingerprint": self.task_fingerprint,
        }

    def validate_resume_state(self, state):
        # Historical checkpoints did not record this contract and used frame_uniform.
        if state is None and self.sampling_mode == "frame_uniform":
            return
        if state != self.sampling_state_dict():
            raise ValueError(
                "Training-state sampling contract differs from this run. "
                "Resume with the original sampling mode/data/batch/world size, or "
                "load a weights .pt checkpoint into a new run to change sampling."
            )

    def normalize_resume_position(self, epoch: int, batch_in_epoch: int):
        if epoch < 0 or batch_in_epoch < 0:
            raise ValueError("Resume epoch and batch offset must be nonnegative.")
        if self.sampling_mode == "task_uniform":
            batches_per_epoch = self.num_samples // (self.batch_size * self.num_processes)
            completed, batch_in_epoch = divmod(batch_in_epoch, batches_per_epoch)
            epoch += completed
        return epoch, batch_in_epoch

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        if self.weights is None:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        if self.sampling_mode == "task_uniform" and self.epoch == 0:
            return max(0, self.num_samples - self.resume_batch_offset * self.batch_size * self.num_processes)
        return self.num_samples
