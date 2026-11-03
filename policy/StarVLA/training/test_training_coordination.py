"""CPU numeric contract test, not a substitute for a real 8-GPU DeepSpeed run."""
from copy import deepcopy
from types import SimpleNamespace

import torch

from engine_trainer import engine_step, make_trainer


class CpuEngine:
    """Small DeepSpeed-protocol stand-in backed by a real CPU AdamW optimizer."""
    def __init__(self, model, ga):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-8)
        self.micro = self.updates = self.clips = 0
        self.ga = ga

    def forward(self, batch):
        x, target = batch
        return {"action_loss": (self.model(x) - target).square().mean()}

    def backward(self, loss):
        (loss / self.ga).backward()

    def is_gradient_accumulation_boundary(self):
        return (self.micro + 1) % self.ga == 0

    def step(self):
        if self.is_gradient_accumulation_boundary():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.clips += 1
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.micro += 1


def check(ga):
    torch.set_num_threads(1)
    torch.manual_seed(917)
    reference = torch.nn.Linear(3, 2).double()
    engine = CpuEngine(deepcopy(reference), ga)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-8)
    schedule = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda step: 1 / (step + 1))
    ref_schedule = torch.optim.lr_scheduler.LambdaLR(reference_optimizer, lambda step: 1 / (step + 1))
    batches = [(torch.randn(32 * ga, 3, dtype=torch.float64), torch.randn(32 * ga, 2, dtype=torch.float64)) for _ in range(3)]
    for x, y in batches:
        before = deepcopy(engine.model.state_dict())
        for micro in range(ga):
            result = engine_step(engine, (x[micro * 32:(micro + 1) * 32], y[micro * 32:(micro + 1) * 32]), schedule)
            assert result["_optimizer_step"] == (micro == ga - 1)
            if micro < ga - 1:
                for name, value in engine.model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        reference_optimizer.zero_grad()
        (reference(x) - y).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        reference_optimizer.step()
        ref_schedule.step()
        for actual, expected in zip(engine.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    assert engine.micro == 3 * ga and engine.updates == engine.clips == schedule.last_epoch == 3
    assert all(int(state["step"]) == 3 for state in engine.optimizer.state.values())

    class Base:
        def _log_training_config(self): pass
        def _create_data_iterators(self): pass
        def _get_next_batch(self): return next(self.batches)
        def _log_metrics(self, metrics): self.logs += 1
        def eval_action_model(self, metrics): self.evals += 1; return metrics
        def _save_checkpoint(self): self.saves += 1
        def _finalize_training(self): self.finalized += 1

    trainer = make_trainer(SimpleNamespace(VLATrainer=Base))()
    trainer.model = CpuEngine(deepcopy(reference), ga)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.model.optimizer, lambda step: 1.0)
    trainer.config = SimpleNamespace(output_dir="/cpu-test-no-weights",
                                    trainer=SimpleNamespace(max_train_steps=3, eval_interval=1, save_interval=1,
                                                            gradient_accumulation_steps=ga))
    trainer.accelerator = SimpleNamespace(is_local_main_process=False)
    trainer.completed_steps = trainer.logs = trainer.evals = trainer.saves = trainer.finalized = 0
    trainer.batches = iter([(x[i * 32:(i + 1) * 32], y[i * 32:(i + 1) * 32])
                            for x, y in batches for i in range(ga)])
    class Loader:
        dataset = SimpleNamespace(set_epoch=lambda epoch: None)
        def __iter__(self): return trainer.batches
    trainer.vla_train_dataloader = Loader()
    trainer.checkpoint_dir = "/cpu-test-no-weights/checkpoints"
    trainer.sidecars = []
    trainer._write_sidecars = lambda path: trainer.sidecars.append(str(path))
    trainer.train()
    assert trainer.completed_steps == trainer.logs == trainer.evals == trainer.saves == 3
    assert trainer.finalized == 1 and trainer.model.micro == 3 * ga and trainer.model.updates == 3
    assert len(trainer.sidecars) == 4  # Three complete checkpoints plus final weights.
    return {"ga": ga, "global_batch_simulated": 32 * ga, "optimizer_updates": 3,
            "microbatches": trainer.model.micro, "clip_Adam_LR_calls": 3,
            "callbacks": "once per optimizer boundary", "numeric_reference": "matched full batch at rtol1e-11/atol1e-12"}


def main():
    import json
    print(json.dumps({"status": "passed", "GPU_used": False,
                      "cases": [check(ga) for ga in (1, 2)]}, indent=2))


if __name__ == "__main__":
    main()
