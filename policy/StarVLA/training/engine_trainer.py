"""Coordinate native DeepSpeed GA without Accelerate.no_sync on ZeRO-2.

The successful SparkArena source snapshot carries the same engine-owned GA fix.
Keep it in the adapter so the official clone remains an unchanged reference.
"""
import math
import time
import json
from pathlib import Path

import torch
from tqdm import tqdm


def engine_step(engine, batch, scheduler):
    """DeepSpeed owns gradient averaging, clipping, optimizer and zero_grad."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        loss = engine.forward(batch)["action_loss"]
    if not math.isfinite(loss.detach().float().item()):
        raise FloatingPointError("Non-finite action FM loss; preserve the run for diagnosis")
    engine.backward(loss)
    boundary = bool(engine.is_gradient_accumulation_boundary())
    engine.step()
    if boundary:
        scheduler.step()
    return {"action_dit_loss": loss.detach().float().item(), "_optimizer_step": boundary}


def make_trainer(native):
    class SparkArenaTrainer(native.VLATrainer):
        def _plain_config(self):
            from omegaconf import OmegaConf
            cfg = self.config.unwrap() if hasattr(self.config, "unwrap") else self.config
            return OmegaConf.to_container(cfg, resolve=True)

        def _write_sidecars(self, weight):
            if self.accelerator.is_main_process:
                import sys
                policy = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(policy.parents[2]))
                from XPolicyLab.policy.StarVLA.ee58_runtime import save_sidecars, sha256
                cfg = self._plain_config()
                data = Path(cfg["datasets"]["vla_data"]["data_root_dir"]) / cfg["sparkarena_contract"]["dataset_name"]
                stats_path = data / "meta/stats_gr00t.json"
                expected = cfg["sparkarena_contract"]["metadata_sha256"]["meta/stats_gr00t.json"]
                if sha256(stats_path) != expected:
                    raise RuntimeError("Private absolute statistics changed after CPU preflight; refusing checkpoint sidecars")
                stats = json.loads(stats_path.read_text())["statistics"]
                save_sidecars(self.config.output_dir, weight, cfg, stats, self.completed_steps)
            self.accelerator.wait_for_everyone()

        def _save_checkpoint(self):
            super()._save_checkpoint()
            self._write_sidecars(Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_pytorch_model.pt")

        def _finalize_training(self):
            super()._finalize_training()
            self._write_sidecars(Path(self.config.output_dir) / "final_model/pytorch_model.pt")

        def _set_data_epoch(self, epoch):
            loader = self.vla_train_dataloader
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)

        def _create_data_iterators(self):
            self.vla_epoch_count = 0
            self._set_data_epoch(0)
            self.vla_iter = iter(self.vla_train_dataloader)

        def _get_next_batch(self):
            try:
                return next(self.vla_iter)
            except StopIteration:
                self.vla_epoch_count += 1
                self._set_data_epoch(self.vla_epoch_count)
                self.vla_iter = iter(self.vla_train_dataloader)
                return next(self.vla_iter)

        def prepare_training(self):
            # Called before wrapping in DeepSpeed. Qwen does not read the YAML flag.
            vlm = self.model.qwen_vl_interface.model
            # Fresh upstream loads SDPA unconditionally. Restore the successful
            # SparkArena FA2 backend through the supported Transformers API.
            vlm.set_attn_implementation("flash_attention_2")
            for part in (vlm.config, vlm.config.text_config, vlm.config.vision_config):
                if part._attn_implementation != "flash_attention_2":
                    raise RuntimeError("Qwen language/vision attention did not become FA2")
            vlm.config.use_cache = False
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(vlm, "enable_input_require_grads"):
                vlm.enable_input_require_grads()
            if not vlm.is_gradient_checkpointing:
                raise RuntimeError("Qwen gradient checkpointing did not become active")
            super().prepare_training()
            if not all(hasattr(self.model, name) for name in
                       ("backward", "step", "is_gradient_accumulation_boundary", "gradient_accumulation_steps")):
                raise RuntimeError("This recipe requires a prepared DeepSpeed engine")
            expected_ga = self.config.trainer.gradient_accumulation_steps
            if self.model.gradient_accumulation_steps() != expected_ga:
                raise RuntimeError(f"DeepSpeed engine GA is not {expected_ga}")
            expected_batch = self.config.sparkarena_contract.effective_batch
            if self.total_batch_size != expected_batch:
                raise RuntimeError(f"Unexpected effective batch: {self.total_batch_size}")
            if self.model.train_micro_batch_size_per_gpu() != 4 or self.model.train_batch_size() != expected_batch:
                raise RuntimeError("DeepSpeed actual micro/global batch metadata does not match the recipe")
            if self.model.gradient_clipping() != 1.0:
                raise RuntimeError("DeepSpeed must clip once per update at norm 1.0")
            native.accelerator.print(f"DexBench: Qwen checkpointing active; ZeRO2 engine owns GA{expected_ga}; effective batch={expected_batch}")

        def _train_step(self, batch_vla, batch_vlm=None):
            return engine_step(self.model, batch_vla, self.lr_scheduler)

        def train(self):
            self._log_training_config()
            self._create_data_iterators()
            progress = tqdm(total=self.config.trainer.max_train_steps, initial=self.completed_steps,
                            disable=not self.accelerator.is_local_main_process)
            micro_count = 0
            losses = []
            data_seconds = model_seconds = 0.0
            while self.completed_steps < self.config.trainer.max_train_steps:
                start = time.perf_counter()
                batch = self._get_next_batch()
                after_data = time.perf_counter()
                metrics = self._train_step(batch)
                after_model = time.perf_counter()
                data_seconds += after_data - start
                model_seconds += after_model - after_data
                micro_count += 1
                losses.append(metrics["action_dit_loss"])
                if not metrics.pop("_optimizer_step"):
                    continue
                expected_ga = self.config.trainer.gradient_accumulation_steps
                if micro_count != expected_ga:
                    raise RuntimeError(f"Expected {expected_ga} microbatches per optimizer update, got {micro_count}")
                self.completed_steps += 1
                progress.update(1)
                metrics["action_dit_loss"] = sum(losses) / len(losses)
                metrics["timing/data"] = data_seconds
                metrics["timing/model"] = model_seconds
                if self.accelerator.is_local_main_process:
                    progress.set_postfix(step_seconds=f"{data_seconds + model_seconds:.3f}")
                # Eval, logging, save, LR and step counters run only on optimizer boundaries.
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    metrics = self.eval_action_model(metrics)
                self._log_metrics(metrics)
                if self.completed_steps % self.config.trainer.save_interval == 0:
                    self._save_checkpoint()
                micro_count = 0
                losses.clear()
                data_seconds = model_seconds = 0.0
            progress.close()
            self._finalize_training()

    return SparkArenaTrainer
