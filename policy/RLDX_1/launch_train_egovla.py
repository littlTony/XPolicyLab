#!/usr/bin/env python3
"""Launch RLDX training with isolated EgoVLA/ZeRO-3 compatibility fixes.

RLDX passes ``transformers.BatchFeature`` objects between model modules.
DeepSpeed 0.17.6 only walks concrete ``dict`` containers when installing its
ZeRO-3 backward hooks, so parameters can be released before a checkpointed
backward recomputation.  BatchFeature is a MutableMapping, not a dict.  Teach
DeepSpeed's tensor walker about such mappings before loading the unchanged
upstream launcher.
"""

from __future__ import annotations

import os
import pathlib
import runpy
from collections.abc import MutableMapping
from typing import Any, Callable


def install_zero3_mutable_mapping_patch() -> None:
    import deepspeed.runtime.zero.parameter_offload as parameter_offload
    import deepspeed.runtime.zero.stage3 as stage3
    import deepspeed.runtime.zero.utils as zero_utils

    current = zero_utils.apply_to_tensors_only
    if getattr(current, "_xpolicylab_mutable_mapping_patch", False):
        return

    original = current

    def apply_to_tensors_only(
        function: Callable[[Any], Any], value: Any, warning_msg_fn: Callable[[Any], str] | None = None
    ) -> Any:
        if isinstance(value, MutableMapping) and not isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = apply_to_tensors_only(function, value[key], warning_msg_fn)
            return value
        return original(function, value, warning_msg_fn)

    apply_to_tensors_only._xpolicylab_mutable_mapping_patch = True  # type: ignore[attr-defined]
    zero_utils.apply_to_tensors_only = apply_to_tensors_only
    parameter_offload.apply_to_tensors_only = apply_to_tensors_only
    stage3.apply_to_tensors_only = apply_to_tensors_only

    # Fail closed if the installed Transformers container is not traversed.
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    visited: list[tuple[int, ...]] = []
    probe = BatchFeature(data={"tensor": torch.zeros(2, 3)})
    parameter_offload.apply_to_tensors_only(lambda tensor: visited.append(tuple(tensor.shape)) or tensor, probe)
    if visited != [(2, 3)]:
        raise RuntimeError(f"ZeRO-3 MutableMapping patch self-test failed: {visited}")
    visited.clear()
    nested_probe = ({"nested": BatchFeature(data={"tensor": torch.zeros(4, 5)})},)
    parameter_offload.apply_to_tensors_only(
        lambda tensor: visited.append(tuple(tensor.shape)) or tensor, nested_probe
    )
    if visited != [(4, 5)]:
        raise RuntimeError(f"ZeRO-3 nested MutableMapping patch self-test failed: {visited}")
    if not (
        zero_utils.apply_to_tensors_only
        is parameter_offload.apply_to_tensors_only
        is stage3.apply_to_tensors_only
    ):
        raise RuntimeError("ZeRO-3 MutableMapping patch did not replace every imported tensor walker")


def install_resume_save_cadence_patch() -> None:
    """Keep checkpoint saves on the requested cadence when resuming.

    Transformers 4.57 rebuilds ``TrainingArguments`` from the CLI, then
    reloads ``trainer_state.json`` inside ``_inner_training_loop``.  The
    latter restores the old ``state.save_steps`` value, so merely passing a
    new ``--save-steps`` does not change the callback cadence on resume.
    This private callback runs at ``on_train_begin`` (after that reload) and
    restores the explicitly requested integer cadence.  It is opt-in via an
    environment variable so ordinary launches retain upstream behavior.
    """

    raw_target = os.environ.get("RLDX_SAVE_STEPS_OVERRIDE")
    if raw_target is None:
        return

    try:
        target = int(raw_target)
    except (TypeError, ValueError) as exc:
        raise ValueError("RLDX_SAVE_STEPS_OVERRIDE must be a positive integer") from exc
    if target <= 0:
        raise ValueError("RLDX_SAVE_STEPS_OVERRIDE must be a positive integer")

    from transformers import TrainerCallback
    from rldx.experiment.trainer import RLDXTrainer

    original_init = RLDXTrainer.__init__
    if getattr(original_init, "_xpolicylab_save_cadence_patch", False):
        return

    class _ResumeSaveCadenceCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            before = state.save_steps
            state.save_steps = target
            # Keep the two sources of truth aligned for callbacks that inspect
            # TrainingArguments rather than TrainerState.
            args.save_steps = target
            if getattr(state, "is_world_process_zero", True):
                print(
                    "[XPolicyLab] resume save cadence override: "
                    f"state.save_steps {before!r} -> {state.save_steps}; "
                    f"args.save_steps={args.save_steps}",
                    flush=True,
                )
            if state.save_steps != target or args.save_steps != target:
                raise RuntimeError("resume save cadence override self-check failed")

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(_ResumeSaveCadenceCallback())

    patched_init._xpolicylab_save_cadence_patch = True  # type: ignore[attr-defined]
    RLDXTrainer.__init__ = patched_init


def install_video_backend_override() -> None:
    """Allow machines without FFmpeg/torchcodec to select another backend."""
    backend = os.environ.get("RLDX_VIDEO_BACKEND")
    if not backend:
        return

    from rldx.experiment import assembly

    original = assembly._apply_training_overrides
    if getattr(original, "_xpolicylab_video_backend_patch", False):
        return

    def patched(cli, run_config, base_model_path):
        original(cli, run_config, base_model_path)
        run_config.data.video_backend = backend
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[XPolicyLab] video_backend override: {backend}", flush=True)

    patched._xpolicylab_video_backend_patch = True  # type: ignore[attr-defined]
    assembly._apply_training_overrides = patched


def main() -> None:
    install_zero3_mutable_mapping_patch()
    install_resume_save_cadence_patch()
    install_video_backend_override()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("[XPolicyLab] enabled ZeRO-3 tensor hooks for BatchFeature/MutableMapping", flush=True)

    upstream = pathlib.Path(__file__).resolve().parent / "RLDX-1" / "rldx" / "experiment" / "launch_train.py"
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
