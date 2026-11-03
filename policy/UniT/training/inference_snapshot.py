"""Bind every policy checkpoint to the exact serving configuration/statistics."""
from pathlib import Path
from transformers import TrainerCallback

from XPolicyLab.policy.UniT.ee58_runtime import write_inference_bundle


class InferenceSnapshot(TrainerCallback):
    def __init__(self, prepared):
        self.prepared = prepared

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_inference_bundle(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                                   self.prepared, state.global_step)


def save_final(output, prepared, step):
    write_inference_bundle(output, prepared, step)
