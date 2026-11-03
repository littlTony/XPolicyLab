"""Launch the official H-RDT trainer with the independent compact EE58 backend."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

from ee58_data import POLICY, sha256, validate_language
from native_assets import native_paths, build_vision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    contract = config["ee58_contract"]
    if config["training_profile"] != {"ckpt_name": "tianji_wuji", "optimizer_updates": 60000, "global_batch_size": 128,
                                       "num_processes": 8, "micro_batch_size": 16, "gradient_accumulation_steps": 1}:
        raise ValueError("Training profile differs from the reviewed 60000-update global128 recipe")
    if contract["optimizer_updates"] != 60000 or contract["global_batch_size"] != 128 or contract["micro_batch"] != 16 or contract["gradient_accumulation"] != 1:
        raise ValueError("Expected 60000 optimizer updates, global128 = 8 ranks x micro16 x GA1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("Expected 8 ranks; use train_ee58.sh")
    if config["common"]["state_dim"] != 58 or config["common"]["action_dim"] != 58 or config["model"]["hrdt"]["output_size"] != 58:
        raise ValueError("Compact58 input/output configuration required")
    if contract["rotation_6d"] != "matrix_rows_0_then_1":
        raise ValueError("This dataset uses row6D, not the human48 column6D decoder")
    root = Path(contract["dataset_root"])
    for name, digest in contract["metadata_sha256"].items():
        if sha256(root / "meta" / name) != digest:
            raise ValueError(f"Dataset metadata changed after preflight: {name}")
    from XPolicyLab.utils.dexbench_ee58_contract import validate_dataset
    validate_language(contract["language_cache"], validate_dataset(root))
    from runtime_bundle import validate_bundle
    bundle = validate_bundle(Path(contract["output_dir"]))
    if sha256(args.config) != bundle["contract"]["launch_config_sha256"]:
        raise ValueError("Generated config differs from checkpoint-bound launch snapshot")
    upstream = native_paths()
    os.chdir(upstream)
    from datasets.dataset import VLAConsumerDataset
    from ee58_data import consumer_class
    import train.train as native_train
    native_train.VLAConsumerDataset = consumer_class(VLAConsumerDataset)
    # Every full saved model is cryptographically bound to its immutable run
    # snapshots. A partially written save cannot be selected by serving.
    from models.hrdt_runner import HRDTRunner
    from runtime_bundle import write_weight_binding
    native_save = HRDTRunner.save_pretrained

    def save_bound(model, directory, *values, **options):
        result = native_save(model, directory, *values, **options)
        if int(os.environ.get("RANK", "0")) == 0:
            write_weight_binding(directory, Path(contract["output_dir"]))
        return result

    HRDTRunner.save_pretrained = save_bound
    # With GA1 the native loop already calls scheduler.step once per update.
    # Avoid AcceleratedScheduler advancing the underlying schedule once per rank.
    native_accelerator = native_train.Accelerator

    def accelerator(*values, **options):
        options["step_scheduler_with_optimizer"] = False
        instance = native_accelerator(*values, **options)
        if instance.num_processes != 8 or instance.gradient_accumulation_steps != 1:
            raise RuntimeError("Actual Accelerator must be 8 ranks / GA1 so micro16 means global128")
        return instance

    native_train.Accelerator = accelerator

    def vision(*values, **options):
        if options.get("image_resize_strategy") != "letterbox" or options.get("default_image_size") != 384:
            raise ValueError("Checkpoint image preprocessing must remain RGB384 letterbox")
        return build_vision(contract["vision_root"])

    native_train.DinoSigLIPViTBackbone = vision
    spec = importlib.util.spec_from_file_location("hrdt_ee58_native_main", upstream / "main.py")
    native_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_main)
    cli = [
        "--pretrained_vision_encoder_name_or_path=dino-siglip", "--deepspeed=configs/zero1.json",
        f"--config_path={args.config}", f"--output_dir={contract['output_dir']}",
        "--train_batch_size=16", "--sample_batch_size=8", "--gradient_accumulation_steps=1",
        "--max_train_steps=60000", "--checkpointing_period=10000", "--checkpoints_total_limit=6",
        "--lr_scheduler=constant_with_warmup", "--lr_warmup_steps=500", "--learning_rate=1e-4",
        "--mixed_precision=bf16", "--dataloader_num_workers=4", "--dataset_type=finetune",
        # Sampling is outside the update budget; native also supports -1 via a >0 guard.
        "--sample_period=60001", "--upsample_rate=1", "--precomp_lang_embed",
        "--training_mode=lang", "--mode=finetune", "--task_name=dexbench_12tasks_ee58",
        f"--seed={contract['seed']}", f"--pretrained_backbone_path={contract['pretrained_backbone']}",
    ]
    native_args = native_main.parse_args(cli)
    native_args.report_to = None
    if int(os.environ.get("RANK", "0")) == 0:
        (Path(contract["output_dir"]) / "native_args.json").write_text(json.dumps(vars(native_args), indent=2))
    from accelerate.logging import get_logger
    native_train.train(native_args, get_logger("hrdt_dexbench_ee58"))


if __name__ == "__main__":
    main()
