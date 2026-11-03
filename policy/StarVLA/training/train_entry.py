"""Register DexBench EE58 data and run QwenPI_v3 with the SparkArena recipe."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

import yaml

from prepare_sparkarena import POLICY, sha256, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config_yaml.read_text())
    validate_config(cfg)
    contract = cfg["sparkarena_contract"]
    for relative, digest in contract["adapter_sha256"].items():
        if sha256(POLICY / relative) != digest:
            raise RuntimeError(f"Adapter changed after CPU preflight: {relative}")
    private = Path(cfg["datasets"]["vla_data"]["data_root_dir"]) / contract["dataset_name"]
    for relative, digest in contract["metadata_sha256"].items():
        if sha256(private / relative) != digest:
            raise RuntimeError(f"Dataset metadata changed after CPU preflight: {relative}")
    expected_world = contract["num_processes"]
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world:
        raise RuntimeError(f"Expected {expected_world} ranks; launch via train_sparkarena.sh")
    expected_ga = cfg["trainer"]["gradient_accumulation_steps"]
    if os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS") != str(expected_ga):
        raise RuntimeError(f"Accelerate GA must match the validated training config ({expected_ga})")
    upstream = Path(contract["upstream_root"])
    sys.path.insert(0, str(upstream))
    from sparkarena_registry import register

    register(contract["dataset_name"])
    from data_pipeline import install_dataset_pipeline
    install_dataset_pipeline(cfg["seed"])
    os.chdir(upstream)
    from omegaconf import OmegaConf
    from engine_trainer import make_trainer
    spec = importlib.util.spec_from_file_location("sparkarena_native_trainer", upstream / "starVLA/training/train_starvla.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if native.accelerator.num_processes != expected_world or native.accelerator.gradient_accumulation_steps != expected_ga:
        raise RuntimeError("Accelerate world size / GA does not match the launch contract")
    native.VLATrainer = make_trainer(native)
    config = native.apply_config_compat(OmegaConf.create(cfg))
    config.config_yaml = str(args.config_yaml)
    native.main(config)


if __name__ == "__main__":
    main()
