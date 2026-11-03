"""Read-only CPU data and pretrained-shape audit; never starts GPU training."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml

from ee58_data import EE58Backend, POLICY, sha256, validate_language
from native_assets import native_paths

PRETRAIN = "/personal/xiangpc/0812_Xpolicylab_bench/H_RDT/pretrain_model/hrdt_pretrain_500k/pytorch_model.bin"
VISION_ROOT = "/personal/xiangpc/training_hrdt_sparkarena_samplerfix_0921/H_RDT/bak/dino-siglip"


def pretrained_shapes(config, checkpoint):
    import torch
    from unittest.mock import patch
    native_paths()
    from models.hrdt_runner import HRDTRunner
    state = torch.load(checkpoint, map_location="meta", mmap=True, weights_only=True)
    state = state.get("state_dict", state.get("model", state))
    # Distribution argument validation calls Tensor.item(); keep only these two
    # scalar sampler bounds on CPU while all billion-parameter model tensors use meta.
    uniform = lambda low, high: torch.distributions.Uniform(torch.tensor(low, device="cpu"), torch.tensor(high, device="cpu"))
    with torch.device("meta"), patch("models.hrdt_runner.Uniform", uniform):
        model = HRDTRunner(state_dim=58, action_dim=58, pred_horizon=16, config=config["model"],
                           act_pos_emb_config=[("state", 1), ("action", 16)],
                           img_pos_emb_config=[("image", (1, 3, -729))],
                           lang_pos_emb_config=[("language", -1024)],
                           max_img_len=2187, max_lang_len=1024,
                           training_mode="lang", mode="pretrain", dtype=torch.float32)
    current = model.state_dict()
    loaded, skipped, mismatches, unexpected, ignored_extra = [], [], [], [], []
    for key, value in state.items():
        if any(part in key for part in ("st_adaptor", "act_adaptor", "action_encoder", "final_layer", "action_decoder")):
            skipped.append(key)
        elif key == "model.img_pos_emb" and value.shape != current[key].shape:
            skipped.append(key)
        elif key not in current:
            # The human checkpoint also stores an unused video adapter. Native
            # strict=False loading ignores exactly these four extra tensors.
            if key in {"video_adapter.0.weight", "video_adapter.0.bias", "video_adapter.2.weight", "video_adapter.2.bias"}:
                ignored_extra.append(key)
            else:
                unexpected.append(key)
        elif value.shape != current[key].shape:
            mismatches.append([key, list(value.shape), list(current[key].shape)])
        else:
            loaded.append(key)
    if mismatches or unexpected or not loaded:
        raise ValueError(f"Native backbone incompatibility: mismatches={mismatches}; unexpected={unexpected}")
    missing = [key for key in current if key not in loaded]
    if any(not any(part in key for part in ("action_encoder", "action_decoder", "model.img_pos_emb")) for key in missing):
        raise ValueError(f"Unexpected missing backbone keys: {missing}")
    return {"method": "official runner allocated on meta + pretrained checkpoint header tensors on meta",
            "state_dim": 58, "action_dim": 58, "padding_dim": 0,
            "loaded_backbone_keys": len(loaded), "fresh_interface_keys": skipped,
            "shape_mismatches": mismatches, "unexpected_keys": unexpected, "known_unused_checkpoint_keys": ignored_extra,
            "gpu_used": False, "weights_not_materialized": True}


def main():
    parser = argparse.ArgumentParser()
    for name in ("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed", "gpu_ids"):
        parser.add_argument(name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.bench_name != "DexBench" or args.env_cfg_type != "tianji_marvin_wuji" or args.action_type != "ee":
        raise ValueError("This entry is DexBench/tianji_marvin_wuji/ee58 only")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.ckpt_name) or args.ckpt_name in (".", ".."):
        raise ValueError("Use a simple checkpoint name")
    if re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(size)?[_-]?\d+)", args.ckpt_name):
        raise ValueError("New checkpoint names must not contain batch-size labels; use tianji_wuji")
    gpu_ids = args.gpu_ids.split(",")
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or not all(key.isdigit() for key in gpu_ids):
        raise ValueError("This recipe requires 8 distinct GPU IDs")
    if int(args.seed) < 0:
        raise ValueError("Seed must be nonnegative")
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    run_id = build_run_dir_name(vars(args))
    output = POLICY / "checkpoints" / run_id
    if output.exists():
        raise FileExistsError(f"Preserve existing output; choose another ckpt_name: {output}")
    generated = POLICY / "training/generated"
    generated.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=run_id+"_", dir=generated))
    backend = EE58Backend(os.environ.get("HRDT_EE58_DATASET", "/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21"), require_language=False)
    spec = backend.spec
    config = yaml.safe_load((POLICY / "configs/hrdt_dexbench_ee58.yaml").read_text())
    pretrain = Path(os.environ.get("HRDT_PRETRAIN", PRETRAIN)).resolve(strict=True)
    vision = Path(os.environ.get("HRDT_VISION_ROOT", VISION_ROOT)).resolve(strict=True)
    language = Path(os.environ.get("HRDT_LANG_CACHE", str(POLICY / "data/dexbench_ee58_t5_60")))
    assets = [vision / "vit_large_patch14_reg4_dinov2.lvd142m/pytorch_model.bin",
              vision / "vit_so400m_patch14_siglip_384/open_clip_pytorch_model.bin"]
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(f"Local vision asset missing (no download is attempted): {path}")
    rows = []
    for task, episodes in spec["physical_task_groups"].items():
        for episode in (episodes[0], episodes[-1]):
            step = min(7, int(backend.lengths[episode])-1)
            state, action, task_index = backend.raw_step(episode, step)
            from ee58_data import normalize
            normalized_state = normalize(state, spec["stats"]["observation.state"])
            normalized_action = normalize(action, spec["stats"]["action"])
            if state.shape != (1, 58) or action.shape != (16, 58):
                raise ValueError("Native state/action shape mismatch")
            if not np.isfinite(normalized_state).all() or not np.isfinite(normalized_action).all():
                raise ValueError("Non-finite normalized sample")
            images, decoded = backend.rgb_frames(episode, step)
            if images.shape[0:2] != (3, 1) or images.shape[-1] != 3:
                raise ValueError("Expected three RGB camera histories")
            tail = backend.raw_step(episode, int(backend.lengths[episode])-1)[1]
            np.testing.assert_array_equal(tail, np.repeat(tail[:1], 16, axis=0))
            rows.append({"physical_task": task, "episode": episode, "task_index": task_index,
                         "local_step": step, "state_shape": list(state.shape), "action_shape": list(action.shape),
                         "valid_action_dimensions": 58, "decoded_camera_offsets": decoded})
    language_status = "passed"
    try:
        validate_language(language, spec)
    except (FileNotFoundError, ValueError) as error:
        language_status = f"pending: {error}"
    (workspace / "data_samples.json").write_text(json.dumps(rows, indent=2)+"\n")
    log_path = workspace / "pretrained_header.log"
    with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
        compatibility = pretrained_shapes(config, pretrain)
    config["ee58_contract"] = {
        "dataset_root": str(spec["root"]), "language_cache": str(language), "pretrained_backbone": str(pretrain),
        "vision_root": str(vision), "source_contract_id": spec["contract"]["id"],
        "state_action_dim": 58, "rotation_6d": "matrix_rows_0_then_1", "camera_order": list(spec["offsets"]["0"]),
        "metadata_sha256": {name: sha256(spec["root"] / "meta" / name) for name in
                            ("info.json", "action_contract.json", "stats.json", "tasks.jsonl", "episodes.jsonl", "video_offsets.json", "modality.json")},
        "pretrain_header": compatibility, "seed": int(args.seed), "run_id": run_id, "output_dir": str(output),
        "upstream_commit": subprocess.check_output(["git", "-C", str(POLICY / "upstream"), "rev-parse", "HEAD"], text=True).strip(),
        "optimizer_updates": 60000, "global_batch_size": 128, "micro_batch": 16, "gradient_accumulation": 1,
    }
    report = {"data_status": "passed", "sampled_episodes": len(rows), "samples": rows,
              "episodes": 3000, "frames": 751348, "physical_tasks": 12, "instruction_variants": 60,
              "state_action_dim": 58, "compact_no_padding": True, "normalization": "separate state/action q01/q99 clip to [-1,1]",
              "pretrained_header_status": "passed", "language_cache": language_status,
              "optimizer_updates": 60000, "global_batch_size": 128, "micro_batch_size": 16,
              "gradient_accumulation_steps": 1,
              "training_ready": language_status == "passed", "GPU_forward_backward": "not run", "training_started": False}
    (workspace / "preflight.json").write_text(json.dumps(report, indent=2)+"\n")
    path = workspace / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"H_RDT EE58 preflight: {workspace / 'preflight.json'}; language={language_status}", file=sys.stderr)
    if not args.dry_run:
        if language_status != "passed":
            raise RuntimeError("Training not launched: generate and verify the full 60-instruction T5 cache first")
        output.parent.mkdir(exist_ok=True)
        output.mkdir()
        from runtime_bundle import write_sidecars
        write_sidecars(output, config, spec, language)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        import shutil
        shutil.copy2(workspace / "preflight.json", output / "preflight.json")
    print(path)


if __name__ == "__main__":
    main()
