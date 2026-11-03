"""CPU-only preflight for UniT EE58; no pretrained tensor loading or training."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

POLICY = Path(__file__).resolve().parent.parent
UPSTREAM = POLICY / "UniT"
sys.path.insert(0, str(POLICY.parents[2]))
sys.path.insert(0, str(UPSTREAM))
from XPolicyLab.utils.dexbench_ee58_contract import DEFAULT_DATASET, CONTRACT_ID, validate_dataset
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_headers(path):
    headers = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with file.open("rb") as fp:
            size = struct.unpack("<Q", fp.read(8))[0]
            entries = json.loads(fp.read(size))
        last = max((value["data_offsets"][1] for key, value in entries.items() if key != "__metadata__"), default=0)
        require(file.stat().st_size >= 8 + size + last, f"Truncated safetensors shard: {file}")
        headers.update({key: value["shape"] for key, value in entries.items() if key != "__metadata__"})
    require(headers, f"No safetensors weights in {path}")
    return headers


def inspect_tokenizer_shape(path):
    cfg = json.loads((Path(path) / "config.json").read_text())
    require(cfg["action_dim"] == cfg["state_dim"] == 128 and cfg["action_horizon"] == 16,
            "Tokenizer shape must remain padded128/H16")
    headers = checkpoint_headers(path)
    checks = {
        "action_branch.action_conv_encoder.stem_conv.W": [32, 1024, 128, 5],
        "action_branch.state_encoder.layer1.W": [32, 128, 1024],
        "action_decoder.resnet_decoder.upsample_layer.proj_conv.W": [32, 128, 1024, 5],
        "vq.layers.0.embedding.weight": [128, 32],
        "vq.layers.1.embedding.weight": [128, 32],
    }
    for key, shape in checks.items():
        require(headers.get(key) == shape, f"Unexpected checkpoint tensor {key}: {headers.get(key)}")
    return checks


def native_probe(prepared, validated):
    import numpy as np
    import pandas as pd
    import torch
    from ee58_data import EE58DataConfig, EE58Dataset, PARTS, CAMERAS
    from runtime_compat import encode_views, FullBatchSampler
    from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer_Config
    from gr00t.model.gr00t_n1_unit import GR00T_N1_5_UniT_Config

    saved_contract = {}
    for stage, config_class in (("tokenizer", GR00T_Tokenizer_Config), ("policy", GR00T_N1_5_UniT_Config)):
        model_dict = json.loads((Path(prepared["workspace"]) / f"{stage}_config.json").read_text())
        # Config-only: no DINO/Qwen model or pretrained tensor is constructed.
        roundtrip = config_class(**model_dict).to_dict()
        require(roundtrip["action_dim"] == 128 and roundtrip["ee58_contract"]["real_dim"] == 58,
                "HF config serialization lost the EE58 contract")
        if stage == "tokenizer":
            require(roundtrip["state_dim"] == 128, "Tokenizer model state width changed")
        else:
            require(roundtrip["action_head_cfg"]["max_state_dim"] == 128
                    and roundtrip["unit_cfg"]["num_bridge_tokens"] == 8, "Policy architecture changed")
        saved_contract[stage] = roundtrip["ee58_contract"]

    cfg = EE58DataConfig(eagle_path=prepared["vlm_path"], use_bridge=False, tokenizer_only=True)
    dataset = EE58Dataset(dataset_path=prepared["dataset_path"], modality_configs=cfg.modality_config(),
                          transforms=cfg.transform(augment=False), embodiment_tag="new_embodiment",
                          split="[:]", video_backend="torchvision_av")
    require(len(dataset.trajectory_ids) == 3000 and len(dataset) == 751348, "Native loader lost episodes/frames")
    stats = validated["stats"]
    samples = []
    for task, episodes in validated["physical_task_groups"].items():
        for episode_id in (episodes[0], episodes[-1]):
            episode_id = int(episode_id)
            parquet = Path(prepared["dataset_path"]) / validated["info"]["data_path"].format(
                episode_chunk=episode_id // 1000, episode_index=episode_id)
            trajectory = pd.read_parquet(parquet)
            base = min(7, len(trajectory) - 1)
            raw = dataset.get_step_data(episode_id, base)
            state = np.concatenate([raw[f"state.{part}"] for part in PARTS], axis=1)
            action = np.concatenate([raw[f"action.{part}"] for part in PARTS], axis=1)
            np.testing.assert_array_equal(state, np.stack(trajectory["observation.state"])[[base]])
            indices = np.minimum(np.arange(base, base + 16), len(trajectory) - 1)
            np.testing.assert_array_equal(action, np.stack(trajectory["action"])[indices])
            queries = {}
            for camera in CAMERAS:
                key = f"video.{camera}"
                require(raw[key].shape == (2, 480, 640, 3), f"Expected current+goal RGB frames: {raw[key].shape}")
                indices, offset, seconds = dataset.video_query(episode_id, key, base)
                np.testing.assert_array_equal(indices, [base, min(base + 16, len(trajectory) - 1)])
                queries[camera] = {"offset_frames": offset, "current_goal_seconds": seconds.tolist()}
            # Transform mutates its input; keep the raw oracle arrays above.
            sample = dataset.transforms(raw)
            for key, original, source_key in (("state", state, "observation.state"), ("action", action, "action")):
                packed = np.asarray(sample[key])
                mask = np.asarray(sample[f"{key}_mask"])
                require(packed.shape == mask.shape == (1 if key == "state" else 16, 128), "Bad padded shape")
                require(mask[:, :58].all() and not mask[:, 58:].any(), "Expected mask58 true / 70 false")
                require(np.count_nonzero(packed[:, 58:]) == 0, "Padding must be zero")
                mean = np.asarray(stats[source_key]["mean"], dtype=np.float32)
                std = np.asarray(stats[source_key]["std"], dtype=np.float32)
                expected = original.copy()
                expected[:, std != 0] = (original[:, std != 0] - mean[std != 0]) / std[std != 0]
                np.testing.assert_allclose(packed[:, :58], expected, atol=2e-6, rtol=2e-6)
                require(np.isfinite(packed).all(), "Non-finite normalization")
            require(sample["embodiment_id"] == 31, "Wrong new-embodiment slot")
            require(tuple(sample["imagenet_obs_images"].shape) == (3, 3, 224, 224), "Three obs views lost")
            require(tuple(sample["imagenet_goal_images"].shape) == (3, 3, 224, 224), "Three goal views lost")
            samples.append({"episode": episode_id, "task": task, "base": base, "queries": queries})

    # Exercise the exact runtime pooling helper with a tiny CPU vision stand-in.
    images = torch.stack([torch.full((3, 2, 2), float(view)) for view in (1, 2, 6)])[None]
    def vision_model(flat):
        return flat.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 4, 2)
    pooled = encode_views(vision_model, images, batch_size=1, hidden_size=2)
    torch.testing.assert_close(pooled, torch.full((1, 4, 2), 3.0))
    full_sampler = FullBatchSampler(range(751348))
    require(len(full_sampler) == 751296 and len(list(full_sampler)) % 64 == 0, "Sampler creates partial optimizer batches")
    # Policy-side processor only: tokenization/image grids, not the Qwen model.
    policy_data = EE58DataConfig(eagle_path=prepared["vlm_path"], use_bridge=True,
                                 ignore_lang_prefix=True, num_bridge_tokens=8, tokenizer_only=False)
    policy_transform = policy_data.transform(augment=False)
    policy_transform.set_metadata(dataset.metadata)
    policy_sample = policy_transform(dataset.get_step_data(episode_id, base))
    from gr00t.model.transforms import collate
    processor = policy_transform.transforms[-1].eagle_processor
    policy_batch = collate([policy_sample], processor)
    require(policy_batch["eagle_image_grid_thw"].shape[0] == 3, "Qwen processor did not receive all three current views")
    bridge_ids = [processor.tokenizer.convert_tokens_to_ids(f"<|bridge_{index}|>") for index in range(8)]
    require(len(set(bridge_ids)) == 8 and all(token in policy_batch["eagle_input_ids"] for token in bridge_ids),
            "Qwen bridge-marker registration/prompt mismatch")
    require(np.asarray(policy_sample["action_mask"])[:, :58].all()
            and not np.asarray(policy_sample["action_mask"])[:, 58:].any(), "Policy mask differs from tokenizer mask")
    old = Path('/personal/xiangpc/0811_Xpolicylab_bench/UniT/policy/UniT/checkpoints_0908-SparkArena/tokenizer_0908-SparkArena/checkpoint-40000')
    headers = inspect_tokenizer_shape(old)
    return {"status": "passed", "native_episodes": 3000, "native_frames": 751348,
            "physical_tasks": 12, "source_instruction_variants": 60, "samples": samples,
            "state_action_real": 58, "state_action_padded": 128, "masked_padding": 70,
            "normalization": "separate state/action mean_std", "embodiment_id": 31,
            "three_view_pool_cpu": "passed", "tokenizer_header_shapes": headers,
            "full_epoch_draw_count": 751296, "reshuffled_epoch_tail_omitted": 52,
            "HF_config_roundtrip_contracts": saved_contract,
            "policy_processor_cpu": "three current RGB views / native bridge-token processor passed",
            "policy_bridge_ids": bridge_ids,
            "header_is_shape_only_not_semantic_compatibility": True,
            "GPU_or_large_pretrained_tensor_loading": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_name")
    parser.add_argument("gpu_ids")
    parser.add_argument("--run-id", default=os.environ.get("UNIT_EE58_RUN_ID", ""),
                        help="Unique output suffix for an explicitly approved fresh attempt; ckpt_name is unchanged")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.ckpt_name), "Use a simple new checkpoint name")
    require(re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(?:size)?[_-]?\d+)", args.ckpt_name) is None,
            "New checkpoint names must not contain batch-size tags; use tianji_wuji")
    require(not args.run_id or re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id), "Use a simple run ID")
    require(re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(?:size)?[_-]?\d+)", args.run_id) is None,
            "Run IDs must not contain batch-size tags")
    gpu_ids = args.gpu_ids.split(",")
    require(len(gpu_ids) == len(set(gpu_ids)) == 8 and all(item.isdigit() for item in gpu_ids), "Expected eight distinct GPU IDs")
    source = Path(os.environ.get("UNIT_EE58_DATASET", DEFAULT_DATASET)).expanduser().resolve(strict=True)
    validated = validate_dataset(source)
    recipe = json.loads((POLICY / "configs/ee58_recipe.json").read_text())
    for stage, steps, micro, ga in (("tokenizer", 40000, 8, 1), ("policy", 60000, 4, 2)):
        hyper = recipe[stage]
        require((hyper["max_steps"], hyper["micro_batch"], hyper["ga"], hyper["effective_batch"])
                == (steps, micro, ga, 64), "Expected tokenizer40k/policy60k with global64")
    run_name = build_run_dir_name({"bench_name": "DexBench", "ckpt_name": args.ckpt_name,
                                   "env_cfg_type": "tianji_marvin_wuji", "action_type": "ee", "seed": recipe["seed"]})
    if args.run_id:
        run_name += "__" + args.run_id
    output = POLICY / "checkpoints" / run_name
    require(not output.exists(), f"Preserve existing run; select a new explicitly approved run ID: {output}")
    generated = POLICY / ".generated"
    generated.mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=run_name + "_", dir=generated))
    dataset = workspace / "data" / "dexbench_ee58_unit"
    (dataset / "meta").mkdir(parents=True)
    copied = ["info.json", "modality.json", "episodes.jsonl", "tasks.jsonl", "video_offsets.json", "action_contract.json"]
    hashes = {name: sha(source / "meta" / name) for name in copied + ["stats.json"]}
    for name in copied:
        shutil.copy2(source / "meta" / name, dataset / "meta" / name)
    info = dict(validated["info"])
    info["codebase_version"] = "v2.0"  # Metadata compatibility only; values/videos are unchanged.
    (dataset / "meta/info.json").write_text(json.dumps(info, indent=2))
    stats = {key: {stat: validated["stats"][key][stat] for stat in ("mean", "std", "min", "max", "q01", "q99")}
             for key in ("observation.state", "action")}
    (dataset / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    (dataset / "meta/relative_stats.json").write_text("{}\n")
    modality = validated["modality"]
    for group in ("state", "action"):
        for item in modality[group].values():
            item["dtype"] = "float32"
    (dataset / "meta/modality.json").write_text(json.dumps(modality, indent=2))
    for name in ("data", "videos"):
        (dataset / name).symlink_to(source / name, target_is_directory=True)
    dino = os.environ.get("UNIT_DINO_PATH", "/mnt/xspark-data/xiangpc/_shared_caches/unit/dinov2-large")
    vlm = os.environ.get("UNIT_VLM_PATH", "/personal/xspark_shared/model_weights/Qwen2.5-VL-3B-Instruct")
    for name, model_path in (("DINO", dino), ("Qwen", vlm)):
        require((Path(model_path) / "config.json").is_file(), f"Local {name} pretrained assets missing")
    for stage in ("tokenizer", "policy"):
        model = json.loads((POLICY / f"configs/ee58_{stage}.json").read_text())
        model["backbone_cfg"]["eagle_path"] = vlm
        model["backbone_cfg"]["dinov2_path"] = dino if stage == "tokenizer" else model["backbone_cfg"].get("dinov2_path", dino)
        if stage == "policy":
            model["backbone_cfg"].pop("dinov2_path", None)
            model["backbone_cfg"].update(select_layer=36, tune_llm=True, tune_visual=False,
                                          tune_bridge_embedding=True, load_bf16=False)
            model["unit_cfg"].update(bridge_type="vision_lang_obs", compute_bridge_loss=True,
                                      use_image_type_embedding=True, detach_vl_for_action=True,
                                      groot_tokenizer_path=str(output / "tokenizer/checkpoint-40000"))
            model["compute_dtype"] = "float32"
        model["ee58_contract"] = {"id": CONTRACT_ID, "real_dim": 58, "padded_dim": 128,
                                   "rotation": "matrix_rows_0_then_1", "normalization": "mean_std",
                                   "source_stats_sha256": hashes["stats.json"]}
        (workspace / f"{stage}_config.json").write_text(json.dumps(model, indent=2))
    prepared = {"dataset_path": str(dataset), "source_dataset": str(source), "vlm_path": vlm,
                "dino_path": dino, "recipe": recipe, "output_dir": str(output), "gpu_ids": args.gpu_ids,
                "workspace": str(workspace), "source_metadata_sha256": hashes,
                "ckpt_name": args.ckpt_name, "run_id": args.run_id, "run_dir_name": run_name,
                "created_at": datetime.now(timezone.utc).isoformat(), "contract_id": CONTRACT_ID}
    warmstart = os.environ.get("UNIT_TOKENIZER_WARMSTART")
    if warmstart:
        inspect_tokenizer_shape(warmstart)
        prepared["tokenizer_warmstart"] = str(Path(warmstart).resolve(strict=True))
        prepared["warmstart_semantics"] = "weights-only initializer, new optimizer/step0, retrain EE58 tokenizer40000; not a finished EE58 tokenizer"
    logpath = workspace / "native_preflight.log"
    print(f"UniT EE58 CPU probe log: {logpath}", file=sys.stderr)
    with logpath.open("w") as log, redirect_stdout(log), redirect_stderr(log):
        report = native_probe(prepared, validated)
    for name, digest in hashes.items():
        require(sha(source / "meta" / name) == digest, f"Source metadata changed: {name}")
    (workspace / "preflight.json").write_text(json.dumps(report, indent=2))
    prepared["bridge_ids"] = report["policy_bridge_ids"]
    prepared["adapter_sha256"] = {str(file.relative_to(POLICY)): sha(file)
                                    for file in [*(POLICY / "training").glob("*.py"), *(POLICY / "configs").glob("*.json")]}
    path = workspace / "prepared.json"
    path.write_text(json.dumps(prepared, indent=2))
    print(f"UniT EE58 native CPU checks passed: {workspace / 'preflight.json'}", file=sys.stderr)
    print(path)


if __name__ == "__main__":
    main()

