"""CPU preparation: current DexBench EE58 data with SparkArena hyperparameters."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

POLICY = Path(__file__).resolve().parent.parent
REVISION_ID = "r20261031_fast"
WORLD_SIZE, MICRO_BATCH, GRADIENT_ACCUMULATION, GLOBAL_BATCH = 8, 4, 1, 32
UPSTREAM_COMMIT = "985c2238627481f25476fbee478083812dd82a21"
DEFAULT_DATASET = "/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21"
CONTRACT_ID = "dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1"
ROBOT = "tianji_marvin_wuji"
PARTS = ("left_eef", "left_hand", "right_eef", "right_hand")
CAMERAS = ("front", "left_wrist", "right_wrist")
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_helpers():
    sys.path.insert(0, str(POLICY.parents[2]))
    from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
    from XPolicyLab.utils.process_data import get_robot_action_dim_info
    dims = get_robot_action_dim_info(ROBOT)
    require(len(dims["arm_dim"]) == 2 and dims["ee_dim"] == [20, 20], f"Expected dual Wuji20: {dims}")
    # Native EEF9 is xyz3 + row-rot6D, not the helper's arm-joint7.
    return build_run_dir_name, (3 + 6, dims["ee_dim"][0], 3 + 6, dims["ee_dim"][1])


def validate_config(cfg):
    d, f, t, c = cfg["datasets"]["vla_data"], cfg["framework"], cfg["trainer"], cfg["sparkarena_contract"]
    require(f["name"] == "QwenPI_v3" and c["bench_name"] == "DexBench", "QwenPI_v3 / DexBench required")
    require(c["env_cfg_type"] == ROBOT and c["action_type"] == "ee", "Expected Wuji EE58")
    _, widths = shared_helpers()
    require(f["action_model"]["action_dim"] == f["action_model"]["state_dim"] == sum(widths) == 58, "Both state/action must be58")
    require(c["data_contract_id"] == CONTRACT_ID and c["rot6d"] == "matrix_rows_0_then_1", "Wrong EE6D contract")
    require(f["action_model"]["action_horizon"] == 16 and f["action_model"]["diffusion_model_cfg"]["interleave_self_attention"] is True, "H16 canonical DiT required")
    require(f["qwenvl"]["attn_implementation"] == "flash_attention_2", "FA2 required")
    require(d["include_state"] is True and d["action_mode"] == "abs" and d["action_temporal_offset"] == 0 and d["action_derived_from_state"] is False, "Expected state[t], absolute action[t:t+16]")
    require(d["image_size"] == d["obs_image_size"] == [224, 224] and d["state_action_dtype"] == "float16", "Wrong image/packing settings")
    require(d["lerobot_version"] == "v2.0" and d["data_mix"] == "xpolicylab_dexbench_ee58_qwen_pi_v3", "Wrong native schema/registry")
    require(d["per_device_batch_size"] == MICRO_BATCH and t["gradient_accumulation_steps"] == GRADIENT_ACCUMULATION and c["num_processes"] == WORLD_SIZE and c["effective_batch"] == GLOBAL_BATCH, "Expected 8xmicro4xGA1=32")
    require(c["adapter_revision"] == REVISION_ID, "Wrong immutable adapter revision")
    require(t["max_train_steps"] == 60000 and t["repeated_diffusion_steps"] == 16, "Wrong update/noise-repeat recipe")
    require(t["freeze_modules"] is None and t["gradient_checkpointing"] is True, "Expected full training with checkpointing")
    require(not d["delete_pause_frame"] and not d["drop_incomplete_action_chunks"], "Keep all frames and last-action padding")


def inspect_source(source, widths):
    import numpy as np
    read = lambda name: json.loads((source / "meta" / name).read_text())
    info, contract, modality, stats, offsets = [read(n) for n in ("info.json", "action_contract.json", "modality.json", "stats.json", "video_offsets.json")]
    require(info["codebase_version"] == "v2.1" and info["robot_type"] == "dexbench_wujihand2", "Wrong dataset/robot")
    require((info["total_episodes"], info["total_frames"], info["total_tasks"], info["fps"]) == (3000, 751348, 60, 25) and info["splits"] == {"train": "0:3000"}, "Wrong DexBench export")
    require(contract["id"] == CONTRACT_ID and contract["rot6d"] == "matrix_rows_0_then_1" and contract["RGB"] is True, "Wrong EE58/RGB contract")
    require(contract["state"] == contract["action"] == "absolute_world_xyz_rot6d_hand20" and contract["layout"] == ["left_eef9", "left_hand20", "right_eef9", "right_hand20"], "Wrong representation/layout")
    require(contract["action_alignment"] == "export_action_t_no_extra_shift", "Wrong temporal alignment")
    for group, original in (("state", "observation.state"), ("action", "action")):
        require(info["features"][original]["shape"] == [sum(widths)] and list(modality[group]) == list(PARTS), "Bad feature/order")
        start = 0
        for name, width in zip(PARTS, widths):
            item = modality[group][name]
            require((item["start"], item["end"], item["absolute"], item["original_key"]) == (start, start + width, True, original) and item.get("rotation_type") is None, "Bad EE58 mapping or second rotation transform")
            start += width
        for stat in STAT_KEYS:
            values = np.asarray(stats[original][stat])
            require(values.shape == (58,) and np.isfinite(values).all(), f"Invalid {original}.{stat}")
    require(list(modality["video"]) == list(CAMERAS) and modality["annotation"]["human.task_description"]["original_key"] == "task_index", "Wrong camera/language aliases")
    originals = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
    for name, original in zip(CAMERAS, originals):
        require(modality["video"][name]["original_key"] == original, "Wrong camera mapping")
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    require([e["episode_index"] for e in episodes] == list(range(3000)) and sum(e["length"] for e in episodes) == 751348 and all(e["length"] > 0 for e in episodes), "Bad episode IDs/lengths")
    groups = {}
    for ep in episodes:
        require(len(ep["tasks"]) == 1, "One instruction per episode required")
        match = re.match(r"^\[[^/]+/([^\]]+)\]", ep["tasks"][0])
        require(match is not None, f"Unrecognized task instruction: {ep['tasks']}")
        groups.setdefault(match.group(1), []).append(ep["episode_index"])
        for original in originals:
            v = offsets[str(ep["episode_index"])][original]
            require(isinstance(v, (float, int)) and np.isfinite(v) and v >= 0 and int(v) == v, "Invalid per-camera frame offset")
    require(len(groups) == 12 and all(len(ids) == 250 for ids in groups.values()), "Expected12 tasks x250eps")
    return info, contract, stats, offsets, episodes, groups


def prepare_native_metadata(private, stats, episodes):
    # State/action statistics remain independent; relative_stats.json is not used.
    native = {key: {stat: stats[key][stat] for stat in STAT_KEYS} for key in ("action", "observation.state")}
    (private / "meta/stats_gr00t.json").write_text(json.dumps({"__format_version": 2, "__cache_config": {"mode": "abs"}, "statistics": native}, indent=2))
    steps = [(e["episode_index"], t) for e in episodes for t in range(e["length"])]
    cache_key = hashlib.md5(str(sorted({"delete_pause_frame": False, "dataset_name": private.name}.items())).encode()).hexdigest()[:12]
    with (private / "meta/steps_data_index.pkl").open("wb") as fp:
        pickle.dump({"config_key": cache_key, "steps": steps, "num_trajectories": len(episodes), "total_steps": len(steps), "delete_pause_frame": False}, fp, protocol=pickle.HIGHEST_PROTOCOL)


def native_smoke(cfg, source, stats, offsets, episodes, groups, workspace):
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    sys.path.insert(0, cfg["sparkarena_contract"]["upstream_root"])
    from sparkarena_registry import register
    register(cfg["sparkarena_contract"]["dataset_name"])
    from data_pipeline import install_dataset_pipeline
    install_dataset_pipeline(cfg["seed"])
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.gr00t_lerobot.datasets import get_frames_by_timestamps
    mixture = get_vla_dataset(data_cfg=OmegaConf.create(cfg["datasets"]["vla_data"]), seed=cfg["seed"])
    require(mixture.seed == cfg["seed"] and hasattr(mixture, "shared_epoch"), "Dataset seed/shared epoch pipeline missing")
    ds = mixture.datasets[0]
    require(len(ds.trajectory_ids) == 3000 and len(ds.all_steps) == 751348, "Native loader lost data")
    require(ds.all_steps == [(e["episode_index"], t) for e in episodes for t in range(e["length"])], "Native step cache incomplete")
    mixture.save_dataset_statistics(workspace / "dataset_statistics.json")
    saved = json.loads((workspace / "dataset_statistics.json").read_text())["new_embodiment"]
    require(saved["action"]["mask"] == [True] * 58, "All58 dimensions must participate")
    for group, original in (("state", "observation.state"), ("action", "action")):
        for stat in STAT_KEYS:
            np.testing.assert_allclose(saved[group][stat], stats[original][stat], atol=1e-7)
    info = json.loads((source / "meta/info.json").read_text())
    records, proof = [], None
    for task, index in [(name, idx) for name, ids in groups.items() for idx in (ids[0], ids[-1])]:
        trajectory = pd.read_parquet(source / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index))
        require(len(trajectory) == episodes[index]["length"], "Parquet/episode lengths disagree")
        base = min(7, len(trajectory) - 1)
        raw = ds.get_step_data(index, base)
        sample = ds._pack_sample(ds.transforms({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in raw.items()}))
        require(sample["action"].shape == (16, 58) and sample["state"].shape == (1, 58), "Wrong native EE58 shapes")
        require(sample["action"].dtype == sample["state"].dtype == np.float16, "Wrong packing dtype")
        require(len(sample["image"]) == 3 and all(im.size == (224, 224) and im.mode == "RGB" for im in sample["image"]), "Wrong RGB pack")
        require(sample["lang"] == episodes[index]["tasks"][0], "Language/episode mismatch")
        times = np.minimum(np.arange(base, base + 16), len(trajectory) - 1)
        for group, original, indices in (("state", "observation.state", [base]), ("action", "action", times)):
            native = np.concatenate([raw[f"{group}.{key}"] for key in PARTS], axis=1)
            expected = np.stack(trajectory[original].to_numpy())[indices]
            np.testing.assert_array_equal(native, expected)
            lo, hi = (np.asarray(stats[original][name], np.float32) for name in ("q01", "q99"))
            normalized, mask = expected.copy(), hi != lo
            normalized[:, mask] = 2 * (expected[:, mask] - lo[mask]) / (hi[mask] - lo[mask]) - 1
            np.testing.assert_allclose(sample[group], np.clip(normalized, -2.2, 2.2).astype(np.float16), atol=0.001, rtol=0.001)
            require(np.isfinite(sample[group]).all(), f"Non-finite {group}")
        camera_offsets = {}
        for camera in CAMERAS:
            original = ds.lerobot_modality_meta.video[camera].original_key
            offset = offsets[str(index)][original]
            camera_offsets[camera] = offset
            if offset > 0 and proof is None:
                video = source / info["video_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index, video_key=original)
                local_time = float(trajectory.timestamp.iloc[base])
                ref = get_frames_by_timestamps(str(video), np.asarray([local_time + offset / info["fps"]]), video_backend="torchvision_av", video_backend_kwargs=None)
                np.testing.assert_array_equal(raw[f"video.{camera}"], ref)
                wrong = get_frames_by_timestamps(str(video), np.asarray([local_time]), video_backend="torchvision_av", video_backend_kwargs=None)
                difference = float(np.mean(np.abs(ref.astype(float) - wrong.astype(float))))
                require(difference > 0, "Offset proof must distinguish the packed-video start")
                proof = {"episode": index, "camera": camera, "offset_frames": offset, "requested_seconds": local_time + offset / info["fps"], "mean_pixel_difference_without_offset": difference, "frame_sha256": hashlib.sha256(ref.tobytes()).hexdigest()}
        records.append({"episode": index, "task": task, "base_index": base, "action_shape": [16, 58], "state_shape": [1, 58], "dtype": "float16", "camera_offset_frames": camera_offsets})
    require(proof is not None, "No nonzero packed-video offset verified")
    tail = ds.get_step_data(0, episodes[0]["length"] - 1)
    for key in PARTS:
        np.testing.assert_array_equal(tail[f"action.{key}"], np.repeat(tail[f"action.{key}"][:1], 16, axis=0))
    return {"status": "passed", "dataset": str(source), "contract": CONTRACT_ID, "samples": records, "episodes": 3000, "frames": 751348, "physical_tasks": 12, "instruction_variants": 60, "state_dim": 58, "action_dim": 58, "action_mask": [True] * 58, "raw_action_offset": 0, "tail_padding": "repeat_last", "q99_clip": [-2.2, 2.2], "video_offset_proof": proof, "gpu_used": False, "scope": "Native CPU loader:24 real episodes, vectors/normalization/RGB/packed-video offsets. No pretrained model forward or GPU training."}


def resolve_run(args, build_run_dir_name):
    """Fresh independent output without changing ckpt_name or overwriting a run."""
    base = build_run_dir_name(vars(args))
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_suffix) is not None,
            "Use a simple nonempty run suffix")
    run_id = args.run_id or (base + "-" + args.run_suffix)
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is not None and run_id.startswith(base + "-"),
            "Independent run-id must preserve the benchmark/ckpt/robot/action/seed prefix")
    require(re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(size)?[_-]?\d+)", run_id) is None,
            "Checkpoint output names must not contain batch-size labels")
    output = args.output_dir or (POLICY / "checkpoints" / run_id)
    require(output.is_absolute() and output.name == run_id, "Output-dir must be absolute and named exactly run-id")
    require(all(re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(size)?[_-]?\d+)", part) is None
                for part in output.parts), "Checkpoint path must not contain batch-size labels")
    return run_id, output


def main():
    parser = argparse.ArgumentParser()
    for name in ("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed", "gpu_ids"):
        parser.add_argument(name)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-suffix", default=REVISION_ID)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    require(args.bench_name == "DexBench", "Use DexBench; SparkArena supplies hyperparameters, not this dataset")
    require(args.env_cfg_type == ROBOT and args.action_type == "ee", "Only tianji_marvin_wuji EE58 supported")
    require(args.ckpt_name == "tianji_wuji", "This revision keeps ckpt_name=tianji_wuji")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", args.ckpt_name) is not None and args.ckpt_name not in (".", ".."), "Use a simple ckpt_name")
    require(re.search(r"(?i)(bs[_-]?\d+|batch[_-]?(size)?[_-]?\d+)", args.ckpt_name) is None,
            "New checkpoint names must not contain batch-size labels; use tianji_wuji")
    ids = args.gpu_ids.split(",")
    require(len(ids) == len(set(ids)) == 8 and all(re.fullmatch(r"\d+", value) for value in ids), "Eight distinct GPU IDs required")
    seed = int(args.seed)
    require(seed >= 0, "Seed must be nonnegative")
    build_run_dir_name, widths = shared_helpers()
    upstream = POLICY / "upstream"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    require(commit == UPSTREAM_COMMIT, "Upstream changed; re-audit before training")
    source = Path(os.environ.get("STARVLA_EE58_DATASET", DEFAULT_DATASET)).resolve(strict=True)
    source_hashes = {str(p.relative_to(source)): sha256(p) for p in sorted((source / "meta").rglob("*")) if p.is_file()}
    info, contract, stats, offsets, episodes, groups = inspect_source(source, widths)
    run_id, output = resolve_run(args, build_run_dir_name)
    require(not output.exists(), f"Refusing overwrite/resume of existing run: {output}")
    (POLICY / ".generated").mkdir(exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=run_id + "_", dir=POLICY / ".generated"))
    private = workspace / "data" / source.name
    private.mkdir(parents=True)
    shutil.copytree(source / "meta", private / "meta")
    for folder in ("data", "videos"):
        (private / folder).symlink_to(source / folder, target_is_directory=True)
    prepare_native_metadata(private, stats, episodes)
    cfg = yaml.safe_load((POLICY / "configs/sparkarena_qwen_pi_v3.yaml").read_text())
    cfg["run_id"], cfg["run_root_dir"], cfg["seed"] = run_id, str(output.parent), seed
    cfg["framework"]["qwenvl"]["base_vlm"] = os.environ.get("STARVLA_BASE_VLM", cfg["framework"]["qwenvl"]["base_vlm"])
    require((Path(cfg["framework"]["qwenvl"]["base_vlm"]) / "config.json").is_file(), "Local Qwen weights missing")
    cfg["framework"]["action_model"].update(action_dim=sum(widths), state_dim=sum(widths))
    cfg["datasets"]["vla_data"]["data_root_dir"] = str(private.parent)
    cfg["sparkarena_contract"] = {"bench_name": "DexBench", "env_cfg_type": ROBOT, "action_type": "ee", "data_contract_id": CONTRACT_ID, "rot6d": contract["rot6d"], "actual_dataset_version": info["codebase_version"], "source_dataset": str(source), "dataset_name": source.name, "source_metadata_sha256": source_hashes, "upstream_root": str(upstream), "upstream_commit": commit, "num_processes": WORLD_SIZE, "effective_batch": GLOBAL_BATCH, "adapter_revision": REVISION_ID, "recipe_reference": "SparkArena QwenPI_v3; user-authorized global32 fresh run; current DexBench EE58 data", "state_input": "q99_then_native_state2str", "normalization": "q99", "noise_repeats": 16, "created_at": datetime.now(timezone.utc).isoformat()}
    validate_config(cfg)
    print(f"DexBench EE58 CPU preflight log: {workspace / 'native_loader.log'}", file=sys.stderr)
    with (workspace / "native_loader.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        report = native_smoke(cfg, source, stats, offsets, episodes, groups, workspace)
    for relative, digest in source_hashes.items():
        require(sha256(source / relative) == digest, f"Source metadata changed: {relative}")
    cfg["sparkarena_contract"]["metadata_sha256"] = {str(p.relative_to(private)): sha256(p) for p in sorted((private / "meta").rglob("*")) if p.is_file()}
    cfg["sparkarena_contract"]["adapter_sha256"] = {str(p.relative_to(POLICY)): sha256(p) for p in [*(POLICY / "training").glob("*.py"), POLICY / "ee58_runtime.py", POLICY / "model.py", POLICY / "train_sparkarena.sh", POLICY / "configs/sparkarena_qwen_pi_v3.yaml"]}
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (workspace / "preflight.json").write_text(json.dumps(report, indent=2))
    print(f"DexBench EE58 CPU preflight passed: {workspace / 'preflight.json'}", file=sys.stderr)
    if not args.dry_run:
        output.parent.mkdir(exist_ok=True)
        output.mkdir()
        shutil.copy2(config_path, output / "launch_config.yaml")
        shutil.copy2(workspace / "preflight.json", output / "preflight.json")
    print(config_path)


if __name__ == "__main__":
    main()
