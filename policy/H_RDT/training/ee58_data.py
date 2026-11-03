"""Compact EE58 backend for native H-RDT; no old joint/HDF5 sampler code."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY.parents[2]))
from XPolicyLab.utils.dexbench_ee58_contract import DEFAULT_DATASET, VIDEO_KEYS, validate_dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(values, statistics):
    lo = np.asarray(statistics["q01"], dtype=np.float32)
    hi = np.asarray(statistics["q99"], dtype=np.float32)
    scale = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    return ((np.clip(values, lo, hi) - lo) / scale * 2 - 1).astype(np.float32)


def validate_language(cache, spec, load_tensors=True):
    """Bind exact 60 instruction strings to local T5-XXL embeddings."""
    import torch
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing complete 60-instruction T5 cache: {manifest_path}; run training/prepare_language.py")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_fixture") or manifest.get("not_real_T5"):
        raise ValueError("Synthetic CPU language fixtures are forbidden for training")
    if manifest.get("dataset_tasks_sha256") != sha256(spec["root"] / "meta/tasks.jsonl"):
        raise ValueError("Language cache belongs to different task/instruction metadata")
    if manifest.get("feature_dim") != 4096 or manifest.get("max_length") != 1024:
        raise ValueError("Expected native T5-v1_1-XXL 4096 features / max_length1024")
    encoder = Path(manifest["encoder_path"])
    if not (encoder / "config.json").is_file() or sha256(encoder / "config.json") != manifest["encoder_config_sha256"]:
        raise ValueError("T5 encoder configuration is missing or changed")
    for name, record in manifest.get("encoder_weights", {}).items():
        file = encoder / name
        if not file.is_file() or file.stat().st_size != record["bytes"] or file.stat().st_mtime_ns != record["mtime_ns"]:
            raise ValueError("T5 source weights changed since cache generation")
    if not manifest.get("encoder_weights"):
        raise ValueError("Language cache lacks source weight provenance")
    records = manifest["texts"]
    if set(records) != {str(task["task_index"]) for task in spec["tasks"]}:
        raise ValueError("Language cache must cover all 60 exact instruction variants")
    for task in spec["tasks"]:
        record = records[str(task["task_index"])]
        if record["file"] != f"task_{task['task_index']:03d}.pt":
            raise ValueError("Unexpected language cache file naming")
        file = cache / record["file"]
        if record["text_sha256"] != text_hash(task["task"]) or sha256(file) != record["sha256"]:
            raise ValueError(f"Language embedding/text digest mismatch for {task['task_index']}")
        if load_tensors:
            value = torch.load(file, map_location="cpu", weights_only=True)
            embeddings = value["embeddings"]
            if value["instruction"] != task["task"] or embeddings.ndim != 2 or embeddings.shape[1] != 4096:
                raise ValueError(f"Bad language embedding for task {task['task_index']}")
            if not (0 < embeddings.shape[0] <= 1024) or not torch.isfinite(embeddings).all():
                raise ValueError("Invalid T5 sequence length or non-finite embedding")
    return manifest


def decode_frame(path, frame_index, fps):
    """MP4 AV1 -> RGB. Offsets are global frame indices inside shared videos."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target_seconds = frame_index / fps
        container.seek(int(target_seconds / float(stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            decoded_index = int(round(float(frame.pts * frame.time_base) * fps))
            if decoded_index < frame_index:
                continue
            if decoded_index != frame_index:
                raise ValueError(f"Video frame mismatch: requested {frame_index}, decoded {decoded_index}: {path}")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Expected decoded RGB uint8 video frame")
            return rgb, decoded_index
    raise IndexError(f"Video frame {frame_index} missing: {path}")


class EE58Backend:
    DATASET_NAME = "dexbench_ee58"

    def __init__(self, dataset=DEFAULT_DATASET, language_cache=None, require_language=True):
        self.spec = validate_dataset(dataset)
        self.root = self.spec["root"]
        self.episodes = sorted(self.spec["episodes"], key=lambda row: row["episode_index"])
        self.lengths = np.asarray([row["length"] for row in self.episodes], dtype=np.int64)
        self.ends = self.lengths.cumsum()
        self.language_cache = Path(language_cache) if language_cache else POLICY / "data/ee58_t5"
        if require_language:
            validate_language(self.language_cache, self.spec)
        self.parquet_cache = OrderedDict()
        self.language_tensors = {}

    def __len__(self):
        return int(self.ends[-1])

    def trajectory(self, episode_index):
        if episode_index not in self.parquet_cache:
            template = self.spec["info"]["data_path"]
            path = self.root / template.format(episode_chunk=episode_index // 1000, episode_index=episode_index)
            frame = pd.read_parquet(path)
            if len(frame) != self.lengths[episode_index]:
                raise ValueError(f"Episode rows do not match metadata: {episode_index}")
            if frame.episode_index.nunique() != 1 or int(frame.episode_index.iloc[0]) != episode_index:
                raise ValueError("Wrong episode parquet identity")
            self.parquet_cache[episode_index] = frame
            if len(self.parquet_cache) > 8:
                self.parquet_cache.popitem(last=False)
        self.parquet_cache.move_to_end(episode_index)
        return self.parquet_cache[episode_index]

    def raw_step(self, episode_index, step):
        frame = self.trajectory(episode_index)
        state = np.stack(frame["observation.state"].to_numpy())
        actions = np.stack(frame["action"].to_numpy())
        times = np.minimum(np.arange(step, step + 16), len(frame) - 1)
        if state.shape[1] != 58 or actions.shape[1] != 58:
            raise ValueError("Expected compact58 state/action")
        if not np.isfinite(state[step]).all() or not np.isfinite(actions[times]).all():
            raise ValueError("Non-finite raw EE58 sample")
        task = int(frame["task_index"].iloc[step])
        if task != int(frame["task_index"].iloc[0]):
            raise ValueError("Instruction changes within episode")
        return state[step:step+1].astype(np.float32), actions[times].astype(np.float32), task

    def rgb_frames(self, episode_index, step):
        images, decoded = [], {}
        for key in VIDEO_KEYS:
            offset = int(self.spec["offsets"][str(episode_index)][key])
            template = self.spec["info"]["video_path"]
            path = self.root / template.format(episode_chunk=episode_index // 1000,
                                              episode_index=episode_index, video_key=key)
            rgb, index = decode_frame(path, offset + step, self.spec["info"]["fps"])
            images.append(rgb[None])
            decoded[key] = {"offset": offset, "local_frame": step, "decoded_frame": index}
        return np.stack(images), decoded

    def get_item(self, index):
        import torch
        episode = int(np.searchsorted(self.ends, index, side="right"))
        step = int(index - (self.ends[episode-1] if episode else 0))
        state, action, task = self.raw_step(episode, step)
        images, _ = self.rgb_frames(episode, step)
        if task not in self.language_tensors:
            file = self.language_cache / f"task_{task:03d}.pt"
            self.language_tensors[task] = torch.load(file, map_location="cpu", weights_only=True)["embeddings"]
        return {"current_images": images, "current_images_mask": np.ones((3, 1), dtype=bool),
                "states": normalize(state, self.spec["stats"]["observation.state"]),
                "actions": normalize(action, self.spec["stats"]["action"]),
                "action_norm": np.ones((16, 58), dtype=np.float32),
                "instruction": self.language_tensors[task], "dataset_name": self.DATASET_NAME}


def consumer_class(native_base):
    class EE58Consumer(native_base):
        def __init__(self, config, image_transform, num_cameras, image_size=None,
                     auto_adjust_image_brightness=False, image_aug=False, image_corrupt_severity=None,
                     dataset_type=None, state_noise_snr=None, use_precomp_lang_embed=True,
                     upsample_rate=None, val=False, **kwargs):
            if num_cameras != 3 or config["common"]["state_dim"] != 58 or config["common"]["action_dim"] != 58:
                raise ValueError("Expected 3-view compact EE58 model")
            if image_aug or not use_precomp_lang_embed or upsample_rate != 1:
                raise ValueError("This recipe requires no image augmentation, precomputed T5, stride1")
            contract = config["ee58_contract"]
            self.hdf5_dataset = EE58Backend(contract["dataset_root"], contract["language_cache"])
            self.dataset_name = EE58Backend.DATASET_NAME
            self.dataset_name2id = {self.dataset_name: 0}
            self.dataset_id2name = {0: self.dataset_name}
            self.state_noise_snr = state_noise_snr
            self.num_cameras, self.img_history_size = 3, 1
            self.image_transform, self.image_size = image_transform, image_size
            self.auto_adjust_image_brightness = auto_adjust_image_brightness
            self.image_aug, self.image_corrupt_severity = False, image_corrupt_severity
            self.use_precomp_lang_embed, self.dataset_type = True, dataset_type
    return EE58Consumer
