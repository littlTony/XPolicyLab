"""One saved UniT EE58 train/serve contract; no model weights are initialized here."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

from XPolicyLab.utils.dexbench_ee58_contract import CONTRACT_ID, pack_ee54, unpack_ee58
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

POLICY = Path(__file__).resolve().parent
UPSTREAM = POLICY / "UniT"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
PARTS = ("left_eef", "left_hand", "right_eef", "right_hand")
WIDTHS = (9, 20, 9, 20)
CAMERAS = ("front", "left_wrist", "right_wrist")
WIRE_CAMERAS = (("cam_head", "cam_high", "front"), ("cam_left_wrist", "left_wrist"),
                ("cam_right_wrist", "right_wrist"))
SCHEMA = "unit-ee58-inference-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def processor_inventory(path):
    names = ("config.json", "preprocessor_config.json", "preprocessor.json", "processor_config.json",
             "video_preprocessor.json", "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.json", "chat_template.jinja")
    result = {name: digest(Path(path) / name) for name in names if (Path(path) / name).is_file()}
    require("config.json" in result and "tokenizer_config.json" in result, "Missing local Qwen processor assets")
    return result


def weight_inventory(path):
    result = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with file.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            raw = stream.read(length)
        header = json.loads(raw)
        end = max((value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"), default=0)
        require(file.stat().st_size >= 8 + length + end, f"Truncated checkpoint: {file}")
        result[file.name] = {"bytes": file.stat().st_size, "header_sha256": hashlib.sha256(raw).hexdigest(),
                             "content_sha256": digest(file)}
    require(result, f"No complete safetensors checkpoint: {path}")
    return result


def vectors(metadata, kind):
    groups = metadata["statistics"][kind]
    values = {}
    for stat in ("mean", "std"):
        parts = [np.asarray(groups[key][stat], dtype=np.float64) for key in PARTS]
        require([len(value) for value in parts] == list(WIDTHS), f"Wrong {kind}/{stat} slice order")
        values[stat] = np.concatenate(parts)
        require(np.isfinite(values[stat]).all(), f"Non-finite {kind} statistics")
    require(np.all(values["std"] >= 0), "Negative standard deviation")
    return values


def validate_architecture(config):
    require(config["action_dim"] == 128 and config["action_horizon"] == 16, "UniT policy must be padded128/H16")
    head = config["action_head_cfg"]
    require(head["action_dim"] == head["max_state_dim"] == head["max_action_dim"] == 128,
            "Checkpoint changes the pretrained128-wide action/state architecture")
    require(head["max_num_embodiments"] == 32, "Expected32 embodiment slots")
    require(config["unit_cfg"]["num_bridge_tokens"] == 8, "Expected8 bridge markers")
    token = config["unit_cfg"]["tokenizer_cfg"]
    require(token["num_codebooks"] == 2 and token["codebook_size"] == 128
            and token["codebook_dim"] == 32, "Unexpected RVQ/CE-head architecture")
    require(config["unit_cfg"]["bridge_type"] == "vision_lang_obs", "Unexpected UniT action context")
    require(config["backbone_cfg"]["select_layer"] == 36, "Expected36 Qwen language layers")
    contract = config["ee58_contract"]
    require(contract["id"] == CONTRACT_ID and contract["real_dim"] == 58
            and contract["padded_dim"] == 128 and contract["rotation"] == "matrix_rows_0_then_1"
            and contract["normalization"] == "mean_std", "Checkpoint is not the new absolute EE58 contract")


def write_inference_bundle(checkpoint, prepared, step):
    """Called after native HF checkpoint/config/metadata saving has completed."""
    checkpoint = Path(checkpoint)
    config = read_json(checkpoint / "config.json")
    validate_architecture(config)
    metadata = read_json(checkpoint / "experiment_cfg/metadata.json")["new_embodiment"]
    source = Path(prepared["source_dataset"]) / "meta"
    expected = prepared["source_metadata_sha256"]
    for original, target in (("stats.json", "source_stats.json"), ("action_contract.json", "source_action_contract.json")):
        require(digest(source / original) == expected[original], "Source metadata changed during training")
        shutil.copy2(source / original, checkpoint / target)
    source_stats = read_json(checkpoint / "source_stats.json")
    for kind, key in (("state", "observation.state"), ("action", "action")):
        for name, actual in vectors(metadata, kind).items():
            np.testing.assert_array_equal(actual, np.asarray(source_stats[key][name]))
    processor_path = prepared["vlm_path"]
    sidecar = {
        "schema": SCHEMA, "policy_name": "UniT", "contract_id": CONTRACT_ID,
        "optimizer_step": int(step), "target_optimizer_updates": 60000,
        "real_state_dim": 58, "real_action_dim": 58, "model_dim": 128,
        "layout": list(PARTS), "widths": list(WIDTHS), "rotation": "matrix_rows_0_then_1",
        "normalization": "separate_state_action_mean_std", "embodiment": "new_embodiment", "embodiment_id": 31,
        "image_color_order": "rgb", "camera_order": list(CAMERAS), "image_size": [224, 224],
        "preprocess": "EE58DataConfig.transform(augment=True).eval(); crop0.95 center/resize224/no random jitter",
        "ignore_lang_prefix": True, "bridge_markers": 8, "action_horizon": 16,
        "bridge_ids": prepared["bridge_ids"],
        "num_inference_timesteps": int(config["action_head_cfg"]["num_inference_timesteps"]),
        "inference_compute_bridge_loss": False, "inference_needs_future_images": False,
        "processor_path": processor_path, "processor_sha256": processor_inventory(processor_path),
        "config_sha256": digest(checkpoint / "config.json"),
        "metadata_sha256": digest(checkpoint / "experiment_cfg/metadata.json"),
        "source_stats_sha256": digest(checkpoint / "source_stats.json"),
        "source_contract_sha256": digest(checkpoint / "source_action_contract.json"),
        "weights": weight_inventory(checkpoint),
        "teacher_tokenizer_path": config["unit_cfg"]["groot_tokenizer_path"],
        "training_preprocess_sha256": digest(POLICY / "training/ee58_data.py"),
    }
    require(sidecar["source_stats_sha256"] == config["ee58_contract"]["source_stats_sha256"], "Wrong source normalization snapshot")
    target = checkpoint / "inference_contract.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2))
    tmp.replace(target)
    return sidecar


@dataclass
class Bundle:
    checkpoint: Path
    config: dict
    metadata: dict
    sidecar: dict
    processor_path: str


def load_bundle(checkpoint, processor_override=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    sidecar = read_json(checkpoint / "inference_contract.json")
    require(sidecar["schema"] == SCHEMA and sidecar["contract_id"] == CONTRACT_ID, "Missing/mismatched EE58 inference sidecar")
    for filename, key in (("config.json", "config_sha256"), ("experiment_cfg/metadata.json", "metadata_sha256"),
                          ("source_stats.json", "source_stats_sha256"), ("source_action_contract.json", "source_contract_sha256")):
        require(digest(checkpoint / filename) == sidecar[key], f"Checkpoint contract snapshot changed: {filename}")
    require(weight_inventory(checkpoint) == sidecar["weights"], "Checkpoint weights changed/truncated after sidecar saving")
    config = read_json(checkpoint / "config.json")
    validate_architecture(config)
    require(config["ee58_contract"]["source_stats_sha256"] == sidecar["source_stats_sha256"], "Normalization snapshot differs from the saved model config")
    require(digest(POLICY / "training/ee58_data.py") == sidecar["training_preprocess_sha256"], "UniT preprocessing source changed after training; re-audit before serving")
    require(sidecar["optimizer_step"] <= 60000 and sidecar["target_optimizer_updates"] == 60000, "Expected the new60k policy schedule")
    require(sidecar["inference_compute_bridge_loss"] is False and sidecar["inference_needs_future_images"] is False,
            "Online inference must not require future teacher images")
    require(sidecar["camera_order"] == list(CAMERAS) and sidecar["image_color_order"] == "rgb"
            and sidecar["embodiment_id"] == 31, "Wrong cameras/color/embodiment")
    require(sidecar["layout"] == list(PARTS) and sidecar["widths"] == list(WIDTHS)
            and sidecar["rotation"] == "matrix_rows_0_then_1" and sidecar["normalization"] == "separate_state_action_mean_std"
            and sidecar["ignore_lang_prefix"] is True, "Wrong state/action/preprocessing contract")
    metadata = read_json(checkpoint / "experiment_cfg/metadata.json")["new_embodiment"]
    stats = read_json(checkpoint / "source_stats.json")
    for kind, source in (("state", "observation.state"), ("action", "action")):
        for name, actual in vectors(metadata, kind).items():
            np.testing.assert_array_equal(actual, np.asarray(stats[source][name]))
    contract = read_json(checkpoint / "source_action_contract.json")
    require(contract["id"] == CONTRACT_ID and contract["rot6d"] == "matrix_rows_0_then_1", "Wrong rotation/source action contract")
    processor_path = str(processor_override or sidecar["processor_path"])
    require(processor_inventory(processor_path) == sidecar["processor_sha256"], "Qwen processor differs from training")
    return Bundle(checkpoint, config, metadata, sidecar, processor_path)


class Runtime:
    def __init__(self, bundle, robot_dims):
        from .training.ee58_data import EE58DataConfig
        import gr00t
        require(Path(gr00t.__file__).resolve().is_relative_to(UPSTREAM.resolve()), "Wrong gr00t checkout loaded; start a fresh UniT process")
        from gr00t.data.schema import DatasetMetadata
        from gr00t.model.transforms import collate
        self.bundle, self.robot_dims, self.collate = bundle, robot_dims, collate
        cfg = EE58DataConfig(eagle_path=bundle.processor_path, use_bridge=True, ignore_lang_prefix=True,
                             num_bridge_tokens=8, tokenizer_only=False)
        self.transforms = cfg.transform(augment=True)
        self.transforms.set_metadata(DatasetMetadata.model_validate(bundle.metadata))
        self.transforms.eval()
        self.processor = self.transforms.transforms[-1].eagle_processor
        ids = [self.processor.tokenizer.convert_tokens_to_ids(f"<|bridge_{index}|>") for index in range(8)]
        require(ids == bundle.sidecar["bridge_ids"] and len(set(ids)) == 8, "Bridge marker IDs differ from training")
        self.action_stats = vectors(bundle.metadata, "action")

    def observation(self, obs):
        wire = np.asarray(pack_robot_state(obs, "ee", self.robot_dims), dtype=np.float32)
        if wire.shape == (1, 54):
            wire = wire[0]
        require(wire.shape == (54,), f"Expected one EE54 wire state, got{wire.shape}")
        state = pack_ee54(wire)
        features = {}
        start = 0
        for key, width in zip(PARTS, WIDTHS):
            features[f"state.{key}"] = state[None, start:start + width]
            start += width
        vision = obs["vision"]
        for key, aliases in zip(CAMERAS, WIRE_CAMERAS):
            camera = next((vision[name] for name in aliases if name in vision), None)
            require(camera is not None, f"Missing RGB camera {key}; expected one of{aliases}")
            image = camera["color"] if isinstance(camera, dict) else camera
            image = np.asarray(image)
            require(image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8,
                    f"Expected decoded RGB uint8 HWC for {key}, got{image.shape}/{image.dtype}")
            resolution = self.bundle.metadata["modalities"]["video"][key]["resolution"]
            require(tuple(image.shape[:2]) == (resolution[1], resolution[0]), "Camera resolution differs from the saved training crop")
            features[f"video.{key}"] = image[None].copy()
        language = obs.get("instruction", obs.get("instructions"))
        if isinstance(language, (list, tuple, np.ndarray)):
            language = language[0]
        require(isinstance(language, str) and bool(language.strip()), "Missing instruction")
        features["annotation.human.task_description"] = np.asarray([language])
        return features

    def batch(self, observations):
        return self.collate([self.transforms(self.observation(obs)) for obs in observations], self.processor)

    def actions(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        require(prediction.ndim == 3 and prediction.shape[1:] == (16, 128) and np.isfinite(prediction).all(),
                f"Expected finite [B,16,128] predictions, got{prediction.shape}")
        real = prediction[..., :58] * self.action_stats["std"] + self.action_stats["mean"]
        wire = unpack_ee58(real)
        return [[unpack_robot_state(action.astype(np.float32), "ee", self.robot_dims) for action in chunk] for chunk in wire]
