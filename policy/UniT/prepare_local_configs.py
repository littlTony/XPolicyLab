#!/usr/bin/env python3
"""Materialize official UniT configs with local, equivalent weight paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


TOKENIZER_RELATIVE = Path(
    "gr00t/model/configs/shared_tokenizer/gr00t_tokenizer_mix_unified_eef_dino.json"
)
DUAL_RELATIVE = Path(
    "gr00t/model/configs/shared_dual_system/gr00t_n1.5_unit_tokenizer.json"
)
SPARK0_EMBODIMENT_ID = 31
MIN_EMBODIMENT_CAPACITY = SPARK0_EMBODIMENT_ID + 1


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_embodiment_capacity(config: dict[str, Any], key: str) -> int:
    section = config.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"model config is missing object {key!r}")
    configured = section.get("max_num_embodiments")
    # Missing and explicit JSON null both mean "use the model class default".
    # UniT's current runtime default is 32, which is the minimum needed for ID 31.
    current = MIN_EMBODIMENT_CAPACITY if configured is None else int(configured)
    capacity = max(current, MIN_EMBODIMENT_CAPACITY)
    section["max_num_embodiments"] = capacity
    if not 0 <= SPARK0_EMBODIMENT_ID < capacity:
        raise ValueError(
            f"Spark0 embodiment id {SPARK0_EMBODIMENT_ID} is outside {key} capacity {capacity}"
        )
    return capacity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, required=True)
    parser.add_argument("--dinov2-path", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_root.resolve()
    qwen = args.qwen_path.resolve()
    dinov2 = args.dinov2_path.resolve()
    if not (qwen / "config.json").is_file():
        raise FileNotFoundError(f"Qwen checkpoint is incomplete: {qwen}")
    if not (dinov2 / "config.json").is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint is incomplete: {dinov2}")

    tokenizer = read_json(upstream / TOKENIZER_RELATIVE)
    dual = read_json(upstream / DUAL_RELATIVE)
    tokenizer["backbone_cfg"]["dinov2_path"] = str(dinov2)
    tokenizer["backbone_cfg"]["eagle_path"] = str(qwen)
    dual["backbone_cfg"]["eagle_path"] = str(qwen)
    # The official JSON keeps this documentation-only entry beside runtime
    # constructor arguments. UniT forwards backbone_cfg with **kwargs, so the
    # generated runtime copy must not pass the comment into EagleBackboneUniT.
    dual["backbone_cfg"].pop("_comment_eagle_path", None)
    tokenizer_encoder_capacity = ensure_embodiment_capacity(tokenizer, "action_encoder_cfg")
    tokenizer_decoder_capacity = ensure_embodiment_capacity(tokenizer, "action_decoder_cfg")
    dual_action_capacity = ensure_embodiment_capacity(dual, "action_head_cfg")

    tokenizer_out = args.output_dir / "gr00t_tokenizer_local_weights.json"
    dual_out = args.output_dir / "gr00t_dual_system_local_weights.json"
    write_atomic(tokenizer_out, tokenizer)
    write_atomic(dual_out, dual)
    persisted_tokenizer = read_json(tokenizer_out)
    persisted_dual = read_json(dual_out)
    if "_comment_eagle_path" in persisted_dual["backbone_cfg"]:
        raise RuntimeError("dual runtime config retained _comment_eagle_path")
    for key in ("action_encoder_cfg", "action_decoder_cfg"):
        capacity = int(persisted_tokenizer[key]["max_num_embodiments"])
        if capacity <= SPARK0_EMBODIMENT_ID:
            raise RuntimeError(
                f"persisted {key} capacity {capacity} cannot address embodiment id "
                f"{SPARK0_EMBODIMENT_ID}"
            )
    print(
        "validated new_embodiment id 31: "
        f"tokenizer encoder={tokenizer_encoder_capacity}, "
        f"tokenizer decoder={tokenizer_decoder_capacity}, "
        f"dual action head={dual_action_capacity}"
    )
    print(tokenizer_out)
    print(dual_out)


if __name__ == "__main__":
    main()
