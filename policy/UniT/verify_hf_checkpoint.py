#!/usr/bin/env python3
"""Verify that a Hugging Face Trainer checkpoint is complete enough to reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty file: {path}")


def validate_weights(checkpoint: Path) -> list[Path]:
    direct = [
        checkpoint / "model.safetensors",
        checkpoint / "pytorch_model.bin",
    ]
    for path in direct:
        if path.is_file() and path.stat().st_size > 0:
            return [path]

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"empty weight_map in {index_path}")
        shard_names = sorted({str(name) for name in weight_map.values()})
        shards: list[Path] = []
        for name in shard_names:
            path = (checkpoint / name).resolve()
            try:
                path.relative_to(checkpoint.resolve())
            except ValueError as exc:
                raise ValueError(f"weight shard escapes checkpoint: {name}") from exc
            require_nonempty(path)
            shards.append(path)
        return [index_path, *shards]

    raise ValueError(
        f"{checkpoint} has no non-empty model.safetensors/pytorch_model.bin "
        "or complete sharded-weight index"
    )


def validate_checkpoint(checkpoint: Path, expected_step: int) -> list[Path]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")

    config_path = checkpoint / "config.json"
    state_path = checkpoint / "trainer_state.json"
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    load_json(config_path)
    state = load_json(state_path)
    load_json(metadata_path)
    try:
        actual_step = int(state["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid global_step in {state_path}") from exc
    if actual_step != expected_step:
        raise ValueError(
            f"{state_path}: global_step={actual_step}, expected {expected_step}"
        )
    weights = validate_weights(checkpoint)
    return [config_path, state_path, metadata_path, *weights]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        files = validate_checkpoint(args.checkpoint, args.expected_step)
    except ValueError as exc:
        if not args.quiet:
            print(f"checkpoint incomplete: {exc}")
        return 1
    if not args.quiet:
        print(
            f"checkpoint complete: {args.checkpoint.resolve()} "
            f"(step={args.expected_step}, verified_files={len(files)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
