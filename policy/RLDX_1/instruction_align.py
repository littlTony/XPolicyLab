"""Align SparkArena RLDX inference language, color, and action with training.

The 0908 SparkArena 7-task run stored directory slugs in ``tasks.jsonl``
(``hammer_beat``, …) and read videos with OpenCV (BGR). DexBench eval
feeds full English ``gen_instruction()`` sentences and RGB frames.
EgoVLA / old Spark0 checkpoints that trained on full English + RGB stay
on ``as_is`` / no channel swap.

GitHub RLDX training uses arm-relative / hand-absolute actions. Relarm
SparkArena weights must register that contract so ``decode_action`` adds
the current arm qpos back before SparkArena executes the chunk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SPARKARENA_SHORT_NAMES = (
    "click_mouse",
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "put_food_in_microwave",
    "retrieve_gap",
    "stack_bowls",
)

# Official DexBench eval sentences for the SparkArena 7 tasks.
DEXBENCH_INSTRUCTION_TO_SHORT_NAME = {
    "pick up the hammer and beat the cube with the hammer head.": "hammer_beat",
    "pick up both bottles on the table.": "dual_bottles_pick",
    "stack the three bowls together.": "stack_bowls",
    "put all the objects into the basket.": "collect_objects",
    "move the two cubes aside upright, then place the garage on the cushion.": "retrieve_gap",
    "place the mouse on the mouse pad, then click the left button.": "click_mouse",
    "put the sandwich in the microwave, then close the door.": "put_food_in_microwave",
}

_WS = re.compile(r"\s+")


def normalize_instruction(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).casefold()


def _is_sparkarena_checkpoint(checkpoint_path: str) -> bool:
    return "sparkarena" in str(checkpoint_path or "").casefold()


def instruction_style_from_checkpoint(
    checkpoint_path: str,
    explicit: str | None = None,
) -> str:
    """``short_name`` for SparkArena 7-task weights; otherwise leave text alone."""

    value = (
        explicit
        if explicit is not None
        else os.environ.get("RLDX_INSTRUCTION_STYLE", "")
    )
    value = str(value or "").strip().casefold()
    if value in {"short_name", "short", "task_name"}:
        return "short_name"
    if value in {"as_is", "full", "off", "none"}:
        return "as_is"
    if _is_sparkarena_checkpoint(checkpoint_path):
        return "short_name"
    return "as_is"


def action_contract_from_checkpoint(
    checkpoint_path: str,
    explicit: str | None = None,
) -> str:
    """Return the relative, absolute, or non-Spark action contract."""

    value = explicit if explicit is not None else os.environ.get("RLDX_ACTION_CONTRACT", "")
    value = str(value or "").strip().casefold()
    if value in {"relative", "relarm", "rel"}:
        return "relative"
    if value in {"absolute", "abs", "legacy"}:
        return "absolute"
    if value in {"none", "off", "egovla"}:
        return "none"

    text = str(checkpoint_path or "").casefold()
    if "relarm" in text or "relative" in text:
        return "relative"
    for candidate in _checkpoint_config_candidates(checkpoint_path):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("use_relative_action"):
            return "relative"
        break
    if "sparkarena" in text or "joint54" in text:
        return "absolute"
    return "none"


def _checkpoint_config_candidates(checkpoint_path: str) -> list[Path]:
    root = Path(str(checkpoint_path or "")).expanduser()
    candidates = [root / "config.json"]
    if root.name.startswith("checkpoint-"):
        return candidates
    numbered = []
    if root.is_dir():
        for child in root.glob("checkpoint-*"):
            if not child.is_dir():
                continue
            try:
                numbered.append((int(child.name.rsplit("-", 1)[1]), child / "config.json"))
            except ValueError:
                continue
    if numbered:
        candidates.insert(0, max(numbered)[1])
    return candidates


def swap_rgb_bgr_enabled(
    explicit: bool | None = None,
    *,
    checkpoint_path: str = "",
) -> bool:
    """Swap server RGB to OpenCV BGR for SparkArena A800 opencv training."""

    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("RLDX_INFER_SWAP_RGB_BGR", "").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return _is_sparkarena_checkpoint(checkpoint_path)


def align_instruction(
    text: str,
    *,
    style: str,
    task_name: str | None = None,
) -> str:
    raw = str(text or "").strip()
    fallback = str(task_name or "").strip()
    if style != "short_name":
        return raw
    if raw in SPARKARENA_SHORT_NAMES:
        return raw
    mapped = DEXBENCH_INSTRUCTION_TO_SHORT_NAME.get(normalize_instruction(raw))
    if mapped:
        return mapped
    if fallback in SPARKARENA_SHORT_NAMES:
        return fallback
    return raw
