"""Private runtime helpers for the RLDX_1 EgoVLA adapter.

The public XPolicyLab checkout normally resolves robot dimensions through
``env_cfg/robot/_robot_info.json``.  On the split A800 evaluation host that
file is a symlink into ``/personal`` and is therefore unavailable.  Keep a
small, policy-local fallback so evaluation never has to mutate the shared
checkout or depend on the unavailable mount.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


POLICY_DIR = Path(__file__).resolve().parent
XPL_ROOT = POLICY_DIR.parents[1]
PRIVATE_ROBOT_INFO = XPL_ROOT / "utils" / "robot" / "_robot_info.json"
_ENV_CFG_ROOT = XPL_ROOT.parent / "env_cfg"


def _load_private_registry() -> dict[str, dict[str, list[int]]]:
    """Load and validate the registry shipped with this private adapter."""

    try:
        raw = json.loads(PRIVATE_ROBOT_INFO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid private robot registry: {PRIVATE_ROBOT_INFO}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"private robot registry must be an object: {PRIVATE_ROBOT_INFO}")

    registry: dict[str, dict[str, list[int]]] = {}
    for robot_name, info in raw.items():
        if not isinstance(robot_name, str) or not isinstance(info, dict):
            raise ValueError(f"invalid robot entry in {PRIVATE_ROBOT_INFO}: {robot_name!r}")
        normalized: dict[str, list[int]] = {}
        for key in ("arm_dim", "ee_dim"):
            values = info.get(key)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(
                    f"robot {robot_name!r} must define a non-empty {key} list"
                )
            try:
                dims = [int(value) for value in values]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"robot {robot_name!r} has invalid {key}: {values!r}") from exc
            if any(value <= 0 for value in dims):
                raise ValueError(f"robot {robot_name!r} has non-positive {key}: {dims!r}")
            normalized[key] = dims
        if len(normalized["arm_dim"]) != len(normalized["ee_dim"]):
            raise ValueError(f"robot {robot_name!r} arm/ee counts do not match")
        registry[robot_name] = normalized
    return registry


def _env_robot_name(env_cfg_type: str) -> str | None:
    """Read the robot name from a local env config when the type is an alias."""

    cfg_path = _ENV_CFG_ROOT / f"{env_cfg_type}.yml"
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return None

    # The env files only need a scalar ``config: robot: ...`` value here.  A
    # regex keeps this fallback independent of PyYAML (the policy bare Python
    # intentionally has only the shared site-packages on PYTHONPATH).
    match = re.search(r"(?m)^\s*robot\s*:\s*([^#\s]+)", text)
    return match.group(1).strip() if match else None


def get_robot_action_dim_info(env_cfg_type: str) -> dict[str, list[int]]:
    """Return dual-arm action dimensions from the private registry.

    ``env_cfg_type`` is usually also the robot name (``ego_h1_inspire``), but
    aliases such as ``tianji_marvin_wuji`` may point to another robot in an
    env YAML.  A fresh dictionary is returned so callers cannot mutate the
    registry cache or affect another evaluation process.

    Raises:
        FileNotFoundError: if the private registry is missing.
        KeyError: if neither the env type nor its configured robot is present.
    """

    env_cfg_type = str(env_cfg_type).strip()
    if not env_cfg_type:
        raise KeyError("empty env_cfg_type")
    registry = _load_private_registry()
    candidates = [env_cfg_type]
    robot_name = _env_robot_name(env_cfg_type)
    if robot_name and robot_name not in candidates:
        candidates.append(robot_name)
    # Be forgiving of case-only aliases while retaining the canonical key in
    # the returned error message.
    lowered = {name.lower(): name for name in registry}
    for candidate in candidates:
        info = registry.get(candidate)
        if info is None:
            canonical = lowered.get(candidate.lower())
            info = registry.get(canonical) if canonical else None
        if info is not None:
            return {"arm_dim": list(info["arm_dim"]), "ee_dim": list(info["ee_dim"])}

    available = ", ".join(sorted(registry)) or "(empty)"
    raise KeyError(
        f"{env_cfg_type!r} is not present in private robot registry "
        f"{PRIVATE_ROBOT_INFO}; available: {available}"
    )


def resolve_python_executable(*references: str | Path | None) -> Path | None:
    """Resolve an executable Python from a file, env directory, or uv dir."""

    candidates: list[Path] = []
    for reference in references:
        if not reference:
            continue
        path = Path(str(reference)).expanduser()
        candidates.extend(
            [
                path,
                path / "bin" / "python",
                path / "bin" / "python3",
                path / ".venv" / "bin" / "python",
                path / ".venv" / "bin" / "python3",
            ]
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None


def as_jsonable_dims(info: dict[str, Any]) -> dict[str, list[int]]:
    """Normalize a registry result for diagnostics and shell-side callers."""

    return {
        "arm_dim": [int(value) for value in info["arm_dim"]],
        "ee_dim": [int(value) for value in info["ee_dim"]],
    }
