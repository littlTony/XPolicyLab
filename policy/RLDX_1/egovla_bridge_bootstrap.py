#!/usr/bin/env python3
"""Pin the private flat XPolicyLab checkout before entering EgoVLA's bridge.

The benchmark bridge intentionally discovers ``XPolicyLab.policy.<name>``
through an import.  Its generic path helper inserts ``xpolicy_root.parent``
after the selected root, however, and a split checkout may have a sibling
``XPolicyLab`` symlink for another policy.  This tiny private entrypoint pins
the selected flat-module checkout in ``sys.modules`` before calling the
bridge.  No benchmark/common source is modified.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable


_POLICY_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _arg_value(argv: list[str], names: Iterable[str]) -> str | None:
    wanted = set(names)
    for index, value in enumerate(argv):
        if value in wanted and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _remove_bootstrap_args(argv: list[str]) -> list[str]:
    """Remove launcher-only flags before handing argv to bridge argparse."""

    removed: list[str] = []
    index = 0
    # ``--policy-name`` is also a real bridge option for ``env-client``.  Only
    # consume it when it appears before the bridge subcommand (where it can be
    # supplied as a bootstrap convenience); preserve the post-subcommand form.
    command_seen = False
    bridge_commands = {"doctor", "env-client", "policy", "tasks"}
    while index < len(argv):
        value = argv[index]
        if value in {"--check-only"}:
            index += 1
            continue
        if value in {"--policy-name", "--policy_name"} and not command_seen:
            index += 2
            continue
        removed.append(value)
        if value in bridge_commands:
            command_seen = True
        index += 1
    return removed


def _normalize_private_root_arg(argv: list[str], root: Path) -> list[str]:
    """Keep bridge ``--xpolicy-root`` bound to this private checkout.

    The Web evaluator's historical command line may still carry the shared
    benchmark checkout (for example ``/personal/wenwei/.../XPolicyLab``).
    ``egovla_xpolicy`` uses the explicit argument in preference to
    ``XPOLICYLAB_ROOT``; allowing that value through would make its lazy
    ``load_policy_deploy`` import search a different ``XPolicyLab`` package
    after the preflight has already pinned the private one.  The RLDX wrapper
    is intentionally private, so normalize both spellings to the already
    validated root before invoking the unmodified bridge.
    """

    expected = root.expanduser().resolve()
    normalized = str(expected)
    result = list(argv)
    for index, value in enumerate(result):
        if value not in {"--xpolicy-root", "--xpolicy_root"}:
            continue
        if index + 1 >= len(result):
            raise RuntimeError(f"{value} requires a path")
        requested_raw = result[index + 1]
        try:
            requested = Path(requested_raw).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid {value} path {requested_raw!r}"
            ) from exc
        if requested != expected:
            print(
                "[PRIVATE-BOOTSTRAP] overriding stale "
                f"{value}={requested} with private root {expected}",
                file=sys.stderr,
                flush=True,
            )
            result[index + 1] = normalized
    return result


def _private_root() -> Path:
    raw = (
        os.environ.get("RLDX_XPOLICYLAB_PACKAGE_ROOT")
        or os.environ.get("XPOLICYLAB_ROOT")
        or os.environ.get("EGOVLA_POLICY_ADAPTER_ROOT")
    )
    if not raw:
        raise RuntimeError(
            "RLDX_XPOLICYLAB_PACKAGE_ROOT/XPOLICYLAB_ROOT is required"
        )
    root = Path(raw).expanduser().resolve()
    # This adapter deliberately uses the flat XPolicyLab.py compatibility
    # module.  Accepting a guessed parent here would re-introduce shadowing.
    if not (root / "XPolicyLab.py").is_file():
        raise RuntimeError(f"private flat XPolicyLab.py is missing: {root}")
    if not (root / "policy").is_dir():
        raise RuntimeError(f"private policy directory is missing: {root / 'policy'}")
    return root


def _bridge_src() -> Path:
    raw = os.environ.get("EGOVLA_BRIDGE_ROOT")
    if raw:
        candidate = Path(raw).expanduser().resolve() / "src"
    else:
        workspace = os.environ.get("EGOVLA_WORKSPACE_ROOT")
        if not workspace:
            raise RuntimeError(
                "EGOVLA_BRIDGE_ROOT or EGOVLA_WORKSPACE_ROOT is required"
            )
        candidate = Path(workspace).expanduser().resolve() / "integration" / "src"
    if not (candidate / "egovla_xpolicy").is_dir():
        raise RuntimeError(f"EgoVLA bridge package is missing: {candidate}")
    return candidate


def _ordered_sys_path(root: Path, bridge_src: Path) -> list[str]:
    """Put private roots first and discard the implicit current-directory slot.

    Keep the same order advertised by the shell entrypoints: flat checkout,
    policy adapter, policy environment site-packages, upstream source, bridge,
    then any inherited/legacy paths.  This matters when a dependency ships a
    same-named module in the historical checkout.
    """

    # ``python -m`` and an absolute script both put the caller's directory (or
    # this script's directory) ahead of PYTHONPATH.  Keep those entries only
    # after the private root; the empty entry is especially dangerous because
    # the benchmark root contains a different XPolicyLab symlink.
    policy = os.environ.get("EGOVLA_POLICY_NAME", "RLDX_1")
    policy_root = root / "policy" / policy
    upstream_root = Path(
        os.environ.get("RLDX_UPSTREAM_ROOT", str(policy_root / "RLDX-1"))
    ).expanduser()
    site_packages = os.environ.get("RLDX_POLICY_SITE_PACKAGES") or os.environ.get(
        "RLDX_SITE_PACKAGES"
    )
    candidates = [str(root), str(policy_root)]
    if site_packages:
        candidates.append(site_packages)
    candidates.extend(
        [
            str(upstream_root),
            str(bridge_src),
            str(Path(__file__).resolve().parent),
            *sys.path,
        ]
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        # Conda's editable installs add synthetic path-hook entries such as
        # ``__editable__.omni_isaac_lab-*.finder.__path_hook__``.  They are
        # consumed by a meta-path finder, not filesystem paths; resolving
        # them through ``Path.resolve()`` turns the token into a bogus
        # ``/cwd/__editable__...`` path and makes ``omni.isaac`` disappear
        # after this private bootstrap rewrites ``sys.path``.  Preserve these
        # tokens verbatim while canonicalizing real paths as before.
        if item.startswith("__editable__."):
            normalized = item
        else:
            try:
                normalized = str(Path(item).expanduser().resolve())
            except (OSError, RuntimeError):
                normalized = item
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _purge_stale_modules() -> None:
    # A sitecustomize or an embedding launcher may have imported the generic
    # benchmark package before this script starts.  Remove only the two module
    # families that this private bridge must pin; all unrelated imports stay
    # untouched.
    for name in tuple(sys.modules):
        if name == "XPolicyLab" or name.startswith("XPolicyLab."):
            sys.modules.pop(name, None)
        elif name == "client_server" or name.startswith("client_server."):
            sys.modules.pop(name, None)


def _assert_origin(name: str, module: object, expected: Path) -> str:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"{name} has no file origin")
    actual = Path(raw).expanduser().resolve()
    expected = expected.expanduser().resolve()
    try:
        actual.relative_to(expected)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} resolved outside private checkout: {actual} (expected below {expected})"
        ) from exc
    return str(actual)


def _assert_path(name: str, actual: object, expected: Path) -> str:
    """Assert a package ``__path__`` is rooted in the selected checkout."""

    values = [Path(str(item)).expanduser().resolve() for item in (actual or [])]
    expected = expected.expanduser().resolve()
    if not values:
        raise RuntimeError(f"{name} has no package path")
    if any(value != expected and expected not in value.parents for value in values):
        raise RuntimeError(
            f"{name} package path escaped the private checkout: {values} "
            f"(expected below {expected})"
        )
    return ",".join(str(value) for value in values)


def _pin_private_imports(root: Path, policy: str, bridge_src: Path) -> None:
    if not _POLICY_RE.fullmatch(policy):
        raise RuntimeError(f"unsafe policy name: {policy!r}")
    policy_dir = (root / "policy" / policy).resolve()
    deploy_file = policy_dir / "deploy.py"
    if not policy_dir.is_dir() or not (policy_dir / "__init__.py").is_file():
        raise RuntimeError(f"private adapter package is missing: {policy_dir}")
    if not deploy_file.is_file():
        raise RuntimeError(f"private deploy module is missing: {deploy_file}")

    sys.path[:] = _ordered_sys_path(root, bridge_src)
    _purge_stale_modules()
    importlib.invalidate_caches()

    # Import all policy entrypoints explicitly.  In particular, importing only
    # ``XPolicyLab`` is insufficient for a flat module shim: a stale package
    # can still be cached under ``XPolicyLab.policy`` by the caller's cwd.
    # Import client_server.ws too so create_ws_model_client() cannot fall
    # through to another checkout.
    modules = {
        "XPolicyLab": importlib.import_module("XPolicyLab"),
        "XPolicyLab.policy": importlib.import_module("XPolicyLab.policy"),
        f"XPolicyLab.policy.{policy}": importlib.import_module(
            f"XPolicyLab.policy.{policy}"
        ),
        f"XPolicyLab.policy.{policy}.deploy": importlib.import_module(
            f"XPolicyLab.policy.{policy}.deploy"
        ),
        f"XPolicyLab.policy.{policy}.model": importlib.import_module(
            f"XPolicyLab.policy.{policy}.model"
        ),
        "client_server.ws": importlib.import_module("client_server.ws"),
        "egovla_xpolicy": importlib.import_module("egovla_xpolicy"),
    }
    xpolicy_origin = _assert_origin("XPolicyLab", modules["XPolicyLab"], root)
    xpolicy_path = _assert_path(
        "XPolicyLab", getattr(modules["XPolicyLab"], "__path__", []), root
    )
    _assert_origin("XPolicyLab.policy", modules["XPolicyLab.policy"], root / "policy")
    adapter_origin = _assert_origin(
        f"XPolicyLab.policy.{policy}",
        modules[f"XPolicyLab.policy.{policy}"],
        policy_dir,
    )
    deploy_origin = _assert_origin(
        f"XPolicyLab.policy.{policy}.deploy",
        modules[f"XPolicyLab.policy.{policy}.deploy"],
        policy_dir,
    )
    model_origin = _assert_origin(
        f"XPolicyLab.policy.{policy}.model",
        modules[f"XPolicyLab.policy.{policy}.model"],
        policy_dir,
    )
    client_origin = _assert_origin(
        "client_server.ws", modules["client_server.ws"], root / "client_server"
    )
    bridge_origin = _assert_origin(
        "egovla_xpolicy", modules["egovla_xpolicy"], bridge_src
    )

    # The policy environment owns the upstream RLDX package.  The evaluation
    # environment intentionally does not need to install its heavyweight
    # dependencies, so make this extra check opt-in for the server side.
    rldx_origin = "skipped (client environment)"
    if os.environ.get("RLDX_BOOTSTRAP_REQUIRE_RLDX", "0") == "1":
        rldx_root = Path(
            os.environ.get("RLDX_UPSTREAM_ROOT", str(root / "policy" / policy / "RLDX-1"))
        ).expanduser().resolve()
        rldx = importlib.import_module("rldx")
        rldx_origin = _assert_origin("rldx", rldx, rldx_root)

    print(
        "[PRIVATE-BOOTSTRAP] imports: "
        f"XPolicyLab={xpolicy_origin}; XPolicyLab.__path__={xpolicy_path}; "
        f"policy={getattr(modules['XPolicyLab.policy'], '__file__', '')}; "
        f"adapter={adapter_origin}; "
        f"deploy={deploy_origin}; "
        f"model={model_origin}; client_server.ws={client_origin}; "
        f"egovla_xpolicy={bridge_origin}; rldx={rldx_origin}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    policy = (
        _arg_value(args, ("--policy-name", "--policy_name"))
        or os.environ.get("EGOVLA_POLICY_NAME")
        or os.environ.get("RLDX_POLICY_NAME")
    )
    if not policy:
        raise RuntimeError("--policy-name/--policy_name (or EGOVLA_POLICY_NAME) is required")
    root = _private_root()
    bridge_src = _bridge_src()
    os.environ["XPOLICYLAB_ROOT"] = str(root)
    _pin_private_imports(root, policy, bridge_src)

    check_only = "--check-only" in args
    if check_only:
        return 0
    args = _remove_bootstrap_args(args)
    args = _normalize_private_root_arg(args, root)

    # Import only after the private modules are pinned.  The bridge remains
    # the unmodified benchmark implementation; this process is its private
    # compatibility boundary.
    bridge_cli = importlib.import_module("egovla_xpolicy.cli")
    return int(bridge_cli.main(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PRIVATE-BOOTSTRAP][ERROR] {exc}", file=sys.stderr)
        raise
