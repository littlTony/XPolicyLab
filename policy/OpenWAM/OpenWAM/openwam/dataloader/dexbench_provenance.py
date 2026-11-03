"""Bind physical raw58 statistics to the exact LeRobot numeric source.

The fingerprint definition matches data_preprocess/prepare_native_ee58.py.
Video bytes are excluded: normalization is computed from numeric parquet data.
Only the standard library is imported, so guard callers do not load models.
"""
from pathlib import Path
import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(source):
    source = Path(source)
    paths = [source / "meta/info.json", source / "meta/tasks.parquet"]
    paths += sorted((source / "meta/episodes").rglob("*.parquet"))
    paths += sorted((source / "data").rglob("*.parquet"))
    records = [dict(path=str(p.relative_to(source)), size=p.stat().st_size, sha256=sha256(p)) for p in paths]
    value = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(source_root=str(source.resolve()), sha256=value, files=records)


def validate_stats_source(stats, source, info=None):
    """Reject stale training stats, including changes that preserve row counts.

    Deployment intentionally does not call this guard: a checkpoint must retain
    its own training-time stats regardless of the current dataset on disk.
    """
    source = Path(source)
    if info is None:
        info = json.loads((source / "meta/info.json").read_text())
    if stats.get("num_timesteps") != info["total_frames"]:
        raise ValueError("Stale OpenWAM stats: source frame count changed; run process_data.sh")
    if stats.get("num_episodes") != info["total_episodes"]:
        raise ValueError("Stale OpenWAM stats: source episode count changed; run process_data.sh")
    saved = stats.get("source_fingerprint", {})
    if not isinstance(saved, dict) or not saved.get("sha256"):
        raise ValueError("OpenWAM stats lack source SHA provenance; run process_data.sh")
    current = source_fingerprint(source)
    if saved["sha256"] != current["sha256"]:
        raise ValueError("Stale OpenWAM stats: source SHA changed; run process_data.sh")
    return current
