"""Bind training normalization to the current numeric LeRobot export."""
from pathlib import Path
import hashlib
import json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root):
    root = Path(root)
    paths = [root / 'meta/info.json', root / 'meta/tasks.parquet']
    paths += sorted((root / 'meta/episodes').rglob('*.parquet'))
    paths += sorted((root / 'data').rglob('*.parquet'))
    records = [dict(path=str(p.relative_to(root)), size=p.stat().st_size,
                    sha256=sha256_file(p)) for p in paths]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_summary(root, assets, horizon):
    root, assets = Path(root), Path(assets)
    info = json.loads((root / 'meta/info.json').read_text())
    return dict(dataset=str(root.resolve()), frames=info['total_frames'],
                episodes=info['total_episodes'], action_horizon=int(horizon), action_dim=54,
                method='full-parquet, exact forward-chunk weights, empirical quantiles',
                source_sha256=source_fingerprint(root),
                normalization_sha256=sha256_file(assets / 'norm_stats.json'))


def check_summary(root, assets, horizon):
    root, assets = Path(root), Path(assets)
    info = json.loads((root / 'meta/info.json').read_text())
    summary = json.loads((assets / 'dataset_summary.json').read_text())
    expected = dict(frames=info['total_frames'], episodes=info['total_episodes'],
                    action_horizon=int(horizon), action_dim=54)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f'Pi0.5 statistics {key} differs from current data; run process_data.sh')
    if summary.get('source_sha256') != source_fingerprint(root):
        raise ValueError('Pi0.5 statistics refer to another numeric dataset; run process_data.sh')
    if summary.get('normalization_sha256') != sha256_file(assets / 'norm_stats.json'):
        raise ValueError('Pi0.5 normalization file changed after preparation')
    return summary
