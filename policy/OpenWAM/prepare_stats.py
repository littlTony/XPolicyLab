"""Prepare alpha raw58 stats from the existing LeRobot v3 numeric columns."""
from pathlib import Path
import json
import os
import numpy as np
import pyarrow.parquet as pq
from openwam.dataloader.dexbench_contract import CONTRACT, ROT6D_DIMS, pack54
from openwam.dataloader.dexbench_provenance import source_fingerprint

POLICY = Path(__file__).resolve().parent
DATA = Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks')


def stat(values):
    # Preserve the existing raw58 full-row weighting and rotation bounds.
    values = pack54(values).astype(np.float64)
    result = dict(mean=values.mean(axis=0), std=np.maximum(values.std(axis=0), 1e-6),
                  min=values.min(axis=0), max=values.max(axis=0))
    result['q01'], result['q99'] = np.quantile(values, [0.01, 0.99], axis=0)
    for lo, hi in [('min', 'max'), ('q01', 'q99')]:
        result[lo][ROT6D_DIMS] = -1
        result[hi][ROT6D_DIMS] = 1
    return {key: np.asarray(value, dtype=np.float32) for key, value in result.items()}


def main():
    fingerprint = source_fingerprint(DATA)
    info = json.loads((DATA / 'meta/info.json').read_text())
    tables = [pq.read_table(p, columns=['observation.state', 'action'])
              for p in sorted((DATA / 'data').rglob('*.parquet'))]
    arrays = {key: np.concatenate([np.asarray(t[key].to_pylist(), dtype=np.float32) for t in tables])
              for key in ['observation.state', 'action']}
    for key, values in arrays.items():
        if values.shape != (info['total_frames'], 54):
            raise ValueError(f'{key} frame/dimension mismatch')
    target = POLICY / 'data/meta'
    target.mkdir(parents=True, exist_ok=True)
    stats = dict(eef=stat(arrays['action']), eef_state=stat(arrays['observation.state']),
                 action_contract=CONTRACT, num_timesteps=info['total_frames'],
                 num_episodes=info['total_episodes'], source_fingerprint=fingerprint,
                 stats_recipe='full_rows_raw58_float64_moments_and_quantiles_rot6d_bounds_pm1_v1')
    if source_fingerprint(DATA)['sha256'] != fingerprint['sha256']:
        raise ValueError('Source changed while computing OpenWAM stats; output was not replaced')
    temporary = target / f'.normalization_stats.{os.getpid()}.tmp'
    try:
        with temporary.open('wb') as handle:
            np.save(handle, stats, allow_pickle=True)
        temporary.replace(target / 'normalization_stats.npy')
    finally:
        temporary.unlink(missing_ok=True)
    (target / 'action_contract.json').write_text(json.dumps(CONTRACT, indent=2) + '\n')
    print(f'Prepared full-data state/action raw58 statistics: {info["total_episodes"]} episodes / '
          f'{info["total_frames"]} frames -> {target}; SHA256 {fingerprint["sha256"]}', flush=True)


if __name__ == '__main__':
    main()
