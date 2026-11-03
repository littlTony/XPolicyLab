"""Compute absolute state/action norm stats from prepared LeRobot v3 parquet.

No image decoding or dataset rewriting. Action statistics weight each frame by
its exact multiplicity in forward action chunks (last action pads an episode),
matching LeRobot delta_timestamps=[0,...,H-1]/fps without crossing episodes.
Quantiles use the weighted empirical distribution rather than histogram bins.
"""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi.policies.dexbench_contract import ACTION_CONTRACT
from openpi.shared import normalize
from openpi.training import config
from XPolicyLab.policy.Pi_05.stats_provenance import build_summary


def chunk_weights(length, horizon):
    counts = np.minimum(np.arange(length, dtype=np.int64) + 1, horizon)
    m = min(horizon, length)
    counts[-1] = m * horizon - m * (m - 1) // 2
    return counts


def statistics(values, weights=None):
    values = values.astype(np.float64)
    if weights is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    else:
        total = weights.sum()
        mean = np.average(values, axis=0, weights=weights)
        std = np.sqrt(np.average((values - mean)**2, axis=0, weights=weights))
        quantiles = np.empty((2, values.shape[1]))
        for col in range(values.shape[1]):
            order = np.argsort(values[:,col])
            ordered = values[order,col]
            cumulative = np.cumsum(weights[order])
            for j,q in enumerate([0.01,0.99]):
                position = (total-1)*q
                lo = np.searchsorted(cumulative, np.floor(position), side='right')
                hi = np.searchsorted(cumulative, np.ceil(position), side='right')
                quantiles[j,col] = ordered[lo] + (position % 1)*(ordered[hi]-ordered[lo])
        q01,q99 = quantiles
    return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def main():
    cfg = config.get_config('pi05_wuji_marvin_54d')
    root = Path(os.environ['HF_LEROBOT_HOME']) / 'dexbench_12tasks'
    tables = [pq.read_table(p, columns=['observation.state','action','episode_index'])
              for p in sorted((root/'data').glob('chunk-*/*.parquet'))]
    if not tables:
        raise ValueError('No LeRobot v3 parquet found')
    states = np.concatenate([np.asarray(t['observation.state'].to_pylist(),dtype=np.float32) for t in tables])
    actions = np.concatenate([np.asarray(t['action'].to_pylist(),dtype=np.float32) for t in tables])
    episodes = np.concatenate([t['episode_index'].to_numpy() for t in tables])
    info = json.loads((root/'meta/info.json').read_text())
    if states.shape != (info['total_frames'],54) or actions.shape != states.shape:
        raise ValueError('State/action frame dimensions differ from dataset metadata')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Dataset contains non-finite state/action values')
    weights = np.zeros(len(actions),dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(episodes[1:]!=episodes[:-1])+1, len(episodes)]
    if len(boundaries)-1 != info['total_episodes']:
        raise ValueError('Episode indices are not contiguous by episode')
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        weights[start:end] = chunk_weights(end-start,cfg.model.action_horizon)
    assert weights.sum() == len(actions)*cfg.model.action_horizon
    assets = cfg.assets_dirs / 'dexbench_12tasks'
    print(f'Computing 54D normalization: {len(states)} frames, {len(boundaries)-1} episodes',flush=True)
    norm_stats = dict(state=statistics(states),actions=statistics(actions,weights))
    normalize.save(assets,norm_stats)
    (assets/'action_contract.json').write_text(json.dumps(ACTION_CONTRACT,indent=2)+'\n')
    (assets/'dataset_summary.json').write_text(json.dumps(
        build_summary(root, assets, cfg.model.action_horizon), indent=2)+'\n')
    print(f'Normalization ready: {assets}',flush=True)


if __name__ == '__main__':
    main()
