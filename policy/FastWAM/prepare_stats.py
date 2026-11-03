"""Compute DexBench normalization with FastWAM's unmodified native reducer.

LeRobot's EE54 statistics cannot normalize the model's column-rot6d EE58.
This numeric-only view converts the current train split and invokes the same
BaseLerobotDataset.get_dataset_stats used by native RobotVideoDataset. Each
episode has equal weight; action statistics include forward 32-step chunks
with final-row replication; per-episode variance uses PyTorch's correction=1.
No images, training jobs, checkpoints, or source dataset files are written.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.datasets.dexbench import DexBenchIndex
from fastwam.datasets.dexbench_contract import CONTRACT, pack54
from fastwam.datasets.lerobot.base_lerobot_dataset import (
    BaseLerobotDataset, sliding_window_with_replication,
)
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json

POLICY = Path(__file__).resolve().parent
ROOT = Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks')
ACTION_HORIZON = 32
METHOD = 'fastwam-native-episode-equal-forward32-sample-variance-v1'


class NativeNumericView:
    """Minimal native stats interface over already-converted physical EE58."""
    def __init__(self, episodes, action_horizon=ACTION_HORIZON):
        self.episodes = episodes
        self.action_horizon = int(action_horizon)
        self.multi_dataset = SimpleNamespace(
            num_episodes=len(episodes),
            num_frames=sum(len(state) for state, _ in episodes),
        )
        self.state_meta = [{'key': 'default'}]
        self.action_meta = [{'key': 'default'}]

    def _get_episode_data(self, episode_index):
        state, action = self.episodes[episode_index]
        if len(state) != len(action) or len(state) < 2:
            raise ValueError('Native sample variance requires >=2 aligned rows per episode')
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        return {
            'state': {'default': state.unsqueeze(1)},
            'action': {'default': sliding_window_with_replication(action, self.action_horizon)},
        }


class IdentityTransform:
    # The actual DexBench processor has action_state_transforms=None and no
    # delta transform; pack54 above already supplies its physical EE58 fields.
    @staticmethod
    def action_state_transform(batch):
        return batch


def native_statistics(episodes, action_horizon=ACTION_HORIZON):
    view = NativeNumericView(episodes, action_horizon)
    return BaseLerobotDataset.get_dataset_stats(view, IdentityTransform())


def load_training_episodes(root):
    index = DexBenchIndex(root, 'train', val_set_proportion=0.01)
    expected = {int(r.episode_index): int(r.length) for r in index.episodes.itertuples()}
    episodes = {}
    hashes = {}
    for path in sorted((Path(root) / 'data').rglob('*.parquet')):
        df = pq.read_table(path, columns=[
            'episode_index', 'frame_index', 'observation.state', 'action',
        ]).to_pandas()
        for episode_id, group in df.groupby('episode_index', sort=True):
            episode_id = int(episode_id)
            if episode_id not in expected:
                continue
            if episode_id in episodes:
                raise ValueError(f'Episode {episode_id} spans multiple files; refusing partial stats')
            group = group.sort_values('frame_index')
            if len(group) != expected[episode_id] or not np.array_equal(
                    group['frame_index'].to_numpy(), np.arange(len(group))):
                raise ValueError(f'Episode {episode_id} row order/length mismatch')
            state = pack54(np.stack(group['observation.state'].to_numpy()))
            action = pack54(np.stack(group['action'].to_numpy()))
            if state.shape != (len(group), 58) or action.shape != state.shape:
                raise ValueError(f'Episode {episode_id} is not aligned physical EE58')
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f'Non-finite EE58 in episode {episode_id}')
            episodes[episode_id] = (state, action)
            digest = hashlib.sha256()
            for values in (state, action):
                digest.update(np.ascontiguousarray(values, dtype='<f4').tobytes())
            hashes[episode_id] = digest.hexdigest()
    ordered_ids = [int(value) for value in index.episodes.episode_index]
    if set(episodes) != set(ordered_ids):
        raise ValueError('Missing training episodes')
    fingerprint = hashlib.sha256(json.dumps(
        [(value, expected[value], hashes[value]) for value in ordered_ids],
        separators=(',', ':'),
    ).encode()).hexdigest()
    return [episodes[value] for value in ordered_ids], ordered_ids, fingerprint


def main():
    torch.set_num_threads(1)
    episodes, ids, fingerprint = load_training_episodes(ROOT)
    print(f'Computing native statistics: {len(ids)} train episodes, '
          f'{sum(len(s) for s, _ in episodes)} physical rows', flush=True)
    output = native_statistics(episodes)
    rows = sum(len(state) for state, _ in episodes)
    reducer_source = inspect.getsource(BaseLerobotDataset.get_dataset_stats)
    output.update(
        action_contract=CONTRACT,
        train_rows=rows,
        train_episodes=len(ids),
        stats_population='native FastWAM: equal-weight training episodes; '
                         'state T=1; action T=32 forward/end-replicated; '
                         'per-episode variance correction=1; validation excluded',
        stats_provenance={
            'method': METHOD,
            'native_reducer': 'fastwam.datasets.lerobot.base_lerobot_dataset.'
                              'BaseLerobotDataset.get_dataset_stats',
            'native_reducer_sha256': hashlib.sha256(reducer_source.encode()).hexdigest(),
            'dataset': str(ROOT),
            'physical_layout': 'absolute-world-column-rot6d-EE58-Wuji20',
            'training_episode_ids': ids,
            'validation_split_seed': 42,
            'validation_proportion': 0.01,
            'train_numeric_sha256': fingerprint,
            'action_horizon': ACTION_HORIZON,
            'state_horizon_for_native_stats': 1,
            'variance_correction': 1,
            'normalization_mode': 'z-score',
            'use_stepwise_action_norm': False,
            'quantiles': 'per-episode/per-offset q01 and q99, native min/max '
                         'envelope across episodes/offsets; unused by z-score',
            'precision': 'native torch.float32, native parallel episode reduction',
        },
    )
    for section, horizon in [('state', 1), ('action', ACTION_HORIZON)]:
        for key, value in output[section]['default'].items():
            expected = (horizon, 58) if key.startswith('stepwise_') else (58,)
            if tuple(value.shape) != expected or not torch.isfinite(value).all():
                raise ValueError(f'Invalid native {section}.{key} statistics')
    target = POLICY / 'data/dataset_stats.json'
    temporary = target.with_name(target.name + '.tmp')
    target.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(output, str(temporary))
    os.replace(temporary, target)
    print(f'Saved {target}: {len(ids)} training episodes / {rows} rows; '
          f'{METHOD}; SHA256={fingerprint}', flush=True)


if __name__ == '__main__':
    main()
