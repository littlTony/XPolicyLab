"""One physical EE58 contract for new DexBench UniT/H-RDT/StarVLA adapters.

This describes stored absolute values, before model-specific normalization.
It deliberately does not read GR00T's separate relative-action statistics.
"""
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_DATASET = Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21')
CONTRACT_ID = 'dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1'
ROT6D = 'matrix_rows_0_then_1'
LAYOUT = (('left_eef', 0, 9), ('left_hand', 9, 29),
          ('right_eef', 29, 38), ('right_hand', 38, 58))
VIDEO_KEYS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
              'observation.images.cam_right_wrist')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(values, width):
    values = np.asarray(values, dtype=np.float32)
    _require(values.ndim >= 1 and values.shape[-1] == width and np.isfinite(values).all(),
             f'Expected finite last dimension {width}, got {values.shape}')
    return values


def pack_ee54(values):
    """[L xyz,wxyz,hand20,R xyz,wxyz,hand20] -> stored row-rotation EE58."""
    values = _finite(values, 54)
    pieces = []
    for start in (0, 27):
        q = values[..., start+3:start+7]
        _require(np.all(np.linalg.norm(q, axis=-1) > 1e-8), 'Zero input quaternion')
        matrix = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix()
        matrix = matrix.reshape(*q.shape[:-1], 3, 3)
        pieces += [values[..., start:start+3], matrix[..., 0, :], matrix[..., 1, :],
                   values[..., start+7:start+27]]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def unpack_ee58(values):
    """Project predicted row6D to SO(3), returning environment xyz,wxyz,hand20."""
    values = _finite(values, 58)
    pieces = []
    for start in (0, 29):
        a = values[..., start+3:start+6].astype(np.float64)
        b = values[..., start+6:start+9].astype(np.float64)
        norm = np.linalg.norm(a, axis=-1, keepdims=True)
        _require(np.all(norm > 1e-8), 'Degenerate first predicted rotation row')
        a = a / norm
        b = b - np.sum(a*b, axis=-1, keepdims=True)*a
        norm = np.linalg.norm(b, axis=-1, keepdims=True)
        _require(np.all(norm > 1e-8), 'Degenerate second predicted rotation row')
        b = b / norm
        matrix = np.stack([a, b, np.cross(a, b)], axis=-2)
        q = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat()
        q = q[:, [3, 0, 1, 2]].reshape(*matrix.shape[:-2], 4)
        q = np.where(q[..., :1] < 0, -q, q)
        pieces += [values[..., start:start+3], q, values[..., start+9:start+29]]
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def physical_task(instruction):
    match = re.match(r'^\[([^/\]]+)/([^\]]+)\]\s+', instruction)
    _require(match is not None, f'Missing source-embodiment/task prompt prefix: {instruction!r}')
    return match.group(2)


def validate_dataset(root=DEFAULT_DATASET):
    """Read metadata only. Return raw JSON dicts/lists plus physical task groups."""
    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    dims = get_robot_action_dim_info('tianji_marvin_wuji')
    _require(dims['ee_dim'] == [20, 20] and len(dims['arm_dim']) == 2,
             f'Expected dual Wuji20 metadata, got {dims}')
    root = Path(root).resolve(strict=True)
    meta = root/'meta'
    read = lambda name: json.loads((meta/name).read_text())
    info, contract, stats = read('info.json'), read('action_contract.json'), read('stats.json')
    offsets, modality = read('video_offsets.json'), read('modality.json')
    episodes = [json.loads(line) for line in (meta/'episodes.jsonl').read_text().splitlines() if line.strip()]
    tasks = [json.loads(line) for line in (meta/'tasks.jsonl').read_text().splitlines() if line.strip()]
    _require(contract['id'] == CONTRACT_ID and contract['rot6d'] == ROT6D, 'Wrong EE58 rotation contract')
    _require(contract['state'] == contract['action'] == 'absolute_world_xyz_rot6d_hand20', 'Expected absolute state/action')
    _require(contract['layout'] == [key+('9' if 'eef' in key else '20') for key, _, _ in LAYOUT], 'Wrong EE58 ordering')
    _require(contract['action_alignment'] == 'export_action_t_no_extra_shift' and contract['RGB'] is True,
             'Wrong action alignment or RGB contract')
    _require(info['codebase_version'] == 'v2.1' and info['total_episodes'] == 3000, 'Expected DexBench3000 v2.1 export')
    _require(info['splits'] == {'train': '0:3000'}, 'Unexpected dataset split')
    _require(len(episodes) == 3000 and sorted(row['episode_index'] for row in episodes) == list(range(3000)),
             'Missing or duplicate episodes')
    _require(sum(row['length'] for row in episodes) == info['total_frames'] == 751348,
             'Frame count changed; re-audit the source export')
    _require(len(tasks) == info['total_tasks'] == 60, 'Expected 60 source-embodiment/task prompts')
    physical_task_groups = {}
    for row in episodes:
        _require(len(row['tasks']) == 1, 'Expected one instruction per episode')
        task = physical_task(row['tasks'][0])
        physical_task_groups.setdefault(task, []).append(row['episode_index'])
        offset = offsets[str(row['episode_index'])]
        _require(set(offset) == set(VIDEO_KEYS), 'Missing per-camera video offset')
        _require(all(isinstance(n, int) and n >= 0 for n in offset.values()), 'Invalid frame offset')
    _require(len(physical_task_groups) == 12, 'Expected 12 physical tasks; 60 prompts are not 60 tasks')
    for group, original in (('state', 'observation.state'), ('action', 'action')):
        _require(info['features'][original]['shape'] == [58], f'Wrong {group} width')
        _require(list(modality[group]) == [key for key, _, _ in LAYOUT], f'Wrong {group} modality order')
        for key, start, end in LAYOUT:
            item = modality[group][key]
            _require((item['start'], item['end'], item['absolute'], item['original_key']) ==
                     (start, end, True, original), f'Wrong {group}.{key} mapping')
        for name in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
            _finite(stats[original][name], 58)
        _require(np.all(np.array(stats[original]['q99']) >= stats[original]['q01']), f'Invalid {group} quantiles')
    return dict(root=root, info=info, contract=contract, stats=stats, offsets=offsets,
                episodes=episodes, tasks=tasks, modality=modality, physical_task_groups=physical_task_groups)
