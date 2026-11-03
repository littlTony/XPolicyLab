"""Small LeRobot v3 index; no model, HDF5 or image-codec dependencies."""
from collections import OrderedDict
from pathlib import Path
import bisect
import hashlib
import json
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from vitra.datasets.lerobot_video_io import decode_video_frames

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def metadata_signature(root):
    root = Path(root)
    digest = hashlib.sha256()
    paths = [root/'meta/info.json', root/'meta/tasks.parquet', root/'meta/preview_report.json',
             *sorted((root/'meta/episodes').rglob('*.parquet'))]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class LeRobotV3Index:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root/'meta/info.json').read_text())
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('VITRA direct reader requires LeRobot v3.0')
        for key in ['observation.state', 'action']:
            if self.info['features'][key]['shape'] != [54]:
                raise ValueError(f'{key} must be absolute xyz+wxyz/Wuji20 EE54')
        columns = ['episode_index', 'length', 'tasks', 'data/chunk_index', 'data/file_index']
        columns += [f'videos/observation.images.{cam}/{key}' for cam in CAMERAS
                    for key in ['chunk_index', 'file_index', 'from_timestamp']]
        self.episodes = pd.concat([pq.read_table(f, columns=columns).to_pandas()
            for f in sorted((self.root/'meta/episodes').rglob('*.parquet'))], ignore_index=True)
        self.episodes = self.episodes.sort_values('episode_index').reset_index(drop=True)
        ids = self.episodes.episode_index.to_numpy()
        if not np.array_equal(ids, np.arange(len(ids))):
            raise ValueError('Episode IDs must be contiguous and unique')
        self.lengths = self.episodes.length.to_numpy(dtype=np.int64)
        self.cumulative = np.cumsum(self.lengths)
        if len(ids) != self.info['total_episodes'] or int(self.cumulative[-1]) != self.info['total_frames']:
            raise ValueError('LeRobot episode metadata/count mismatch')
        self._pid = None
        self._files = OrderedDict()
        self._numeric = OrderedDict()
        self.expected_numeric_sha256 = {}
        self._verified_files = set()

    def __len__(self):
        return int(self.cumulative[-1])

    def locate(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError('Sample index must be an integer')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.cumulative, index)
        return episode, int(index - (self.cumulative[episode-1] if episode else 0))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _files=OrderedDict(), _numeric=OrderedDict(), _verified_files=set())
        return state

    def numeric(self, episode):
        if self._pid != os.getpid():
            self._files.clear(); self._numeric.clear(); self._verified_files.clear(); self._pid = os.getpid()
        if episode in self._numeric:
            value = self._numeric.pop(episode); self._numeric[episode] = value
            return value
        row = self.episodes.iloc[episode]
        path = self.root/self.info['data_path'].format(chunk_index=int(row['data/chunk_index']),
                                                      file_index=int(row['data/file_index']))
        if path not in self._files:
            expected = self.expected_numeric_sha256.get(str(path.relative_to(self.root)))
            if expected and path not in self._verified_files:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'LeRobot numeric content changed: {path}; regenerate VITRA stats')
                self._verified_files.add(path)
            self._files[path] = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state', 'action']).to_pandas()
            if len(self._files) > 2:
                self._files.popitem(last=False)
        df = self._files[path]
        df = df[df.episode_index == episode].sort_values('frame_index')
        n = int(row['length'])
        if len(df) != n or not np.array_equal(df.frame_index, np.arange(n)):
            raise ValueError(f'Invalid numeric rows for episode {episode}')
        value = tuple(np.stack(df[k]).astype(np.float32) for k in ['observation.state', 'action'])
        if not all(np.isfinite(x).all() and x.shape == (n,54) for x in value):
            raise ValueError(f'Invalid EE54 values for episode {episode}')
        self._numeric[episode] = value
        if len(self._numeric) > 8:
            self._numeric.popitem(last=False)
        return value

    def images(self, episode, frames, camera):
        row = self.episodes.iloc[episode]
        key = 'observation.images.' + camera
        path = self.root/self.info['video_path'].format(video_key=key,
            chunk_index=int(row[f'videos/{key}/chunk_index']), file_index=int(row[f'videos/{key}/file_index']))
        offset = int(round(float(row[f'videos/{key}/from_timestamp'])*self.info['fps']))
        height, width = self.info['features'][key]['shape'][-2:]
        return np.stack([np.asarray(image) for image in decode_video_frames(str(path),
            [int(frame)+offset for frame in frames], height, width)])
