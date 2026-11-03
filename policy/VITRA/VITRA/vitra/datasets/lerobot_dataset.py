"""Direct LeRobot-v3 storage adapter; preserve the native VITRA Wuji codec."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import re
import numpy as np
from scipy.spatial.transform import Rotation
from vitra.datasets.lerobot_io import LeRobotV3Index, metadata_signature
from vitra.datasets.spark0_dataset import (
    RoboDatasetCore, DEFAULT_MAPPING_PATH, load_wuji_mapping, pose7_to_matrix,
    matrix_to_pose6, raw_camera_pose_to_vitra_matrix, intrinsics_to_matrix,
    format_vitra_instruction, calculate_fov, _validate_statistics_contract,
    read_dataset_statistics,
)


COMMAND_STEP_CONTRACT_ID='dexbench-vitra-wuji20-sparse-camera-command-step-v2'
COMMAND_STEP_HORIZON=16


def action_slot_count(length,horizon=COMMAND_STEP_HORIZON):
    return horizon*length-horizon*(horizon-1)//2 if length>=horizon else length*(length+1)//2


def native_vectors(state, action, camera_raw, mapping, include_state=True):
    """Camera state and one transition, before normalization.

    ``state`` here means the transition's previous pose. For command chunks,
    only the first transition starts from measured state; later transitions
    start from the previous absolute command target.
    """
    inverse = np.linalg.inv(raw_camera_pose_to_vitra_matrix(camera_raw))
    states, actions = [], []
    for side, start in [('left',0), ('right',27)]:
        calibration = np.asarray(mapping['calibration'][side+'_ee_to_vitra_wrist'])
        current = inverse @ pose7_to_matrix(state[...,start:start+7]) @ calibration
        target = inverse @ pose7_to_matrix(action[...,start:start+7]) @ calibration
        delta = target[...,:3,:3] @ np.swapaxes(current[...,:3,:3],-1,-2)
        shape = delta.shape[:-2]
        euler = Rotation.from_matrix(delta.reshape(-1,3,3)).as_euler('xyz').reshape(*shape,3)
        if include_state:states.append(np.concatenate([matrix_to_pose6(current),state[...,start+7:start+27]],axis=-1))
        actions.append(np.concatenate([target[...,:3,3]-current[...,:3,3],euler,action[...,start+7:start+27]],axis=-1))
    return (np.concatenate(states,axis=-1).astype(np.float32) if include_state else None), np.concatenate(actions,axis=-1).astype(np.float32)


def native_chunk_actions(current_state,targets,camera_raw,mapping):
    """Encode one ordered command trajectory in a single camera anchor."""
    targets=np.asarray(targets,dtype=np.float32)
    if not len(targets):return np.empty((0,52),np.float32)
    previous=np.concatenate([np.asarray(current_state)[None],targets[:-1]],axis=0)
    return native_vectors(previous,targets,camera_raw,mapping,include_state=False)[1]


def native_episode_values(state,action,cameras,mapping,horizon=COMMAND_STEP_HORIZON):
    """Exactly the state rows and valid command slots sampled by the loader.

    Action slots are ordered by sample anchor, then horizon position. Masked
    terminal padding never enters the Gaussian moments. Every slot uses its
    sample's C0 even if the head camera moves within an episode.
    """
    n=len(state);state_values=native_vectors(state,action,cameras,mapping)[0]
    grid=np.arange(n)[:,None]+np.arange(horizon)[None]
    valid=grid<n
    anchors=np.broadcast_to(np.arange(n)[:,None],grid.shape)[valid]
    targets=grid[valid]
    previous=action[np.maximum(targets-1,0)].copy()
    first=targets==anchors
    previous[first]=state[anchors[first]]
    values=native_vectors(previous,action[targets],np.asarray(cameras)[anchors],mapping,include_state=False)[1]
    assert len(values)==action_slot_count(n,horizon)
    return state_values,values


class LeRobotDatasetCore(RoboDatasetCore):
    # Reuse only the original native normalization/mapping functions, never HDF5 IO.
    def __init__(self, root_dir, statistics_path=None, action_past_window_size=0,
                 action_future_window_size=15, image_past_window_size=0,
                 image_future_window_size=0, load_images=True, mapping_path=None,
                 representation='wuji20', **kwargs):
        if representation != 'wuji20':
            raise ValueError('The EE54 LeRobot export contains Wuji20, not fitted MANO45/Inspire12')
        self.root = str(Path(root_dir).resolve())
        self.index = LeRobotV3Index(self.root)
        self.representation = representation
        self.state_hand_dim = self.action_hand_dim = 26
        self.state_dual_dim = self.action_dual_dim = 52
        self.mapping_path = str(Path(mapping_path or DEFAULT_MAPPING_PATH).resolve())
        self.mapping = load_wuji_mapping(self.mapping_path)
        self.action_past_window_size = int(action_past_window_size)
        self.action_future_window_size = int(action_future_window_size)
        self.image_past_window_size = int(image_past_window_size)
        self.image_future_window_size = int(image_future_window_size)
        if self.action_past_window_size!=0 or self.action_future_window_size!=COMMAND_STEP_HORIZON-1:
            raise ValueError('VITRA command-step-v2 stats require zero action past and a 16-step future chunk')
        if self.image_past_window_size or self.image_future_window_size:
            raise ValueError('VITRA three-view recipe uses current images only; temporal image windows are unsupported')
        if min(self.action_past_window_size,self.action_future_window_size,
               self.image_past_window_size,self.image_future_window_size) < 0:
            raise ValueError('Window sizes must be non-negative')
        self.load_images = bool(load_images)
        self.metadata_dir = Path(self.root)/'meta/vitra'
        manifest = json.loads((self.metadata_dir/'manifest.json').read_text())
        if manifest['action_contract']['id']!=COMMAND_STEP_CONTRACT_ID:
            raise ValueError('Stale VITRA step codec; regenerate command-step-v2 statistics')
        if manifest['lerobot_metadata_sha256'] != metadata_signature(self.root):
            raise ValueError('LeRobot metadata changed; rerun VITRA process_data.sh')
        if manifest['mapping_sha256'] != hashlib.sha256(Path(self.mapping_path).read_bytes()).hexdigest():
            raise ValueError('Wuji mapping changed; regenerate VITRA native stats')
        for record in manifest['numeric_files']:
            path = Path(self.root)/record['path']; info = path.stat()
            if info.st_size != record['size']:
                raise ValueError(f'LeRobot numeric file changed: {path}; regenerate VITRA stats')
        self.index.expected_numeric_sha256 = {r['path']:r['sha256'] for r in manifest['numeric_files']}
        self.records = manifest['episodes']
        if len(self.records) != len(self.index.episodes) or manifest['frame_count'] != len(self.index):
            raise ValueError('VITRA calibration/LeRobot coverage mismatch')
        self.episode_lengths = self.index.lengths.tolist()
        self._cumulative_lengths = self.index.cumulative.tolist()
        self.num_valid_frames = len(self.index)
        self.episode_paths = [f'lerobot://episode/{i}' for i in range(len(self.records))]
        self.data_statistics = read_dataset_statistics(str(statistics_path or self.metadata_dir/'normalization_stats.json'))
        _validate_statistics_contract(str(statistics_path or self.metadata_dir/'normalization_stats.json'),
                                      representation,self.data_statistics)
        stats = json.loads(Path(statistics_path or self.metadata_dir/'normalization_stats.json').read_text())
        if stats['action_contract']['id']!=COMMAND_STEP_CONTRACT_ID:
            raise ValueError('VITRA statistics use the old measured-state delta codec')
        if stats.get('num_action_samples')!=sum(action_slot_count(n) for n in self.episode_lengths):
            raise ValueError('VITRA statistics do not cover the actual valid command-window slots')
        if stats['source_metadata_sha256'] != manifest['lerobot_metadata_sha256']:
            raise ValueError('VITRA stats belong to different LeRobot metadata')
        if stats['action_contract']['mapping_sha256'] != manifest['mapping_sha256']:
            raise ValueError('VITRA stats use another hand mapping')
        if stats['source_calibration_sha256'] != manifest['calibration_sha256']:
            raise ValueError('VITRA stats use another camera calibration')
        if hashlib.sha256(Path(statistics_path or self.metadata_dir/'normalization_stats.json').read_bytes()).hexdigest() != manifest['statistics_sha256']:
            raise ValueError('VITRA statistics checksum mismatch')
        self.global_data_statistics = None
        self._calibration_cache = OrderedDict()
        self._calibration_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_calibration_cache=OrderedDict(),_calibration_pid=None)
        return state

    def calibration(self, episode):
        if self._calibration_pid != os.getpid():
            self._calibration_cache.clear(); self._calibration_pid = os.getpid()
        if episode not in self._calibration_cache:
            path = self.metadata_dir/self.records[episode]['calibration']
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.records[episode]['calibration_sha256']:
                raise ValueError(f'Camera calibration checksum mismatch: {path}')
            with np.load(path,allow_pickle=False) as data:
                value = {key:data[key] for key in data.files}
            n = self.episode_lengths[episode]
            for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
                if value[cam+'_extrinsics'].shape != (n,4,4) or value[cam+'_intrinsics'].shape != (n,3,3):
                    raise ValueError(f'Invalid calibration shape in {path}')
            self._calibration_cache[episode] = value
            if len(self._calibration_cache)>8:
                self._calibration_cache.popitem(last=False)
        return self._calibration_cache[episode]

    def __getitem__(self, index):
        episode, frame = self.index.locate(index)
        state, action = self.index.numeric(episode)
        n = len(state); calibration = self.calibration(episode)
        anchor = calibration['cam_head_extrinsics'][frame]
        current_state,_ = native_vectors(state[frame],action[frame],anchor,self.mapping)
        indices = np.arange(frame-self.action_past_window_size,frame+self.action_future_window_size+1)
        valid = (indices>=0)&(indices<n)
        actions = np.zeros((len(indices),52),np.float32)
        if valid.any():
            actions[valid] = native_chunk_actions(state[frame],action[indices[valid]],anchor,self.mapping)
        intrinsics = intrinsics_to_matrix(calibration['cam_head_intrinsics'][frame])
        record = self.records[episode]
        # Exported task prefixes identify robot/task for other policies. VITRA's
        # native prompt uses the plain simulator instruction, as its old reader did.
        instruction = record['instruction']
        sample = dict(instruction=format_vitra_instruction(instruction),action_list=actions,
            action_mask=np.repeat(valid[:,None],2,axis=1), current_state=current_state,
            current_state_mask=np.ones(2,bool), intrinsics=intrinsics,
            episode_path=self.episode_paths[episode],frame_index=frame,
            representation='wuji20',episode_sample_length=n,episode_source_length=n)
        height,width = self.index.info['features']['observation.images.cam_high']['shape'][-2:]
        sample['fov'] = calculate_fov(height,width,intrinsics).astype(np.float32)
        if self.load_images:
            image_indices = np.arange(frame-self.image_past_window_size,frame+self.image_future_window_size+1)
            image_mask = (image_indices>=0)&(image_indices<n)
            head = self.index.images(episode,np.clip(image_indices,0,n-1),'cam_high')
            # As in FrameDataset, the last head image is the current frame when
            # the official image past/future windows are both zero.
            sample.update(image_list=head,image_mask=image_mask,multi_view_images=[head[-1],
                self.index.images(episode,[frame],'cam_left_wrist')[0],
                self.index.images(episode,[frame],'cam_right_wrist')[0]])
        return sample
