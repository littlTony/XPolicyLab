"""Read prepared DexBench LeRobot v3 through OpenWAM's existing v3 reader.

The exporter already stores a following-pose target at action[t]. Reading it
directly is essential: shifting the action by another row would be incorrect.
"""
import json
from pathlib import Path
import numpy as np
from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.dexbench_contract import CONTRACT,CONTRACT_ID,UNIFY_MAP,pack54
from openwam.dataloader.dexbench_provenance import validate_stats_source
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.dataloader.utils.video_io import decode_video_frames
from openwam.dataloader.utils import get_cfg


class DexBenchLeRobotDataset(LeRobotV3Reader):
    DATASET_NAME='DexBench SparkArena EE/Wuji20'
    NEEDED_COLS=('observation.state','action','task_index')
    HEAD_CAMERA='observation.images.cam_high'
    LEFT_WRIST_CAMERA='observation.images.cam_left_wrist'
    RIGHT_WRIST_CAMERA='observation.images.cam_right_wrist'
    ACTION_DIM=58
    DEFAULT_NORMALIZE_MODE='min-max'
    DEPLOY_ACTION_MODE='eef'

    def __init__(self,dataset_dir,*,normalization_stats_path,**kwargs):
        if kwargs.get('normalize_mode','min-max')!='min-max':raise ValueError('DexBench requires normalize_mode=min-max')
        self._stats_source_path=str(normalization_stats_path)
        self.contract_id=CONTRACT_ID
        self.source_frame='world_xyz_wxyz_wuji20'
        super().__init__(dataset_dir,**kwargs)

    def _resolve_cameras(self,info):
        for key in ['observation.state','action']:
            if info['features'][key]['shape']!=[54]:raise ValueError(f'{key} must be 54D')
        for key in [self.HEAD_CAMERA,self.LEFT_WRIST_CAMERA,self.RIGHT_WRIST_CAMERA]:
            if key not in info['features']:raise ValueError(f'Missing camera {key}')
        self._source_resolution={key:tuple(info['features'][key]['shape'][-2:]) for key in
                                 [self.HEAD_CAMERA,self.LEFT_WRIST_CAMERA,self.RIGHT_WRIST_CAMERA]}
        return self.HEAD_CAMERA,self.LEFT_WRIST_CAMERA,self.RIGHT_WRIST_CAMERA

    def _load_stats(self,info):
        self.normalization_stats_path=self._stats_source_path
        raw=np.load(self.normalization_stats_path,allow_pickle=True).item()
        if raw.get('action_contract')!=CONTRACT:raise ValueError('Incompatible normalization action contract')
        validate_stats_source(raw,self._dataset_dir,info)
        self._action_normalizer=Normalizer('min_max',raw['eef'])
        self._state_normalizer=Normalizer('min_max',raw['eef_state'])
        return raw['eef']

    def _action_20d(self,win):
        return self._action_normalizer.normalize(pack54(np.stack(win['action'].to_numpy())))

    def _proprio_20d(self,win):
        return self._state_normalizer.normalize(pack54(np.asarray(win['observation.state'].iloc[0])[None]))

    def _resolve_prompt(self,row,win):
        return format_prompt_for_inference(super()._resolve_prompt(row,win))

    def _decode_one_camera(self,camera,row,ep_local,offset,real_local_indices,*,kind):
        # Decode all three cameras strictly at source resolution. Exactly one
        # shared multiview resize/composition is applied in training and eval.
        path=self._dataset_dir/self._video_path_template.format(video_key=camera,
            chunk_index=int(row[f'videos/{camera}/chunk_index']),file_index=int(row[f'videos/{camera}/file_index']))
        base=int(self._ep_video_frame_offsets[camera][ep_local])+offset
        height,width=self._source_resolution[camera]
        return decode_video_frames(str(path),(real_local_indices+base).tolist(),height,width)

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self._getitem_impl(index)  # fail visibly; never substitute another episode

    @classmethod
    def from_config(cls,cfg,split='train'):
        if get_cfg(cfg,'contract_id',None)!=CONTRACT_ID:raise ValueError('Expected DexBench world/wxyz contract')
        if get_cfg(cfg,'action_mode','eef')!='eef':raise ValueError('Expected EEF mode')
        if get_cfg(cfg,'normalize_mode','min-max')!='min-max':raise ValueError('DexBench requires normalize_mode=min-max')
        if get_cfg(cfg,'binary_action_dims',None):raise ValueError('DexBench must not use binary_action_dims')
        for key in ['unify_action_map','unify_state_map']:
            if list(get_cfg(cfg,key,[]))!=UNIFY_MAP:raise ValueError(f'{key} differs from alpha80 slots')
        if not get_cfg(cfg,'unify_action',False):raise ValueError('The 154000 foundation requires unified80')
        kwargs={k:get_cfg(cfg,k,None) for k in ['num_frames','video_stride','window_stride','height','width',
                    'multiview','normalize_mode','camera_layout','unify_action','unify_action_map','unify_state_map','color_jitter']}
        kwargs={k:v for k,v in kwargs.items() if v is not None}
        return cls(get_cfg(cfg,'dataset_dir'),normalization_stats_path=get_cfg(cfg,'normalization_stats_path'),split=split,**kwargs)
