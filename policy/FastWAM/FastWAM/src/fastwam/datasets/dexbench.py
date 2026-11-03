"""Direct LeRobot v3 reader for the existing DexBench export, RGB via PyAV."""
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import json
import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from .dexbench_contract import CONTRACT,CAMERAS,DEFAULT_PROMPT,pack54,image_tensor
from .dexbench_video_io import decode_video_frames

class DexBenchIndex:
    def __init__(self,root,split='train',val_set_proportion=0.01):
        self.root=Path(root)
        self.info=json.loads((self.root/'meta/info.json').read_text())
        if self.info['codebase_version']!='v3.0':raise ValueError('Expected LeRobot v3')
        for key in ['action','observation.state']:
            if self.info['features'][key]['shape']!=[54]:raise ValueError(f'{key} must be EE54')
        for k in CAMERAS:
            if 'observation.images.'+k not in self.info['features']:raise ValueError(f'Missing camera {k}')
        columns=['episode_index','length','data/chunk_index','data/file_index','tasks']
        columns += [f'videos/observation.images.{k}/{field}' for k in CAMERAS for field in ['chunk_index','file_index','from_timestamp']]
        import pandas as pd
        eps=pd.concat([pq.read_table(f,columns=columns).to_pandas() for f in sorted((self.root/'meta/episodes').rglob('*.parquet'))],ignore_index=True).sort_values('episode_index')
        if eps['episode_index'].duplicated().any():raise ValueError('Duplicate episodes')
        if split!='all':
            n_val=max(1,int(round(len(eps)*val_set_proportion)))
            val_ids=np.random.default_rng(42).permutation(len(eps))[:n_val]
            is_val=np.isin(np.arange(len(eps)),val_ids)
            eps=eps.loc[~is_val if split=='train' else is_val]
        self.episodes=eps.reset_index(drop=True)
        self.ends=np.cumsum(self.episodes['length'].to_numpy(dtype=np.int64))

    def __len__(self):return int(self.ends[-1])

    def locate(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        e=int(np.searchsorted(self.ends,index,side='right'))
        return self.episodes.iloc[e],int(index-(self.ends[e-1] if e else 0))

    @lru_cache(maxsize=8)
    def _read_episode(self,episode_index,chunk,file):
        f=self.root/self.info['data_path'].format(chunk_index=chunk,file_index=file)
        table=pq.read_table(f,columns=['episode_index','frame_index','action','observation.state','task_index'],filters=[('episode_index','=',episode_index)])
        data=table.to_pandas().sort_values('frame_index').reset_index(drop=True)
        if not np.array_equal(data['frame_index'],np.arange(len(data))):raise ValueError('Noncontiguous episode frames')
        return data

    def episode_data(self,row):
        data=self._read_episode(int(row['episode_index']),int(row['data/chunk_index']),int(row['data/file_index']))
        if len(data)!=int(row['length']):raise ValueError('Episode length mismatch')
        return data

    def numeric(self,index,num_frames=33):
        row,t=self.locate(index); data=self.episode_data(row); n=len(data)
        a_ids=np.minimum(np.arange(t,t+num_frames-1),n-1)
        s_ids=np.minimum(np.arange(t,t+num_frames),n-1)
        return row,t,pack54(np.stack(data['action'].iloc[a_ids])),pack54(np.stack(data['observation.state'].iloc[s_ids])),a_ids,s_ids

    def cameras(self,row,indices):
        output={}
        for key in CAMERAS:
            full='observation.images.'+key
            path=self.root/self.info['video_path'].format(video_key=full,chunk_index=int(row[f'videos/{full}/chunk_index']),file_index=int(row[f'videos/{full}/file_index']))
            offset=int(round(float(row[f'videos/{full}/from_timestamp'])*self.info['fps']))
            h,w=self.info['features'][full]['shape'][-2:]
            output[key]=np.stack([np.asarray(x) for x in decode_video_frames(str(path),(np.asarray(indices)+offset).tolist(),h,w)])
        return output

class DexBenchDataset(torch.utils.data.Dataset):
    _get_cached_text_context=RobotVideoDataset._get_cached_text_context

    def __init__(self,dataset_dirs,shape_meta,processor,pretrained_norm_stats,text_embedding_cache_dir,
                 num_frames=33,action_video_freq_ratio=4,video_size=(544,320),context_len=128,
                 is_training_set=True,val_set_proportion=0.01,contract_id=None,**kwargs):
        if contract_id!=CONTRACT['id']:raise ValueError('Wrong DexBench contract')
        if len(dataset_dirs)!=1:raise ValueError('Expected one DexBench dataset')
        self.index=DexBenchIndex(dataset_dirs[0],'train' if is_training_set else 'val',val_set_proportion)
        self.num_frames=int(num_frames); self.ratio=int(action_video_freq_ratio)
        if self.num_frames!=33 or self.ratio!=4:raise ValueError('Expected action32/video9 temporal contract')
        self.video_size=tuple(video_size); self.context_len=int(context_len)
        self.text_embedding_cache_dir=str(text_embedding_cache_dir)
        if isinstance(processor,DictConfig):processor=instantiate(processor)
        if json.loads(Path(pretrained_norm_stats).read_text()).get('action_contract')!=CONTRACT:raise ValueError('Wrong stats action contract')
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(pretrained_norm_stats))
        processor.train() if is_training_set else processor.eval()
        self.processor=processor
        self.lerobot_dataset=SimpleNamespace(processor=processor)

    def __len__(self):return len(self.index)

    def get_sample_task_ids(self):
        return np.repeat([str(row['tasks'][0]) for _,row in self.index.episodes.iterrows()],self.index.episodes['length'].to_numpy()).tolist()

    def __getitem__(self,index):
        row,t,action,state,a_ids,s_ids=self.index.numeric(index,self.num_frames)
        action=self.processor.normalizer.normalizers['action']['default'].forward(torch.from_numpy(action))
        state=self.processor.normalizer.normalizers['state']['default'].forward(torch.from_numpy(state))
        vi=np.arange(0,self.num_frames,self.ratio)
        video=image_tensor(self.index.cameras(row,s_ids[vi]),self.video_size).permute(1,0,2,3)
        prompt=DEFAULT_PROMPT.format(task=str(row['tasks'][0]))
        context,mask=self._get_cached_text_context(prompt)
        context=context.clone(); context[~mask]=0
        return dict(video=video,action=action,proprio=state[:-1],prompt=prompt,context=context,context_mask=torch.ones_like(mask),
                    image_is_pad=torch.from_numpy(t+vi>=int(row['length'])),
                    action_is_pad=torch.from_numpy(t+np.arange(self.num_frames-1)>=int(row['length'])),
                    proprio_is_pad=torch.from_numpy(t+np.arange(self.num_frames)>=int(row['length'])))
