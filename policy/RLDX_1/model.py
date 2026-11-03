"""Absolute world EE58 / three-RGB-view adapter for the official native policy."""
from pathlib import Path
from collections import deque
import json,sys
import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info,pack_robot_state
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from .native_contract import pack54,unpack58,task_prompt
from XPolicyLab.utils.eval_checkpoint import configure_eval_cuda,task_name_for_prompt,native_checkpoint_metadata

P=Path(__file__).resolve().parent
IS_GROOT=P.name=='GR00T_N17'
GROUPS=[('left_eef',0,9),('left_hand',9,29),('right_eef',29,38),('right_hand',38,58)]
CONTRACT_ID='dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1'

class Model(ModelTemplate):
    def __init__(self,model_cfg):
        self.model_cfg=dict(model_cfg)
        if self.model_cfg.get('action_type','ee')!='ee' or self.model_cfg.get('env_cfg_type')!='tianji_marvin_wuji':raise ValueError('Expected tianji_marvin_wuji absolute EE')
        self.robot_dims=get_robot_action_dim_info('tianji_marvin_wuji')
        assert self.robot_dims['ee_dim']==[20,20]
        self.model=None
        self.histories={}
        self._latest_active_ids=[]
        self.dummy=str(self.model_cfg.get('allow_dummy_policy',False)).lower() in ['true','1']
        raw_replan=self.model_cfg.get('replan_steps',16)
        self.replan_steps=16 if raw_replan is None else int(raw_replan)
        if not 1<=self.replan_steps<=16:raise ValueError('RLDX replan_steps must be in [1,16]')
        if not self.dummy:
            configure_eval_cuda(self.model_cfg.get('seed'))
            root=resolve_checkpoint_root(self.model_cfg,P/'checkpoints',policy_dir=P)
            ckpts=list(root.rglob('checkpoint-*')) if not root.name.startswith('checkpoint-') else [root]
            ckpts=[x for x in ckpts if x.is_dir() and x.name.split('-')[-1].isdigit()]
            if not ckpts:raise FileNotFoundError(f'No native checkpoint under {root}')
            requested=self.model_cfg.get('checkpoint_num')
            if requested is not None:ckpts=[x for x in ckpts if int(x.name.split('-')[-1])==int(requested)]
            if not ckpts:raise FileNotFoundError('Requested checkpoint step is missing')
            root=max(ckpts,key=lambda x:int(x.name.split('-')[-1]))
            self._metadata=native_checkpoint_metadata(root,P.name,self.replan_steps)
            marker=next((x/'action_contract.json' for x in [root,*root.parents] if (x/'action_contract.json').exists()),None)
            if marker is None or json.loads(marker.read_text()).get('id')!=CONTRACT_ID:raise ValueError('Checkpoint lacks DexBench absolute EE58 contract')
            import importlib.util
            spec=importlib.util.spec_from_file_location('dexbench_modalities',P/'dexbench_config.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            if IS_GROOT:
                from gr00t.policy import Gr00tPolicy
                from gr00t.data.embodiment_tags import EmbodimentTag
                self.model=Gr00tPolicy(model_path=str(root),embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,device=self.model_cfg.get('device','cuda'),strict=True)
            else:
                from rldx.policy.rldx_policy import RLDXPolicy
                from rldx.data.embodiment_tags import EmbodimentTag
                self.model=RLDXPolicy(model_path=str(root),embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,device=self.model_cfg.get('device','cuda'),strict=True)
            assert list(self.model.modality_configs['state'].modality_keys)==[x[0] for x in GROUPS]
            for cfg in self.model.modality_configs['action'].action_configs:
                if cfg.rep.name!='ABSOLUTE':raise ValueError('Checkpoint has relative targets')

    def get_metadata(self):
        if self.dummy:raise RuntimeError('No real checkpoint loaded')
        return dict(self._metadata)

    def _encode(self,obs):
        x=pack54(pack_robot_state(obs,'ee',self.robot_dims,source_type='obs'))
        images={}
        for target,source in [('front','cam_head'),('left_wrist','cam_left_wrist'),('right_wrist','cam_right_wrist')]:
            image=np.asarray(obs['vision'][source]['color'])
            if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:raise ValueError(f'Expected decoded RGB {source}')
            images[target]=np.ascontiguousarray(image)
        text=obs.get('instruction',obs.get('instructions',''))
        if isinstance(text,(list,tuple)):text=text[0]
        embodiment=self.model_cfg.get('prompt_embodiment','tianji_wujihand2_real')
        task=task_name_for_prompt(text,self.model_cfg.get('task_name'),embodiment)
        prompt=task_prompt(str(text),task,embodiment)
        return dict(state=x,images=images,prompt=prompt)

    def _update(self,idx,obs):
        if idx not in self.histories:self.histories[idx]=deque(maxlen=7)
        self.histories[idx].append(self._encode(obs))

    def update_obs(self,obs):
        idx=int(obs.get('env_idx',0));self._update(idx,obs)
        self._latest_active_ids=[idx]

    def update_obs_batch(self,obs_list):
        ids=[int(obs['env_idx']) for obs in obs_list]
        if not ids:raise ValueError('Empty observations')
        if len(set(ids))!=len(ids):raise ValueError('Duplicate env index')
        for idx,obs in zip(ids,obs_list):self._update(idx,obs)
        self._latest_active_ids=ids

    def _infer(self,ids):
        if self.dummy:return [unpack58(np.repeat(self.histories[i][-1]['state'][None],self.replan_steps,axis=0)) for i in ids]
        states={k:np.stack([self.histories[i][-1]['state'][a:b][None] for i in ids]) for k,a,b in GROUPS}
        ages=[0] if IS_GROOT else [6,4,2,0]
        video={cam:np.stack([np.stack([self.histories[i][max(0,len(self.histories[i])-1-age)]['images'][cam] for age in ages]) for i in ids]) for cam in ['front','left_wrist','right_wrist']}
        native=dict(state=states,video=video,language={'annotation.human.task_description':[[self.histories[i][-1]['prompt']] for i in ids]})
        if IS_GROOT:action,_=self.model.get_action(native)
        else:action,_=self.model.get_action(native,options={'session_ids':[f'dexbench-{i}' for i in ids]})
        output=[]
        for j in range(len(ids)):
            raw=np.concatenate([np.asarray(action[k][j]) for k,_,_ in GROUPS],axis=-1)
            output.append(unpack58(raw[:self.replan_steps]))
        return output

    def get_action(self):
        if not self._latest_active_ids:raise ValueError('No active observation; call update_obs first')
        return self._infer(self._latest_active_ids[:1])[0]
    def get_action_batch(self,env_idx_list=None):
        ids=list(self._latest_active_ids) if env_idx_list is None else [int(i) for i in env_idx_list]
        if not ids or len(set(ids))!=len(ids):raise ValueError('Invalid active env indices')
        if any(i not in self._latest_active_ids for i in ids):raise ValueError('Requested inactive environment ID')
        return self._infer(ids)
    def reset(self):
        self.histories={}
        self._latest_active_ids=[]
        if self.model is not None and hasattr(self.model,'reset'):self.model.reset()
