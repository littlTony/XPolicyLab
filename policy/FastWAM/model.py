from pathlib import Path
import json,sys,re,hashlib
import numpy as np
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info,pack_robot_state
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR=Path(__file__).resolve().parent
for p in [POLICY_DIR/'FastWAM',POLICY_DIR/'FastWAM/src']:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from fastwam.datasets.dexbench_contract import CONTRACT,pack54,unpack58,task_prompt,inference_image

def truth(x):return str(x).lower() in ['true','1','yes']

def _weight_step(path):
    match=re.fullmatch(r'step_(\d+)\.pt',Path(path).name)
    return int(match.group(1)) if match else None


def _requested_step(value):
    if value is None or str(value).strip().lower() in ['', 'none', 'null', 'latest']:return None
    match=re.fullmatch(r'(?:steps?[_-]?)?(\d+)(?:\.pt)?',str(value).strip())
    if match is None:raise ValueError(f'Invalid FastWAM checkpoint_num: {value}')
    return int(match.group(1))

def resolve_runtime(model_cfg):
    from omegaconf import OmegaConf
    root=Path(resolve_checkpoint_root(model_cfg,POLICY_DIR/'checkpoints',policy_dir=POLICY_DIR))
    requested=_requested_step(model_cfg.get('checkpoint_num'))
    if root.is_file():
        weights=root
        if requested is not None and _weight_step(weights)!=requested:raise FileNotFoundError('Explicit FastWAM weights do not match checkpoint_num')
        roots=[root.parent,*root.parents]
        root=next((p for p in roots if (p/'config.yaml').exists()),root.parent)
    else:
        candidates=[p for p in (root/'checkpoints/weights').glob('step_*.pt') if p.is_file() and _weight_step(p) is not None]
        if requested is not None:candidates=[p for p in candidates if _weight_step(p)==requested]
        weights=max(candidates,key=_weight_step) if candidates else None
    if weights is None:raise FileNotFoundError(f'No FastWAM weights matching checkpoint_num={requested} under {root}')
    cfg=OmegaConf.load(root/'config.yaml')
    if OmegaConf.to_container(cfg.get('action_contract',{}),resolve=True)!=CONTRACT:raise ValueError('Checkpoint is not absolute DexBench EE58')
    for split in ['train','val']:
        ds=cfg.data[split]
        if ds.contract_id!=CONTRACT['id'] or ds.processor.action_output_dim!=58 or ds.processor.proprio_output_dim!=58:raise ValueError('Wrong saved processor contract')
        if ds.processor.action_state_transforms or ds.processor.get('delta_action_dim_mask'):raise ValueError('Absolute contract forbids delta transforms')
    for n in ['action_dit_config','video_dit_config']:
        if cfg.model[n].action_dim!=58:raise ValueError('Wrong saved action head')
    if cfg.model.proprio_dim!=58:raise ValueError('Wrong saved state head')
    stats=root/'dataset_stats.json'
    if json.loads(stats.read_text()).get('action_contract')!=CONTRACT:raise ValueError('Wrong checkpoint statistics contract')
    return str(weights),str(stats),OmegaConf.to_container(cfg,resolve=True)

class Model(ModelTemplate):
    def __init__(self,model_cfg):
        self.model_cfg=dict(model_cfg)
        self.action_type=str(self.model_cfg.get('action_type') or 'ee')
        self.env_cfg_type=self.model_cfg.get('env_cfg_type','tianji_marvin_wuji')
        if self.action_type!='ee' or self.env_cfg_type!='tianji_marvin_wuji':raise ValueError('Expected tianji_marvin_wuji absolute EE')
        self.robot_action_dim_info=get_robot_action_dim_info(self.env_cfg_type)
        if self.robot_action_dim_info['ee_dim']!=[20,20]:raise ValueError('Expected two Wuji20 hands')
        raw_replan=self.model_cfg.get('replan_steps',24)
        self.replan_steps=24 if raw_replan is None else int(raw_replan)
        if not 1<=self.replan_steps<=32:raise ValueError('FastWAM replan_steps must be in [1,32]')
        self.action_dim=58; self.action_horizon=32
        if truth(self.model_cfg.get('allow_dummy_policy',False)):
            raise ValueError('Production FastWAM requires real checkpoint weights')
        self.model=None
        self.reset()
        self.model=self._load_model()
        self.action_horizon=int(self.model.action_horizon)
        if self.action_horizon!=32:raise ValueError('Expected saved action horizon32')

    def _load_model(self):
        import torch
        from experiments.robotwin.fastwam_policy.deploy_policy import get_model
        if str(self.model_cfg.get('device') or 'cuda').startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for this FastWAM server')
        fraction=float(self.model_cfg.get('cuda_memory_fraction') or 1.0)
        if not 0<fraction<=1:raise ValueError('cuda_memory_fraction must be in (0,1]')
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(fraction)
        weights,stats,cfg=resolve_runtime(self.model_cfg)
        args=dict(self.model_cfg,ckpt_setting=weights,dataset_stats_path=stats,checkpoint_config=cfg,
                  action_dim=58,image_tensor_builder=inference_image,sim_task='dexbench_ee58')
        engine=get_model(args)
        payload=torch.load(weights,map_location='cpu',mmap=True,weights_only=False)
        step=int(payload['step'])
        if step!=_weight_step(weights):raise ValueError('Checkpoint internal step differs from filename')
        for name in ['mot','proprio_encoder']:
            expected=getattr(engine.model,name).state_dict(); actual=payload[name]
            if set(expected)!=set(actual) or any(expected[k].shape!=actual[k].shape for k in expected):
                raise ValueError(f'Incomplete checkpoint coverage: {name}')
        self._metadata=dict(policy_name='FastWAM',checkpoint_path=weights,checkpoint_optimizer_step=step,
                            dataset_stats_path=stats,dataset_stats_sha256=hashlib.sha256(Path(stats).read_bytes()).hexdigest(),
                            action_dim=54,model_action_dim=58,action_horizon=32,replan_steps=self.replan_steps,
                            task_name=self.model_cfg.get('task_name'),action_contract=CONTRACT,
                            video_size=cfg['data']['train']['video_size'],device=str(engine.model.device))
        return engine

    def get_metadata(self):return dict(self._metadata)

    def _encode(self,obs):
        cameras={}
        for source,target in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
            x=np.asarray(obs['vision'][source]['color'])
            if x.ndim!=3 or x.shape[-1]!=3 or x.dtype!=np.uint8:raise ValueError(f'Expected decoded uint8 RGB: {source}')
            cameras[target]={'rgb':np.ascontiguousarray(x)}
        state=pack_robot_state(obs,'ee',self.robot_action_dim_info,source_type='obs',state_type='state')
        vector=pack54(state)
        text=obs.get('instruction',obs.get('instructions',self.model_cfg.get('default_instruction','follow the instruction')))
        if isinstance(text,(list,tuple)):text=text[0]
        task=self.model_cfg.get('task_name') or 'auto'
        if task=='auto':
            match=re.fullmatch(r'\[([^/\]]+)/([^\]]+)\] (.+)',str(text),re.DOTALL)
            if match is None:raise ValueError('task_name=auto requires the exported [embodiment/task] instruction')
            task=match[2]
            if obs.get('task_name') not in [None,task]:raise ValueError('Observation task disagrees with instruction prefix')
        prompt=task_prompt(str(text),task,self.model_cfg.get('prompt_embodiment','tianji_wujihand2_real'))
        return {'observation':cameras,'joint_action':{'vector':vector}},prompt

    def update_obs(self,obs):
        self.last_obs,self.last_instruction=self._encode(obs)
        self._batch={}

    def update_obs_batch(self,obs_list):
        batch={}
        for obs in obs_list:
            idx=int(obs['env_idx'])
            if idx in batch:raise ValueError('Duplicate env index')
            batch[idx]=self._encode(obs)
        if not batch:raise ValueError('Empty observations')
        self._batch=batch; self.last_obs=None

    def _infer(self,obs,instruction):
        if obs is None:raise ValueError('Call update_obs first')
        # Upstream engine applies saved action unnormalization exactly once.
        raw=np.asarray(self.model._infer_action_chunk(obs,instruction),dtype=np.float32)
        if raw.ndim!=2 or raw.shape[1]!=58 or len(raw)<self.replan_steps or not np.isfinite(raw).all():raise ValueError('Expected finite physical EE58 action chunk')
        return unpack58(raw[:self.replan_steps])

    def get_action(self):return self._infer(self.last_obs,self.last_instruction)

    def get_action_batch(self,env_idx_list=None):
        ids=list(self._batch) if env_idx_list is None else [int(i) for i in env_idx_list]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Invalid active environment indices')
        if any(i not in self._batch for i in ids):raise ValueError('Requested inactive environment ID')
        return [self._infer(*self._batch[i]) for i in ids]

    def reset(self):
        self.last_obs=None; self.last_instruction=''; self._batch={}
        if self.model is not None:self.model.reset()
