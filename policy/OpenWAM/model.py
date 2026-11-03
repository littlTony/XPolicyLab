"""DexBench OpenWAM-alpha: absolute world EE+wxyz/Wuji20, model raw58/80D.

Use the original SparkArena generation engine; share packing and preprocessing
with its LeRobot reader. No legacy base-origin or xyzw compatibility path.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.eval_checkpoint import configure_eval_cuda,task_name_for_prompt
from XPolicyLab.utils.process_data import get_robot_action_dim_info,pack_robot_state
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root

POLICY_DIR=Path(__file__).resolve().parent
OPENWAM_ROOT=POLICY_DIR/'OpenWAM'
sys.path.insert(0,str(OPENWAM_ROOT))
from openwam.dataloader.dexbench_contract import CONTRACT,CONTRACT_ID,UNIFY_MAP,pack54,unpack58,task_prompt
from openwam.dataloader.transforms.multiview import assemble_multiview_layout,format_prompt_for_inference


def rgb(image):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3:raise ValueError('Expected decoded RGB uint8 image')
    if image.shape[-1]==3:return Image.fromarray(image)
    if image.shape[0]==3:return Image.fromarray(image.transpose(1,2,0))
    raise ValueError(f'Expected RGB image, got {image.shape}')


def _resolve_ckpt_dir(cfg):
    root=resolve_checkpoint_root(cfg,POLICY_DIR/'checkpoints',policy_dir=POLICY_DIR,
        explicit_keys=('ckpt_dir','checkpoint_path','ckpt_path','model_dir'),must_exist=True)
    if (root/'config.yaml').is_file():return root
    # Native OpenWAM creates one timestamp directory below the named run.
    # Evaluation receives that named run from eval.sh, so resolve its saved
    # assets without importing training code or consulting the live dataset.
    from datetime import datetime
    candidates=[]
    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir() or not (child/'config.yaml').is_file():continue
            try:datetime.strptime(child.name,'%Y-%m-%d_%H-%M-%S')
            except ValueError:continue
            candidates.append(child)
    if len(candidates)==1:return candidates[0]
    if len(candidates)>1:
        raise ValueError(f'Multiple checkpoint timestamps under {root}; pass the exact timestamp directory')
    raise FileNotFoundError(f'Checkpoint config missing: {root}')

def _validate_checkpoint_contract(saved,stats):
    """Validate checkpoint-local representation without reading the live dataset."""
    from omegaconf import OmegaConf
    if OmegaConf.select(saved,'dataloader.contract_id')!=CONTRACT_ID:
        raise ValueError('Checkpoint is not DexBench world-wxyz alpha80')
    for key in ['unify_action_map','unify_state_map']:
        if list(OmegaConf.select(saved,'dataloader.'+key) or [])!=UNIFY_MAP:
            raise ValueError(f'Wrong {key}')
    for key in ['action_dim','state_dim']:
        if OmegaConf.select(saved,'model.architecture.'+key)!=80:
            raise ValueError(f'Expected model {key}=80')
    if OmegaConf.select(saved,'dataloader.normalize_mode')!='min-max':
        raise ValueError('DexBench OpenWAM requires normalize_mode=min-max')
    if OmegaConf.select(saved,'dataloader.action_mode')!='eef':
        raise ValueError('DexBench OpenWAM requires action_mode=eef')
    if OmegaConf.select(saved,'dataloader.binary_action_dims'):
        raise ValueError('DexBench absolute hand joints must not use binary_action_dims')
    if stats.get('action_contract')!=CONTRACT:
        raise ValueError('Normalization action contract mismatch')
    for section in ['eef','eef_state']:
        for key in ['min','max','mean','std']:
            value=np.asarray(stats.get(section,{}).get(key))
            if value.shape!=(58,) or not np.isfinite(value).all():
                raise ValueError(f'Expected finite checkpoint {section}/{key} raw58 statistics')


class Model(ModelTemplate):
    def __init__(self,model_cfg):
        self.cfg=dict(model_cfg)
        if self.cfg.get('action_type','ee')!='ee':raise ValueError('DexBench OpenWAM requires action_type=ee')
        self.robot_dims=get_robot_action_dim_info(self.cfg.get('env_cfg_type') or 'tianji_marvin_wuji')
        if len(self.robot_dims['arm_dim'])!=2 or self.robot_dims['ee_dim']!=[20,20]:raise ValueError('Two Wuji20 hands required')
        self.task_name=self.cfg['task_name']
        self.prompt_embodiment=self.cfg.get('prompt_embodiment','tianji_wujihand2_real')
        self.replan_steps=self.cfg.get('replan_steps')
        self.allow_dummy_policy=str(self.cfg.get('allow_dummy_policy',False)).lower() in {'1','true'}
        self._batch={};self._order=[]
        if self.allow_dummy_policy:return
        configure_eval_cuda(self.cfg.get('seed'))
        from omegaconf import OmegaConf
        root=_resolve_ckpt_dir(self.cfg)
        saved=OmegaConf.load(root/'config.yaml')
        stats=np.load(root/'normalization_stats.npy',allow_pickle=True).item()
        _validate_checkpoint_contract(saved,stats)
        from openwam.deploy.server import _load_deploy_yaml,build_server_from_config
        deploy=_load_deploy_yaml(str(OPENWAM_ROOT/'configs/deploy.yaml'))
        for key in ['optimization.dit_cache.enabled','optimization.compile.enabled','optimization.decode_video']:
            OmegaConf.update(deploy,key,False,merge=False)
        OmegaConf.update(deploy,'inference.inference_mode','sync',merge=False)
        step=int(self.cfg.get('checkpoint_num') or 60000)
        weight=root/f'checkpoint_step_{step}.safetensors'
        if not weight.is_file():raise FileNotFoundError(weight)
        server=build_server_from_config(deploy,str(root),device=self.cfg.get('device','cuda'),ckpt_name=weight.name)
        server._init_policy()
        self._engine=server.engine;self._policy=server._policy;self._preprocessor=server._obs_preprocessor
        if self.replan_steps is None:self.replan_steps=32
        self._metadata=dict(policy_name='OpenWAM',checkpoint_path=str(weight),checkpoint_optimizer_step=step,
                            step_source='native checkpoint filename and completed training log',
                            action_horizon=32,replan_steps=self.replan_steps,action_dim=54,model_action_dim=80,
                            stats_sha256=hashlib.sha256((root/'normalization_stats.npy').read_bytes()).hexdigest(),
                            real_checkpoint=True,device='cuda')

    def get_metadata(self):
        if self.allow_dummy_policy:raise RuntimeError('No real checkpoint loaded')
        return dict(self._metadata)

    def _encode_obs(self,obs):
        state=pack_robot_state(obs,'ee',self.robot_dims,source_type='obs')
        state58=pack54(state)
        vision=obs['vision'];images={}
        for target,candidates in [('head_camera',['cam_head','cam_high']),
                                  ('left_wrist_camera',['cam_left_wrist']),('right_wrist_camera',['cam_right_wrist'])]:
            for key in candidates:
                if key in vision:
                    value=vision[key]
                    images[target]=rgb(value['color'] if isinstance(value,dict) else value)
                    break
            if target not in images:raise KeyError(f'Missing camera {candidates}')
        instruction=obs.get('instruction',obs.get('instructions',obs.get('prompt')))
        task=task_name_for_prompt(instruction,self.task_name,self.prompt_embodiment)
        prompt=format_prompt_for_inference(task_prompt(instruction,task,self.prompt_embodiment))
        return dict(images=images,prompt=prompt,state=state58.tolist())

    def update_obs(self,obs):self.update_obs_batch([obs])

    def update_obs_batch(self,obs_list):
        order=[int(obs.get('env_idx',i)) for i,obs in enumerate(obs_list)]
        if len(order)!=len(set(order)):raise ValueError('Duplicate env indices')
        batch={idx:self._encode_obs(obs) for idx,obs in zip(order,obs_list)}
        self._order=order;self._batch=batch

    def _conditions(self,payload):
        obs=self._preprocessor.preprocess(payload)
        return dict(observation=obs,first_frame_image=[obs['image']],prompt=obs['prompt'],proprio=obs['state'])

    def get_action(self):
        if not self._order:raise RuntimeError('Call update_obs first')
        return self.get_action_batch([self._order[0]])[0]

    def get_action_batch(self,env_idx_list=None):
        if env_idx_list is None:env_idx_list=self._order
        if not env_idx_list:return []
        payloads=[self._batch[int(idx)] for idx in env_idx_list]
        if self.allow_dummy_policy:
            return [unpack58(np.repeat(np.asarray(p['state'],dtype=np.float32)[None],self.replan_steps or 8,axis=0)) for p in payloads]
        result=self._engine.generate_batch([self._conditions(p) for p in payloads])['actions']
        if hasattr(result,'detach'):result=result.detach().cpu().numpy()
        actions=np.asarray(result,dtype=np.float32)
        # The copied engine applies its checkpoint normalizer once: gather80
        # -> physical raw58. Never unnormalize or change the frame a second time.
        if actions.ndim!=3 or actions.shape[0]!=len(payloads) or actions.shape[-1]!=58:
            raise ValueError(f'Expected physical raw58 from engine, got {actions.shape}')
        actions=self._policy._project_binary_dims(actions)
        steps=actions.shape[1] if self.replan_steps is None else min(self.replan_steps,actions.shape[1])
        return [unpack58(chunk[:steps]) for chunk in actions]

    def reset(self):self._batch={};self._order=[]
