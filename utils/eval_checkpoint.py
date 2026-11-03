"""Small serving guards for checkpoint-bound DexBench evaluation."""
import hashlib,json,os,re
from pathlib import Path

def configure_eval_cuda(seed=42):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for real policy evaluation')
    fraction=float(os.environ.get('DEXBENCH_EVAL_CUDA_MEMORY_FRACTION','1'))
    if not 0<fraction<=1:raise ValueError('CUDA memory fraction must be in (0,1]')
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.manual_seed(int(seed or 42))
    torch.cuda.manual_seed_all(int(seed or 42))

def task_name_for_prompt(text,configured,embodiment):
    task=str(configured or 'auto')
    if task=='auto':
        match=re.fullmatch(r'\[([^/\]]+)/([^\]]+)\] (.+)',str(text),re.DOTALL)
        if match is None or match[1]!=embodiment:
            raise ValueError('auto requires the matching exported [embodiment/task] instruction')
        task=match[2]
    return task

def native_checkpoint_metadata(root,policy,horizon):
    root=Path(root)
    state=json.loads((root/'trainer_state.json').read_text())
    step=int(state['global_step'])
    if step!=int(root.name.split('-')[-1]):raise ValueError('Checkpoint step differs from trainer state')
    files=[x for x in [root/'statistics.json',root/'processor/statistics.json',root/'processor/processor_config.json'] if x.is_file()]
    return dict(policy_name=policy,checkpoint_path=str(root),checkpoint_optimizer_step=step,
                action_horizon=horizon,replan_steps=horizon,action_dim=54,model_action_dim=58,
                stats_sha256={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
                real_checkpoint=True,device='cuda')
