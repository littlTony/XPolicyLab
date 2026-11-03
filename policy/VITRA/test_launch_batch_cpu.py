"""Run real launchers/config builders in temporary trees with fake torchrun.

No neural weights, CUDA calls, training process, or real checkpoint directory
are used. Actual data/stats guards run before the final command is captured.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

P=Path('/personal/wenwei/DexBench/XPolicyLab/policy')
RESULT=P/'VITRA/logs/launch_bs128_cpu_20261027.json'


def main():
    before={name:sorted(x.name for x in (P/name/'checkpoints').iterdir())
            for name in ['VITRA','GR00T_N17']}
    cases=[]
    with tempfile.TemporaryDirectory(prefix='dexbench-launch-cpu-') as directory:
        base=Path(directory);binary=base/'bin';binary.mkdir()
        fake=binary/'torchrun'
        fake.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
v=dict(args=sys.argv[1:],environment={k:os.environ.get(k) for k in ['GLOBAL_BATCH_SIZE','NUM_GPUS','MAX_STEPS']},fake_torchrun=True)
if '--config' in sys.argv:
 v['config']=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
Path(os.environ['DEXBENCH_CPU_CAPTURE']).write_text(json.dumps(v))
print('CPU FAKE TORCHRUN: command captured; no training launched')
''');fake.chmod(0o755)
        for policy in ['VITRA','GR00T_N17']:
            actual=P/policy;temp=base/policy;temp.mkdir()
            shutil.copy2(actual/'train.sh',temp/'train.sh')
            if policy=='VITRA':
                for name in ['make_train_config.py','mapping_wuji20_mano45.json']:
                    shutil.copy2(actual/name,temp/name)
                overrides='export VITRA_POLICY_DIR='+shlex.quote(str(temp))+'\n'
            else:
                (temp/'prepare_relative_stats.py').symlink_to(actual/'prepare_relative_stats.py')
                overrides='export NATIVE_POLICY_DIR='+shlex.quote(str(temp))+'\n'
            (temp/'runtime.sh').write_text('source '+shlex.quote(str(actual/'runtime.sh'))+'\n'+overrides+
                'export PATH='+shlex.quote(str(binary))+':"$PATH"\n')
            for accumulation in [1,2,4,8,16,None]:
                env=os.environ.copy()
                for key in ['DEXBENCH_GLOBAL_BATCH_SIZE','DEXBENCH_GRAD_ACCUM','VITRA_PER_DEVICE_BATCH','NATIVE_ACCUMULATION']:
                    env.pop(key,None)
                env['CUDA_VISIBLE_DEVICES']=''
                env['VITRA_RUN_DATE']='2026-10-27'
                env['NO_ALBUMENTATIONS_UPDATE']='1'
                env['DEXBENCH_CPU_CAPTURE']=str(base/'capture.json')
                if accumulation is not None:
                    env['DEXBENCH_GLOBAL_BATCH_SIZE']='128'
                    env['DEXBENCH_GRAD_ACCUM']=str(accumulation)
                run='cpu_capture_'+('legacy256' if accumulation is None else f'bs128_ga{accumulation}')
                result=subprocess.run(['bash',str(temp/'train.sh'),'DexBench',run,'tianji_marvin_wuji','ee','42','0,1,2,3,4,5,6,7'],
                    env=env,capture_output=True,text=True,timeout=120)
                if result.returncode:raise RuntimeError(policy+' '+run+'\n'+result.stdout+'\n'+result.stderr)
                capture=json.loads((base/'capture.json').read_text())
                assert capture['fake_torchrun']
                args=capture['args']
                if policy=='VITRA':
                    config=capture['config'];launch=config['dexbench_launch']
                    effective=config['total_batch_size'];micro=config['batch_size'];ga=launch['gradient_accumulation_steps'];steps=config['trainer']['max_steps']
                    assert config['fwd_pred_next_n']==16 and config['data_representation']=='wuji20'
                    assert launch['world_size']==8 and launch['per_device_batch_size']==micro and launch['max_optimizer_steps']==60000
                else:
                    def arg(name):return args[args.index(name)+1]
                    micro_global=int(arg('--global_batch_size'));ga=int(arg('--gradient_accumulation_steps'))
                    world=int(arg('--num_gpus'));steps=int(arg('--max_steps'))
                    assert world==8
                    micro=micro_global//world;effective=micro*world*ga
                    assert capture['environment']['GLOBAL_BATCH_SIZE']==str(micro_global)
                if accumulation is None:
                    assert effective==256
                    assert (micro,ga)==((8,4) if policy=='VITRA' else (32,1))
                else:
                    assert effective==128 and ga==accumulation and micro==16//accumulation
                assert micro*8*ga==effective and steps==60000
                cases.append(dict(policy=policy,effective_batch=effective,per_device_batch=micro,
                                  world_size=8,gradient_accumulation=ga,max_optimizer_steps=steps,fake_torchrun=True))
                print('PASS',cases[-1],flush=True)
            invalid=os.environ.copy();invalid.update(DEXBENCH_GLOBAL_BATCH_SIZE='128',DEXBENCH_GRAD_ACCUM='3',CUDA_VISIBLE_DEVICES='')
            result=subprocess.run(['bash',str(temp/'train.sh'),'DexBench','invalid_ga3','tianji_marvin_wuji','ee','42','0,1,2,3,4,5,6,7'],env=invalid,capture_output=True,text=True,timeout=10)
            assert result.returncode!=0 and 'divide exactly' in result.stderr
    after={name:sorted(x.name for x in (P/name/'checkpoints').iterdir()) for name in before}
    assert before==after,'Real checkpoint directory was modified'
    report=dict(status='PASS',cases=cases,invalid_accumulation_rejected=True,real_checkpoint_dirs_unchanged=True,
                boundary='Real launchers and CPU data/stats checks; torchrun intercepted, no GPU/model/training',
                affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0))
    RESULT.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS12 launch captures and invalid GA3 rejection; no actual checkpoints/training/GPU; report',RESULT)


if __name__=='__main__':main()
