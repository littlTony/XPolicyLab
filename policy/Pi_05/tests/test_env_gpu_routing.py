import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('gpu,marker', [(1,'pair-v1'),(3,'physical-v1'),(5,'physical-v2'),(7,'')])
def test_native_client_uses_explicit_physical_gpu(tmp_path, gpu, marker):
    root=tmp_path/'policy checkout'
    policy=root/'policy/Pi_05';policy.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'setup_eval_env_client.sh'
    script=policy/source.name;shutil.copyfile(source,script)
    (policy/'deploy.yml').write_text('eval_batch: false\nprotocol: ws\n')
    utils=root/'utils';utils.mkdir()
    (utils/'resolve_eval_env_type.sh').write_text('resolve_eval_env_type() { echo simulator; }\n')
    benchmark=tmp_path/'benchmark space';(benchmark/'scripts').mkdir(parents=True)
    entry=benchmark/'scripts/eval_policy.sh'
    entry.write_text('#!'+sys.executable+'\nimport os,json,sys\nprint("CAPTURE="+json.dumps({"argv":sys.argv[1:],"mask":os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
    entry.chmod(0o755)
    env={**os.environ,'EVAL_MAIN_ROOT':str(benchmark),'CUDA_VISIBLE_DEVICES':'0,1',
         'EVAL_WEB_GPU_ISOLATION':marker,'EVAL_WEB_PHYSICAL_ENV_GPU':'1'}
    for key in ['PI05_DEPLOY_CONFIG','PI05_EVAL_CONDA_ENV']:env.pop(key,None)
    args=['bash',str(script),'EgoVLA','Humanoid-Close-Drawer-v0','weights with spaces','ego_h1_inspire','joint','42',str(gpu),'/unused/env','meta','12345']
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    captured=json.loads(next(line.removeprefix('CAPTURE=') for line in result.stdout.splitlines() if line.startswith('CAPTURE=')))
    argv=captured['argv']
    assert argv[argv.index('--device_id')+1]==str(gpu)
    assert argv[argv.index('--checkpoint')+1]=='weights with spaces'
    assert captured['mask'] is None
