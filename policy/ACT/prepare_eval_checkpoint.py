"""Prepare ACT's saved configuration and milestone filenames, without inference."""
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

import yaml


def prepare(checkpoint, deploy_file, task, runtime):
    selected = Path(checkpoint).resolve(strict=True)
    weight = selected / 'policy.ckpt' if selected.is_dir() else selected
    root = weight.parent.parent if weight.parent.name.startswith('step_') else weight.parent
    saved = json.loads((root / 'config_0908-real.json').read_text())
    if saved.get('task') != task or saved.get('action_dim') != 54 or saved.get('action_type') != 'joint':
        raise ValueError('ACT checkpoint must match the requested task and joint54 configuration')
    if weight.parent.name.startswith('step_'):
        step = re.fullmatch(r'step_(\d+)(?:_0908-real)?', weight.parent.name)
        marker = json.loads((weight.parent / 'complete_0908-real.json').read_text())
        if step is None or marker.get('optimizer_step') != int(step.group(1)):
            raise ValueError('ACT checkpoint completion marker does not match its step')
    statistics = root / 'dataset_stats.pkl'
    if not statistics.is_file() or not statistics.stat().st_size:
        raise ValueError(f'Missing ACT statistics: {statistics}')
    with zipfile.ZipFile(weight) as archive:
        members = archive.infolist()
        if not any(item.filename.endswith('/data.pkl') and item.file_size for item in members):
            raise ValueError('ACT checkpoint has no serialized policy')
        if not any('/data/' in item.filename and item.file_size for item in members):
            raise ValueError('ACT checkpoint has no tensor storage')
    with Path(deploy_file).open() as stream:
        config = yaml.safe_load(stream)
    config.update(saved)
    runtime = Path(runtime).resolve()
    if runtime == root or root in runtime.parents:
        raise ValueError('ACT runtime configuration must be outside its source checkpoint run')
    runtime.mkdir(parents=True, exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix=task + '-', dir=runtime))
    try:
        (view / 'policy_last.ckpt').symlink_to(weight)
        (view / 'dataset_stats.pkl').symlink_to(statistics)
        config.update(ckpt_dir=str(view), temporal_agg=os.environ.get('ACT_TEMPORAL_AGG', '0') == '1')
        config_file = view / 'deploy.yml'
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    except BaseException:
        shutil.rmtree(view)
        raise
    return config_file


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('usage: prepare_eval_checkpoint.py CHECKPOINT DEPLOY_YAML TASK RUNTIME_DIRECTORY')
    print(prepare(*sys.argv[1:]))
