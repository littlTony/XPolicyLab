"""Fail before training if data/config/statistics do not match the deployment."""
import argparse
import json
import os
from pathlib import Path

from openpi.policies.dexbench_contract import ACTION_CONTRACT
from openpi.training import config
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from XPolicyLab.policy.Pi_05.stats_provenance import check_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-cfg-type', default='tianji_marvin_wuji')
    parser.add_argument('--require-stats', action='store_true')
    args = parser.parse_args()
    root = Path(os.environ['HF_LEROBOT_HOME']) / 'dexbench_12tasks'
    info = json.loads((root / 'meta/info.json').read_text())
    modality = json.loads((root / 'meta/modality.json').read_text())
    for key in ['observation.state', 'action']:
        if info['features'][key]['shape'] != [54]:
            raise ValueError(f'{key} must have 54 dimensions')
        expected_names = [f'{name}_{i}' for name, n in [
            ('left_ee_poses', 7), ('left_ee_joint_states', 20),
            ('right_ee_poses', 7), ('right_ee_joint_states', 20)] for i in range(n)]
        if info['features'][key]['names'] != [expected_names]:
            raise ValueError(f'{key} dimension names/order differ from the exporter')
    for group in ['state', 'action']:
        for name, start, end in [('left_ee_poses',0,7),('left_ee_joint_states',7,27),
                                 ('right_ee_poses',27,34),('right_ee_joint_states',34,54)]:
            entry = modality[group][name]
            if entry['start'] != start or entry['end'] != end or entry['absolute'] is not True:
                raise ValueError(f'Dataset {group}/{name} is not the absolute EE/hand layout')
    for camera in ACTION_CONTRACT['cameras']:
        if 'observation.images.' + camera not in info['features']:
            raise ValueError(f'Missing camera {camera}')
    robot = get_robot_action_dim_info(args.env_cfg_type)
    if len(robot['arm_dim']) != 2 or robot['ee_dim'] != [20,20]:
        raise ValueError('Runtime robot must use two Wuji20 hands')
    cfg = config.get_config('pi05_wuji_marvin_54d')
    if cfg.model.action_dim != 54 or cfg.data.extra_delta_transform:
        raise ValueError('Training config differs from absolute EE54 contract')
    if args.require_stats:
        assets = cfg.assets_dirs / 'dexbench_12tasks'
        if json.loads((assets / 'action_contract.json').read_text()) != ACTION_CONTRACT:
            raise ValueError('Missing or incompatible action contract; run process_data.sh')
        if not (assets / 'norm_stats.json').is_file():
            raise ValueError('Run process_data.sh to compute normalization statistics first')
        check_summary(root, assets, cfg.model.action_horizon)
    print(f'Contract verified: {info["total_episodes"]} episodes, {info["total_frames"]} frames; absolute world EE xyz+wxyz + Wuji20, RGB x3')


if __name__ == '__main__':
    main()
