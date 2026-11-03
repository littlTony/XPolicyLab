"""GR00T target-space contract, distinct from the absolute physical dataset."""
from pathlib import Path
import hashlib
import json
import numpy as np

CONTRACT_ID='dexbench-groot-anchor-relative-eef9-absolute-wuji20-rowrot6d-h40-v3'
STORAGE_CONTRACT_ID='dexbench-native-absolute-ee58-rowrot6d-wuji20-rgb-v1'
HORIZON=40
GROUPS=[('left_eef',0,9),('left_hand',9,29),('right_eef',29,38),('right_hand',38,58)]
POLICY_CONTRACT=dict(id=CONTRACT_ID,storage_contract_id=STORAGE_CONTRACT_ID,
    state='absolute_world_xyz_rot6d_hand20',eef_action='inv(T_current_state) @ T_absolute_target',
    reference_frame='current_end_effector_local_frame_fixed_for_all_40_targets',
    pretrained_action_horizon=40,max_state_dim=132,max_action_dim=132,
    statistics_window_policy='native_complete_windows_max(0,N-39)_no_padding',
    hand_action='absolute_wuji20',horizon=HORIZON,rot6d='matrix_rows_0_then_1',
    groups=['left_eef','left_hand','right_eef','right_hand'],representations=['RELATIVE','ABSOLUTE','RELATIVE','ABSOLUTE'],
    simulator_output='absolute_world_xyz_wxyz_hand20',native_processor_unapplies_relative_once=True)

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def relative_values_digest(statistics):
    values={}
    for group in ['left_eef','right_eef']:
        values[group]={}
        for key in ['min','max','q01','q99','mean','std']:
            value=np.asarray(statistics[group][key],dtype=np.float64)
            if value.shape!=(HORIZON,9) or not np.isfinite(value).all():raise ValueError(f'Invalid relative stats {group}/{key}: expected ({HORIZON},9)')
            values[group][key]=value.tolist()
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def absolute_group_values(raw_statistics):
    return {modality:{group:{key:raw_statistics[raw][key][a:b]
        for key in ['min','max','q01','q99','mean','std']} for group,a,b in GROUPS}
        for modality,raw in [('state','observation.state'),('action','action')]}

def absolute_values_digest(statistics):
    values={}
    for modality in ['state','action']:
        values[modality]={}
        for group,a,b in GROUPS:
            values[modality][group]={}
            for key in ['min','max','q01','q99','mean','std']:
                value=np.asarray(statistics[modality][group][key],dtype=np.float64)
                if value.shape!=(b-a,) or not np.isfinite(value).all():raise ValueError('Invalid saved absolute state/hand statistics')
                values[modality][group][key]=value.tolist()
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def native_merged_statistics(statistics):
    # The native mixture recomputes even a single dataset's variance; bind those
    # exact saved processor values separately from the raw source-stat file.
    from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
    return {modality:merge_statistics([statistics[modality]],[1.],
        is_relative_stats=(modality=='relative_action'))
        for modality in ['state','action','relative_action']}

def validate_policy_contract(value):
    for key,expected in POLICY_CONTRACT.items():
        if value.get(key)!=expected:raise ValueError(f'GR00T checkpoint target contract mismatch: {key}; old absolute/H16 checkpoints are incompatible')
    for key in ['source_sha256','relative_values_sha256','relative_stats_sha256','absolute_stats_sha256','absolute_values_sha256']:
        if not isinstance(value.get(key),str) or len(value[key])!=64:raise ValueError(f'Missing bound GR00T statistics: {key}')
    return value

def validate_saved_processor(checkpoint,contract):
    """Reject absolute/foreign statistics before a GPU model is constructed."""
    checkpoint=Path(checkpoint);validate_policy_contract(contract)
    model_config=json.loads((checkpoint/'config.json').read_text())
    for key,expected in [('action_horizon',HORIZON),('max_state_dim',132),('max_action_dim',132)]:
        if model_config.get(key)!=expected:raise ValueError(f'Saved GR00T model {key} differs from the pretrained architecture')
    if not (checkpoint/'processor_config.json').is_file() and (checkpoint/'processor').is_dir():checkpoint=checkpoint/'processor'
    kwargs=json.loads((checkpoint/'processor_config.json').read_text())['processor_kwargs']
    for key,expected in [('max_action_horizon',HORIZON),('max_state_dim',132),('max_action_dim',132)]:
        if kwargs.get(key)!=expected:raise ValueError(f'Saved GR00T processor {key} differs from the pretrained architecture')
    if kwargs.get('use_relative_action') is not True:raise ValueError('Saved GR00T processor does not enable relative actions')
    configurations=kwargs['modality_configs'];statistics=json.loads((checkpoint/'statistics.json').read_text())
    matches=[]
    for tag,config in configurations.items():
        if config.get('state',{}).get('modality_keys')!=POLICY_CONTRACT['groups']:continue
        action=config.get('action',{})
        if action.get('modality_keys')!=POLICY_CONTRACT['groups']:continue
        reps=[str(c['rep']).upper() for c in action.get('action_configs',[])]
        if reps!=POLICY_CONTRACT['representations']:raise ValueError('Saved GR00T EEF/hand representations differ')
        if config['state'].get('delta_indices')!=[0]:raise ValueError('Saved GR00T state anchor differs')
        for (key,_,_),action_config in zip(GROUPS,action['action_configs']):
            eef=key.endswith('eef')
            if str(action_config['type']).upper()!=('EEF' if eef else 'NON_EEF'):raise ValueError('Saved GR00T EEF/hand action type differs')
            if str(action_config['format']).upper()!=('XYZ_ROT6D' if eef else 'DEFAULT'):raise ValueError('Saved GR00T action format differs')
            if action_config.get('state_key') not in [None,key]:raise ValueError('Saved GR00T action reference state differs')
        if action.get('delta_indices')!=list(range(HORIZON)):raise ValueError('Saved GR00T action horizon differs')
        value=statistics[tag]
        if relative_values_digest(value['relative_action'])!=contract.get('processor_relative_values_sha256'):raise ValueError('Saved processor relative stats do not match the native merged run contract')
        if absolute_values_digest(value)!=contract.get('processor_absolute_values_sha256'):raise ValueError('Saved processor absolute state/hand stats do not match the native merged run contract')
        matches.append(tag)
    if len(matches)!=1:raise ValueError('Saved processor lacks one unambiguous DexBench relative embodiment')
    return matches[0]
