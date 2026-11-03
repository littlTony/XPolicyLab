"""Lightweight CPU checks of the actual VITRA inference boundary/guards."""
from pathlib import Path
import copy
import hashlib
import importlib
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'VITRA'),str(P.parents[2]),str(P)]
module=importlib.import_module('XPolicyLab.policy.VITRA.model')
from vitra.datasets.lerobot_dataset import LeRobotDatasetCore
from validate_dexbench import adapter


def main():
    root=Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks')
    stats_path=root/'meta/vitra/normalization_stats.json'
    statistics=json.loads(stats_path.read_text());mapping=P/'mapping_wuji20_mano45.json'
    digest=hashlib.sha256(stats_path.read_bytes()).hexdigest()
    mapping_sha=hashlib.sha256(mapping.read_bytes()).hexdigest()
    config=dict(data_representation='wuji20',action_contract=statistics['action_contract'],
        statistics_path=str(stats_path),statistics_sha256=digest,mapping_sha256=mapping_sha,fwd_pred_next_n=16)
    module._validate_dexbench_inference_contract(config,stats_path,mapping)
    # Exercise actual Model.__init__ metadata routing and stop before CUDA/model.
    with tempfile.TemporaryDirectory(prefix='vitra-inference-guard-') as folder:
        temp=Path(folder);cfg=temp/'config.json';bad_stats=temp/'wrong_stats.json';bad_map=temp/'wrong_mapping.json'
        altered=copy.deepcopy(statistics);altered['action_left']['mean'][0]+=.001;bad_stats.write_text(json.dumps(altered))
        altered=json.loads(mapping.read_text());altered['calibration']['left_ee_to_vitra_wrist'][0][3]+=.01;bad_map.write_text(json.dumps(altered))
        cases=[]
        old=copy.deepcopy(config);old['action_contract']['id']='dexbench-vitra-wuji20-sparse-camera-step-v1'
        cases.append(('old codec',old,stats_path,None))
        cases.append(('wrong statistics',config,bad_stats,None))
        cases.append(('wrong wrist mapping',config,stats_path,bad_map))
        absent=copy.deepcopy(config);absent.pop('statistics_sha256')
        cases.append(('unbound statistics',absent,stats_path,None))
        for name,saved,selected_stats,selected_map in cases:
            cfg.write_text(json.dumps(saved))
            model_cfg=dict(env_cfg_type='tianji_marvin_wuji',action_type='ee')
            if selected_map:model_cfg['mapping_path']=str(selected_map)
            with patch.object(module,'_resolve_weights',return_value=(temp/'unused.pt',temp)), \
                 patch.object(module,'_resolve_config_and_statistics',return_value=(cfg,selected_stats)), \
                 patch('torch.cuda.is_available',side_effect=AssertionError('CUDA reached before metadata rejection')):
                try:module.Model(model_cfg)
                except ValueError:pass
                else:raise AssertionError(name+' accepted')
        # Actual constructor must not treat foundation/SparkArena config as a
        # DexBench fine-tune simply because its head has compatible dimensions.
        for benchmark in ('absent', None, 'DexBench'):
            for marker in (None, 'spark0-wuji20-native'):
                saved=copy.deepcopy(config)
                if marker is None:saved.pop('action_contract')
                else:saved['action_contract']={'id':marker}
                cfg.write_text(json.dumps(saved))
                model_cfg=dict(env_cfg_type='tianji_marvin_wuji',action_type='ee')
                if benchmark!='absent':model_cfg['bench_name']=benchmark
                with patch.object(module,'_resolve_weights',return_value=(temp/'unused.pt',temp)), \
                     patch.object(module,'_resolve_config_and_statistics',return_value=(cfg,stats_path)), \
                     patch('torch.cuda.is_available',side_effect=AssertionError('CUDA reached before benchmark rejection')):
                    try:module.Model(model_cfg)
                    except ValueError as exc:assert 'command-step-v2 trained checkpoint' in str(exc)
                    else:raise AssertionError('Unmarked/foundation checkpoint accepted for DexBench')
        # Historical use remains available when a different benchmark is named.
        saved=copy.deepcopy(config);saved.pop('action_contract');cfg.write_text(json.dumps(saved))
        with patch.object(module,'_resolve_weights',return_value=(temp/'unused.pt',temp)), \
             patch.object(module,'_resolve_config_and_statistics',return_value=(cfg,stats_path)), \
             patch('torch.cuda.is_available',side_effect=AssertionError('explicit historical benchmark reached CUDA boundary')):
            try:module.Model(dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',bench_name='SparkArena'))
            except AssertionError as exc:assert 'explicit historical benchmark reached CUDA boundary' in str(exc)
            else:raise AssertionError('Historical benchmark test did not reach its CUDA boundary')
    print('PASS real VITRA constructor rejects old/unmarked/foundation DexBench checkpoints, wrong stats/mapping before CUDA; explicit historical benchmark preserved')
    core=LeRobotDatasetCore(str(root),load_images=False)
    core.set_global_data_statistics(core.data_statistics)
    state,_=core.index.numeric(0);cal=core.calibration(0)
    def observation(idx):
        return {'env_idx':idx,'instruction':core.records[0]['instruction'],
            'state':{'left_ee_pose':state[0,:7],'left_ee_joint_state':state[0,7:27],
                     'right_ee_pose':state[0,27:34],'right_ee_joint_state':state[0,34:54]},
            'vision':{key:dict(color=np.zeros((480,640,3),np.uint8),
                extrinsic_matrix=cal[key+'_extrinsics'][0],intrinsic_matrix=cal[key+'_intrinsics'][0])
                for key in ['cam_head','cam_left_wrist','cam_right_wrist']}}
    m=adapter(core);m.env_cfg_type='tianji_marvin_wuji'
    m._latest_env_idx_list=[];m._observations={};m._raw_observations={}
    m._mano_pending_keyframes={};m._mano_next_tick={};m._mano_warned_odd_query=set();m._h1_adapter=None
    class ResetModel:
        resets=0;clears=0
        def reset(self):self.resets+=1
        def clear_cache(self):self.clears+=1
    m.model=ResetModel()
    m._predict_one=lambda payload:[{'env_idx':payload['env_idx']}]
    m.update_obs_batch([observation(5),observation(2)])
    assert [x[0]['env_idx'] for x in m.get_action_batch()]==[5,2]
    assert [x[0]['env_idx'] for x in m.get_action_batch([2,5])]==[2,5]
    m.update_obs_batch([observation(5)])
    assert [x[0]['env_idx'] for x in m.get_action_batch()]==[5]
    try:m.get_action_batch([2])
    except RuntimeError:pass
    else:raise AssertionError('Dropped VITRA environment used stale observations')
    m.reset();assert not m._observations and m.model.resets==1 and m.model.clears==1
    try:m.get_action()
    except RuntimeError:pass
    else:raise AssertionError('VITRA inference after reset accepted stale data')
    print('PASS actual VITRA batch active/default/explicit order, environment dropout, reset/cache clearing; no weights loaded')


if __name__=='__main__':main()
