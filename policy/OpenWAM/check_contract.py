"""Verify the 154k foundation header and training data/normalization contract."""
from pathlib import Path
import json
import struct
import numpy as np
import yaml
from openwam.dataloader.dexbench_contract import CONTRACT,CONTRACT_ID,UNIFY_MAP
from openwam.dataloader.dexbench_provenance import validate_stats_source

POLICY=Path(__file__).resolve().parent
PRETRAIN=Path('/personal/xiangpc/0912_Xpolicylab_bench/openwam/pretrain_model/OpenWAM-Alpha-Pretrain-Foundation-Model')


def main():
    cfg=yaml.safe_load((POLICY/'OpenWAM/configs/dataloader/dexbench.yaml').read_text())
    assert cfg['contract_id']==CONTRACT_ID and cfg['data_format']=='lerobot_v3'
    assert cfg['normalize_mode']=='min-max' and not cfg.get('binary_action_dims')
    assert cfg['unify_action_map']==cfg['unify_state_map']==UNIFY_MAP
    info=json.loads((Path(cfg['dataset_dir'])/'meta/info.json').read_text())
    assert info['features']['action']['shape']==info['features']['observation.state']['shape']==[54]
    stats=np.load(cfg['normalization_stats_path'],allow_pickle=True).item()
    assert stats['action_contract']==CONTRACT
    validate_stats_source(stats, cfg['dataset_dir'], info)
    for key in ['eef','eef_state']:
        for stat in ['min','max','std','mean']:
            assert stats[key][stat].shape==(58,) and np.isfinite(stats[key][stat]).all()
    foundation=yaml.safe_load((PRETRAIN/'config.yaml').read_text())
    assert foundation['model']['architecture']['action_dim']==foundation['model']['architecture']['state_dim']==80
    weights=PRETRAIN/'checkpoint_step_154000.safetensors'
    with weights.open('rb') as handle:
        n=struct.unpack('<Q',handle.read(8))[0]
        header=json.loads(handle.read(n))
    end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
    assert weights.stat().st_size==8+n+end
    assert any(80 in v.get('shape',[]) for k,v in header.items() if 'action' in k.lower())
    print(f'Verified foundation step154000: unified80 action/state, {weights.stat().st_size:,} bytes')
    print(f'Verified {info["total_episodes"]} episodes / {info["total_frames"]} frames; raw58 -> 58 valid alpha80 slots; absolute world wxyz/Wuji20')


if __name__=='__main__':main()
