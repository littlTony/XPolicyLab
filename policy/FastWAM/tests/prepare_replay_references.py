"""Export CPU references through the native FastWAM training reader."""
import argparse,json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from fastwam.datasets.dexbench import DexBenchIndex
from fastwam.datasets.dexbench_contract import image_tensor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selected=json.loads(args.selection.read_text())
    cfg=OmegaConf.load(args.checkpoint/'config.yaml')
    index=DexBenchIndex(cfg.data.train.dataset_dirs[0],'all')
    train_ids=set(DexBenchIndex(cfg.data.train.dataset_dirs[0],'train').episodes.episode_index)
    assert len(selected)==24 and all(ep['episode_index'] in train_ids for ep in selected)
    processor=instantiate(cfg.data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.checkpoint/'dataset_stats.json'))
    args.output.mkdir(parents=True,exist_ok=True)
    for ep in selected:
        row=index.episodes[index.episodes.episode_index==ep['episode_index']].iloc[0]
        global_index=int(index.episodes[index.episodes.episode_index<ep['episode_index']]['length'].sum())
        _,t,action,state,_,_=index.numeric(global_index)
        assert t==0
        video=image_tensor(index.cameras(row,[0]),tuple(cfg.data.train.video_size))[0].numpy()
        np.savez_compressed(args.output/f"episode_{ep['episode_index']:06d}.npz",image=video,state=state[0],action=action,
                            normalized_state=processor.normalizer.normalizers['state']['default'].forward(__import__('torch').from_numpy(state))[0].numpy())
        print('REFERENCE',ep['task'],ep['episode_index'],flush=True)
    (args.output/'episode_selection.json').write_text(json.dumps(selected,indent=2)+'\n')

if __name__=='__main__':main()
