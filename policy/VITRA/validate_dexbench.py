"""CPU-only validation of direct LeRobot IO, geometry, stats and live codec."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.transform import Rotation

P=Path(__file__).resolve().parent
sys.path[:0]=[str(P/'VITRA'),str(P.parents[2])]
from vitra.datasets.lerobot_dataset import LeRobotDatasetCore,native_vectors,native_chunk_actions,native_episode_values,action_slot_count
from vitra.datasets.spark0_dataset import (runtime_observation_to_native,
    native_action_chunk_to_xpolicy,native_state_to_human,human_action_to_native,
    RoboDatasetCore)
from XPolicyLab.policy.VITRA.model import Model


def observation(core,episode,frame,sample):
    state,_=core.index.numeric(episode);calibration=core.calibration(episode)
    return dict(instruction=core.records[episode]['instruction'],state={
        'left_ee_pose':state[frame,:7], 'left_ee_joint_state':state[frame,7:27],
        'right_ee_pose':state[frame,27:34], 'right_ee_joint_state':state[frame,34:54]},
        vision={camera:dict(color=image,extrinsic_matrix=calibration[camera+'_extrinsics'][frame],
                            intrinsic_matrix=calibration[camera+'_intrinsics'][frame])
                for camera,image in zip(['cam_head','cam_left_wrist','cam_right_wrist'],sample['multi_view_images'])})


def adapter(core):
    # Use the real runtime input/output methods with a fixed-output test model.
    # No neural network weights, optimizer, CUDA or training are constructed.
    model=Model.__new__(Model)
    model.mapping_path=Path(core.mapping_path)
    model._runtime_observation_to_native=runtime_observation_to_native
    model._native_state_to_human=native_state_to_human
    model._human_action_to_native=human_action_to_native
    model._native_action_chunk_to_xpolicy=native_action_chunk_to_xpolicy
    model._mano_codec=None;model._mano_fit_previous={};model.representation='wuji20'
    model._dexbench_three_view=True;model.default_prompt='';model.default_fov=np.zeros(2)
    model.normalizer=core.gaussian_normalizer
    model.chunk_size=16;model.execute_action_chunk=16
    model._torch=torch;model.num_ddim_steps=10;model.cfg_scale=1.;model.sample_times=1
    model.ee_dims=[20,20]
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--compare-legacy',action='store_true')
    args=parser.parse_args()
    if args.compare_legacy:
        raise ValueError('Legacy measured-state delta data use the old codec; compare-legacy is unsupported for command-step-v2')
    # A storage-only migration must be usable without opening any HDF5 file.
    with patch('h5py.File',side_effect=AssertionError('Direct LeRobot reader opened HDF5')):
        core=LeRobotDatasetCore(args.root,load_images=False)
        assert len(core)==int(core.index.info['total_frames']) and len(core.records)==int(core.index.info['total_episodes'])
        core.set_global_data_statistics(core.data_statistics)
        core[0];core[len(core)-1]
    legacy=None
    if args.compare_legacy:
        oldroot=P/'data/dexbench_vitra'
        legacy=RoboDatasetCore(str(oldroot/'TeleData'),statistics_path=str(oldroot/'teledata_statistics.json'),
                               action_future_window_size=15,load_images=False)
        legacy.set_global_data_statistics(core.data_statistics)
        lookup={int(Path(path).stem.split('_')[-1]):i for i,path in enumerate(legacy.episode_paths)}
    worst_pos=0.;worst_rot=0.
    indices=set(np.linspace(0,len(core)-1,32,dtype=int).tolist())
    indices.update([int(core.index.cumulative[0])-1,int(core.index.cumulative[0]),len(core)-1])
    for index in sorted(indices):
        episode,frame=core.index.locate(index)
        sample=core[index]
        state,action=core.index.numeric(episode)
        n=len(state);valid_count=min(16,n-frame)
        if legacy is not None:
            oldep=lookup[episode];start=0 if oldep==0 else legacy._cumulative_lengths[oldep-1]
            old=legacy[start+frame]
            for key in ['current_state','action_list','action_mask','current_state_mask','fov','intrinsics']:
                np.testing.assert_allclose(sample[key],old[key],atol=2e-6,rtol=0,err_msg=key)
            assert sample['instruction']==old['instruction']
        # Real adapter input and predicted-action decode, with physical oracle targets.
        cal=core.calibration(episode)
        runtime_obs=dict(instruction=core.records[episode]['instruction'],state={
            'left_ee_pose':state[frame,:7],'left_ee_joint_state':state[frame,7:27],
            'right_ee_pose':state[frame,27:34],'right_ee_joint_state':state[frame,34:54]},
            vision={'cam_head':dict(color=np.zeros((480,640,3),np.uint8),
                extrinsic_matrix=cal['cam_head_extrinsics'][frame],intrinsic_matrix=cal['cam_head_intrinsics'][frame])})
        context=runtime_observation_to_native(runtime_obs,core.mapping_path)
        np.testing.assert_allclose(context['native_state'],sample['current_state'],atol=2e-6,rtol=0)
        transformed=core.transform_trajectory(copy.deepcopy(sample))
        assert transformed['current_state'].shape==(212,) and transformed['action_list'].shape==(16,192)
        assert not transformed['action_mask'][valid_count:].any()
        physical=core.gaussian_normalizer.unnormalize_action(human_action_to_native(transformed['action_list'].numpy(),core.mapping))
        decoded=native_action_chunk_to_xpolicy(physical[:valid_count],context['context'])
        for t,command in enumerate(decoded):
            for side,start in [('left',0),('right',27)]:
                target=action[frame+t,start:start+7]
                error=float(np.max(np.abs(command[side+'_ee_pose'][:3]-target[:3])))
                worst_pos=max(worst_pos,error)
                predicted=Rotation.from_quat(command[side+'_ee_pose'][[4,5,6,3]])
                expected=Rotation.from_quat(target[[4,5,6,3]])
                angle=float((predicted.inv()*expected).magnitude());worst_rot=max(worst_rot,angle)
                np.testing.assert_allclose(command[side+'_ee_joint_state'],action[frame+t,start+7:start+27],atol=2e-6,rtol=0)
                assert error<3e-5 and angle<1e-4,(episode,frame,t,error,angle)
    # Validate RGB video offsets using a sequential, independent PyAV read.
    core.load_images=True
    episode=next(i for i,row in core.index.episodes.iterrows()
                 if row['videos/observation.images.cam_high/from_timestamp']*core.index.info['fps']>500)
    start=0 if episode==0 else int(core.index.cumulative[episode-1])
    with patch('h5py.File',side_effect=AssertionError('LeRobot images opened HDF5')):
        sample=core[start+1]
    row=core.index.episodes.iloc[episode]
    import av
    for camera,image in zip(['cam_high','cam_left_wrist','cam_right_wrist'],sample['multi_view_images']):
        key='observation.images.'+camera
        path=core.index.root/core.index.info['video_path'].format(video_key=key,
            chunk_index=int(row[f'videos/{key}/chunk_index']),file_index=int(row[f'videos/{key}/file_index']))
        absolute=int(round(row[f'videos/{key}/from_timestamp']*core.index.info['fps']))+1
        with av.open(str(path)) as stream:
            for i,video_frame in enumerate(stream.decode(video=0)):
                if i==absolute:
                    np.testing.assert_array_equal(image,video_frame.to_ndarray(format='rgb24'));break
            else:raise AssertionError('Missing reference video frame')
    model=adapter(core);obs=observation(core,episode,1,sample)
    payload=model._encode_observation(obs)
    transformed=core.transform_trajectory(copy.deepcopy(sample))
    np.testing.assert_allclose(payload['current_state'],transformed['current_state'].numpy(),atol=2e-6,rtol=0)
    np.testing.assert_array_equal(payload['current_state_mask'],transformed['current_state_mask'])
    np.testing.assert_allclose(payload['fov'],sample['fov'],atol=1e-6,rtol=0)
    for a,b in zip(payload['image'],sample['multi_view_images']):np.testing.assert_array_equal(a,b)
    bad_obs=copy.deepcopy(obs)
    bad_obs['vision']['cam_left_wrist']['color']=bad_obs['vision']['cam_left_wrist']['color'][...,0]
    try:model._encode_observation(bad_obs)
    except ValueError:pass
    else:raise AssertionError('Invalid wrist RGB shape was accepted')
    class FixedModel:
        def predict_action(self,**kwargs):
            assert len(kwargs['image'])==3
            return transformed['action_list'].numpy()[None]
    model.model=FixedModel()
    actual=model._predict_one(payload)
    for t,command in enumerate(actual):
        target=core.index.numeric(episode)[1][1+t]
        for side,base in [('left',0),('right',27)]:
            np.testing.assert_allclose(command[side+'_ee_pose'][:3],target[base:base+3],atol=3e-5,rtol=0)
            np.testing.assert_allclose(command[side+'_ee_joint_state'],target[base+7:base+27],atol=2e-6,rtol=0)
    # Exercise the production FrameDataset and native offline processor.
    from transformers import AutoProcessor
    from vitra.datasets.dataset import FrameDataset
    processor=AutoProcessor.from_pretrained('/personal/xiangpc/0813_Xpolicylab_bench/VITRA/pretrain_model/paligemma2-3b-mix-224-local',local_files_only=True)
    framed=FrameDataset(args.root,'robo_dataset',action_future_window_size=15,processor=processor,data_type='robot')
    assert isinstance(framed.episodic_dataset_core,LeRobotDatasetCore)
    framed.episodic_dataset_core.set_global_data_statistics(core.data_statistics)
    processed=framed[start+1]
    assert tuple(processed['pixel_values'].shape)==(3,3,224,224)
    image_token=processor.tokenizer.convert_tokens_to_ids('<image>')
    assert int((processed['input_ids']==image_token).sum())==3*processor.image_seq_length
    from vitra.utils.data_utils import PaddedCollatorForHandPrediction
    collator=PaddedCollatorForHandPrediction(processor.tokenizer.model_max_length,
                                           processor.tokenizer.pad_token_id)
    loader=torch.utils.data.DataLoader(framed,batch_size=2,num_workers=2,collate_fn=collator)
    iterator=iter(loader);batch=next(iterator)
    assert tuple(batch['pixel_values'].shape)==(6,3,224,224)
    assert tuple(batch['actions'].shape)==(2,16,192)
    assert tuple(batch['current_state'].shape)==(2,212)
    del iterator,loader
    # Intentionally mismatched metadata must be rejected, never silently loaded.
    with tempfile.TemporaryDirectory(prefix='vitra_bad_stats_') as folder:
        path=Path(folder)/'stats.json'
        wrong=json.loads((Path(args.root)/'meta/vitra/normalization_stats.json').read_text())
        wrong['action_contract']['mapping_sha256']='0'*64
        path.write_text(json.dumps(wrong))
        try:LeRobotDatasetCore(args.root,statistics_path=str(path),load_images=False)
        except ValueError:pass
        else:raise AssertionError('Wrong hand-mapping stats were accepted')
    original=core.records[episode]['calibration_sha256']
    core.records[episode]['calibration_sha256']='0'*64;core._calibration_cache.clear()
    try:core.calibration(episode)
    except ValueError:pass
    else:raise AssertionError('Wrong camera calibration checksum was accepted')
    finally:core.records[episode]['calibration_sha256']=original
    print('PASS two-worker DataLoader, malformed RGB rejection, mismatched stats/calibration rejection',flush=True)
    print('PASS real RGB three-view video offsets, native processor, state212/action192 masks, all valid terminal targets, live state/action codec parity',flush=True)
    print(f'PASS full-chunk absolute EE reconstruction: maximum xyz error={worst_pos:.3g} m, rotation={worst_rot:.3g} rad',flush=True)
    if args.full:
        totals={};count=0;action_count=0;core.load_images=False
        for episode in range(len(core.records)):
            state,action=core.index.numeric(episode)
            values=native_episode_values(state,action,core.calibration(episode)['cam_head_extrinsics'],core.mapping)
            count+=len(state)
            action_count+=len(values[1])
            # Cross-check vectorized moments against the actual loader at the
            # first, middle and terminal anchors of every episode.
            for frame in sorted({0,len(state)//2,len(state)-1}):
                start_index=0 if episode==0 else int(core.index.cumulative[episode-1])
                loaded=core[start_index+frame]
                offset=sum(min(16,len(state)-i) for i in range(frame))
                valid=min(16,len(state)-frame)
                np.testing.assert_allclose(loaded['current_state'],values[0][frame],atol=2e-6,rtol=0)
                np.testing.assert_allclose(loaded['action_list'][:valid],values[1][offset:offset+valid],atol=2e-6,rtol=0)
                assert not loaded['action_mask'][valid:].any()
            for key,matrix in zip(['state','action'],values):
                for side,offset in [('left',0),('right',26)]:
                    name=f'{key}_{side}';x=matrix[:,offset:offset+26].astype(np.float64)
                    if name not in totals:totals[name]=[np.zeros(26),np.zeros(26)]
                    totals[name][0]+=x.sum(0);totals[name][1]+=(x*x).sum(0)
        stats=json.loads((Path(args.root)/'meta/vitra/normalization_stats.json').read_text())
        for key,(total,square) in totals.items():
            denominator=action_count if key.startswith('action_') else count
            mean=total/denominator;std=np.maximum(np.sqrt(np.maximum(square/denominator-mean*mean,0)),1e-6)
            np.testing.assert_allclose(mean,stats[key]['mean'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(std,stats[key]['std'],atol=1e-12,rtol=0)
        assert stats['num_action_samples']==action_count==sum(action_slot_count(n) for n in core.episode_lengths)
        print(f'PASS independently recomputed native52 stats: {count} state rows / {action_count} valid command slots; actual loader windows checked for every episode; all camera checksums valid',flush=True)
    if legacy is not None:print('PASS direct LeRobot versus legacy HDF5 native numeric sample parity',flush=True)
    print('No model weights loaded, no training started.')


if __name__=='__main__':main()
