"""Add VITRA camera metadata and native52 stats to the existing LeRobot export.

Only preparation reads source HDF5 calibration. Training never opens HDF5.
No image, source parquet, or generic LeRobot statistics are changed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
import sys
import time
import h5py
import numpy as np

P = Path(__file__).resolve().parent
sys.path[:0] = [str(P/'VITRA'), str(P.parents[2])]
from vitra.datasets.lerobot_io import LeRobotV3Index, metadata_signature
from vitra.datasets.lerobot_dataset import native_episode_values,COMMAND_STEP_CONTRACT_ID,COMMAND_STEP_HORIZON
from vitra.datasets.spark0_dataset import load_wuji_mapping

INDEX = None


def worker_init(root):
    global INDEX
    INDEX = LeRobotV3Index(root)


def expand(value, n, shape):
    value = np.asarray(value,dtype=np.float64)
    if value.shape == shape:
        value = np.broadcast_to(value,(n,*shape)).copy()
    elif value.shape == (n+1,*shape):
        value = value[:n].copy()
    elif value.shape != (n,*shape):
        raise ValueError(f'Unexpected calibration shape {value.shape}, need {n} x {shape}')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite camera calibration')
    return value


def one(job):
    episode, report = job
    state, action = INDEX.numeric(episode)
    n = len(state)
    if n != report['frames']:
        raise ValueError(f'Preview report length mismatch for episode {episode}')
    if str(INDEX.episodes.iloc[episode]['tasks'][0]) != report['task']:
        raise ValueError(f'Preview report instruction/index mismatch {episode}')
    values = {}
    with h5py.File(report['source'],'r') as source:
        for side,start in [('left',0),('right',27)]:
            np.testing.assert_allclose(state[:,start:start+7],source[f'state/{side}_ee_poses'][:n].reshape(n,7),atol=1e-6,rtol=0)
            np.testing.assert_allclose(action[:,start:start+7],source[f'action/{side}_ee_poses'][:n].reshape(n,7),atol=1e-6,rtol=0)
        for camera in ['cam_head','cam_left_wrist','cam_right_wrist']:
            extr = expand(source[f'vision/{camera}/extrinsic_matrix'][()],n,(4,4))
            intr = expand(source[f'vision/{camera}/intrinsic_matrix'][()],n,(3,3))
            np.testing.assert_allclose(extr[:,3,:],np.broadcast_to([0,0,0,1],(n,4)),atol=1e-7)
            np.testing.assert_allclose(np.linalg.det(extr[:,:3,:3]),1,atol=1e-5)
            np.testing.assert_allclose(np.swapaxes(extr[:,:3,:3],-1,-2)@extr[:,:3,:3],np.broadcast_to(np.eye(3),(n,3,3)),atol=1e-5)
            if np.any(intr[:,[0,1],[0,1]]<=0):raise ValueError('Invalid focal length')
            values[camera+'_extrinsics'] = extr
            values[camera+'_intrinsics'] = intr
    relative = f'calibration/episode_{episode:06d}.npz'
    path = INDEX.root/'meta/vitra'/relative
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**values)
    temp.replace(path)
    mapping = load_wuji_mapping(P/'mapping_wuji20_mano45.json')
    states, actions = native_episode_values(state,action,values['cam_head_extrinsics'],mapping)
    moments = {}
    for key, matrix in [('state',states),('action',actions)]:
        for side,start in [('left',0),('right',26)]:
            x = matrix[:,start:start+26].astype(np.float64)
            moments[f'{key}_{side}'] = (x.sum(0),(x*x).sum(0))
    return dict(episode_index=episode,frames=n,action_samples=len(actions),calibration=relative,
                calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                instruction=report['instruction'],task=report['task']), moments


def write_json(path,value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks')
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    index = LeRobotV3Index(args.root)
    meta = index.root/'meta/vitra'
    (meta/'calibration').mkdir(parents=True,exist_ok=True)
    # One writer at a time; release the lock even after an exception.
    lock = meta/'.prepare.lock'
    fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        reports = json.loads((index.root/'meta/preview_report.json').read_text())
        if len(reports)!=len(index.episodes):raise ValueError('Preview report episode count mismatch')
        records=[]; totals={}; count=0; action_count=0
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(index.root),)) as pool:
            for record,moments in pool.map(one,enumerate(reports),chunksize=4):
                records.append(record);count+=record['frames'];action_count+=record['action_samples']
                for key,(total,square) in moments.items():
                    if key not in totals:totals[key]=[np.zeros(26),np.zeros(26)]
                    totals[key][0]+=total;totals[key][1]+=square
                if len(records)%100==0:print(f'Prepared calibration/native stats {len(records)}/{len(reports)} episodes, {count} rows',flush=True)
        assert count==len(index)
        mapping = load_wuji_mapping(P/'mapping_wuji20_mano45.json')
        mapping_sha = hashlib.sha256((P/'mapping_wuji20_mano45.json').read_bytes()).hexdigest()
        signature = metadata_signature(index.root)
        calibration_sha = hashlib.sha256(''.join(r['calibration_sha256'] for r in records).encode()).hexdigest()
        contract = dict(id=COMMAND_STEP_CONTRACT_ID,
            state='camera_anchor_absolute_xyz_eulerxyz_plus_absolute_wuji20',
            action='first_target_minus_measured_state_then_target_minus_previous_target; dR=Rtarget_RpreviousT; absolute_wuji20',
            action_horizon=COMMAND_STEP_HORIZON,action_reference='single_current_head_camera_anchor_for_whole_chunk',
            mapping_codec=mapping['codec_id'],mapping_sha256=mapping_sha,state_dim=212,action_dim=192,
            valid_action_rows='all_exported_rows_no_extra_shift',RGB=True,
            cameras=['cam_head','cam_left_wrist','cam_right_wrist'])
        statistics = dict(dataset_name='dexbench_vitra_lerobot',representation='wuji20',
            state_dimension_per_hand=26,action_dimension_per_hand=26,num_traj=count,
            num_samples=count,num_action_samples=action_count,num_episodes=len(records),action_contract=contract,
            source_metadata_sha256=signature,
            source_calibration_sha256=calibration_sha,
            statistics_space='native_camera_state52_and_step_action52_before_sparse_mapping',
            sampling='state: each current row once; action: every valid slot of every 16-step chunk; masked terminal padding excluded')
        for key,(total,square) in totals.items():
            denominator=action_count if key.startswith('action_') else count
            mean=total/denominator;std=np.sqrt(np.maximum(square/denominator-mean*mean,0))
            statistics[key]=dict(mean=mean.tolist(),std=np.maximum(std,1e-6).tolist())
        numeric=[]
        for path in sorted((index.root/'data').rglob('*.parquet')):
            stat=path.stat();numeric.append(dict(path=str(path.relative_to(index.root)),size=stat.st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = dict(storage='lerobot_v3',episode_count=len(records),frame_count=count,action_sample_count=action_count,
            lerobot_metadata_sha256=signature,mapping_sha256=mapping_sha,numeric_files=numeric,
            calibration_sha256=calibration_sha,
            calibration_convention='camera_to_env_graphics; native codec applies diag(1,-1,-1,1) once',
            action_contract=contract,episodes=records)
        write_json(meta/'normalization_stats.json',statistics)
        manifest['statistics_sha256']=hashlib.sha256((meta/'normalization_stats.json').read_bytes()).hexdigest()
        write_json(meta/'manifest.json',manifest)
        print(f'READY: {len(records)} episodes / {count} rows; LeRobot Parquet+three RGB MP4 views; standalone camera metadata and native52 stats',flush=True)
    finally:
        lock.unlink()


if __name__=='__main__':main()
