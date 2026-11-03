"""Vectorized equivalent of GR00T calculate_stats_for_key (axis=0, T x D).

Only the two EEF groups use relative statistics. Hands stay absolute. Every
episode is cross-checked against an unmodified native RelativeActionLoader on
one complete 40-target window before its windows enter the statistics. Episodes
shorter than40 contribute no windows, matching native statistics and training.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import importlib.util
import json
import os
import sys
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from relative_contract import POLICY_CONTRACT,HORIZON,relative_values_digest,sha256_file,absolute_group_values,absolute_values_digest,native_merged_statistics

P=Path(__file__).resolve().parent
DEFAULT_DATA=Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_ee58_v21')
ROOT=None
NATIVE_LOADERS={}


def rotation_matrices(row6d):
    # Match native Pose: float32 Gram-Schmidt, followed by scipy SO(3) projection.
    row6d=np.asarray(row6d,dtype=np.float32).reshape(-1,2,3)
    a=row6d[:,0];b=row6d[:,1]
    a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b-np.sum(a*b,axis=-1,keepdims=True)*a
    b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return Rotation.from_matrix(np.stack([a,b,np.cross(a,b)],axis=-2)).as_matrix()


def relative_windows(state9,action9):
    state9=np.asarray(state9,dtype=np.float32);action9=np.asarray(action9,dtype=np.float32)
    n=len(state9)-HORIZON+1
    if n<=0:return np.empty((0,HORIZON,9),np.float32)
    r0=rotation_matrices(state9[:n,3:]);rt=rotation_matrices(action9[:,3:])
    ids=np.arange(n)[:,None]+np.arange(HORIZON)[None]
    inverse=np.swapaxes(r0,-1,-2)
    dp=np.einsum('nij,ntj->nti',inverse,action9[ids,:3].astype(np.float64)-state9[:n,None,:3])
    dr=inverse[:,None]@rt[ids]
    # Native constructs a Pose from the relative homogeneous matrix, then .rot6d.
    dr=Rotation.from_matrix(dr.reshape(-1,3,3)).as_matrix().reshape(n,HORIZON,3,3)
    return np.concatenate([dp,dr[...,:2,:].reshape(n,HORIZON,6)],axis=-1).astype(np.float32)


class _OneWindow:
    def __init__(self,base,start):self.base=base;self.start=start
    def __getitem__(self,episode):return self.base[episode].iloc[self.start:self.start+HORIZON]


def worker_init(data):
    global ROOT,NATIVE_LOADERS
    ROOT=Path(data)
    import dexbench_config  # registers relative EEF / absolute hand modality config
    from gr00t.data.stats import RelativeActionLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    NATIVE_LOADERS={k:RelativeActionLoader(ROOT,EmbodimentTag.NEW_EMBODIMENT,k) for k in ['left_eef','right_eef']}


def one(record):
    episode=int(record['episode_index'])
    file=ROOT/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
    df=pq.read_table(file,columns=['observation.state','action']).to_pandas()
    state=np.stack(df['observation.state']);action=np.stack(df['action'])
    if len(df)!=record['length']:raise ValueError(f'Invalid episode length {episode}')
    output={};error=0.
    for key,start in [('left_eef',0),('right_eef',29)]:
        values=relative_windows(state[:,start:start+9],action[:,start:start+9])
        if len(values):
            anchor=episode%len(values)
            native=NATIVE_LOADERS[key];base=native.loader
            try:
                native.loader=_OneWindow(base,anchor)
                expected=np.asarray(native.load_relative_actions(episode))
            finally:native.loader=base
            np.testing.assert_allclose(values[anchor:anchor+1],expected,atol=8e-7,rtol=2e-6)
            error=max(error,float(np.max(np.abs(values[anchor:anchor+1]-expected))))
        else:
            # Native stats and ShardedSingleStepDataset both exclude N<H.
            # Actually call the original loader for every short episode too.
            expected=NATIVE_LOADERS[key].load_relative_actions(episode)
            if len(expected):raise ValueError(f'Native short episode unexpectedly produced windows: {episode}')
        output[key]=values
    return output,error


def check_ready(data,allow_legacy_binding=False):
    data=Path(data)
    prep=Path('/personal/wenwei/DexBench/data_preprocess/prepare_native_ee58.py')
    spec=importlib.util.spec_from_file_location('prepare_ee58',prep);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=Path(json.loads((data/'meta/source_fingerprint.json').read_text())['source_root'])
    source_value=module.check_view(source,data)
    marker=json.loads((data/'meta/groot_relative_provenance.json').read_text())
    relative=json.loads((data/'meta/relative_stats.json').read_text())
    info=json.loads((data/'meta/info.json').read_text())
    records=[json.loads(line) for line in (data/'meta/episodes.jsonl').read_text().splitlines()]
    expected_windows=sum(max(0,int(record['length'])-HORIZON+1) for record in records)
    short=sum(int(record['length'])<HORIZON for record in records)
    if marker['source_sha256']!=source_value['sha256'] or marker['frame_count']!=info['total_frames']:raise ValueError('Stale GR00T relative stats/source binding')
    if marker['relative_stats_sha256']!=sha256_file(data/'meta/relative_stats.json'):raise ValueError('GR00T relative statistics file changed')
    if marker['absolute_stats_sha256']!=sha256_file(data/'meta/stats.json'):raise ValueError('GR00T absolute state/hand statistics changed')
    if marker['absolute_values_sha256']!=absolute_values_digest(absolute_group_values(json.loads((data/'meta/stats.json').read_text()))):raise ValueError('GR00T grouped state/hand statistics differ')
    if marker['modality_config_sha256']!=sha256_file(P/'dexbench_config.py'):raise ValueError('GR00T modality config changed; regenerate relative stats')
    if marker['relative_values_sha256']!=relative_values_digest(relative):raise ValueError('Wrong GR00T relative target statistics')
    if marker['native_checked_episodes']!=info['total_episodes']:raise ValueError('Incomplete native-loader cross-check')
    if marker.get('episode_count')!=len(records) or marker.get('window_count')!=expected_windows:raise ValueError('GR00T complete-window coverage differs from native training')
    if marker.get('short_episode_count')!=short or marker.get('eligible_episode_count')!=len(records)-short:raise ValueError('GR00T short-episode accounting differs')
    from relative_contract import validate_policy_contract
    validate_policy_contract(marker)
    merged=native_merged_statistics(dict(absolute_group_values(json.loads((data/'meta/stats.json').read_text())),relative_action=relative))
    if not allow_legacy_binding:
        if marker.get('processor_relative_values_sha256')!=relative_values_digest(merged['relative_action']):raise ValueError('Missing/wrong native-merged relative stats binding')
        if marker.get('processor_absolute_values_sha256')!=absolute_values_digest(merged):raise ValueError('Missing/wrong native-merged state/hand stats binding')
    return marker


def write_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--write-policy-contract',type=Path)
    parser.add_argument('--refresh-native-binding',action='store_true',help='Rebind verified raw stats after native mixture arithmetic; no statistics recomputation')
    args=parser.parse_args();data=args.data.resolve()
    if args.refresh_native_binding:
        marker=check_ready(data,allow_legacy_binding=True)
        merged=native_merged_statistics(dict(absolute_group_values(json.loads((data/'meta/stats.json').read_text())),
            relative_action=json.loads((data/'meta/relative_stats.json').read_text())))
        marker.update(processor_relative_values_sha256=relative_values_digest(merged['relative_action']),
            processor_absolute_values_sha256=absolute_values_digest(merged))
        write_json(data/'meta/groot_relative_provenance.json',marker);check_ready(data)
        print('PASS verified raw data and native single-dataset merged processor statistics binding',flush=True)
        return
    if args.check or args.write_policy_contract:
        marker=check_ready(data)
        if args.write_policy_contract:write_json(args.write_policy_contract,marker)
        print(f'PASS current source-bound native GR00T relative stats ({HORIZON},9) per EEF group',flush=True)
        return
    if not 1<=args.workers<=8:raise ValueError('Preparation is limited to 1..8 CPU workers')
    lock=data/'meta/.groot_relative_stats.lock'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        records=[json.loads(line) for line in (data/'meta/episodes.jsonl').read_text().splitlines()]
        source=json.loads((data/'meta/source_fingerprint.json').read_text())
        parts={k:[] for k in ['left_eef','right_eef']};max_error=0.
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(str(data),)) as pool:
            for count,(values,error) in enumerate(pool.map(one,records,chunksize=4),1):
                for key,x in values.items():parts[key].append(x)
                max_error=max(max_error,error)
                if count%100==0:print(f'Relative windows/native checked {count}/{len(records)} episodes',flush=True)
        statistics={};windows=None
        for key,items in parts.items():
            x=np.concatenate(items,axis=0)
            if not len(x):raise ValueError('No complete 40-target windows')
            if windows is not None and windows!=len(x):raise ValueError('EEF window counts differ')
            windows=len(x)
            # Exact native reduction semantics: do not flatten horizon into samples.
            q=np.quantile(x,[.01,.99],axis=0)
            statistics[key]={name:fn(x,axis=0).tolist() for name,fn in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]}
            statistics[key].update(q01=q[0].tolist(),q99=q[1].tolist())
            print('Computed native axis0 statistics',key,x.shape,flush=True)
        path=data/'meta/relative_stats.json';write_json(path,statistics)
        info=json.loads((data/'meta/info.json').read_text())
        marker=dict(POLICY_CONTRACT,source_sha256=source['sha256'],frame_count=info['total_frames'],episode_count=info['total_episodes'],
            window_count=windows,native_checked_episodes=len(records),native_max_error=max_error,
            short_episode_count=sum(int(r['length'])<HORIZON for r in records),
            eligible_episode_count=sum(int(r['length'])>=HORIZON for r in records),
            relative_values_sha256=relative_values_digest(statistics),relative_stats_sha256=sha256_file(path),
            absolute_stats_sha256=sha256_file(data/'meta/stats.json'),modality_config_sha256=sha256_file(P/'dexbench_config.py'),
            absolute_values_sha256=absolute_values_digest(absolute_group_values(json.loads((data/'meta/stats.json').read_text()))),
            stats_generator='vectorized native Pose SO3 projection; each episode checked by original RelativeActionLoader',
            statistics_axes='axis0 over native complete windows, shape40x9 per EEF; short episodes excluded; hands use absolute stats')
        merged=native_merged_statistics(dict(absolute_group_values(json.loads((data/'meta/stats.json').read_text())),relative_action=statistics))
        marker.update(processor_relative_values_sha256=relative_values_digest(merged['relative_action']),
            processor_absolute_values_sha256=absolute_values_digest(merged))
        write_json(data/'meta/groot_relative_provenance.json',marker)
        check_ready(data)
        print('READY',windows,'complete windows, native max error',max_error,flush=True)
    finally:lock.unlink()


if __name__=='__main__':main()
