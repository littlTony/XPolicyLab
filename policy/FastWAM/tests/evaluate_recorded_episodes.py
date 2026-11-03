"""Replay 2 recorded training episodes per task through the real WebSocket server.

This measures action agreement under recorded observations, not closed-loop
task success. Each query predicts32 actions and evaluates the returned first24; padded episode tails are
excluded. The hold-state baseline holds the observation at the chunk boundary.
"""
import argparse

import json
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from client_server.ws.model_client import WsModelClient

from openpi.policies.wuji_policy import _parse_image

from openpi.training import config, data_loader
from XPolicyLab.policy.Pi_05 import model as pi_model
from XPolicyLab.policy.FastWAM import model
from fastwam.datasets.dexbench_contract import pack54,inference_image

from XPolicyLab.utils.process_data import encode_image_bit, get_robot_action_dim_info, pack_robot_state, unpack_robot_state


def summarize(prediction, target):
    position = np.stack([np.linalg.norm(prediction[:, s:s+3] - target[:, s:s+3], axis=1) * 100 for s in [0, 27]], axis=1)
    angles = []
    for s in [3, 30]:
        p, t = prediction[:, s:s+4], target[:, s:s+4]
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        angles.append(np.rad2deg(2 * np.arccos(np.clip(np.abs(np.sum(p * t, axis=1)), 0, 1))))
    angles = np.stack(angles, axis=1)
    joints = np.abs(np.concatenate([prediction[:, 7:27] - target[:, 7:27], prediction[:, 34:54] - target[:, 34:54]], axis=1))
    return dict(frames=len(target), position_mean_cm=float(position.mean()), position_p95_cm=float(np.quantile(position, .95)),
                rotation_mean_deg=float(angles.mean()), rotation_p95_deg=float(np.quantile(angles, .95)),
                hand_mae_rad=float(joints.mean()), hand_p95_rad=float(np.quantile(joints, .95)),
                left_position_cm=float(position[:, 0].mean()), right_position_cm=float(position[:, 1].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks'))
    parser.add_argument('--embodiment', default='tianji_wujihand2_real')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'summary.json').exists(), 'Use a unique report directory'
    columns = ['episode_index', 'tasks', 'length', 'dataset_from_index', 'dataset_to_index']
    rows = []
    for p in sorted((args.dataset / 'meta/episodes').rglob('*.parquet')):
        rows.extend(pq.read_table(p, columns=columns).to_pylist())
    tasks = {}
    for row in rows:
        prompt = row['tasks'][0]
        m = re.fullmatch(r'\[([^/\]]+)/([^\]]+)\] (.+)', prompt, re.DOTALL)
        if m and m[1] == args.embodiment:
            tasks.setdefault(m[2], []).append({**row, 'task': m[2], 'prompt': prompt})
    assert len(tasks) == 12, list(tasks)
    selected=json.loads((args.references/'episode_selection.json').read_text())
    (args.output / 'episode_selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    cfg = config.get_config(pi_model.TRAIN_CONFIG)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(dc, 32, cfg.model)
    dims = get_robot_action_dim_info('tianji_marvin_wuji')
    adapter = model.Model.__new__(model.Model)
    adapter.model_cfg = dict(task_name='auto',prompt_embodiment=args.embodiment)
    adapter.robot_action_dim_info = dims
    all_predictions, all_targets, all_baselines, all_first = [], [], [], []
    per_episode = []
    latencies = []
    parity_checks = 0
    protocol_checks = []
    with WsModelClient(url=args.url, evaluation_id='fastwam-60000-training-replay', trial_id='24-recorded-episodes',
                       request_timeout_s=600, connect_timeout_s=30, max_connect_attempts=2) as client:
        metadata = client.call('get_metadata')
        assert Path(metadata['checkpoint_path']).resolve() == args.checkpoint.resolve()
        checkpoint_optimizer_step = metadata['checkpoint_optimizer_step']
        assert checkpoint_optimizer_step == 60000
        assert metadata['action_horizon'] == 32 and metadata['replan_steps'] == 24 and metadata['action_dim'] == 54 and metadata['task_name'] == 'auto'
        (args.output / 'server_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        for ep in selected:
            client.call('reset')
            predictions, targets, baselines, firsts = [], [], [], []
            ep_latency = []
            for frame in range(0, ep['length'], 24):
                row = dataset[ep['dataset_from_index'] + frame]
                assert int(np.asarray(row['episode_index'])) == ep['episode_index']
                assert int(np.asarray(row['frame_index'])) == frame
                assert row['prompt'] == ep['prompt']
                state = np.asarray(row['observation.state'], dtype=np.float32)
                obs = dict(env_idx=7, state=unpack_robot_state(state, 'ee', dims), instruction=row['prompt'],
                           task_name=ep['task'], vision={camera: {'color': _parse_image(row['observation.images.' + camera])}
                           for camera in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']})
                obs['vision']['cam_head'] = obs['vision'].pop('cam_high')
                if frame == 0:
                    native_obs,prompt = adapter._encode(obs)
                    np.testing.assert_allclose(native_obs['joint_action']['vector'],pack54(state),atol=0,rtol=0)
                    assert prompt == ep['prompt']
                    reference=np.load(args.references/f"episode_{ep['episode_index']:06d}.npz")
                    np.testing.assert_allclose(reference['state'],native_obs['joint_action']['vector'],atol=1e-6)
                    np.testing.assert_allclose(reference['action'],pack54(np.asarray(row['action'],dtype=np.float32)),atol=1e-6)
                    runtime=inference_image(native_obs,video_size=tuple(metadata['video_size']))[0]
                    torch.testing.assert_close(torch.from_numpy(reference['image']),runtime,atol=0,rtol=0)
                    parity_checks += 1
                    if not protocol_checks:
                        # Exercise encoded-image transport with an actual trained
                        # model, separately from quantitative raw-RGB replay.
                        encoded = {**obs, 'vision': {k: {'color': encode_image_bit(v['color'])} for k, v in obs['vision'].items()}}
                        client.call('update_obs', encoded)
                        warm = client.call('get_action')
                        assert len(warm) == 24
                        protocol_checks.append('encoded_RGB_real_model_inference')
                        client.call('reset')
                start = time.perf_counter()
                client.call('update_obs_batch', [obs])
                chunks = client.call('get_action_batch', [7])
                elapsed = time.perf_counter() - start
                assert len(chunks) == 1 and len(chunks[0]) == 24
                prediction = np.stack([pack_robot_state({'state': a}, 'ee', dims) for a in chunks[0]])
                assert prediction.shape == (24, 54) and np.isfinite(prediction).all()
                valid = min(24, ep['length'] - frame)
                target = np.asarray(row['action'], dtype=np.float32)[:valid]
                baseline = np.repeat(state[None], valid, axis=0)
                predictions.append(prediction[:valid]); targets.append(target); baselines.append(baseline)
                firsts.append((prediction[:1], target[:1], baseline[:1]))
                ep_latency.append(elapsed); latencies.append(elapsed)
            pred, target, baseline = map(np.concatenate, [predictions, targets, baselines])
            assert len(pred) == ep['length']
            first_pred, first_target, first_base = [np.concatenate([x[i] for x in firsts]) for i in range(3)]
            result = dict(episode_index=ep['episode_index'], task=ep['task'], prompt=ep['prompt'],
                          chunks=len(predictions), frames=len(pred), action=summarize(pred, target),
                          hold_baseline=summarize(baseline, target), first_action=summarize(first_pred, first_target),
                          first_hold_baseline=summarize(first_base, first_target), latency_mean_s=float(np.mean(ep_latency)))
            per_episode.append(result)
            all_predictions.append(pred); all_targets.append(target); all_baselines.append(baseline); all_first.append((first_pred, first_target, first_base))
            np.savez_compressed(args.output / f'episode_{ep["episode_index"]:06d}.npz', prediction=pred, target=target, hold_baseline=baseline)
            (args.output / 'episode_results.json').write_text(json.dumps(per_episode, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        client.call('reset')
    pred, target, baseline = map(np.concatenate, [all_predictions, all_targets, all_baselines])
    first_pred, first_target, first_base = [np.concatenate([x[i] for x in all_first]) for i in range(3)]
    per_task = []
    for task in sorted(tasks):
        ids = [i for i, ep in enumerate(selected) if ep['task'] == task]
        per_task.append(dict(task=task, episodes=[selected[i]['episode_index'] for i in ids],
                             action=summarize(np.concatenate([all_predictions[i] for i in ids]), np.concatenate([all_targets[i] for i in ids])),
                             hold_baseline=summarize(np.concatenate([all_baselines[i] for i in ids]), np.concatenate([all_targets[i] for i in ids]))))
    summary = dict(checkpoint=str(args.checkpoint), checkpoint_optimizer_step=checkpoint_optimizer_step, seed=args.seed,
                   embodiment=args.embodiment, tasks=12, episodes=24, frames=len(pred), queries=len(latencies),
                   action=summarize(pred, target), hold_baseline=summarize(baseline, target),
                   first_action=summarize(first_pred, first_target), first_hold_baseline=summarize(first_base, first_target),
                   latency_mean_s=float(np.mean(latencies)), latency_p95_s=float(np.quantile(latencies, .95)),
                   train_runtime_preprocessing_parity_checks=parity_checks, protocol_checks=protocol_checks,
                   server_metadata=metadata, per_task=per_task,
                   evaluation='Recorded training observations; non-overlapping 24-step action chunks; padding excluded; not a closed-loop success rate')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('EVALUATION_COMPLETE ' + json.dumps({k:v for k,v in summary.items() if k not in ['per_task', 'server_metadata']}), flush=True)


if __name__ == '__main__':
    main()
