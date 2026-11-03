"""Real LeRobot sample + websocket integration checks, without loading weights."""
import asyncio
import copy
import json
import os
from pathlib import Path
import re

import numpy as np
from openpi import transforms
from openpi.policies.wuji_policy import _parse_image
from openpi.training import config, data_loader
from XPolicyLab.policy.Pi_05 import model
from XPolicyLab.utils.process_data import encode_image_bit, unpack_robot_state
from client_server.ws.model_client import WsModelClient
from client_server.ws.model_server import PolicyServer,PolicyServerConfig
from test_dexbench_contract import adapter,observation


def check_real_dataset():
    cfg=config.get_config(model.TRAIN_CONFIG)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    root=Path(os.environ['HF_LEROBOT_HOME'])/'dexbench_12tasks'
    assert len(ds)==json.loads((root/'meta/info.json').read_text())['total_frames']
    import pyarrow.parquet as pq
    episodes=[]
    for path in sorted((root/'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path,columns=['tasks','dataset_from_index']).to_pylist())
    offset=next(row['dataset_from_index'] for row in episodes
                if row['tasks'][0].startswith('[tianji_wujihand2_real/'))
    for index in [0,offset]:
        row=ds[index]
        training=transforms.compose([*dc.repack_transforms.inputs,*dc.data_transforms.inputs])(row)
        np.testing.assert_array_equal(training['actions'],np.asarray(row['action']))
        prefix=re.match(r'^\[([^/]+)/([^\]]+)\] ',row['prompt'])
        embodiment,task=prefix.groups()
        obs={'state':unpack_robot_state(np.asarray(row['observation.state']),'ee',{'arm_dim':[7,7],'ee_dim':[20,20]}),
             'instruction':row['prompt'],'vision':{
                 camera: {'color':_parse_image(row['observation.images.'+camera])}
                 for camera in ['cam_high','cam_left_wrist','cam_right_wrist']}}
        runtime=model.encode_obs(obs,{'arm_dim':[7,7],'ee_dim':[20,20]},task_name=task,prompt_embodiment=embodiment)
        inference=transforms.compose([*model._runtime_repack_transforms(cfg).inputs,*dc.data_transforms.inputs])(runtime)
        np.testing.assert_array_equal(training['state'],inference['state'])
        assert training['prompt']==inference['prompt']
        for name in training['image']:
            np.testing.assert_array_equal(training['image'][name],inference['image'][name])
        print(f'Real data index={index}: state(54), actions{training["actions"].shape}, RGB cameras, prompt and runtime repack match',flush=True)


async def check_websocket():
    policy=adapter()
    server=PolicyServer(policy,PolicyServerConfig(host='127.0.0.1',port=0))
    await server.start()
    def client_check():
        with WsModelClient(url=server.url,evaluation_id='dexbench-pi05-contract',trial_id='offline') as client:
            for encoded in [False,True]:
                obs=observation()
                if encoded:
                    for camera in obs['vision'].values():
                        camera['color']=encode_image_bit(camera['color'])
                client.call('update_obs',obs)
                chunk=client.call('get_action')
                assert len(chunk)==3
                for key,value in observation()['state'].items():
                    np.testing.assert_allclose(chunk[0][key],value)
                # Verify the server-side decoder preserves RGB even for encoded transport.
                pixel=policy.observation_window[0]['images']['cam_high'][:,0,0]
                assert pixel[0]>200 and 60<pixel[1]<80 and pixel[2]<30
                client.call('reset')
                print(f'Websocket round trip: encoded={encoded}, EE/wxyz + Wuji20 output verified',flush=True)
    try:
        await asyncio.to_thread(client_check)
    finally:
        await server.stop()


if __name__=='__main__':
    check_real_dataset()
    asyncio.run(check_websocket())
