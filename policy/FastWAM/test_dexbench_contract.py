import unittest,tempfile,json,hashlib
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.datasets.dexbench_contract import *
from fastwam.datasets.dexbench import DexBenchIndex
from XPolicyLab.policy.FastWAM.model import Model,resolve_runtime

P=Path(__file__).resolve().parent

def cfg():
    with initialize_config_dir(config_dir=str(P/'FastWAM/configs'),version_base='1.3'):
        return compose(config_name='train',overrides=['task=dexbench_ee58'])

def obs(value=0.2,idx=0):
    return {'env_idx':idx,'instruction':'test','state':{
        'left_ee_pose':np.array([value,0.1,1,1,0,0,0],np.float32),
        'left_ee_joint_state':np.arange(20,dtype=np.float32)/100,
        'right_ee_pose':np.array([-value,0.2,1,1,0,0,0],np.float32),
        'right_ee_joint_state':np.arange(20,dtype=np.float32)/50},
        'vision':{k:{'color':np.full((480,640,3),[13,101,217],np.uint8)} for k in ['cam_head','cam_left_wrist','cam_right_wrist']}}

class HoldEngine:
    action_horizon=32
    def _infer_action_chunk(self,obs,prompt):return np.repeat(obs['joint_action']['vector'][None],32,axis=0)
    def reset(self):pass

def adapter(**overrides):
    with patch.object(Model,'_load_model',return_value=HoldEngine()):
        base=dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test',replan_steps=2)
        return Model(dict(base,**overrides))

class ContractTests(unittest.TestCase):
    def test_auto_prompt_and_real_weights_required(self):
        m=adapter(task_name='auto')
        x=obs();x['instruction']='[tianji_wujihand2_real/test] test'
        m.update_obs(x)
        self.assertEqual(m.last_instruction,x['instruction'])
        x['instruction']='test'
        with self.assertRaises(ValueError):m.update_obs(x)
        x['instruction']='[other/test] test'
        with self.assertRaises(ValueError):m.update_obs(x)
        with self.assertRaises(ValueError):adapter(allow_dummy_policy=True)

    def test_roundtrip_absolute_poses_and_hand_order(self):
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(23)
        x=rng.normal(size=(20,54)).astype(np.float32)
        for s in [0,27]:x[:,s+3:s+7]=Rotation.random(20,random_state=rng).as_quat()[:,[3,0,1,2]]
        y=pack54(x);result=unpack58(y)
        for i,row in enumerate(result):
            np.testing.assert_allclose(row['left_ee_pose'][:3],x[i,:3])
            np.testing.assert_allclose(row['right_ee_pose'][:3],x[i,27:30])
            np.testing.assert_allclose(row['left_ee_joint_state'],x[i,7:27])
            np.testing.assert_allclose(row['right_ee_joint_state'],x[i,34:54])
            for s,key in [(0,'left_ee_pose'),(27,'right_ee_pose')]:
                self.assertAlmostEqual(abs(float(row[key][3:]@x[i,s+3:s+7])),1,places=5)

    def test_actual_rows_alignment_and_end_padding(self):
        index=DexBenchIndex('/personal/wenwei/DexBench/data_preprocess/data/dexbench_12tasks','all')
        self.assertEqual(len(index),index.info['total_frames'])
        for sample in [0,179,180,len(index)-1]:
            row,t,action,state,a,s=index.numeric(sample)
            data=index.episode_data(row)
            np.testing.assert_allclose(action[0],pack54(data['action'].iloc[t]),atol=1e-6)
            np.testing.assert_allclose(state[0],pack54(data['observation.state'].iloc[t]),atol=1e-6)
            self.assertLessEqual(int(a[-1]),len(data)-1)

    def test_batch_order_and_missing_camera(self):
        m=adapter();m.update_obs_batch([obs(0.2,3),obs(0.4,1)])
        result=m.get_action_batch([3,1])
        self.assertAlmostEqual(float(result[0][0]['left_ee_pose'][0]),0.2,places=5)
        self.assertAlmostEqual(float(result[1][0]['left_ee_pose'][0]),0.4,places=5)
        default=m.get_action_batch()
        self.assertAlmostEqual(float(default[0][0]['left_ee_pose'][0]),0.2,places=5)
        self.assertAlmostEqual(float(default[1][0]['left_ee_pose'][0]),0.4,places=5)
        m.update_obs_batch([obs(0.6,3)])
        self.assertEqual(len(m.get_action_batch()),1)
        with self.assertRaises(ValueError):m.get_action_batch([1])
        m.reset()
        with self.assertRaises(ValueError):m.get_action_batch()
        broken=obs();del broken['vision']['cam_left_wrist']
        with self.assertRaises(KeyError):m.update_obs(broken)

    def test_real_sample_image_prompt_and_normalizer_parity(self):
        c=cfg()
        with tempfile.TemporaryDirectory() as d:
            c.data.train.text_embedding_cache_dir=d
            ds=instantiate(c.data.train)
            row,t,a,s,ai,si=ds.index.numeric(0)
            prompt=DEFAULT_PROMPT.format(task=str(row['tasks'][0]))
            h=hashlib.sha256(prompt.encode()).hexdigest()
            # Only a text-cache fixture; image/action samples below are real exported data.
            torch.save({'context':torch.ones(128,4096),'mask':torch.ones(128,dtype=torch.bool)},Path(d)/f'{h}.t5_len128.wan22ti2v5b.pt')
            sample=ds[0]
            self.assertEqual(tuple(sample['action'].shape),(32,58))
            self.assertEqual(tuple(sample['video'].shape),(3,9,384,320))
            self.assertEqual(sample['prompt'],prompt)
            cameras=ds.index.cameras(row,[0])
            live={'observation':{key:{'rgb':cameras[cam][0]} for key,cam in zip(['head_camera','left_camera','right_camera'],CAMERAS)}}
            torch.testing.assert_close(sample['video'][:,0],inference_image(live,video_size=(384,320))[0],rtol=0,atol=0)
            normalized=ds.processor.normalizer.normalizers['action']['default'].forward(torch.from_numpy(a))
            torch.testing.assert_close(sample['action'],normalized)
            from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
            engine=WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
            engine.processor=ds.processor
            current=engine._normalize_state(s[0])
            torch.testing.assert_close(current[0],sample['proprio'][0])
            self.assertFalse(np.allclose(a[0],s[0]))  # current measurement differs from its target
            physical=ds.processor.normalizer.normalizers['action']['default'].backward(normalized)
            np.testing.assert_allclose(physical,a,atol=1e-5)
            train=ds.index;val=DexBenchIndex(c.data.val.dataset_dirs[0],'val')
            self.assertFalse(set(train.episodes.episode_index)&set(val.episodes.episode_index))
            self.assertEqual(len(train)+len(val),train.info['total_frames'])

    def test_saved_contract_rejects_old_and_accepts_new(self):
        c=cfg()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoints/weights').mkdir(parents=True)
            (root/'checkpoints/weights/step_000001.pt').touch()
            (root/'dataset_stats.json').write_text((P/'data/dataset_stats.json').read_text())
            OmegaConf.save(c,root/'config.yaml')
            weights,stats,loaded=resolve_runtime({'ckpt_name':d})
            self.assertEqual(loaded['model']['action_dit_config']['action_dim'],58)
            self.assertTrue(weights.endswith('step_000001.pt'))
            (root/'checkpoints/weights/step_000020.pt').touch()
            (root/'checkpoints/weights/step_9.pt').touch()
            weights,_,_=resolve_runtime({'ckpt_name':d})
            self.assertTrue(weights.endswith('step_000020.pt'))
            weights,_,_=resolve_runtime({'ckpt_name':d,'checkpoint_num':1})
            self.assertTrue(weights.endswith('step_000001.pt'))
            with self.assertRaises(FileNotFoundError):resolve_runtime({'ckpt_name':d,'checkpoint_num':2})
            with self.assertRaises(FileNotFoundError):resolve_runtime({'ckpt_name':str(root/'checkpoints/weights/step_000020.pt'),'checkpoint_num':1})
            del c.action_contract;OmegaConf.save(c,root/'config.yaml')
            with self.assertRaises(ValueError):resolve_runtime({'ckpt_name':d})

    def test_execution_horizon(self):
        base=dict(env_cfg_type='tianji_marvin_wuji',action_type='ee',task_name='test')
        for value in [0,-1,33]:
            with self.assertRaises(ValueError):Model(dict(base,replan_steps=value))
        for value,expected in [(None,24),(1,1),(32,32)]:
            with patch.object(Model,'_load_model',return_value=HoldEngine()):m=Model(dict(base,replan_steps=value))
            m.update_obs(obs())
            self.assertEqual(len(m.get_action()),expected)

    def test_websocket_raw_and_encoded(self):
        import asyncio
        from client_server.ws.model_server import PolicyServer,PolicyServerConfig
        from client_server.ws.model_client import WsModelClient
        from XPolicyLab.utils.process_data import encode_image_bit
        async def run():
            m=adapter();server=PolicyServer(m,PolicyServerConfig(host='127.0.0.1',port=0));await server.start()
            def call():
                with WsModelClient(url=server.url,evaluation_id='fastwam-contract',trial_id='offline') as client:
                    for encoded in [False,True]:
                        x=obs()
                        if encoded:
                            for camera in x['vision'].values():camera['color']=encode_image_bit(camera['color'])
                        client.call('update_obs',x);result=client.call('get_action')
                        self.assertEqual(set(result[0]),set(obs()['state']))
                        for k,v in obs()['state'].items():np.testing.assert_allclose(result[0][k],v,atol=1e-6)
            try:await asyncio.to_thread(call)
            finally:await server.stop()
        asyncio.run(run())

if __name__=='__main__':unittest.main(verbosity=2)
