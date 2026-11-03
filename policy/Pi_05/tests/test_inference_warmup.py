import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('pi_warmup',Path(__file__).parents[1]/'inference_warmup.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Policy:
    def __init__(self, error=None):
        self._rng=object()
        self.error=error
        self.observation=None
    def infer(self, observation):
        self._rng=object()
        self.observation=observation
        if self.error:
            raise self.error
        return {'actions':np.zeros((50,38))}

def test_warmup_preserves_first_rollout_rng_and_compiles_runtime_shapes():
    p=Policy(); rng=p._rng
    module.warmup_egovla_policy(p,action_dim=38)
    assert p._rng is rng
    assert p.observation['state'].shape==(38,)
    assert all(x.shape==(3,384,384) and x.dtype==np.uint8 for x in p.observation['images'].values())
    assert p.observation['camera_mask'].dtype==bool

def test_compile_error_propagates_before_readiness_and_restores_rng():
    error=RuntimeError('compile failed'); p=Policy(error); rng=p._rng
    with pytest.raises(RuntimeError,match='compile failed'):
        module.warmup_egovla_policy(p,action_dim=38)
    assert p._rng is rng

@pytest.mark.parametrize('value',[np.full((50,38),np.nan),np.zeros(38)])
def test_bad_warmup_output_is_not_announced_ready(value):
    p=Policy(); p.infer=lambda obs:{'actions':value}
    with pytest.raises(RuntimeError,match='invalid actions'):
        module.warmup_egovla_policy(p,action_dim=38)
