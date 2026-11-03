"""A predicted chunk must stop at the first terminal action."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("terminal_step", [1, 2, 5])
def test_single_environment_stops_inside_chunk(terminal_step):
    deploy = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy.py"))

    class Environment:
        steps = 0

        def is_episode_end(self):
            return self.steps >= terminal_step

        def get_obs(self):
            assert not self.is_episode_end(), "observation requested after termination"
            return {"step": self.steps}

        def take_action(self, action):
            assert not self.is_episode_end(), "action applied after termination"
            self.steps += 1

    class Client:
        def call(self, func_name, obs=None):
            if func_name == "get_action":
                return [{} for _ in range(4)]

    env = Environment()
    deploy["eval_one_episode"](env, Client())
    assert env.steps == terminal_step
