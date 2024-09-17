from typing import Tuple
import jbw  # noqa: F401
from jbw.environment import JBWEnv
from gym.wrappers import FlattenObservation
import gym
import numpy as np
from typing import Any
from itertools import cycle
from gym.spaces import Discrete

def collect_jellybean_reward(prev_items, items):
    """
    Reference for item indicies:
    0 - Banana: 0 reward
    1 - Onion: -1 reward for every one collected
    2 - JellyBean: +1 reward for every one collected
    3 - Wall: 0 reward, cannot collect
    4 - Tree: 0 reward, cannot collect
    5 - Truffle: 0 reward
    """
    reward_array = np.array([0, -1, 1, 0, 0, 0])
    diff = items - prev_items
    return (diff * reward_array).sum().astype(np.float32)


def collect_onion_reward(prev_items, items):
    """
    Reference for item indicies:
    0 - Banana: 0 reward
    1 - Onion: +1 reward for every one collected
    2 - JellyBean: -1 reward for every one collected
    3 - Wall: 0 reward, cannot collect
    4 - Tree: 0 reward, cannot collect
    5 - Truffle: 0 reward
    """
    reward_array = np.array([0, 1, -1, 0, 0, 0])
    diff = items - prev_items
    return (diff * reward_array).sum().astype(np.float32)


class CyclingJbwWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cycle_period: int = 2):
        super().__init__(env)
        self._step_idx = 0
        self._reward_functions = cycle([collect_jellybean_reward, collect_onion_reward])
        self._target_items = cycle(["jellybean", "onion"])
        self.target_item = "jellybean" # the env starts in jellybean mode
        self._cycle_period = cycle_period
        self._last_obs = None

        # create a copy of the underlying obs space with an extra dim for the target item
        self.observation_space = self.env.observation_space
        self.observation_space["target_item"] = Discrete(2)

    def _add_target_item(self, obs: dict):
        if self.target_item == "jellybean":
            obs["target_item"] = 0
        else:
            obs["target_item"] = 1


    def reset(self, **kwargs) -> Any | Tuple[Any | dict]:
        # embed a lifetime learning environment in an episodic framework by not resetting
        # reset needs to return the last observation
        if self._last_obs is None: 
            # first reset is a real reset
            reset_result = super().reset(**kwargs)
        else:
            # first reset will be a real reset
            reset_result = self._last_obs
            
        self._add_target_item(reset_result)
        return reset_result
    
    def render(self, mode="matplotlib", **kwargs):
        return super().render(mode, **kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        if self._step_idx % self._cycle_period == 0:
            self.unwrapped._reward_fn = next(self._reward_functions)
            self.target_item = next(self._target_items)

        self._step_idx += 1

        obs, reward, done, info = super().step(action)
        self._add_target_item(obs)

        return obs, reward, done, info
    
        
def make_jbw(render: bool):
    # a version of JBW that cycles between collect jellybean & collect onion
    return FlattenObservation(CyclingJbwWrapper(gym.make("JBW-render-v1", render=render)))
   