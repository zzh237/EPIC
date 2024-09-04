import numpy as np
import gym

class TorqueConvert(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self, mode="human"):
        self.unwrapped.last_u = float(self.unwrapped.last_u) if self.unwrapped.last_u is not None else None
        return super().render(mode)

def make_pendulum(seed, toy=False):
    if toy:
        gravity = 10.0
    else:
        gravity = np.random.uniform(1.0, 20.0)

    env = TorqueConvert(gym.make("Pendulum-v1", g=gravity))
    env.seed(seed)
    return env