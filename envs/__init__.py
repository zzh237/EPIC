import numpy as np
import gym

def make_pendulum(seed, toy=False):
    if toy:
        gravity = 10.0
    else:
        gravity = np.random.uniform(1.0, 20.0)

    env = gym.make("Pendulum-v1", g=gravity)
    env.seed(seed)
    return env