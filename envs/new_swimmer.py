import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import swimmer
from gym.envs.mujoco import walker2d
import copy

class new_Swimmer(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, goal):
        self.angle = np.random.uniform(low=0, high=np.pi)
        self.direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 5)
        utils.EzPickle.__init__(self)



    def step(self, a):
        ctrl_cost_coeff = 0.0001
        posbefore = copy.deepcopy(self.sim.data.qpos[0:2])
        self.do_simulation(a, self.frame_skip)
        posafter = copy.deepcopy(self.sim.data.qpos[0:2])
        reward_fwd = np.sum((posafter-posbefore)*self.direction) / self.dt

        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def reset(self):
        # *modification*
        # self._goal_vel = self.sample_goals()
        self.sim.reset()
        ob = self.reset_model()
        return ob
