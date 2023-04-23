import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnvRandVel(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._goal_vel = self.sample_goals()  # *modification*
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)

    def sample_goals(self):
        # *modification*
        return np.random.uniform(0.1, 0.2)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        # reward_fwd = (xposafter - xposbefore) / self.dt
        vel_x = (xposafter - xposbefore) / self.dt  # *modification*
        reward_fwd = - 1.5 * np.abs(vel_x - self._goal_vel)  # *modification*
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
        self._goal_vel = self.sample_goals()
        self.sim.reset()
        ob = self.reset_model()
        return ob