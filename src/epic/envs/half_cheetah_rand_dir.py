import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnvRandDir(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self._goal_vel = self.sample_goals()  # *modification*
        self._goal_direction = -1.0 if self._goal_vel < 1.0 else 1.0  # *modification*
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def sample_goals(self):
        # *modification*
        return np.random.uniform(0.0, 2.0)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        vel_x = (xposafter - xposbefore) / self.dt  # *modification*
        reward_run = self._goal_direction * vel_x  # *modification*
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        # *modification*
        self._goal_vel = self.sample_goals()
        self._goal_direction = -1.0 if self._goal_vel < 1.0 else 1.0
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5