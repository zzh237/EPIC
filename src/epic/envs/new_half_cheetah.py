import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class new_HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        # self.goal_velocity = np.random.choice(np.array([0.5, 1.0, 1.5, 2.0, 2.5]), p=[0.15, 0.18, 0.34, 0.18, 0.15])
        # self.goal_velocity = self.goal_velocity + 0.05*np.random.randn()
        self.goal_velocity = np.random.uniform(low=-2.0, high=2.0)
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    # def sample_direction(self):
    #     angle = np.random.choice(np.array([0.0, np.pi/2, np.pi, np.pi*1.5]), p=[0.3, 0.18, 0.34, 0.18])
    #     self.goal_dir = np.array([np.cos(angle), np.sin(angle)])

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        vel_x = (xposafter - xposbefore) / self.dt
        print(vel_x)
        reward_run = np.exp(-5*np.abs(vel_x-self.goal_velocity))  # *modification*
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == '__main__':
    env = new_HalfCheetahEnv()


    import time

    # Number of steps you run the agent for
    num_steps = 1500

    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs)
        action = env.action_space.sample()

        # apply the action
        obs, reward, done, info = env.step(action)

        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()
