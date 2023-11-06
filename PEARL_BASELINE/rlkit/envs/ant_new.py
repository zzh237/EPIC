# import numpy as np
# from gym import utils
# from gym.envs.mujoco import mujoco_env

import numpy as np

from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

class AntDirection(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):

        self._task = task
        self._n_tasks = n_tasks
        self.tasks = self.sample_tasks()
        # theta = np.random.uniform(0, 2*np.pi)
        # self._goal = np.array([np.sin(theta), np.cos(theta)])

        # mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        # utils.EzPickle.__init__(self)

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
        )
        MuJocoPyEnv.__init__(
            self, "ant.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def sample_tasks(self):
        thetas = np.random.uniform(0, 2 * np.pi, size=self._n_tasks)
        tasks = []
        for theta in thetas:
            tasks.append({'goal_direction': theta})
        return tasks

    def sample_tasks_itself(self):
        self.tasks = self.sample_tasks()

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def step(self, a):
        xposbefore = self.get_body_com("torso")[:2]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[:2]
        forward_reward = np.sum(self._goal * (xposafter - xposbefore)) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        return self._get_obs()


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        theta = self._task['goal_direction']
        self._goal = np.array([np.sin(theta), np.cos(theta)])

        # self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.reset()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


