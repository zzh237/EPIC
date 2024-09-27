"""
Basic SAC agent without gaussian parameterization.
"""

from __future__ import annotations
from operator import itemgetter

import torch
from torch import Tensor, nn
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.distributions import TanhNormal
from typing import cast

from algos.logging import track_config
from algos.types import Action, EPICModel
from ..memory import ReplayMemory
from rlkit.torch.networks import ConcatMlp
from copy import deepcopy
import gym
import wandb
from torchrl.data import ReplayBuffer, LazyTensorStorage


class FlattenMlp(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], output_size: int):
        super().__init__()
        self.layers = nn.Sequential()

        current_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(
                nn.Linear(
                    current_size,
                    hidden_size,
                )
            )
            self.layers.append(nn.ReLU())
            current_size = hidden_size

        self.layers.append(nn.Linear(current_size, output_size))

    def forward(self, *inputs):
        cat = torch.cat(inputs, dim=1)
        return self.layers(cat)

    def copy(self):
        return deepcopy(self)


class VanillaSAC(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        device: str,
        lr: float,
        use_automatic_entropy_tuning: bool,
        sac_steps: int = 1,
        batch_size: int = 512,
        capacity: int = 100_000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sac_steps = sac_steps
        self.device = device
        self.lr = lr
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.qf1 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.qf2 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.target_qf1 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.target_qf2 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)

        self.policy = TanhGaussianPolicy(hidden_sizes=(256, 256), obs_dim=state_dim, action_dim=action_dim)

        self.trainer = SACTrainer(
            env=env,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_lr=lr,
            qf_lr=lr,
            use_automatic_entropy_tuning=use_automatic_entropy_tuning,
        )

        # self.replay_buffer = ReplayMemory(capacity=capacity)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=capacity,
            ),
            batch_size=batch_size,
            pin_memory=True,
            prefetch=2
        )
        self.to(torch.device(device))

    def initialize_policy_m(self):
        """Initialize MC worker policies."""
        # do nothing because this is a baseline

    def act_policy_m(self, obs, j: int):
        """Take an action with the mth MC worker."""
        # this doesn't support MC workers so we'll just take an action with the current policy.
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        if obs.ndim < 2:
            obs = obs.unsqueeze(0)

        dist: TanhNormal = self.policy(obs.to(torch.device(self.device)))
        value, logprob = dist.rsample_and_logprob()
        return obs, value, logprob

    def update_mu_theta_for_default(self, meta_memories, meta_update_every, H):
        """Do one step of updating. hello"""
        # if self.replay_buffer.size() < self.batch_size:
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.sac_steps):
            # state, action, reward, next_state, done = self.replay_buffer.sample(
            #     batch_size=self.batch_size, device=self.device, as_tensors=True
            # )
            state, action, reward, next_state, done = self.replay_buffer.sample()
            self.trainer.train_from_torch(
                {
                    "rewards": reward,
                    "terminals": done,
                    "observations": state,
                    "actions": action,
                    "next_observations": next_state,
                }
            )
        self.trainer.end_epoch(1)
        wandb.log(self.trainer.eval_statistics)

    def update_default_and_prior_policy(self):
        """Don't do anything"""


class VanillaSACv2(EPICModel):
    @track_config
    def __init__(
        self,
        m: int,
        batch_size: int,
        device: str,
        env: gym.Env,
        lr: float,
        use_automatic_entropy_tuning: bool,
        replay_capacity: int,
        optimizer_class,
        sac_steps: int = 1,
    ):
        super().__init__()
        self._m = m  # number of MC workers
        self.batch_size = batch_size
        self.sac_steps = sac_steps
        self.device = device
        self.lr = lr
        state_dim = env.observation_space.shape[0]  # type: ignore
        action_dim = env.action_space.shape[0]  # type: ignore

        self.qf1 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.qf2 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.target_qf1 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)
        self.target_qf2 = ConcatMlp(input_size=state_dim + action_dim, hidden_sizes=(256, 256), output_size=1)

        self.policy = TanhGaussianPolicy(hidden_sizes=(256, 256), obs_dim=state_dim, action_dim=action_dim)

        self.trainer = SACTrainer(
            env=env,
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_lr=lr,
            qf_lr=lr,
            use_automatic_entropy_tuning=use_automatic_entropy_tuning,
            optimizer_class=optimizer_class
        )

        # self.replay_buffer = ReplayMemory(capacity=replay_capacity)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=replay_capacity),
            batch_size=batch_size,
            pin_memory=True,
            prefetch=2
        )
        self.to(torch.device(device))

    @property
    def m(self) -> int:
        return self._m

    def act_m(self, m, state) -> Action:
        """
        Since we only maintain a single copy of the policy, just act with the policy itself.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state)

        if state.ndim < 2:
            state = state.unsqueeze(0)

        dist: TanhNormal = self.policy(state.to(torch.device(self.device)))
        value, logprob = dist.rsample_and_logprob()
        return Action(state=state, action=value.flatten(), log_prob=logprob)
    
    def add_sample(self, state, action, reward, new_state, done):
        if not isinstance(state, Tensor):
            state = torch.from_numpy(state)
        if not isinstance(action, Tensor):
            action = torch.from_numpy(action)
        if not isinstance(new_state, Tensor):
            new_state = torch.from_numpy(new_state)

        state = state.squeeze().type(torch.float32).detach()
        action = torch.atleast_1d(action.squeeze()).detach()
        new_state = new_state.squeeze().type(torch.float32).detach()
        done = torch.atleast_1d(torch.tensor(done)).type(torch.int8).detach()
        reward = torch.atleast_1d(torch.tensor(reward)).type(torch.float32).detach()

        self.replay_buffer.add((state, action, reward, new_state, done))

    def per_step_m(self, m: int, meta_episode, step, action_dict, reward, new_state, done):
        state, action = itemgetter("state", "action")(action_dict)
        self.add_sample(state, action, reward, new_state, done)
        # self.replay_buffer.push(state, action, reward, new_state, done)
        # if self.replay_buffer.size() < self.batch_size:
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.sac_steps):
            # state, action, reward, next_state, done = self.replay_buffer.sample(
            #     batch_size=self.batch_size, device=self.device, as_tensors=True
            # )
            state, action, reward, next_state, done = self.replay_buffer.sample()
            self.trainer.train_from_torch(
                {
                    "rewards": reward,
                    "terminals": done,
                    "observations": state,
                    "actions": action,
                    "next_observations": next_state,
                }
            )
        self.trainer.end_epoch(1)
        eval_stats = deepcopy(self.trainer.eval_statistics)
        eval_stats["meta_episode"] = meta_episode
        wandb.log(eval_stats)

    def pre_meta_episode(self):
        pass

    def post_episode(self):
        pass

    def update_default(self):
        pass

    def update_prior(self):
        pass
