from __future__ import annotations

import math
import typing
from typing import NamedTuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
from rlkit.torch.distributions import TanhNormal
from torch import Tensor, nn
from torch.autograd import Variable

import wandb
from algos.logging import track_config
from algos.memory import ReplayMemory
from algos.types import Action, EPICModel
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy

from tools import register_hooks

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class StochasticLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        log_var_mean: float = -10.0,
        log_var_std: float = 0.1,
        eps_std: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weights_mean = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.weights_log_var = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.log_var_mean = log_var_mean
        self.log_var_std = log_var_std
        if use_bias:
            self.bias_mean = nn.Parameter(torch.empty(out_dim))
            self.bias_log_var = nn.Parameter(torch.empty(out_dim))
        self.eps_std = eps_std

        self.reset_layer()

    def reset_layer(self):
        n = self.weights_mean.size(1)
        stdv = np.sqrt(1.0 / n)

        with torch.no_grad():
            self.weights_mean.data.uniform_(-stdv, stdv)
            self.weights_log_var.data.normal_(self.log_var_mean, self.log_var_std)

            if self.bias_mean is not None:
                self.bias_mean.uniform_(-stdv, stdv)
                self.bias_log_var.normal_(self.log_var_mean, self.log_var_std)

    def __str__(self):
        return f"StochasticLinear({self.in_dim} -> {self.out_dim})"

    def forward(self, x: Tensor):
        bias_mean = bias_var = None
        if self.bias_mean is not None:
            bias_mean = self.bias_mean
            bias_var = torch.exp(self.bias_log_var)
        out_mean = F.linear(x, self.weights_mean, bias_mean)

        if self.eps_std == 0.0:
            return out_mean
        else:
            weights_var = torch.exp(self.weights_log_var)
            out_var = F.linear(x.pow(2.0), weights_var, bias_var)
            noise = out_mean.data.new(out_mean.size()).normal_(0, self.eps_std)
            noise = Variable(noise, requires_grad=False)
            out_var = F.relu(out_var)
            return out_mean + noise * torch.sqrt(out_var)


class StochasticTanhGaussianPolicy(nn.Module):
    """
    A policy network that uses a series of stochasticLinear layers to predict the logstd
    and mean of a gaussian output.
    """

    def __init__(
        self, hidden_sizes: tuple[int, ...] | int, obs_dim: int, action_dim: int, device: str, std: float | None = None
    ):
        super().__init__()
        self.device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        # init fc layers
        in_size = obs_dim
        # self.fcs = nn.ModuleList()
        self.hidden_layers = nn.Sequential()
        for _, next_size in enumerate(hidden_sizes):
            self.hidden_layers.append(StochasticLinear(in_size, next_size))
            self.hidden_layers.append(nn.ReLU())
            in_size = next_size

        self.last_fc = StochasticLinear(in_size, action_dim)

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = StochasticLinear(last_hidden_size, action_dim)
        else:
            self.log_std = np.log(std)

    def load_from(self, other: "StochasticTanhGaussianPolicy"):
        """Reload own parameters with those from another network."""
        self.load_state_dict(other.state_dict())

    def get_action(self, obs, deterministic=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # state, detached action, action logprob
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        state, action, log_prob = self.get_actions(obs, deterministic=deterministic)
        return state, action, log_prob

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor, deterministic=False):
        action, _, _, log_prob, _, _, _, _ = self.forward(
            obs, deterministic=deterministic, reparameterize=True, return_log_prob=True
        )
        return obs, action.detach(), log_prob

    def forward(
        self, obs, reparameterize=False, deterministic=False, return_log_prob=False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # h = obs
        # for fc in self.fcs:
        #     h = F.relu(fc(h))
        # mean = self.last_fc(h)
        h = self.hidden_layers(obs)
        mean = self.last_fc(h)

        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, log_prob, pre_tanh_value = tanh_normal.rsample_logprob_and_pretanh()
                else:
                    raise ValueError("This is bad")
            else:
                # no logprob
                if reparameterize:
                    action, _ = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)  # type: ignore


class StochasticMlp(nn.Module):
    """A generic MLP using stochasticLinear layers."""

    def __init__(self, input_size: int, output_size: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        current_dim = input_size
        # self.fcs = nn.ModuleList()
        self.layers = nn.Sequential()
        for next_dim in hidden_dims:
            self.layers.append(StochasticLinear(current_dim, next_dim))
            self.layers.append(nn.ReLU())
            current_dim = next_dim

        self.last_fc = StochasticLinear(current_dim, output_size)
        self.layers.append(self.last_fc)

    def forward(self, x):
        return self.layers(x)


class FlattenStochasticMlp(StochasticMlp):
    def forward(self, *x):
        flat = torch.cat(x, dim=1)
        return super().forward(flat)


def KL_div(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    term2 = torch.sum(sigma1 / sigma2)
    term3 = torch.sum((mu2 - mu1).pow(2) / sigma2)

    return 0.5 * (term1 + term2 + term3)


def model_kl_div(default: nn.Module, prior: nn.Module):
    # calculate KL div between a default model and a prior
    kl = []
    for default_layer, prior_layer in zip(
        (layer1 for layer1 in default.modules() if isinstance(layer1, StochasticLinear)),
        (layer2 for layer2 in prior.modules() if isinstance(layer2, StochasticLinear)),
    ):
        kl.append(
            KL_div(
                mu1=default_layer.w_mu,
                sigma1=default_layer.w_log_var,
                mu2=prior_layer.w_mu,
                sigma2=prior_layer.w_log_var,
            )
        )
        kl.append(
            KL_div(
                mu1=default_layer.b_mu,
                sigma1=default_layer.b_log_var,
                mu2=prior_layer.b_mu,
                sigma2=prior_layer.b_log_var,
            )
        )

    return torch.stack(kl).sum()


def kl_regularizer(kl, N, H, c=torch.tensor(1.5), delta=torch.tensor(0.01)):
    epsilon = torch.log(torch.tensor(2.0)) / (2 * torch.log(c)) * (1 + torch.log(kl / torch.log(2.0 / delta)))
    reg = (1 + c) / 2 * torch.sqrt(torch.tensor(2.0)) * torch.sqrt((kl + torch.log(2.0 / delta) + epsilon) * N * H**2)
    return reg


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class OptimizerFactory(typing.Protocol):
    def __init__(self, params, lr): ...

    def zero_grad(self): ...
    def step(self): ...


class Losses(NamedTuple):
    qf1_loss: Tensor
    qf2_loss: Tensor
    policy_loss: Tensor


class EpicSACMcActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_capacity: int,
        sac_steps: int,
        batch_size: int,
        optimizer_class: type[OptimizerFactory],
        policy_lr: float,
        qf_lr: float,
        tau: float,
        target_update_period: int,
        reward_scale: float,
        discount: float,
        device: str,
        q_network_hiddens: tuple[int, ...] = (256, 256),
        policy_hiddens: tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self.sac_steps = sac_steps
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.tau = tau
        self.reward_scale = reward_scale
        self.discount = discount
        self.device = device

        # self.qf1 = ConcatMlp(input_size=state_dim+action_dim, hidden_sizes=(256, 256), output_size =1)
        # self.qf2 = ConcatMlp(input_size=state_dim+action_dim, hidden_sizes=(256, 256), output_size =1)
        # self.target_qf1 = ConcatMlp(input_size=state_dim+action_dim, hidden_sizes=(256, 256), output_size =1)
        # self.target_qf2 = ConcatMlp(input_size=state_dim+action_dim, hidden_sizes=(256, 256), output_size =1)

        self.qf1 = FlattenStochasticMlp(input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1)
        self.qf2 = FlattenStochasticMlp(input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1)

        self.target_qf1 = FlattenStochasticMlp(
            input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1
        )
        self.target_qf2 = FlattenStochasticMlp(
            input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1
        )
        # self.policy = TanhGaussianPolicy(hidden_sizes=(256, 256), obs_dim=state_dim, action_dim=action_dim)

        self.policy = StochasticTanhGaussianPolicy(
            hidden_sizes=policy_hiddens, obs_dim=state_dim, action_dim=action_dim, device=device
        )

        # TODO possibly share this between MC actors
        self.replay_buffer = ReplayMemory(replay_capacity)

        self.qf_criterion = nn.MSELoss()
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)

    def act(self, state: Tensor | np.ndarray) -> Action:
        if not isinstance(state, Tensor):
            state = torch.from_numpy(state).to(torch.device(self.device))

        if state.ndim < 2:
            state = state.unsqueeze(0)

        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        action, _, _, log_prob, *_ = self.policy(state, reparameterize=True, return_log_prob=True)
        # dist = self.policy(state)
        # action, log_prob = dist.rsample_and_logprob()


        return Action(state=state, action=action.flatten(), log_prob=log_prob)

    def push_sample(self, state, action, reward, new_state, done):
        self.replay_buffer.push(state, action, reward, new_state, done)

    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(
            batch_size=self.batch_size, device=self.device, as_tensors=True
        )
        losses = self.compute_loss(state, action, reward, next_state, done)

        # get_dot = register_hooks(z)
        # z.backward()
        # dot = get_dot()
        # dot.save('tmp.dot')

        # get_dot = register_hooks(losses.policy_loss)
        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        # get_dot().save("policy_grad.dot")
        
        # get_dot = register_hooks(losses.qf1_loss)
        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()
        # get_dot().save("qf1_grad.dot")

        # get_dot = register_hooks(losses.qf2_loss)
        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()
        # get_dot().save("qf2_grad.dot")

        # raise ValueError("debug hello")

    def per_step(self, state, action, reward, new_state, done, meta_episode: int, step: int):
        # every step, the MC actor adds a new sample and takes some number of trainsteps. then it may update its
        # target networks
        self.push_sample(state, action, reward, new_state, done)

        for _ in range(self.sac_steps):
            self.train_step()

        if step % self.target_update_period == 0:
            self.update_targets()

    def update_targets(self):
        soft_update_from_to(self.qf1, self.target_qf1, self.tau)
        soft_update_from_to(self.qf2, self.target_qf2, self.tau)

    def compute_loss(self, state, action, reward, new_state, done) -> Losses:
        # policy
        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        state_next_actions, _, _, log_prob, *_ = self.policy(state, reparameterize=True, return_log_prob=True)
        log_prob = log_prob.unsqueeze(-1)
        # dist: TanhNormal = self.policy(state.to(torch.device(self.device)))
        # state_next_actions, log_prob = dist.rsample_and_logprob()
        # log_prob = log_prob.unsqueeze(-1)

        q_new_actions = torch.min(self.qf1(state, state_next_actions), self.qf2(state, state_next_actions))
        policy_loss = (log_prob - q_new_actions).mean()
        
        # QF
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        new_state_next_actions, _, _, new_state_log_prob, *_ = self.policy(
            new_state, reparameterize=True, return_log_prob=True
        )
        new_state_log_prob = new_state_log_prob.unsqueeze(-1)
        # new_dist = self.policy(state.to(torch.device(self.device)))
        # new_state_next_actions, new_state_log_prob = new_dist.rsample_and_logprob()
        # new_state_log_prob = new_state_log_prob.unsqueeze(-1)
        
        target_q_values = (
            torch.min(
                self.target_qf1(new_state, new_state_next_actions), self.target_qf2(new_state, new_state_next_actions)
            )
            - new_state_log_prob
        )

        q_target = self.reward_scale * reward + (1.0 - done) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        wandb.log({
            "qf1_loss": qf1_loss.detach(),
            "qf2_loss": qf2_loss.detach(),
            "policy_loss": policy_loss.detach()
        })

        return Losses(qf1_loss=qf1_loss, qf2_loss=qf2_loss, policy_loss=policy_loss)


class EpicSAC2(EPICModel):
    @track_config(ignore=["env"])
    def __init__(
        self,
        m: int,
        env: gym.Env,
        replay_capacity: int,
        batch_size: int,
        sac_steps: int,
        optimizer_class: type[OptimizerFactory],
        policy_lr: float,
        qf_lr: float,
        tau: float,
        qf_target_update_period: int,
        reward_scale: float,
        discount: float,
        device: str,
        q_network_hiddens: tuple[int, ...] = (256, 256),
        policy_hiddens: tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self._m = m
        self.batch_size = batch_size
        self.sac_steps = sac_steps
        self.device = device

        # MC actor initialization
        assert env.action_space.shape is not None
        assert env.observation_space.shape is not None
        state_dim = math.prod(env.observation_space.shape)
        action_dim = math.prod(env.action_space.shape)
        self.mc_actors = nn.ModuleList(
            [
                EpicSACMcActor(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    replay_capacity=replay_capacity,
                    device=device,
                    q_network_hiddens=q_network_hiddens,
                    policy_hiddens=policy_hiddens,
                    sac_steps=self.sac_steps,
                    batch_size=self.batch_size,
                    optimizer_class=optimizer_class,
                    policy_lr=policy_lr,
                    qf_lr=qf_lr,
                    tau=tau,
                    target_update_period=qf_target_update_period,
                    reward_scale=reward_scale,
                    discount=discount,
                )
                for _ in range(m)
            ]
        )

        self.to(torch.device(device))

    @property
    def m(self):
        return self._m

    def act_m(self, m, state) -> Action:
        return self.mc_actors[m].act(state)

    def per_step_m(self, m: int, state, action, reward, new_state, done, meta_episode: int, step: int):
        self.mc_actors[m].per_step(state, action, reward, new_state, done, meta_episode, step)

    def post_episode(self) -> None:
        pass

    def update_prior(self) -> None:
        pass

    def update_default(self) -> None:
        pass
