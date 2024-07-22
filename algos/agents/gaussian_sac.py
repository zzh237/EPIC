"""
An SAC agent that uses gaussian parameterization of its parameters.
"""

from __future__ import annotations

import copy
import math
from typing import List, Sequence, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Distribution, Normal

from algos.memory import Memory

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def KL_div(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    term2 = torch.sum(sigma1 / sigma2)
    term3 = torch.sum((mu2 - mu1).pow(2) / sigma2)

    return 0.5 * (term1 + term2 + term3)


def get_param(shape: int | tuple[int, ...]):
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.FloatTensor(*shape))


class MeanStd(TypedDict):
    mean: float
    std: float


class StochasticLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True, prm_log_var_init: MeanStd | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        bias_size = out_dim if use_bias else None
        self.create_layer(weights_size, bias_size)
        self.init_layer(prm_log_var_init)
        self.eps_std = 1.0

    def create_layer(self, weights_size: tuple[int, int], bias_size: int | None):
        # create but do not initialize layers
        self.w_mu = get_param(weights_size)
        self.w_log_var = get_param(weights_size)
        if bias_size:
            self.b_mu = get_param(bias_size)
            self.b_log_var = get_param(bias_size)

    def init_layer(self, log_var_init: MeanStd | None):
        if log_var_init is None:
            log_var_init = {"mean": -10, "std": 0.1}
        n = self.w_mu.size(1)
        stdv = math.sqrt(1.0 / n)
        self.w_mu.data.uniform_(-stdv, stdv)
        self.w_log_var.data.normal_(log_var_init["mean"], log_var_init["std"])
        if self.use_bias:
            # TODO use bias size instead?
            self.b_mu.data.uniform(-stdv, stdv)
            self.b_log_var.data.normal_(log_var_init["mean"], log_var_init["std"])

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)

    def forward(self, x):
        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b_mu
        else:
            b_var = None
            bias_mean = None

        out_mean = self.operation(x, self.w_mu, bias_mean)

        if self.eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            out_var = self.operation(x.pow(2), w_var, b_var)

            noise = out_mean.data.new(out_mean.size()).normal_(0, self.eps_std)
            noise = Variable(noise, requires_grad=False)
            out_var = F.relu(out_var)
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    def __str__(self):
        return f"StochasticLinear({self.in_dim} -> {self.out_dim})"


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = self.normal_mean + self.normal_std * Variable(
            Normal(
                self.normal_mean.new().zero_(),
                self.normal_std.new().zero_(),
                # ptu.zeros(self.normal_mean.size()),
                # ptu.ones(self.normal_std.size())
            ).sample()
        )
        # z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class TanhGaussianPolicy(nn.Module):
    """
    A policy network that uses a series of stochasticLinear layers to predict the logstd
    and mean of a gaussian output.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] | int, obs_dim: int, action_dim: int, std: float | None = None):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = tuple(hidden_sizes)
        # init fc layers
        in_size = obs_dim
        self.fcs = []
        for i, next_size in enumerate(hidden_sizes):
            fc = StochasticLinear(in_size, next_size)
            in_size = next_size
            self.fcs.append(fc)
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

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return outputs.cpu().detach().numpy()

    def forward(self, obs, reparameterize=False, deterministic=False, return_log_prob=False):
        h = obs
        for fc in self.fcs:
            h = F.relu(fc(h))
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
                    action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)


class StochasticQNetwork(nn.Module):
    """A Q-network using stochasticLinear layers."""

    def __init__(self, num_inputs: int, num_actions, int, hidden_dims: tuple[int, ...]):
        input_dim = num_inputs + num_actions
        self.fcs = []
        for next_dim in hidden_dims:
            self.fcs.append(StochasticLinear(input_dim, next_dim))
            input_dim = next_dim

        self.last_fc = StochasticLinear(input_dim, 1)

    def forward(self, state, action):
        hidden_act = torch.cat([state, action], 1)
        for fc in self.fcs:
            hidden_act = fc(hidden_act)

        return self.last_fc(hidden_act)


from torch.optim import Optimizer


class EpicOptimizers(TypedDict):
    default_policy: Optimizer
    prior_policy: Optimizer
    q_networks: List[Optimizer]


class EpicSAC(nn.Module):
    """
    Keeps multiple copies of a TanhGaussianPolicy to perform EPIC updates on.
    """

    def __init__(
        self,
        policy_hidden_sizes: tuple[int, ...],
        obs_dim: int,
        action_dim: int,
        optimizer_class: type[Optimizer],
        policy_lr: float,
        q_networks: List[StochasticQNetwork],
        q_network_lr: float,
    ):
        super().__init__()

        self.optimizers = dict()

        # policies
        self.default_policy = TanhGaussianPolicy(
            hidden_sizes=policy_hidden_sizes, obs_dim=obs_dim, action_dim=action_dim
        )
        self.optimizers["default_policy"] = optimizer_class(self.default_policy.parameters(), lr=policy_lr)
        self.prior_policy = copy.deepcopy(self.default_policy)
        self.optimizers["prior_policy"] = optimizer_class(self.prior_policy.parameters(), lr=policy_lr)

        # q networks
        self.q_networks = q_networks
        self.optimizers["q_networks"] = list()
        for network in q_networks:
            self.optimizers["q_networks"].append(optimizer_class(network.parameters(), lr=q_network_lr))

    def initialize_default_policy(self):
        """Initialize the default policy by copying it from the prior."""
        self.default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

    def initialize_policy_m(self):
        self.initialize_default_policy()

    def train(self):
        pass

    def act_policy_m(self, state, task_idx):
        """Take an action using the policy as of task {task_idx}"""
        return self.act(state)

    def act(self, state):
        self.default_policy.get_action(state, deterministic=False)


    def update_mu_theta_for_default(self, meta_memory: Memory, meta_update_every: int, H):
        pass

