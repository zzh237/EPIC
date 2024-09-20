"""
An SAC agent that uses StochasticLinear gaussian-parameterized layers.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import math
from collections import defaultdict
from functools import wraps
from typing import List, Sequence, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.distributions import Distribution, Normal
from torch.nn.parameter import Parameter
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

import wandb
from algos.memory import Memory, ReplayMemory
from algos.types import Action, EPICModel

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def flatmap(func, iterable):
    return itertools.chain.from_iterable(map(func, iterable))


def KL_div(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    term2 = torch.sum(sigma1 / sigma2)
    term3 = torch.sum((mu2 - mu1).pow(2) / sigma2)

    return 0.5 * (term1 + term2 + term3)


def get_param(shape: int | tuple[int, ...]):
    if isinstance(shape, int):
        shape = (shape,)
    return Parameter(torch.FloatTensor(*shape))


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
            self.b_mu.data.uniform_(-stdv, stdv)
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
        z = self.normal.sample() # gradients stop here
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

    def __init__(
        self, hidden_sizes: tuple[int, ...] | int, obs_dim: int, action_dim: int, device: str, std: float | None = None
    ):
        super().__init__()
        self.device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        # init fc layers
        in_size = obs_dim
        self.fcs = nn.ModuleList()
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

    def copy(self):
        # some module types are not cloneable, so you'll have to rebuild the network here if so
        new = copy.deepcopy(self)
        new.load_from(self)
        return new

    def load_from(self, other: "TanhGaussianPolicy"):
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
        action, _, _, log_prob, _, _, _, _ = self.forward(obs, deterministic=deterministic, reparameterize=True, return_log_prob=True)
        return obs, action.detach(), log_prob

    def forward(
        self, obs, reparameterize=False, deterministic=False, return_log_prob=False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value) # gradients can go through here

                log_prob = log_prob.sum()
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)  # type: ignore


class StochasticMlp(nn.Module):
    """A generic MLP using stochasticLinear layers."""

    def __init__(self, input_size: int, output_size: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        current_dim = input_size
        self.fcs = nn.ModuleList()
        for next_dim in hidden_dims:
            self.fcs.append(StochasticLinear(current_dim, next_dim))
            current_dim = next_dim

        self.last_fc = StochasticLinear(current_dim, output_size)

    def forward(self, x):
        hidden_act = x
        for fc in self.fcs:
            hidden_act = fc(hidden_act)
        return self.last_fc(hidden_act)

    def copy(self):
        return copy.deepcopy(self)


class FlattenStochasticMlp(StochasticMlp):
    """MLP which flattens its inputs along the first nonbatch dimension."""

    def forward(self, *x):
        flat = torch.cat(x, dim=1)
        return super().forward(flat)


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


class EpicOptimizers(TypedDict, total=False):
    policy: Optimizer
    q_networks: List[Optimizer]
    v_network: Optimizer


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def track_config(init):
    """Log all arguments to init as config objects in wandb."""

    @wraps(init)
    def wrapper(self, **kwargs):
        default_args = get_default_args(init)
        params = kwargs
        params.update(default_args)
        wandb.config.update(params)
        return init(self, **kwargs)

    return wrapper


class KlRegularizationSettings(TypedDict):
    q_network: bool
    policy: bool


class UpdateMetrics(TypedDict):
    q_loss: float
    q_epic_reg: float
    v_loss: float
    v_epic_reg: float
    policy_loss: float
    policy_epic_reg: float

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class EpicSACActor(nn.Module):
    """
    Combination of q, v, policy.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy_hidden_sizes: tuple[int, ...],
        policy_lr: float,
        discount: float,
        batch_size: int,
        q_networks: Sequence[StochasticMlp],
        q_network_lr: float,
        v_network: StochasticMlp,
        v_network_lr: float,
        soft_target_tau: float,
        policy_mean_reg_weight: float,
        policy_std_reg_weight: float,
        policy_pre_activation_weight: float,
        replay_capacity: int,
        device: str,
        kl_settings: KlRegularizationSettings,
        optimizer_class: type[Optimizer] = Adam,
        alpha: float = 1.,
        target_update_period: int = 1,
    ):
        super().__init__()
        self.replay_buffer = ReplayMemory(capacity=replay_capacity)
        self.device = device
        self.optimizers = EpicOptimizers()
        self.discount = discount
        self.batch_size = batch_size
        self.kl_settings = kl_settings
        self.alpha = alpha
        self.target_update_period = target_update_period

        # policy
        self.policy_default = TanhGaussianPolicy(
            hidden_sizes=policy_hidden_sizes, obs_dim=obs_dim, action_dim=action_dim, device=device
        )

        self.optimizers["policy"] = optimizer_class(self.policy_default.parameters(), lr=policy_lr)

        # q networks
        self.q_networks = nn.ModuleList(q_networks)
        self.optimizers["q_networks"] = list()
        for network in self.q_networks:
            self.optimizers["q_networks"].append(optimizer_class(network.parameters(), lr=q_network_lr))

        self.qf_criterion = nn.MSELoss()

        self.target_q_networks = nn.ModuleList()
        for n in self.q_networks:
            self.target_q_networks.append(n.copy())


        self.total_steps = 0

        # self.v_network_default = v_network
        # self.target_v_network = self.v_network_default.copy()
        # self.v_criterion = nn.MSELoss()
        self.soft_target_tau = soft_target_tau
        # self.optimizers["v_network"] = optimizer_class(self.v_network_default.parameters(), lr=v_network_lr)

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight


    def min_q(self, obs, actions, networks):
        values, _ = torch.min(torch.cat([q(obs, actions) for q in networks], dim=1), dim=1, keepdim=True)
        return values

    def update(self, prior: "EpicSACActor", N: int, H: int) -> UpdateMetrics:
        """
        Perform 1 SAC update on this actor.
        """
        ## Unpack
        metrics = UpdateMetrics()

        states, actions, rewards, succ_states, dones = self.replay_buffer.sample(
            self.batch_size, as_tensors=True, device=self.device
        )
        dones = dones.to(dtype=torch.float32)

        ## policy loss / update
        policy_outputs = self.policy_default(states, return_log_prob=True)
        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        new_actions, policy_mean, policy_log_std, log_pi, *_ = policy_outputs
        q_new_actions = self.min_q(states, new_actions, self.q_networks)
        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        if self.kl_settings["policy"]:
            policy_epic_reg = kl_regularizer(model_kl_div(self.policy_default, prior.policy_default), N, H)
            metrics["policy_epic_reg"] = policy_epic_reg.cpu().detach()
            policy_loss += policy_epic_reg

        self.optimizers["policy"].zero_grad()
        policy_loss.backward()
        self.optimizers["policy"].step()


        ## Q loss
        q_preds = [q(states, actions) for q in self.q_networks]
        next_policy_outputs = self.policy_default(succ_states, return_log_prob=True)
        new_next_actions, new_policy_mean, new_policy_log_std, new_log_pi, *_ = next_policy_outputs
        target_q_values = self.min_q(succ_states, new_next_actions, self.target_q_networks) - self.alpha * new_log_pi
        q_target = rewards + (1. - dones) * self.discount * target_q_values
        q_loss = sum([self.qf_criterion(q_pred, q_target.detach()) for q_pred in q_preds])
        metrics["q_loss"] = q_loss.cpu().detach()

        if self.kl_settings["q_network"]:
            q_kl_sum = 0
            for q_default, q_prior in zip(self.q_networks, prior.q_networks):
                q_kl_sum += kl_regularizer(model_kl_div(q_default, q_prior), N, H)
            q_loss += q_kl_sum
            metrics["q_epic_reg"] = q_kl_sum.cpu().detach()

        q_loss.backward()
        for o in self.optimizers["q_networks"]:
            o.step()


        self.total_steps += 1

        if self.total_steps % self.target_update_period == 0:
            for q, q_target in zip(self.q_networks, self.target_q_networks):
                soft_update_from_to(q, q_target, self.soft_target_tau)


        return metrics

    def forward(self, obs):
        return self.act(obs)

    def copy(self):
        return copy.deepcopy(self)

    def load_from(self, other: "EpicSACActor"):
        self.load_state_dict(other.state_dict())

    def act(self, obs, deterministic=True):
        return self.policy_default.get_action(obs, deterministic=deterministic)



class EpicSAC(nn.Module):
    """
    Keeps multiple copies of an EpicSACActor to do an MC update on. Each Actor is metalearning policy.
    """

    @track_config
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        policy_hidden_sizes: tuple[int, ...],
        policy_lr: float,
        q_networks: List[StochasticMlp],
        q_network_lr: float,
        v_network: StochasticMlp,
        v_network_lr: float,
        discount: float,
        replay_capacity: int,
        batch_size: int,
        device: str,
        m: int,
        c1: float = 1.6,
        lam: float = 0.9,
        lam_decay: float = 0.999,
        soft_target_tau: float = 1e-2,
        policy_mean_reg_weight: float = 1e-3,
        policy_std_reg_weight: float = 1e-3,
        policy_pre_activation_weight: float = 0.0,
        optimizer_class: type[Optimizer] = Adam,
        kl_settings: KlRegularizationSettings = KlRegularizationSettings(q_network=True, policy=True),
    ):
        super().__init__()

        # TODO try larger / smaller versions of this
        self.c1 = c1
        self.lam = lam
        self.lam_decay = lam_decay
        self.m = m
        # this isn't used for anything
        self.KL = torch.tensor(0.)


        self.prior_actor = EpicSACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_sizes=policy_hidden_sizes,
            policy_lr=policy_lr,
            q_networks=q_networks,
            q_network_lr=q_network_lr,
            v_network=v_network,
            v_network_lr=v_network_lr,
            discount=discount,
            replay_capacity=replay_capacity,
            device=device,
            kl_settings=kl_settings,
            batch_size=batch_size,
            soft_target_tau=soft_target_tau,
            policy_mean_reg_weight=policy_mean_reg_weight,
            policy_pre_activation_weight=policy_pre_activation_weight,
            policy_std_reg_weight=policy_std_reg_weight,
            optimizer_class=optimizer_class,
        )
        self.default_actor = self.prior_actor.copy()
        self.new_actor = self.default_actor.copy()
        # instantiate M copies of the SAC actor for MC updates
        self.mc_actors = nn.ModuleList(self.default_actor.copy() for _ in range(m))
        self.to(device=torch.device(device))

    def initialize_policy_m(self):
        """Initialize the MC actors by loading the state from the default actor"""
        actor: EpicSACActor
        for actor in self.mc_actors:
            actor.load_from(self.default_actor)

    def act_policy_m(self, state, m_idx: int):
        """Retrieve action from MC copy {i}"""
        return self.mc_actors[m_idx].act(state)

    def act(self, state):
        return self.default_actor.act(state, deterministic=False)


    def update_mu_theta_for_default(self, meta_memory: Memory, N: int, H: int):
        """Do one SAC update on the default policy."""
        m_metrics = defaultdict(list)  # metrics across the whole MC step
        # key: q_loss, value: [0.0001, 0.0002, ....] for all MC actors

        actor: EpicSACActor
        for m_idx, actor in enumerate(self.mc_actors):
            # load MC actors with default policy's params
            actor_parameters_before = copy.deepcopy(actor.state_dict())
            update_metrics = actor.update(self.prior_actor, N, H)
            for k, v in update_metrics.items():
                m_metrics[k].append(v)

            # accumulate the parameter step, subtract out the previous value
            v = copy.deepcopy(actor.state_dict())
            for key in actor_parameters_before:
                # v = current params - before params
                v[key] -= actor_parameters_before[key]

        # update the new_actor with the MC update
        for name, param in self.new_actor.named_parameters():
            param.data.copy_(param.data + self.c1*v[name]/self.m)

        # wandb.log({name: wandb.Histogram(values) for name, values in m_metrics.items()}, commit=False)
        out_metrics = {}
        for name, values in m_metrics.items():
            out_metrics.update({f"{name}.mean": np.mean(values), f"{name}.std": np.std(values)})

        return out_metrics


    def update_default_and_prior_policy(self):
        self.default_actor.load_from(self.new_actor)

        # update the prior with decay
        # TODO not sure if this will mess up stepping
        wandb.log({"lambda": self.lam}, commit=False)

        for prior_param, new_default_param in zip(self.prior_actor.parameters(),
                                                  self.new_actor.parameters()):
            prior_param.data.copy_((1-self.lam) * new_default_param.data + self.lam * prior_param.data)


        self.lam *= self.lam_decay


