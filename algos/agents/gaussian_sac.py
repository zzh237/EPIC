"""
An SAC agent that uses StochasticLinear gaussian-parameterized layers.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import math
from torch.optim import Optimizer
from typing import List, Sequence, TypedDict
from torch.optim import Adam
import wandb
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Distribution, Normal
from algos.logging import track_config
from algos.memory import Memory, ReplayMemory

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

    def __init__(
        self, hidden_sizes: tuple[int, ...] | int, obs_dim: int, action_dim: int, device: str, std: float | None = None
    ):
        super().__init__()
        self.device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = tuple(hidden_sizes)
        # init fc layers
        in_size = obs_dim
        self.fcs = nn.ModuleList()
        for i, next_size in enumerate(hidden_sizes):
            fc = StochasticLinear(in_size, next_size).to(device=device)
            in_size = next_size
            self.fcs.append(fc)
        self.last_fc = StochasticLinear(in_size, action_dim)

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = StochasticLinear(last_hidden_size, action_dim).to(device=device)
        else:
            self.log_std = np.log(std)

        self.to(device=device)

    def copy(self):
        # some module types are not cloneable, so you'll have to rebuild the network here if so
        new = copy.deepcopy(self)
        new.load_from(self)
        return new
 

    def load_from(self, other: "TanhGaussianPolicy"):
        """Reload own parameters with those from another network."""
        self.load_state_dict(other.state_dict())


    def get_action(self, obs, deterministic=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # state, detached action, action logprob
        obs = torch.from_numpy(obs).float().to(self.device)
        state, action, log_prob = self.get_actions(obs, deterministic=deterministic)
        return state, action, log_prob

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor, deterministic=False):
        action, _, _, log_prob, _, _, _, _ = self.forward(obs, deterministic=deterministic, return_log_prob=True)
        return obs, action.detach(), log_prob

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

                log_prob = log_prob.sum()
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)


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


class EpicOptimizers(TypedDict):
    policy: Optimizer
    q_networks: List[Optimizer]
    v_network: Optimizer


class KlRegularizationSettings(TypedDict):
    q_network: bool
    v_network: bool
    policy: bool

class UpdateMetrics(TypedDict):
    q_loss: float
    q_epic_reg: float
    v_loss: float
    v_epic_reg: float
    policy_loss: float
    policy_epic_reg: float


class EpicSACActor(nn.Module):
    """
    Combination of q, v, policy and priors.
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim :int,
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
                 kl_settings: KlRegularizationSettings = KlRegularizationSettings(q_network=True, v_network=True, 
                                                                                  policy=True),
                 optimizer_class: type[Optimizer] = Adam,
                 ):
        self.replay_buffer = ReplayMemory(capacity=replay_capacity)
        self.device = device
        self.optimizers = EpicOptimizers()
        self.discount = discount
        self.batch_size = batch_size
        self.kl_settings = kl_settings

        # policy
        self.policy_new = TanhGaussianPolicy(hidden_sizes=policy_hidden_sizes,
                                         obs_dim=obs_dim,
                                         action_dim=action_dim)
        self.policy_default = self.policy_new.copy()
        self.policy_prior = self.policy_new.copy()
        self.optimizers["policy"] = optimizer_class(self.policy_default.parameters(), lr=policy_lr)

        # q networks
        self.q_networks = nn.ModuleList(q_networks)
        self.optimizers["q_networks"] = list()
        for network in self.q_networks:
            self.optimizers["q_networks"].append(optimizer_class(network.parameters(), lr=q_network_lr))

        self.q_priors = nn.ModuleList(q.copy() for q in q_networks)

        self.v_network_prior = v_network
        self.v_network_default = self.v_network_prior.copy()
        self.target_v_network = self.v_network_prior.copy()
        self.v_criterion = nn.MSELoss()
        self.soft_target_tau = soft_target_tau
        self.optimizers["v_network"] = optimizer_class(self.v_network_default, lr=v_network_lr)

        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight    



    def update(self, N: int, H: int) -> UpdateMetrics:
        """
        Perform 1 SAC update on this actor.
        """
        metrics = UpdateMetrics()

        states, actions, rewards, succ_states, dones = self.replay_buffer.sample(
                self.batch_size, as_tensors=True, device=self.device
            )
        dones = dones.to(float)

        policy_outputs = self.policy_default(states, return_log_prob=True)
        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        new_actions, policy_mean, policy_log_std, log_pi, *_ = policy_outputs

        # update Q and V networks
        q_preds = [q_net(states, actions) for q_net in self.q_networks]
        v_pred = self.v_network_default(states)

        with torch.no_grad():
            target_v_values = self.target_v_network(succ_states)

        # qf update
        for o in self.optimizers["q_networks"]:
            o.zero_grad()

        q_target = rewards + (1.0 - dones) * self.discount * target_v_values
        q_loss = sum([torch.mean((q_pred - q_target) ** 2) for q_pred in q_preds])
        metrics["q_loss"] = q_loss.cpu().detach()
        if self.kl_settings["q_network"]:
            q_kl_sum = 0
            for q_default, q_prior in zip(self.q_networks, self.q_priors):
                q_kl_sum += kl_regularizer(model_kl_div(q_default, q_prior), N, H)
            q_loss += q_kl_sum
            metrics["q_epic_reg"] = q_kl_sum.cpu().detach()
            # m_metrics["q_epic_reg"].append(q_kl_sum.cpu().detach())

        q_loss.backward()
        for o in self.optimizers["q_networks"]:
            o.step()

        # minq
        min_q_new_actions = self.min_q(states, new_actions)

        # v update
        # I think this one is KL - regularized?
        v_target = min_q_new_actions - log_pi
        v_loss = self.v_criterion(v_pred, v_target.detach())

        # m_metrics["v_loss"].append(v_loss.cpu().detach())

        # kl-divergence for v_function, put default on left and prior on right
        if self.kl_settings["v_network"]:
            v_kl_regularizer = kl_regularizer(model_kl_div(self.v_network_default, self.v_network_prior), N, H)
            v_loss += v_kl_regularizer
            # m_metrics["v_epic_reg"].append(v_kl_regularizer.cpu().detach())
            metrics["v_epic_reg"] = v_kl_regularizer.cpu().detach()

        self.optimizers["v_network"].zero_grad()
        v_loss.backward()
        self.optimizers["v_network"].step()

        # update the target (polyak)
        for target_p, p in zip(self.target_v_network.parameters(), self.v_network_default.parameters()):
            target_p.data.copy_(target_p.data * (1.0 - self.soft_target_tau) + p.data * self.soft_target_tau)

        # policy update
        log_policy_target = min_q_new_actions
        policy_loss = (log_pi - log_policy_target).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (pre_tanh_value**2).sum().mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        metrics["policy_loss"] = policy_loss.cpu().detach()
        if self.kl_settings["policy"]:
            policy_epic_reg = kl_regularizer(model_kl_div(self.policy_default, self.policy_prior), N, H)
            metrics["policy_epic_reg"] = policy_epic_reg.cpu().detach()
            policy_loss += policy_epic_reg

        self.optimizers["policy"].zero_grad()
        policy_loss.backward()
        self.optimizers["policy"].step()
        
        return metrics
    


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
        soft_target_tau: float = 1e-2,
        policy_mean_reg_weight: float = 1e-3,
        policy_std_reg_weight: float = 1e-3,
        policy_pre_activation_weight: float = 0.0,
        optimizer_class: type[Optimizer] = Adam,
        kl_settings: KlRegularizationSettings = KlRegularizationSettings(q_network=True, v_network=True, policy=True),
    ):
        super().__init__()

        # instantiate M copies of the SAC actor for MC updates
        self.default_actor = EpicSACActor(
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
            
            

        )
        self.actors: nn.ModuleList[EpicSACActor] = nn.ModuleList(self.default_actor.copy() 
                                                                 for _ in range(m))

        

        # self.batch_size = batch_size
        # self.discount = discount
        # self.device = device
        # self.m = m  # MC copies
        # self.kl_settings = kl_settings

        # self.optimizers = dict()

        # # policies
        # self.new_default_policy = TanhGaussianPolicy(hidden_sizes=policy_hidden_sizes,
        #                                              obs_dim=obs_dim, action_dim=action_dim, device=device)
        # self.default_policy = self.new_default_policy.copy()
        # self.policies_mc = nn.ModuleList(self.new_default_policy.copy() for _ in range(self.m))
        # self.optimizers["default_policy"] = [optimizer_class(self.default_policy.parameters(), lr=policy_lr) for _ in range(self.m)]
        # self.prior_policy = copy.deepcopy(self.default_policy)
        # self.optimizers["prior_policy"] = optimizer_class(self.prior_policy.parameters(), lr=policy_lr)

        # # q networks
        # self.q_networks = nn.ModuleList(q_networks)

        # self.optimizers["q_networks"] = list()
        # for network in q_networks:
        #     self.optimizers["q_networks"].append(optimizer_class(network.parameters(), lr=q_network_lr))

        # self.q_priors = [q.copy() for q in q_networks]

        # self.prior_vf = v_network.to(device)
        # self.default_vf = self.prior_vf.copy()
        # self.target_vf = self.default_vf.copy()
        # self.v_criterion = nn.MSELoss()
        # self.soft_target_tau = soft_target_tau
        # self.optimizers["v_network"] = optimizer_class(self.default_vf.parameters(), lr=v_network_lr)

        # self.policy_mean_reg_weight = policy_mean_reg_weight
        # self.policy_std_reg_weight = policy_std_reg_weight
        # self.policy_pre_activation_weight = policy_pre_activation_weight

        # # KL summary value
        # self.KL = torch.tensor(0.0)

        # self.replay_buffer = ReplayMemory(capacity=replay_capacity)

        # self.to(device=device)

    def initialize_default_policy(self):
        """Initialize the default policy by copying it from the prior."""
        self.default_policy.load_state_dict(copy.deepcopy(self.default_policy.state_dict()))

    def initialize_policy_m(self):
        """Initialize the MC policies by loading the state from the default policy"""
        self.initialize_default_policy()

    def train(self):
        pass

    def act_policy_m(self, state, task_idx: int):
        """Take an action using the policy as of task {task_idx}"""
        return self.act(state)

    def act(self, state):
        return self.default_policy.get_action(state, deterministic=False)

    def min_q(self, obs, actions):
        values, _ = torch.min(torch.cat([q(obs, actions) for q in self.q_networks], dim=1), dim=1, keepdim=True)
        return values

    def update_mu_theta_for_default(self, meta_memory: Memory, N: int, H: int):
        """Do one SAC update on the default policy."""
        m_metrics = defaultdict(list)  # metrics across the whole MC step

        for m_idx, actor in enumerate(self.actors):
            # load actors with default policy's params
            update_dict = actor.update()
            for k, v in update_dict.items():
                m_metrics[k].append(v)
        
            # states, actions, rewards, succ_states, dones = self.replay_buffer.sample(
            #     self.batch_size, as_tensors=True, device=self.device
            # )
            # dones = dones.to(float)

            # policy_outputs = self.default_policy(states, return_log_prob=True)
            # # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
            # new_actions, policy_mean, policy_log_std, log_pi, *_ = policy_outputs

            # # update Q and V networks
            # q_preds = [q_net(states, actions) for q_net in self.q_networks]
            # v_pred = self.default_vf(states)

            # with torch.no_grad():
            #     target_v_values = self.target_vf(succ_states)

            # # qf update
            # for o in self.optimizers["q_networks"]:
            #     o.zero_grad()

            # q_target = rewards + (1.0 - dones) * self.discount * target_v_values
            # # I think this isn't KL regularized. maybe?
            # q_loss = sum([torch.mean((q_pred - q_target) ** 2) for q_pred in q_preds])
            # m_metrics["q_loss"].append(q_loss.cpu().detach())
            # if self.kl_settings["q_network"]:
            #     q_kl_sum = 0
            #     for q_default, q_prior in zip(self.q_networks, self.q_priors):
            #         q_kl_sum += kl_regularizer(model_kl_div(q_default, q_prior), N, H)
            #     q_loss += q_kl_sum
            #     m_metrics["q_epic_reg"].append(q_kl_sum.cpu().detach())

            # q_loss.backward()
            # for o in self.optimizers["q_networks"]:
            #     o.step()

            # # minq
            # min_q_new_actions = self.min_q(states, new_actions)

            # # v update
            # # I think this one is KL - regularized?
            # v_target = min_q_new_actions - log_pi
            # v_loss = self.v_criterion(v_pred, v_target.detach())

            # m_metrics["v_loss"].append(v_loss.cpu().detach())

            # # kl-divergence for v_function, put default on left and prior on right
            # if self.kl_settings["v_network"]:
            #     v_kl_regularizer = kl_regularizer(model_kl_div(self.default_vf, self.prior_vf), N, H)
            #     v_loss += v_kl_regularizer
            #     m_metrics["v_epic_reg"].append(v_kl_regularizer.cpu().detach())

            # self.optimizers["v_network"].zero_grad()
            # v_loss.backward()
            # self.optimizers["v_network"].step()

            # # update the target (polyak)
            # for target_p, p in zip(self.target_vf.parameters(), self.default_vf.parameters()):
            #     target_p.data.copy_(target_p.data * (1.0 - self.soft_target_tau) + p.data * self.soft_target_tau)

            # # policy update
            # log_policy_target = min_q_new_actions
            # policy_loss = (log_pi - log_policy_target).mean()
            # mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
            # std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
            # pre_tanh_value = policy_outputs[-1]
            # pre_activation_reg_loss = self.policy_pre_activation_weight * (pre_tanh_value**2).sum().mean()
            # policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            # policy_loss = policy_loss + policy_reg_loss
            # m_metrics["policy_loss"].append(policy_loss.cpu().detach())
            # if self.kl_settings["policy"]:
            #     policy_epic_reg = kl_regularizer(model_kl_div(self.default_policy, self.prior_policy), N, H)
            #     m_metrics["policy_epic_reg"].append(policy_epic_reg.cpu().detach())
            #     policy_loss += policy_epic_reg

            # self.optimizers["default_policy"].zero_grad()
            # policy_loss.backward()
            # self.optimizers["default_policy"].step()

        # update the default actor with the MC update
        

        wandb.log({name: wandb.Histogram(values) for name, values in m_metrics.items()})

    def update_default_and_prior_policy(self):
        pass