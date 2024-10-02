from operator import itemgetter
from algos.epic_util import PriorWrapper, soft_update_from_to
from algos.memory import Memory
from algos.types import Action, EPICModel
from torch import nn
import torch
import math
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, MultivariateNormal, Categorical
from torch.autograd import Variable
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import copy
from math import log, sqrt
from collections import deque

from typing import TypedDict

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class MeanStd(TypedDict):
    mean: float
    std: float

def KL_div(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    term2 = torch.sum(sigma1 / sigma2)
    term3 = torch.sum((mu2 - mu1).pow(2) / sigma2)

    return 0.5 * (term1 + term2 + term3)


def get_param(shape: int | tuple[int, ...]):
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.FloatTensor(*shape))

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



class GaussianContActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_std: float, device):
        super().__init__()
        self.device = device

        self.layers = nn.Sequential()
        curr_size = state_dim
        for next_size in hidden_sizes[:-1]:
            self.layers.append(StochasticLinear(curr_size, next_size))
            self.layers.append(activation())
            curr_size = next_size
        self.layers.append(StochasticLinear(curr_size, action_dim))
        self.layers.append(nn.Tanh())
        self.action_var = torch.full((action_dim, ), action_std ** 2.)
    
    def act(self, state):
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        action_mean = self.layers(state)
        cov = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return Action(state=state, action=action.detach(), log_prob=log_prob)
    
    

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, device):
        super().__init__()
        self.device = device

        self.layers = nn.Sequential()
        curr_size = state_dim
        for next_size in hidden_sizes[:-1]:
            self.layers.append(StochasticLinear(curr_size, next_size))
            self.layers.append(activation())
            curr_size = next_size
        self.layers.append(StochasticLinear(curr_size, action_dim))
        self.layers.append(nn.Softmax(dim=-1))
    
    def act(self, state) -> Action:
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.layers(state)
        dist = Categorical(action_probs)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        return Action(state=state, action=action.detach(), log_prob=log_prob)



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


def kl_regularizer(kl, prior_update_every, gamma, max_steps, 
                c = 1.5,
                delta = 0.01
                   ):
    N = prior_update_every
    H = 1. * (1. - gamma ** max_steps) / (1. - gamma)
    
    epsilon = log(2.) / (2. * log(c)) * (1. + torch.log(kl / log(2. / delta)))
    reg = (1. + c ) / 2. * sqrt(2.) * torch.sqrt((kl + log(2. / delta) + epsilon) * N * H ** 2.)
    return reg

class GaussianVPGMC2(EPICModel):
    def __init__(self, m: int, env: Env, hidden_sizes: tuple[int, ...], device, action_std: float,
                 prior_update_every: int, gamma: float, max_steps: int, c: float, delta: float, tau: float,
                 lr: float, c1: float, lam: float, lam_decay: float, enable_epic_regularization: bool,
                 optimizer
            ):
        super().__init__()
        self.gamma = gamma
        self.prior_update_every = prior_update_every
        self.max_steps = max_steps
        self.c1 = c1
        self.lam = lam
        self.lam_decay = lam_decay
        self._m = m
        self.device = device
        self.enable_epic_regularization = enable_epic_regularization

        assert env.action_space.shape is not None
        assert env.observation_space.shape is not None

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        def make_actor():
            if isinstance(env.action_space, Discrete):
                return GaussianActor(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes,
                                     activation=nn.ReLU, device=device)
            elif isinstance(env.action_space, Box):
                return GaussianContActor(state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes,
                                         action_std=action_std,
                                         activation=nn.ReLU, device=device)
            else:
                raise NotImplementedError(f"Unrecognized space type: {env.action_space}")

        self.policies = nn.ModuleList(
            [make_actor() for _ in range(m)]
        )
        self.optimizers = [optimizer(p.parameters(), lr=lr) for p in self.policies]

        self.actor_pair = PriorWrapper(make_actor,
                                       prior_update_every=prior_update_every,
                                       gamma=gamma,
                                       max_steps=max_steps,
                                       c=c,
                                       delta=delta,
                                       tau=tau
                                       )
        
        self.memories = [Memory() for _ in range(m)]
        self.to(device)

    @property
    def m(self) -> int:
        return self._m

    def act_m(self, m: int, state) -> Action:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        state = state.to(self.device)
        return self.policies[m].act(state)
    
    def per_step_m(self, m: int, meta_episode, step, action_dict: Action, reward, new_state, done):
        # populate memory with full trajectories
        state, action, log_prob = itemgetter("state", "action", "log_prob")(action_dict)
        self.memories[m].add(state, action, log_prob, reward, done)

    def policy_loss(self, memory: Memory):
        # compute a policy gradient loss from a memory
        discounted_reward = deque()
        Gt = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if done:
                Gt = 0
            Gt = reward + (self.gamma * Gt)
            discounted_reward.appendleft(Gt)

        policy_gradient = []
        gamma_pow = 1

        for log_prob, Gt, done in zip(memory.logprobs, discounted_reward, memory.is_terminals):
            policy_gradient.append(-log_prob * Gt * gamma_pow)
            if done:
                gamma_pow = 1
            else:
                gamma_pow *= self.gamma
        policy_gradient = torch.stack(policy_gradient).sum()
        return policy_gradient
    
    def get_epic_regularizer(self, default, prior):
        return kl_regularizer(model_kl_div(default, prior), self.prior_update_every, self.gamma, self.max_steps)

    def post_episode(self) -> None:
        pass

    def post_meta_episode(self):
        # accumulate default's update by averaging MC policies
        v = dict()
        for m_idx in range(self.m):
            policy_m = self.policies[m_idx]
            policy_m_before = copy.deepcopy(policy_m.state_dict())
            loss = self.policy_loss(self.memories[m_idx])
            if self.enable_epic_regularization:
                loss = loss + self.get_epic_regularizer(policy_m, self.actor_pair.prior)

            self.optimizers[m_idx].zero_grad()
            loss.backward()
            self.optimizers[m_idx].step()
            
            for key in policy_m_before:
                if key not in v:
                    v[key] = torch.zeros_like(policy_m_before[key])
                v[key] = v[key] + policy_m.state_dict()[key] - policy_m_before[key]

            self.memories[m_idx].clear_memory()

            
        for k, parameter in zip(v, self.actor_pair.default.parameters()):
            with torch.no_grad():
                parameter.data.copy_(parameter.data + self.c1 * v[k] / self.m)
        

    def update_default(self) -> None:
        # we actually need to update the default every meta-episode, so this
        # method is not called frequently enough
        pass
        

    def update_prior(self) -> None:
        # update prior from default
        with torch.no_grad():
            soft_update_from_to(self.actor_pair.default, self.actor_pair.prior, self.lam)
        
        self.lam *= self.lam_decay
    
    
    def pre_meta_episode(self):
        # load all policies from the default
        for m_idx in range(self.m):
            self.policies[m_idx].load_state_dict(self.actor_pair.default.state_dict())
