from __future__ import annotations

import functools
import typing
import gym.spaces
from typing_extensions import Self
from typing import NamedTuple, TypedDict, cast
from math import log, sqrt, prod

import gym
import numpy as np
import torch
import torch.nn.functional as F
from rlkit.torch.distributions import TanhNormal
from torch import Tensor, nn
from torch.autograd import Variable

import wandb
from algos.epic_util import ModuleWithKlDivergence, PriorWrapper, soft_update_from_to
from algos.logging import track_config
from algos.memory import ReplayMemory
from algos.types import Action, EPICModel
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.distributions import Categorical

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

class StochasticCategoricalPolicy(nn.Module):
    """
    Policy network with a categorical distribution head.
    """
    def __init__(self, hidden_sizes: tuple[int, ...], obs_dim: int, action_dim: int, device: str):
        super().__init__()
        self.device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        in_size = obs_dim
        self.hidden_layers = nn.Sequential()
        for next_size in hidden_sizes:
            self.hidden_layers.append(StochasticLinear(in_size, next_size))
            self.hidden_layers.append(nn.ReLU())
            in_size = next_size
        # output of this is logits
        self.last_fc = StochasticLinear(in_size, action_dim)

    def forward(self, x, reparameterize=None, deterministic=None, return_log_prob=None):
        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        x = self.hidden_layers(x)
        logits = self.last_fc(x)

        policy_dist = Categorical(logits=logits)
        # this sample is OK for some reason
        action = torch.atleast_2d(policy_dist.sample())
        # obs, action.detach(), log_prob
        log_prob = F.log_softmax(logits, dim=1)

        return action, None, None, log_prob, None, None, None, None

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor, deterministic=None):
        action, _, _, log_prob, *_ = self.forward(obs)

        return obs, action.detach(), log_prob
    

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
        # self.compile(mode="reduce-overhead")

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
        self.layers = nn.Sequential()
        for next_dim in hidden_dims:
            self.layers.append(StochasticLinear(current_dim, next_dim))
            self.layers.append(nn.ReLU())
            current_dim = next_dim

        self.last_fc = StochasticLinear(current_dim, output_size)
        self.layers.append(self.last_fc)
        # self.compile(mode="reduce-overhead")

    def forward(self, x):
        return self.layers(x)


class FlattenStochasticMlp(StochasticMlp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.compile()

    def forward(self, *x):
        flat = torch.cat(x, dim=1)
        return super().forward(flat)

# @torch.compile
def model_kl_div(default: nn.Module, prior: nn.Module):
    # calculate KL div between a default model and a prior
    kl = []
    for default_layer, prior_layer in zip(
        (layer1 for layer1 in default.modules() if isinstance(layer1, StochasticLinear)),
        (layer2 for layer2 in prior.modules() if isinstance(layer2, StochasticLinear)),
    ):
        kl.append(
            KL_div(
                mu1=default_layer.weights_mean,
                sigma1=default_layer.weights_log_var,
                mu2=prior_layer.weights_mean,
                sigma2=prior_layer.weights_log_var,
            )
        )
        kl.append(
            KL_div(
                mu1=default_layer.bias_mean,
                sigma1=default_layer.bias_log_var,
                mu2=prior_layer.bias_mean,
                sigma2=prior_layer.bias_log_var,
            )
        )

    return torch.stack(kl).sum()

# @torch.compile
def kl_regularizer(kl, prior_update_every, gamma, max_steps, c=1.5, delta=0.01):
    N = prior_update_every
    H = 1.0 * (1.0 - gamma**max_steps) / (1.0 - gamma)

    epsilon = log(2.0) / (2.0 * log(c)) * (1.0 + torch.log(kl / log(2.0 / delta)))
    reg = (1.0 + c) / 2.0 * sqrt(2.0) * torch.sqrt((kl + log(2.0 / delta) + epsilon) * N * H**2.0)
    return reg


# @torch.compile
def KL_div(mu1, sigma1, mu2, sigma2):
    term1 = torch.sum(torch.log(sigma2 / sigma1)) - len(sigma1)
    term2 = torch.sum(sigma1 / sigma2)
    term3 = torch.sum((mu2 - mu1).pow(2.0) / sigma2)

    return 0.5 * (term1 + term2 + term3)


class OptimizerFactory(typing.Protocol):  # type: ignore
    def __init__(self, params, lr): ...

    def zero_grad(self): ...
    def step(self): ...


class Losses(NamedTuple):
    qf1_loss: Tensor
    qf2_loss: Tensor
    policy_loss: Tensor


class EpicRegularizers(TypedDict):
    qf1: Tensor
    qf2: Tensor
    policy: Tensor


class EpicSACMcActor(nn.Module):
    def __init__(
        self,
        # state_dim: int,
        # action_dim: int,
        env: gym.Env,
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
        prior_update_every: int,
        gamma: float,
        max_steps: int,
        device: str,
        c: float,
        delta: float,
        enable_epic_regularization: bool,
        q_network_hiddens: tuple[int, ...] = (256, 256),
        policy_hiddens: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        assert env.action_space.shape is not None
        assert env.observation_space.shape is not None
        state_dim = prod(env.observation_space.shape)
        action_dim = prod(env.action_space.shape)

        self.sac_steps = sac_steps
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.tau = tau
        self.reward_scale = reward_scale
        self.discount = discount
        self.device = device
        self.enable_epic_regularization = enable_epic_regularization

        self.qf1 = FlattenStochasticMlp(input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1)
        self.qf2 = FlattenStochasticMlp(input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1)

        self.target_qf1 = FlattenStochasticMlp(
            input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1
        )
        self.target_qf2 = FlattenStochasticMlp(
            input_size=state_dim + action_dim, hidden_dims=q_network_hiddens, output_size=1
        )
        
        if isinstance(env.action_space, gym.spaces.Box):
            self.policy = StochasticTanhGaussianPolicy(
                hidden_sizes=policy_hiddens, obs_dim=state_dim, action_dim=action_dim, device=device
            )
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.policy = StochasticCategoricalPolicy(
                hidden_sizes=policy_hiddens, obs_dim=state_dim, action_dim=action_dim,
                device=device
            )

        # TODO possibly share this between MC actors
        # self.replay_buffer = ReplayMemory(replay_capacity)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=replay_capacity, device=torch.device(device)),
            batch_size=batch_size,
            prefetch=3,
        )

        # self.qf_criterion = nn.MSELoss()
        self.qf_criterion = F.mse_loss
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)

    def act(self, state: Tensor | np.ndarray) -> Action:
        # if not isinstance(state, Tensor):
        #     state = torch.from_numpy(state).to(torch.device(self.device))

        # if state.ndim < 2:
        #     state = state.unsqueeze(0)

        state = torch.atleast_2d(torch.as_tensor(state, device=self.device, dtype=torch.float32))

        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        action, _, _, log_prob, *_ = self.policy(state, reparameterize=True, return_log_prob=True)

        return Action(state=state, action=action.flatten(), log_prob=log_prob)

    def push_sample(self, state, action, reward, new_state, done):
        state = torch.as_tensor(state).squeeze().type(torch.float32).detach()
        action = torch.atleast_1d(torch.as_tensor(action).squeeze().type(torch.float32)).detach()
        new_state = torch.as_tensor(new_state).squeeze().type(torch.float32).detach()
        done = torch.atleast_1d(torch.tensor(done)).type(torch.int8).detach()
        reward = torch.atleast_1d(torch.tensor(reward)).type(torch.float32).detach()

        self.replay_buffer.add((state, action, reward, new_state, done))

    def train_step(self, kl_divergences: EpicRegularizers):
        # if self.replay_buffer.size() < self.batch_size:
        #     return

        if len(self.replay_buffer) < self.batch_size:
            return

        # state, action, reward, next_state, done = self.replay_buffer.sample(
        #     batch_size=self.batch_size, device=self.device, as_tensors=True
        # )

        state, action, reward, next_state, done = self.replay_buffer.sample()
        losses = self.compute_loss(state, action, reward, next_state, done, kl_divergences)

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

    def per_step(
        self, state, action, reward, new_state, done, meta_episode: int, step: int, kl_divergences: EpicRegularizers
    ):
        # every step, the MC actor adds a new sample and takes some number of trainsteps. then it may update its
        # target networks
        self.push_sample(state, action, reward, new_state, done)

        for _ in range(self.sac_steps):
            self.train_step(kl_divergences)

        if step % self.target_update_period == 0:
            self.update_targets()

    def update_targets(self):
        soft_update_from_to(self.qf1, self.target_qf1, self.tau)
        soft_update_from_to(self.qf2, self.target_qf2, self.tau)

    def compute_loss(self, state, action, reward, new_state, done, kl_divergences: EpicRegularizers) -> Losses:
        # (action, mean, log_std, log_prob, expected_log_prob, std, mean_action_log_prob, pre_tanh_value)
        state_next_actions, _, _, log_prob, *_ = self.policy(state, reparameterize=True, return_log_prob=True)
        log_prob = log_prob.unsqueeze(-1)

        # Policy loss
        q_new_actions = torch.min(self.qf1(state, state_next_actions), 
                                  self.qf2(state, state_next_actions))
        policy_loss = (log_prob - q_new_actions).mean()

        # QF
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        new_state_next_actions, _, _, new_state_log_prob, *_ = self.policy(
            new_state, reparameterize=True, return_log_prob=True
        )
        new_state_log_prob = new_state_log_prob.unsqueeze(-1)

        target_q_values = (
            torch.min(
                self.target_qf1(new_state, new_state_next_actions), 
                self.target_qf2(new_state, new_state_next_actions)
            )
            - new_state_log_prob
        )

        q_target = (self.reward_scale * reward + (1.0 - done) * self.discount * target_q_values).detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        stats = {"qf1_loss": qf1_loss.detach(), "qf2_loss": qf2_loss.detach(), "policy_loss": policy_loss.detach()}

        if self.enable_epic_regularization:
            policy_loss = policy_loss + kl_divergences["policy"]
            qf1_loss = qf1_loss + kl_divergences["qf1"]
            qf2_loss = qf2_loss + kl_divergences["qf2"]

            stats.update({"policy_kl_loss": kl_divergences["policy"].detach(), "qf1_kl_loss": kl_divergences["qf1"].detach(),
                          "qf2_kl_loss": kl_divergences["qf2"].detach()})

        wandb.log(stats)

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
        prior_update_every: int,
        gamma: float,  # TODO does this need to be equal to the discount.
        max_steps: int,
        c: float,
        delta: float,
        enable_epic_regularization: bool,
        prior_lambda: float,
        prior_lambda_decay: float,
        q_network_hiddens: tuple[int, ...] = (256, 256),
        policy_hiddens: tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self._m = m
        self.batch_size = batch_size
        self.sac_steps = sac_steps
        self.prior_lambda = prior_lambda
        self.prior_lambda_decay = prior_lambda_decay
        self.device = device
        self.prior_update_every = prior_update_every
        self.gamma = gamma
        self.max_steps = max_steps
        self.c = c
        self.delta = delta
        self.enable_epic_regularization = enable_epic_regularization

        # MC actor initialization
    
        def make_actor():
            return EpicSACMcActor(
                env=env,
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
                prior_update_every=prior_update_every,
                gamma=gamma,
                max_steps=max_steps,
                c=c,
                delta=delta,
                enable_epic_regularization=enable_epic_regularization,
            )

        self.mc_actors = nn.ModuleList([make_actor() for _ in range(m)])

        self.actor_pair = PriorWrapper(
            make_actor,
            prior_update_every=prior_update_every,
            gamma=gamma,
            max_steps=max_steps,
            c=c,
            delta=delta,
            tau=tau,
        )

        # initialize all networks to be the same so the divergence starts small
        with torch.no_grad():
            for target in self.mc_actors + [self.actor_pair.default]:
                soft_update_from_to(self.actor_pair.prior, target, 1.0)

        self.to(torch.device(device))

    @property
    def m(self):
        return self._m

    def _get_epic_regularizer(self, default, prior) -> Tensor:
        return kl_regularizer(
            model_kl_div(default, prior), self.prior_update_every, self.gamma, self.max_steps, self.c, self.delta
        )

    def get_epic_regularizers(self, actor: EpicSACMcActor) -> EpicRegularizers:
        """Return the divergence between some actor and the prior actor."""
        # default = self.actor_pair.default
        prior = self.actor_pair.prior

        qf1_reg = self._get_epic_regularizer(actor.qf1, prior.qf1)
        qf2_reg = self._get_epic_regularizer(actor.qf2, prior.qf2)
        policy_reg = self._get_epic_regularizer(actor.policy, prior.policy)

        return {"qf1": qf1_reg, "qf2": qf2_reg, "policy": policy_reg}

    def act_m(self, m, state) -> Action:
        return self.mc_actors[m].act(state)

    def per_step_m(self, m: int, meta_episode, step, action_dict: Action, reward, new_state, done):
        actor = cast(EpicSACMcActor, self.mc_actors[m])
        actor.per_step(
            action_dict["state"],
            action_dict["action"],
            reward,
            new_state,
            done,
            meta_episode,
            step,
            kl_divergences=self.get_epic_regularizers(actor),
        )

    def post_episode(self) -> None:
        # not needed.
        pass

    def update_prior(self) -> None:
        # update the prior by doing a polyak update from the current default.
        if self.enable_epic_regularization:
            self.actor_pair.update_prior()

    def pre_meta_episode(self):
        # update the MC workers from the default
        if self.enable_epic_regularization:
            for actor in self.mc_actors:
                with torch.no_grad():
                    soft_update_from_to(self.actor_pair.default, actor, 1.0)

    def update_default(self) -> None:
        # update the default from the MC copies
        # in the 1-worker case, just copy worker 0 into the default
        if self.m == 1:
            if self.enable_epic_regularization:
                with torch.no_grad():
                    soft_update_from_to(self.mc_actors[0], self.actor_pair.default, 1.0)
        else:
            raise NotImplementedError("Didn't implement this yet")
