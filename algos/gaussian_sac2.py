from __future__ import annotations
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions import Distribution, Normal
from torch import Tensor

from algos.types import EPICModel, Action


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class StochasticLinear(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int, 
                 use_bias: bool = True, 
                 log_var_mean: float = -10.,
                 log_var_std: float = 0.1,
                 eps_std: float = 1.):
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

        if self.eps_std == 0.:
            return out_mean
        else:
            weights_var = torch.exp(self.weights_log_var)
            out_var = F.linear(x.pow(2.), weights_var, bias_var)
            noise = out_mean.data.new(out_mean.size()).normal_(0, self.eps_std)
            noise = Variable(noise, requires_grad=False)
            out_var = F.relu(out_var)
            return out_mean + noise * torch.sqrt(out_var)
        
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
        z.requires_grad_(False) # do not try to differentiate sampling from the normal
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z: Tensor = self.normal_mean + self.normal_std * Variable(
            Normal(
                self.normal_mean.new().zero_(),
                self.normal_std.new().zero_(),
            ).sample()
        )
        z.requires_grad_()
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

class FlattenStochasticMlp(StochasticMlp):
    def forward(self, *x):
        flat = torch.cat(x, dim=1),
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



class EpicSAC2(nn.Module, EPICModel):
    def __init__(self, m: int):
        self._m = m
        self.mc_actors = 

    @property
    def m(self):
        return self._m

    def act_m(self, m, state) -> Action:
        

    