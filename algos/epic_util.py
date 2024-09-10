"""
Various utils
"""

from typing import Generic, Protocol, Callable, Any, TypeVar
from torch import nn, Tensor
from typing_extensions import Self
from abc import ABC, abstractmethod
import torch
from math import log, sqrt
from typing import Iterator
from torch.nn import Parameter

# class HasKlDivergence(Protocol):
#     def kl_divergence(self, other: Self) -> Tensor: ...

# class ModuleLike(Protocol):
#     def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...
#     def forward(self, *args, **kwargs) -> Any: ...

# class ModuleWithKlDivergence(ModuleLike, HasKlDivergence, Protocol): ...


class ModuleWithKlDivergence(nn.Module, ABC):
    @abstractmethod
    def kl_divergence(self, other: Self) -> Tensor | dict[str, Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def get_epic_regularizers(self, kl_divergence: Tensor | dict[str, Tensor],
                                prior_update_every: int, gamma: float,
                                max_steps: int, c: float, delta: float) -> Tensor | dict[str, Tensor]:
        """Get the regularizer(s) for all children of this module."""
        raise NotImplementedError


def kl_regularizer(
    kl,
    N,
    H,
    c=1.5,
    delta=0.01,
):
    epsilon = log(2.0) / (2.0 * log(c)) * (1.0 + torch.log(kl / log(2.0 / delta)))
    reg = (1.0 + c) / 2.0 * sqrt(2.0) * torch.sqrt((kl + log(2.0 / delta) + epsilon) * N * H**2.0)
    return reg

T = TypeVar("T", bound=nn.Module)

# class PriorWrapper[T: nn.Module]:
class PriorWrapper(Generic[T]):
    def __init__(
        self,
        module_factory: Callable[[], T],
        prior_update_every: int,
        gamma: float,
        max_steps: int,
        c: float,
        delta: float,
        tau: float
    ):
        super().__init__()
        self.c = c
        self.delta = delta
        self.gamma = gamma
        self.max_steps = max_steps
        self.tau = tau
        self.default = module_factory()
        self.prior = module_factory()
        self.prior_update_every = prior_update_every

    def call_default(self, *args, **kwargs):
        return self.default.forward(*args, **kwargs)
    
    def update_prior(self):
        # update the prior inplace via soft update from the default
        with torch.no_grad():
            soft_update_from_to(self.default, self.prior, tau=self.tau)
    
    def sample_default(self):
        # update the default network with parameters from the prior. since the prior and posterior
        # are both bayesian networks, sampling is done implicitly in the forward pass and 
        # here we just have to copy the parameters over.
        with torch.no_grad():
            soft_update_from_to(self.prior, self.default, tau=1.0)


def soft_update_from_to(source: nn.Module, target: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
