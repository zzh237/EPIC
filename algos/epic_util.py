"""
Various utils
""" 
from typing import Protocol, Callable, Any
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
    def kl_divergence(self, other: Self) -> Tensor:
        raise NotImplementedError

 
def kl_regularizer(kl, N, H, 
                #    c=torch.tensor(1.5), 
                c = 1.5,
                delta=0.01,
                #    delta=torch.tensor(0.01)
                   ):
    epsilon = log(2.) / (2. * log(c)) * (1. + torch.log(kl / log(2. / delta)))
    reg = (1. + c) / 2. * sqrt(2.) * torch.sqrt((kl + log(2. / delta) + epsilon) * N * H ** 2.)
    return reg



class PriorWrapper:
    def __init__(self, module_factory: Callable[[] , ModuleWithKlDivergence],
                 prior_update_every: int,
                 gamma: float,
                 max_steps: int,
                 c: float,
                 delta,
                 ):
        super().__init__()
        self.gamma = gamma
        self.max_steps = max_steps
        self.default = module_factory()
        self.prior = module_factory()
        self.prior_update_every = prior_update_every

    def call_default(self, *args, **kwargs):
        return self.default.forward(*args, **kwargs)

    # def forward(self, *args, **kwargs):
        # return self.default.forward(*args, **kwargs)

    def get_epic_regularizer(self):
        kl_div = self.default.kl_divergence(self.prior)
        H = 1. * (1 - self.gamma ** self.max_steps) / (1. - self.gamma)
        regularizer = kl_regularizer(kl_div, self.prior_update_every, H)

        return regularizer

        

