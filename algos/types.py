from abc import ABC, abstractmethod
from typing import TypedDict

from torch import Tensor


class Action(TypedDict):
    state: Tensor
    action: Tensor
    log_prob: Tensor


class EPICModel(ABC):

    @property
    @abstractmethod
    def m(self) -> int: ...

    @abstractmethod
    def act_m(self, m: int, state) -> Action:
        """
        Take an action using the Mth mc worker.
        """

    @abstractmethod
    def per_step_m(self, m: int, state, action, reward, new_state, done, meta_episode,
                   step):
        """
        Do something per environment step.
        """
        ...

    @abstractmethod
    def post_episode(self) -> None:
        """
        Do something after completion of an episode.
        """
        ...
    @abstractmethod
    def update_prior(self) -> None:
        """
        Update the parameter priors.
        """
        ...

    @abstractmethod
    def update_default(self) -> None:
        """
        Update the default model by drawing from the prior.
        """
        ...