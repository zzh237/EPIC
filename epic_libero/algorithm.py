"""
Perform EPIC updates on a Bayesian policy for lifelong learning.
"""
from libero.lifelong.algos.base import Sequential
from libero.lifelong.models.base_policy import BasePolicy
from typing import Protocol
from epic_libero.policy import *


class PolicyMaker(Protocol):
    def __call__(self, cfg, shape_meta) -> BasePolicy: ...

class EPICAlgorithm(Sequential):
    def __init__(self, n_tasks, cfg, policy_maker: PolicyMaker):
        super().__init__(n_tasks=n_tasks, cfg=cfg)
        self.policy = policy_maker(cfg, cfg.shape_meta)

class MyLifelongAlgo(Sequential):
    """
    The experience replay policy.
    """
    def __init__(self,
                 n_tasks,
                 cfg,
                 **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # define the learning policy
        self.datasets = []
        self.policy = eval(cfg.policy.policy_type)(cfg, cfg.shape_meta)

    def start_task(self, task):
        # what to do at the beginning of a new task
        super().start_task(task)

    def end_task(self, dataset, task_id, benchmark):
        # what to do when finish learning a new task
        self.datasets.append(dataset)

    def observe(self, data):
        # how the algorithm observes a data and returns a loss to be optimized
        loss = super().observe(data)
        return loss