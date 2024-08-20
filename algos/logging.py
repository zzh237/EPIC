import inspect
from functools import wraps

import wandb


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
