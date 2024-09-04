import copy
import inspect
from functools import wraps
from typing import Collection

import wandb


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def doublewrap(f):
    '''
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    '''
    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec

@doublewrap
# def track_config(init, *, ignore: Collection[str] | None = None):
def track_config(*args, **kwargs):
    """Log all arguments to init as config objects in wandb."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        ignore = []
    else:
        ignore = kwargs.get("ignore", [])

    init = args[0]

    @wraps(init)
    def wrapper(self, **kwargs):
        default_args = get_default_args(init)
        params = copy.deepcopy(kwargs)
        params.update(default_args)
        for k in ignore:
            if k in params:
                params.pop(k)

        wandb.config.update(params)
        return init(self, **kwargs)

    return wrapper
