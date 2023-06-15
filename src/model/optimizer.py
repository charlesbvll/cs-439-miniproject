from typing import Dict, Iterable, Tuple, Union

import torch

from .yogi import Yogi

Optimizer = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop, Yogi]


def get_optim(
    optim_name: str,
    params: Iterable,
    args: Dict[str, Union[float, Tuple[float, float]]],
) -> Optimizer:
    """Return a given optimizers with the given arguments."""
    if optim_name == "adam":
        return torch.optim.Adam(
            params=params, **args
        )
    if optim_name == "rmsprop":
        return torch.optim.RMSprop(
            params=params, **args
        )

    return torch.optim.SGD(params=params, **args)
