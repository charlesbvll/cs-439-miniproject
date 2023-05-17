from typing import Dict, Iterable, Tuple, Union

import torch

Optimizer = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]


def get(
    optim_name: str,
    params: Iterable,
    args: Dict[str, Union[float, Tuple[float, float]]],
) -> Optimizer:
    if optim_name == "sgd":
        return torch.optim.Adam(
            params=params, **args
        )
    if optim_name == "rmsprop":
        return torch.optim.RMSprop(
            params=params, **args
        )

    return torch.optim.SGD(params=params, **args)
