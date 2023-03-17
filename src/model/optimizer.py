import torch

def get(optim_name: str) -> torch.optim:
    if optim_name == "sgd":
        return sgd()

def sgd():
    return torch.optim.SGD
