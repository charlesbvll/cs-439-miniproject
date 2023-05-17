import torch

def get(optim_name: str) -> torch.optim:
    if optim_name == "sgd":
        return torch.optim.SGD
    elif optim_name == "adam":
        return torch.optim.Adam
    else:
        return torch.optim.RMSprop
