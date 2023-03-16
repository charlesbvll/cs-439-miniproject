import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


def get_loaders(batch_size, val_ratio, seed):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./.dataset", train=True, download=True, transform=transform)
    testset = MNIST("./.dataset", train=False, download=True, transform=transform)

    len_val = int(len(trainset) / (1 / val_ratio))
    lengths = [len(trainset) - len_val, len_val]
    ds_train, ds_val = random_split(
        trainset, lengths, torch.Generator().manual_seed(seed)
    )

    return (
        DataLoader(ds_train, batch_size=batch_size),
        DataLoader(ds_val, batch_size=batch_size),
        DataLoader(testset, batch_size=batch_size),
    )
