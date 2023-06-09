"""CNN model architecutre, training, and testing functions for MNIST."""


from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optimizer import get_optim

def train(
    net: nn.Module,
    trainloader: DataLoader,
    optim_name: str,
    optim_args: Dict[str, Any],
    device: torch.device,
    epochs: int,
    proximal_mu: float,
    tqdm_disable: bool,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    """
    criterion = torch.nn.CrossEntropyLoss()
    global_params = [val.detach().clone() for val in net.parameters()]
    optimizer = get_optim(optim_name, net.parameters(), optim_args)
    net.train()
    for epoch in tqdm(range(epochs), disable=tqdm_disable):
        running_loss = 0.0
        for batch_number, (images, labels) in tqdm(enumerate(trainloader), disable=tqdm_disable, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_number % 1000 == 999:    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {batch_number + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

def test(
    net: nn.Module, testloader: DataLoader, device: torch.device, tqdm_disable: bool
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader, disable=tqdm_disable):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total
    accuracy = float(correct / total)
    return loss, accuracy
