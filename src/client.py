from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from src.dataset.MNIST import distributed_loaders
from src.model.common import test, train
from src.model.MNIST_CNN import Net


class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        optim_name: str,
        optim_args: Dict[str, Any],
        device: torch.device,
        num_epochs: int,
        staggler_schedule: np.ndarray,
        tqdm_disable: bool,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.optim_name = optim_name
        self.optim_args = optim_args
        self.device = device
        self.num_epochs = num_epochs
        self.staggler_schedule = staggler_schedule
        self.tqdm_disable = tqdm_disable

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)

        # At each round check if the client is a staggler,
        # if so, train less epochs (to simulate partial work)
        if (
            self.staggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
        else:
            num_epochs = self.num_epochs

        train(
            self.net,
            self.trainloader,
            self.optim_name,
            self.optim_args,
            self.device,
            epochs=num_epochs,
            proximal_mu=config["proximal_mu"],
            tqdm_disable=self.tqdm_disable,
        )

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device, self.tqdm_disable)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    device: torch.device,
    iid: bool,
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    optim_name: str,
    optim_args: Dict[str, Union[float, Tuple[float, float]]],
    stragglers: float,
    tqdm_disable: bool,
) -> Tuple[
    Callable[[str], FlowerClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    num_clients : int
        The number of clients present in the setup
    num_rounds : int
        The number of communication rounds the server will perform.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    optim_name : str
        The name of the client optimizer to use (either 'sgd', 'adam', or 'rmsprop').
    optim_args : Dict[str, Union[float, Tuple[float, float]]]
        A dicitonnary containing arguments for the optimizer (can be empty).
    stragglers : float
        The fraction of machines that should be considered as stragglers.
    tqdm_disable : bool
        Used to disable the tqdm progress bar while training.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    trainloaders, valloaders, testloader = distributed_loaders(
        iid=iid, num_clients=num_clients, batch_size=batch_size
    )

    # Defines a straggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of straggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = Net().to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net,
            trainloader,
            valloader,
            optim_name,
            optim_args,
            device,
            num_epochs,
            stragglers_mat[int(cid)],
            tqdm_disable,
        )

    return client_fn, testloader
