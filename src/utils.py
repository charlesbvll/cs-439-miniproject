"""Contains utility functions for CNN FL on MNIST."""


from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import Metrics, NDArrays, Parameters, Scalar, ndarrays_to_parameters
from torch.utils.data import DataLoader

from src.model.common import test
from src.model.MNIST_CNN import Net


def plot_metric_from_dict(
    dict: Dict[str, List[Tuple[int, float]]],
    save_plot_path: Path,
    suffix: str = "",
    metric: str = "accuracy",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    rounds, values = zip(*dict[metric])
    plt.plot(np.asarray(rounds), np.asarray(values), label="FedProx")
    plt.title(f"Validation {metric.capitalize()} - MNIST")
    plt.xlabel("Rounds")
    plt.ylabel(f"{metric.capitalize()}")
    plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric}{suffix}.png"))
    plt.close()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


def gen_evaluate_fn(
    testloader: DataLoader, device: torch.device, tqdm_disable: bool
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""
        # determine device
        net = Net()
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device, tqdm_disable=tqdm_disable)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def get_initial_params(net: torch.nn.Module) -> Parameters:
    return ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in net.state_dict().items()]
    )
