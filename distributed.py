from collections import OrderedDict
from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch

import src.client as client
import src.utils as utils
from src.model.MNIST_CNN import Net
from src.strategy import get_strategy


@hydra.main(config_path="docs/conf", config_name="base_distributed", version_base=None)
def main(params) -> None:
    DEVICE = torch.device(params.DEVICE)

    client_fn, testloader = client.gen_client_fn(
        num_epochs=params.NUM_EPOCHS,
        batch_size=params.BATCH_SIZE,
        device=params.DEVICE,
        num_clients=params.NUM_CLIENTS,
        num_rounds=params.NUM_ROUNDS,
        iid=params.IID,
        optim_name=params.client_optim_name,
        optim_args=params.client_optim_args,
        stragglers=params.STRAGGLERS_FRACTION,
        tqdm_disable=params.TQDM_DISABLE,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE, params.TQDM_DISABLE)

    strategy_class = get_strategy(params.STRATEGY)

    net = Net()
    initial_params = utils.get_initial_params(net)

    strategy = strategy_class(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=int(params.NUM_CLIENTS * (1 - params.STRAGGLERS_FRACTION)),
        min_evaluate_clients=0,
        min_available_clients=params.NUM_CLIENTS,
        initial_parameters=initial_params,
        on_fit_config_fn=lambda curr_round: {"curr_round": curr_round},
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        proximal_mu=params.PROXIMAL_MU,
        optim=params.server_optim_name,
        **params.server_optim_args,
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=params.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=params.NUM_ROUNDS),
        strategy=strategy,
    )

    file_suffix: str = (
        f"{'_iid' if params.IID else ''}"
        f"_C={params.NUM_CLIENTS}"
        f"_B={params.BATCH_SIZE}"
        f"_E={params.NUM_EPOCHS}"
        f"_R={params.NUM_ROUNDS}"
        f"_mu={params.PROXIMAL_MU}"
        f"_strag={params.STRAGGLERS_FRACTION}"
        f"_CO={params.client_optim_name}"
        f"_SO={params.server_optim_name}"
        f"_strat={params.STRATEGY}"
    )

    # Save run history
    np.save(
        Path(params.SAVE_PATH) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_dict(
        history.metrics_centralized,
        Path(params.SAVE_PATH),
        f"_centralized{file_suffix}",
        "accuracy",
    )

    # Save the model
    params_dict = zip(net.state_dict().keys(), strategy.current_weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    torch.save(state_dict, Path(params.SAVE_PATH) / Path(f"model{file_suffix}"))
    net.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    main()
