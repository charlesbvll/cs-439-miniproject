from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch

import src.client as client
import src.model.optimizer as optimizer
import src.utils as utils


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
        balance=params.BALANCE,
        optimizer=optimizer.get(params.OPTIMIZER),
        learning_rate=params.LR,
        stagglers=params.STRAGGLERS_FRACTION,
        tqdm_disable=params.TQDM_DISABLE,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE, params.TQDM_DISABLE)

    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=int(params.NUM_CLIENTS * (1 - params.STRAGGLERS_FRACTION)),
        min_evaluate_clients=0,
        min_available_clients=params.NUM_CLIENTS,
        on_fit_config_fn=lambda curr_round: {"curr_round": curr_round},
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=utils.weighted_average,
        proximal_mu=params.PROXIMAL_MU,
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
        f"{'_balanced' if params.BALANCE else ''}"
        f"_C={params.NUM_CLIENTS}"
        f"_B={params.BATCH_SIZE}"
        f"_E={params.NUM_EPOCHS}"
        f"_R={params.NUM_ROUNDS}"
        f"_mu={params.PROXIMAL_MU}"
        f"_stag={params.STRAGGLERS_FRACTION}"
    )

    np.save(
        Path(params.SAVE_PATH) / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        Path(params.SAVE_PATH),
        (file_suffix),
    )


if __name__ == "__main__":
    main()
