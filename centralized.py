from pathlib import Path

import hydra
import numpy as np
import torch

from src.dataset.MNIST import centralized_loaders
from src.model.common import test, train
from src.model.MNIST_CNN import Net
from src.utils import plot_metric_from_dict


@hydra.main(config_path="docs/conf", config_name="base_centralized", version_base=None)
def main(params):
    DEVICE = torch.device(params.DEVICE)

    train_loader, val_loader, test_loader = centralized_loaders(
        params.BATCH_SIZE, params.VAL_RATIO, params.SEED
    )

    net = Net().to(DEVICE)

    results = {"accuracy": [], "loss": []}

    for round in range(params.NUM_ROUNDS):
        print(f"Strating round {round}...")
        train(
            net,
            train_loader,
            params.optim_name,
            params.optim_args,
            DEVICE,
            params.NUM_EPOCHS,
            params.PROXIMAL_MU,
            params.TQDM_DISABLE,
        )
        loss, accuracy = test(net, val_loader, DEVICE, params.TQDM_DISABLE)
        print(f"Validation results: loss: {loss:.3f}, accuracy: {accuracy:.3f}")
        results["accuracy"].append((round, accuracy))
        results["loss"].append((round, loss))

    file_suffix: str = (
        f"_B={params.BATCH_SIZE}"
        f"_E={params.NUM_EPOCHS}"
        f"_R={params.NUM_ROUNDS}"
        f"_O={params.optim_name}"
    )

    np.save(Path(params.SAVE_PATH) / Path(f"hist{file_suffix}"), results)
    plot_metric_from_dict(results, Path(params.SAVE_PATH), file_suffix, "accuracy")

    loss, accuracy = test(net, test_loader, DEVICE, params.TQDM_DISABLE)
    print(f"Test results: loss: {loss:.3f}, accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
