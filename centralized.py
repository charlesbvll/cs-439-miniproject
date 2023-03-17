import hydra
import torch

import src.model.optimizer as optimizer
from src.dataset.MNIST import centralized_loaders
from src.model.common import test, train
from src.model.MNIST_CNN import Net


@hydra.main(config_path="docs/conf", config_name="base_centralized", version_base=None)
def main(params):
    DEVICE = torch.device(params.DEVICE)

    train_loader, val_loader, test_loader = centralized_loaders(
        params.BATCH_SIZE, params.VAL_RATIO, params.SEED
    )

    net = Net().to(DEVICE)

    for round in range(params.NUM_ROUNDS):
        print(f"Strating round {round}...")
        train(
            net,
            train_loader,
            optimizer.get(params.OPTIMIZER),
            DEVICE,
            params.NUM_EPOCHS,
            params.LR,
            params.PROXIMAL_MU,
            params.TQDM_DISABLE,
        )
        print(
            f"Validation results: {test(net, val_loader, DEVICE, params.TQDM_DISABLE)}"
        )

    print(f"Test results: {test(net, test_loader, DEVICE, params.TQDM_DISABLE)}")


if __name__ == "__main__":
    main()
