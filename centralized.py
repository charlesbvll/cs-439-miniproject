from src.dataset.MNIST import get_loaders
from src.model.MNIST_CNN import Net
from src.model.common import train, test
import conf.parameters as params


def main():
    train_loader, val_loader, test_loader = get_loaders(
        params.BATCH_SIZE, params.VAL_RATIO, params.SEED
    )

    net = Net().to(params.DEVICE)

    for round in range(params.NUM_ROUNDS):
        print(f"Strating round {round}...")
        train(
            net,
            train_loader,
            params.DEVICE,
            params.NUM_EPOCHS,
            params.LR,
            params.PROXIMAL_MU,
        )
        print(f"Validation results: {test(net, val_loader, params.DEVICE)}")

    print(f"Test results: {test(net, test_loader, params.DEVICE)}")


if __name__ == "__main__":
    main()
