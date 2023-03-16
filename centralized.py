from data import get_loaders
from model import Net, train, test
import parameters


def main():
    train_loader, val_loader, test_loader = get_loaders(
        parameters.BATCH_SIZE, parameters.VAL_RATIO, parameters.SEED
    )

    net = Net().to(parameters.DEVICE)

    for round in range(parameters.NUM_ROUNDS):
        print(f"Strating round {round}...")
        train(
            net,
            train_loader,
            parameters.DEVICE,
            parameters.NUM_EPOCHS,
            parameters.LR,
            parameters.PROXIMAL_MU,
        )
        print(f"Validation results: {test(net, val_loader, parameters.DEVICE)}")

    print(f"Test results: {test(net, test_loader, parameters.DEVICE)}")


if __name__ == "__main__":
    main()
