import torch

DEVICE = torch.device("cuda:0")
NUM_ROUNDS = 1
NUM_EPOCHS = 5
LR = 0.001
BATCH_SIZE = 32
VAL_RATIO = 0.2
SEED = 42
PROXIMAL_MU = 0.0
TQDM_DISABLE = False
