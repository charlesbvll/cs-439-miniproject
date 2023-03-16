import torch

TQDM_DISABLE = True 

DEVICE = torch.device("cpu")

# Experiment variables
NUM_ROUNDS = 1
NUM_EPOCHS = 5

# Data parameters
BATCH_SIZE = 32
VAL_RATIO = 0.2
SEED = 42

# Model hyperparameters
LR = 0.001
PROXIMAL_MU = 0.0

# Distributed setting
NUM_CLIENTS = 10
IID = True
BALANCE = True
STAGGLERS_FRACTION = 0.0

SAVE_PATH = "./docs/"
