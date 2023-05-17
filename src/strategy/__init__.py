from .fedavg import FedAvgProxOpt
from .fedkrum import FedKrumProxOpt
from .fedmedian import FedMedProxOpt


def get_strategy(strategy_name: str = "fedavg"):
    if strategy_name == "fedavg":
        return FedAvgProxOpt
    elif strategy_name == "fedmedian":
        return FedMedProxOpt
    else:
        return FedKrumProxOpt
