import torch.nn as nn


class BeliefUnit(nn.Module):
    """
    R^2n * R^3n -> R^2n
    """
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

    def forward(self, u_i_t):
        pass


class ActionUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, u_i_t):
        pass
