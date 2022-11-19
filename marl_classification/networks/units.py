import torch as th
import torch.nn as nn


class Unit(nn.Module):
    """
    Action & Belief units
    """

    def __init__(self, input_size: int, n: int) -> None:
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(input_size, n),
            nn.ReLU(),
        )

    def forward(self, u: th.Tensor) -> th.Tensor:
        return self.__seq_lin(u)
