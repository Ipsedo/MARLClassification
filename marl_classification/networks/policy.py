from typing import cast

import torch as th
import torch.nn as nn
from torchvision.ops import Permute


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """

    def __init__(self, nb_action: int, n: int, hidden_size: int) -> None:
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1),
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_lin(h_caret_t_next))


class Critic(nn.Module):
    def __init__(self, n: int, hidden_size: int):
        super(Critic, self).__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, 1),
            nn.Flatten(-2, -1),
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_lin(h_caret_t_next))
