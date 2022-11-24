import torch as th
import torch.nn as nn


class Prediction(nn.Module):
    """
    q_θ8 : R^n -> R^M
    """

    def __init__(self, n: int, nb_class: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.__n = n
        self.__nb_class = nb_class

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.__nb_class)
        )

    def forward(self, c_t: th.Tensor) -> th.Tensor:
        return self.__seq_lin(c_t)
