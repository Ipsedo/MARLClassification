import torch as th
import torch.nn as nn


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """

    def __init__(self, nb_action, n: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        return self.__seq_lin(h_caret_t_next)


class Critic(nn.Module):
    def __init__(self, n: int, hidden_size: int):
        super(Critic, self).__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, h_caret_t_next: th.Tensor) -> th.Tensor:
        return self.__seq_lin(h_caret_t_next)

