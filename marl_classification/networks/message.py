from typing import cast

import torch as th
from torch import nn
from torchvision.ops import Permute


def aggregate_messages(messages: th.Tensor) -> th.Tensor:
    """
    Communication protocol between agents: each agent receives the mean
    of the messages sent by all the *other* agents.

    messages: [Na, Nb, n_m] -> [Na, Nb, n_m]
    """
    nb_agent = messages.size(0)

    if nb_agent == 1:
        return th.zeros_like(messages)

    return (messages.sum(dim=0) - messages) / (nb_agent - 1)


class MessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int, hidden_size: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = hidden_size

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n_e),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.__n_e),
            Permute([2, 0, 1]),
            nn.Linear(self.__n_e, self.__n_m),
        )

    def forward(self, h_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_lin(h_t))


class MessageReceiver(nn.Module):
    """
    d_θ6 : R^n_m -> R^n
    """

    def __init__(self, n_m: int, n: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n_m, self.__n),
            nn.GELU(),
        )

    def forward(self, m_t: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.__seq_lin(m_t))
