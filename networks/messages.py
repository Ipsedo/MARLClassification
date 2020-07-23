import torch as th
import torch.nn as nn
from torch import device

from typing import Optional, Union


class MessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """
    def __init__(self, n: int, n_m: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = 64

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n_e),
            nn.ReLU(),
            nn.Linear(self.__n_e, self.__n_m)
        )

    def forward(self, h_t):
        return self.seq_lin(h_t)


class MessageReceiver(nn.Module):
    """
    d_θ6 : R^n_m -> R^n
    """
    def __init__(self, n_m: int, n: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__n_m, self.__n),
            nn.ReLU()
        )

    def forward(self, m_t):
        return self.seq_lin(m_t)


###################
# Dummy networks
# for test
###################

class DummyMessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m

        self.register_buffer("w", th.rand(n, n_m, requires_grad=False))
        self.register_buffer("b", th.rand(1, n_m, requires_grad=False))

    def forward(self, h_t):
        return th.matmul(h_t, self.w) + self.b


class DummyMessageReceiver(nn.Module):
    """
    m_θ6 : R^n_m -> R^n
    """

    def __init__(self, n: int, n_m: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m

        self.register_buffer("w", th.rand(n_m, n, requires_grad=False))
        self.register_buffer("b", th.rand(1, n, requires_grad=False))

    def forward(self, h_t):
        return th.matmul(h_t, self.w) + self.b
