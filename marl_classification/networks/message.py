import torch as th
from torch import nn


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


class MessageSender(nn.Sequential):
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int, hidden_size: int) -> None:
        super().__init__(
            nn.Linear(n, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_m),
            nn.LayerNorm(n_m),
            nn.SiLU(),
        )


class MessageReceiver(nn.Sequential):
    """
    d_θ6 : R^n_m -> R^n
    """

    def __init__(self, n_m: int, n: int, hidden_size: int) -> None:
        super().__init__(
            nn.Linear(n_m, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n),
            nn.LayerNorm(n),
            nn.SiLU(),
        )
