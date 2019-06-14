import torch.nn as nn


class MessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """
    def __init__(self, n: int, n_m: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = 2 * self.__n

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n_e),
            nn.ReLU(),
            nn.Linear(self.__n_e, self.__n_m)
        )

    def forward(self, h_i_t):
        return self.__seq_lin(h_i_t)


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
            nn.ReLU()
        )

    def forward(self, m_i_t):
        return self.__seq_lin(m_i_t)
