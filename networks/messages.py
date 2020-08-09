import torch.nn as nn


class MessageSender(nn.Module):
    """
    m_θ4 : R^n -> R^n_m
    """

    def __init__(self, n: int, n_m: int,
                 hidden_size: int) -> None:
        super().__init__()
        self.__n = n
        self.__n_m = n_m
        self.__n_e = hidden_size

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n_e),
            nn.ReLU(),
            nn.Linear(self.__n_e, self.__n_m)
        )

        for m in self.seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

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

        for m in self.seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, m_t):
        return self.seq_lin(m_t)
