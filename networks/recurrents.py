import torch.nn as nn


class BeliefUnit(nn.Module):
    """
    f_θ1 : R^2n * R^3n -> R^2n
    """
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

        self.lstm = nn.LSTM(self.__n * 3, self.__n, batch_first=True)

    def forward(self, h_t, c_t, u_t):
        assert u_t.size(1) == 1, "Only one time iteration is allowed"

        h_t, c_t = h_t[:, :u_t.size(0), :], c_t[:, :u_t.size(0), :]

        _, (h_t_next, c_t_next) = self.lstm(u_t, (h_t, c_t))

        return h_t_next, c_t_next


class ActionUnit(nn.Module):
    """
    f_θ2 : ?
    Supposition : R^2n * R^3n -> R^2n
    R^2n : pas sûr
    """
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

        # TODO find hidden state size in article
        self.lstm = nn.LSTM(self.__n * 3, self.__n, batch_first=True)

    def forward(self, h_caret_t, c_caret_t, u_t):
        assert u_t.size(1) == 1, "Only one time iteration is allowed"

        h_caret_t, c_caret_t = h_caret_t[:, :u_t.size(0), :], c_caret_t[:, :u_t.size(0), :]

        _, (h_caret_t_next, c_caret_t_next) = self.lstm(u_t, (h_caret_t, c_caret_t))

        return h_caret_t_next, c_caret_t_next
