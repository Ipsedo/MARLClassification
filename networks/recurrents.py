import torch.nn as nn


class BeliefUnit(nn.Module):
    """
    f_θ1 : R^2n * R^3n -> R^2n
    """
    def __init__(self, n: int) -> None:
        super().__init__()

        self.__n = n

        self.__lstm = nn.LSTM(self.__n * 3, self.__n, batch_first=True)

    def forward(self, h_i_t, c_i_t, u_i_t):
        assert u_i_t.size(0) == 1, "Only one time iteration is allowed"

        _, (h_i_t_p_one, c_i_t_p_one) = self.__lstm(u_i_t, (h_i_t, c_i_t))

        return h_i_t_p_one, c_i_t_p_one


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
        self.__lstm = nn.LSTM(self.__n * 3, self.__n, batch_first=True)

    def forward(self, h_caret_i_t, c_caret_i_t, u_i_t):
        assert u_i_t.size(0) == 1, "Only one time iteration is allowed"

        _, (h_caret_i_t_p_one, c_caret_i_t_p_one) = self.__lstm(u_i_t, (h_caret_i_t, c_caret_i_t))

        return h_caret_i_t_p_one, c_caret_i_t_p_one
