from typing import Tuple

import torch as th
import torch.nn as nn


class LSTMCellWrapper(nn.Module):
    # f_θ1 : R^2n * R^3n -> R^2n
    #
    # f_θ2 : ?
    # Supposition : R^2n * R^3n -> R^2n
    # R^2n : pas sûr

    def __init__(self, input_size: int, n: int) -> None:
        super().__init__()

        self.__lstm = nn.LSTMCell(input_size, n)

    def forward(self, h: th.Tensor, c: th.Tensor, u: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        TODO

        :param h:
        :type h:
        :param c:
        :type c:
        :param u:
        :type u:
        :return:
        :rtype:
        """

        nb_ag, batch_size, hidden_size = h.size()

        h, c, u = (
            h.flatten(0, 1),
            c.flatten(0, 1),
            u.flatten(0, 1)
        )

        h_next, c_next = self.__lstm(u, (h, c))

        return (
            h_next.view(nb_ag, batch_size, -1),
            c_next.view(nb_ag, batch_size, -1)
        )
