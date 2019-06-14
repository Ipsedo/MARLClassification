import torch.nn as nn
from torch import cat


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """
    def __init__(self, action_size, n: int):
        super().__init__()

        self.__nb_a = action_size
        self.__n = n

        self.__seq_lin = nn.Sequential(
            nn.Linear(action_size + n, (action_size + n) * 2),
            nn.ReLU(),
            nn.Linear((action_size + n) * 2, 1)
        )

    def forward(self, a, h_caret_i_t_p_one):
        assert len(a.size()) == 1, "Action must be a 1-D tensor"
        assert len(h_caret_i_t_p_one.size()) == 1, "action unit lstm hidden state must be 1-D tensor"

        x = cat((a, h_caret_i_t_p_one), dim=0)

        return self.__seq_lin(x)
