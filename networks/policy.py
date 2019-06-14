import torch.nn as nn
from torch import cat, stack


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """
    def __init__(self, action_size, n: int):
        super().__init__()

        self.__nb_a = action_size
        self.__n = n

        self.seq_lin = nn.Sequential(
            nn.Linear(action_size + n, (action_size + n) * 2),
            nn.ReLU(),
            nn.Linear((action_size + n) * 2, 1)
        )

    def forward(self, a, h_caret_t_p_one):
        assert len(a.size()) == 2, "Action must be a 1-D tensor"
        assert len(h_caret_t_p_one.size()) == 1, "action unit lstm hidden state must be 1-D tensor"

        x = cat((a, stack([h_caret_t_p_one] * a.size(0), dim=0)), dim=1)

        return self.seq_lin(x)
