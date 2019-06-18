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

    def forward(self, a, h_caret_t_next):
        assert len(a.size()) == 2, "Action must be a 2-D tensor"
        assert len(h_caret_t_next.size()) == 3, "action unit lstm hidden state must be 3-D tensor"

        a = stack([a] * h_caret_t_next.size(1), dim=1)
        b = cat([h_caret_t_next.squeeze(1)] * a.size(0), dim=0)

        x = cat((a, b), dim=2).permute(1, 0, 2)

        return self.seq_lin(x)
