import torch.nn as nn
from torch import cat, stack


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """
    def __init__(self, nb_action, n: int):
        super().__init__()

        self.__n = n

        self.seq_lin = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, nb_action),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_caret_t_next):
        # assert len(a.size()) == 2, "Action must be a 2-D tensor"
        assert len(h_caret_t_next.size()) == 3, "action unit lstm hidden state must be 3-D tensor"

        #a = stack([a] * h_caret_t_next.size(1), dim=1)
        #b = cat([h_caret_t_next.squeeze(1)] * a.size(0), dim=0)

        #x = cat((a, b), dim=2).permute(1, 0, 2)

        return self.seq_lin(h_caret_t_next).squeeze(0)
