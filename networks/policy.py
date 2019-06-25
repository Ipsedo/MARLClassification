import torch.nn as nn
from torch import cat, stack


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """
    def __init__(self, nb_action, n: int):
        super().__init__()

        self.seq_lin = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, nb_action),
            nn.Softmax(dim=-1)
        )

    def forward(self, h_caret_t_next):
        if len(h_caret_t_next.size()) == 1:
            h_caret_t_next = h_caret_t_next.unsqueeze(0)

        assert len(h_caret_t_next.size()) == 2,\
            "action unit lstm hidden state must be 2-D tensor (curr = {})".format(len(h_caret_t_next.size()))

        return self.seq_lin(h_caret_t_next)
