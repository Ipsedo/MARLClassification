import torch.nn as nn


class Policy(nn.Module):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """

    def __init__(self, nb_action, n: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.seq_lin = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1)
        )

        for m in self.seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_caret_t_next):
        return self.seq_lin(h_caret_t_next)
