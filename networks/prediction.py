import torch.nn as nn


class Prediction(nn.Module):
    """
    q_θ8 : R^n -> R^M
    """

    def __init__(self, n: int, nb_class: int,
                 hidden_size: int) -> None:
        super().__init__()

        self.__n = n
        self.__nb_class = nb_class

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__n, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.__nb_class)
        )

        for m in self.seq_lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, c_t):
        return self.seq_lin(c_t)
