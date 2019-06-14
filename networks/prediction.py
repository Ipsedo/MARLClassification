import torch.nn as nn


class Prediction(nn.Module):
    """
    q_θ8 : R^n -> R^M
    """
    def __init__(self, n: int, M: int) -> None:
        super().__init__()

        self.__n = n
        self.__nb_class = M

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__n, self.__n * 2),
            nn.ReLU(),
            nn.Linear(self.__n * 2, self.__nb_class)
        )

    def forward(self, c_t):
        return self.seq_lin(c_t)
