from torch import nn


class Prediction(nn.Sequential):
    """
    q_θ8 : R^n -> R^M
    """

    def __init__(self, n: int, nb_class: int, hidden_size: int) -> None:
        super().__init__(
            nn.Linear(n, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, nb_class),
        )
