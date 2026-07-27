from torch import nn


class Policy(nn.Sequential):
    """
    π_θ3 : A * R^n
    R^n : pas sûr, voir reccurents.ActionUnit
    """

    def __init__(self, nb_action: int, n: int, hidden_size: int) -> None:
        super().__init__(
            nn.Linear(n, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1),
        )


class Critic(nn.Sequential):
    def __init__(self, n: int, hidden_size: int):
        super().__init__(
            nn.Linear(n, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(-2, -1),
        )
