from torch import nn


############################
# State to features stuff
############################
class StateToFeatures(nn.Sequential):
    """
    λ_θ7 : R^d -> R^n
    """

    def __init__(self, d: int, n_d: int) -> None:
        super().__init__(
            nn.Linear(d, n_d),
            nn.LayerNorm(n_d),
            nn.SiLU(),
        )
