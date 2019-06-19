import torch.nn as nn


class CNN_MNIST(nn.Module):
    """
    b_θ5 : R^f*f -> R^n
    """
    def __init__(self, f: int, n: int) -> None:
        super().__init__()

        self.__f = f
        self.__n = n

        self.seq_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3)
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(16 * (((f - 2) - 2) ** 2), self.__n)
        )

    def forward(self, o_t):
        out = self.seq_conv(o_t.unsqueeze(1))
        out = out.flatten(1, -1)
        return self.seq_lin(out)


class CNN_MNIST_2(nn.Module):
    """
    b_θ5 : R^f*f -> R^n
    """
    def __init__(self, f: int, n: int) -> None:
        super().__init__()

        self.__f = f
        self.__n = n

        self.seq_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(8 * ((f - 2) ** 2), self.__n)
        )

    def forward(self, o_t):
        out = self.seq_conv(o_t.unsqueeze(1))
        out = out.flatten(1, -1)
        return self.seq_lin(out)


class StateToFeatures(nn.Module):
    """
    λ_θ7 : R^d -> R^n
    """
    def __init__(self, d: int, n: int) -> None:
        super().__init__()

        self.__d = d
        self.__n = n

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__d, self.__n)
        )

    def forward(self, p_t):
        return self.seq_lin(p_t)
