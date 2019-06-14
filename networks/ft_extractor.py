import torch.nn as nn


class CNN_MNIST(nn.Module):
    """
    R^f*f -> R^n
    """
    def __init__(self, f: int, n: int) -> None:
        super().__init__()

        self.__f = f
        self.__n = n

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.Conv2d(8, 16, kernel_size=3)
        )

        self.__seq_lin = nn.Sequential(
            nn.Linear(16 * (((f - 2) - 2) ** 2), self.__n)
        )

    def forward(self, o_i_t):
        out = self.__seq_conv(o_i_t.unsqueeze(1))
        out = out.flatten(1, -1)
        return self.__seq_lin(out)
