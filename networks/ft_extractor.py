import torch.nn as nn
import torch as th


############################
# Features extraction stuff
############################

# MNIST Stuff
class TestCNN(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()

        f = 28
        self.__n = n

        self.seq_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(16 * (((f - 2) - 2) ** 2), self.__n),
            nn.Softmax(dim=-1)
        )

    def forward(self, o_t):
        out = self.seq_conv(o_t.unsqueeze(1))
        out = out.flatten(1, -1)
        return self.seq_lin(out)


class MNISTCnn(nn.Module):
    """
    b_θ5 : R^f*f -> R^n
    """
    def __init__(self, f: int, n: int) -> None:
        super().__init__()

        self.__f = f
        self.__n = n

        self.seq_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(16 * (f ** 2), self.__n)
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


# RESISC-45 Stuff

class RESISC45Cnn(nn.Module):
    def __init__(self, f: int = 128, n: int = 1024) -> None:
        super().__init__()

        self.seq_conv = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.lin = nn.Linear(32 * (f // 2 // 2 // 2 // 2) ** 2, n)

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = self.seq_conv(o_t)
        out = out.flatten(1, -1)
        out = self.lin(out)
        return out

############################
# State to features stuff
############################
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
