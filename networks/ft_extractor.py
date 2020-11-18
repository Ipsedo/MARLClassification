import torch.nn as nn
import torch as th

from abc import ABC, abstractmethod


class CNNFtExtract(nn.Module, ABC):

    @property
    @abstractmethod
    def out_size(self) -> int:
        return -1


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


class MNISTCnn(CNNFtExtract):
    """
    b_θ5 : R^f*f -> R^n
    """

    def __init__(self, f: int) -> None:
        super().__init__()

        self.__f = f

        self.seq_conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3,
                      padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(3, 6, kernel_size=3,
                      padding=1, padding_mode='zeros'),
            nn.ReLU()
        )

        self.__out_size = 6 * f ** 2

    def forward(self, o_t):
        o_t = o_t[:, 0, None, :, :]  # grey scale
        out = self.seq_conv(o_t)
        out = out.flatten(1, -1)
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


# RESISC-45 Stuff

class RESISC45Cnn(CNNFtExtract):

    def __init__(self, f: int = 10) -> None:
        super().__init__()

        self.seq_conv = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.__out_size = 16 * f ** 2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = self.seq_conv(o_t)
        out = out.flatten(1, -1)
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


class TestRESISC45(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_conv = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU()
        )

        out_size = 32 * (16 // 2) ** 2

        self.seq_lin = nn.Sequential(
            nn.Linear(out_size, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 45),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: th.Tensor):
        out = th.cat(th.cat(x.split(16, dim=-1)).split(16, dim=-2))
        out = self.seq_conv(out)
        out = out.flatten(1, -1)
        out = self.seq_lin(out)
        return out.view(-1, 45)


# Knee MRI stuff

class KneeMRICnn(CNNFtExtract):
    def __init__(self, f: int = 10):
        super().__init__()

        self.seq_conv = nn.Sequential(
            nn.Conv3d(1, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )

        self.__out_size = 16 * (f // 2) ** 3

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = self.seq_conv(o_t)
        out = out.flatten(1, -1)
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


############################
# State to features stuff
############################
class StateToFeatures(nn.Module):
    """
    λ_θ7 : R^d -> R^n
    """

    def __init__(self, d: int, n_d: int) -> None:
        super().__init__()

        self.__d = d
        self.__n_d = n_d

        self.seq_lin = nn.Sequential(
            nn.Linear(self.__d, self.__n_d),
            nn.CELU()
        )

    def forward(self, p_t):
        return self.seq_lin(p_t)
