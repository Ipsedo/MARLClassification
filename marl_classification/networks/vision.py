from abc import ABC, abstractmethod
from typing import cast

import torch as th
from torch import nn


class VisionCnnModule(nn.Module, ABC):
    @property
    @abstractmethod
    def out_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, o_t: th.Tensor) -> th.Tensor: ...


############################
# Features extraction stuff
############################


class _Generic2dCnnModule(VisionCnnModule):
    def __init__(
        self, f: int, layers: list[tuple[int, int]], group_norm_nums: list[int]
    ) -> None:
        super().__init__()

        self.__layers = nn.Sequential()

        for (c_i, c_o), g in zip(layers, group_norm_nums):
            self.__layers.append(
                nn.Conv2d(c_i, c_o, kernel_size=3, stride=2, padding=1)
            )
            self.__layers.append(nn.GroupNorm(g, c_o))
            self.__layers.append(nn.SiLU())

        self.__layers.append(nn.Flatten(1, -1))

        window_size = f
        for _ in range(len(layers)):
            window_size = (window_size - 3 + 2 * 1) // 2 + 1

        self.__out_size = layers[-1][1] * window_size**2

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        encoded_o_t: th.Tensor = self.__layers(o_t)
        return encoded_o_t

    @property
    def out_size(self) -> int:
        return self.__out_size


class MNISTCnn(_Generic2dCnnModule):
    """
    b_θ5 : R^f*f -> R^n
    """

    def __init__(self, f: int) -> None:
        super().__init__(f, [(1, 8), (8, 16)], [2, 4])

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        o_t = o_t[:, 0, None, :, :]  # grey scale
        return super().forward(o_t)


class Resisc45Cnn(_Generic2dCnnModule):
    def __init__(self, f: int) -> None:
        super().__init__(f, [(3, 16), (16, 32), (32, 64)], [2, 4, 8])


class AidCnn(_Generic2dCnnModule):
    def __init__(self, f: int) -> None:
        super().__init__(
            f, [(3, 16), (16, 32), (32, 64), (64, 128)], [2, 4, 8, 16]
        )


class WorldStratCnn(_Generic2dCnnModule):
    def __init__(self, f: int) -> None:
        super().__init__(
            f,
            [(3, 16), (16, 32), (32, 64), (64, 128), (128, 256)],
            [2, 4, 8, 16, 32],
        )


# Knee MRI stuff


class KneeMriCnn(VisionCnnModule):
    def __init__(self, f: int = 16):
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, (3, 3, 3), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(2, 2),
            nn.BatchNorm3d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 8) ** 3

    def forward(self, o_t: th.Tensor) -> th.Tensor:
        out = cast(th.Tensor, self.__seq_conv(o_t))
        return out

    @property
    def out_size(self) -> int:
        return self.__out_size


class SkinCancerCnn(_Generic2dCnnModule):
    # https://github.com/Ipsedo/MARLClassification/issues/4
    # https://drive.google.com/drive/folders/17g6zFSbCNXTV3VaDKop73W7Cn-NJlTO7?usp=sharing
    def __init__(self, f: int) -> None:
        super().__init__(f, [(3, 16), (16, 32), (32, 64)], [2, 4, 8])
