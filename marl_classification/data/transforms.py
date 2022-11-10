from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch as th


# All the transforms here are based on .dataset.ImgDataset

class ImgTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, img_data: th.Tensor) -> th.Tensor:
        raise NotImplementedError(
            self.__class__.__name__ +
            ".__call__ method is not implemented, must be overridden !"
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


#########################
# Normal normalization
#########################
class UserNormalNorm(ImgTransform):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super().__init__()

        self.__mean = th.tensor(mean)
        self.__std = th.tensor(std)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__mean) / self.__std

    def __repr__(self):
        return self.__class__.__name__ + \
               f"(mean = {str(self.__mean)}, std = {str(self.__std)})"


class ChannelNormalNorm(ImgTransform):

    def __call__(self, x: th.Tensor) -> th.Tensor:
        mean = x.view(3, -1).mean(dim=-1).view(3, 1, 1)
        std = x.view(3, -1).std(dim=-1).view(3, 1, 1)

        return (x - mean) / std


class NormalNorm(ImgTransform):

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x - th.mean(x)) / th.std(x)


#########################
# Uniform normalization
#########################
class UserMinMaxNorm(ImgTransform):
    def __init__(self, min_value: Tuple[float, float, float],
                 max_value: Tuple[float, float, float]):
        self.__min = th.tensor(min_value)
        self.__max = th.tensor(max_value)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__min) / (self.__max - self.__min)

    def __repr__(self):
        return self.__class__.__name__ + \
               f"(min_value = {self.__min}, max_value = {self.__max})"


class MinMaxNorm(ImgTransform):

    def __call__(self, x: th.Tensor) -> th.Tensor:
        x_max = x.max()
        x_min = x.min()
        return (x - x_min) / (x_max - x_min)


class ChannelMinMaxNorm(ImgTransform):
    def __call__(self, x: th.Tensor) -> th.Tensor:
        x_max = x.view(3, -1).max(dim=-1)[0].view(3, 1, 1)
        x_min = x.view(3, -1).min(dim=-1)[0].view(3, 1, 1)
        return (x - x_min) / (x_max - x_min)
