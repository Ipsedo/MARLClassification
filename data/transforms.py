import torch as th

from typing import Tuple, Optional, Union, Callable

from abc import ABCMeta, abstractmethod


# All the transforms here are based on .dataset.ImgDataset

class ImgTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, img_data: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError(self.__class__.__name__ + ".__call__ method is not implemented, must be overridden !")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class NormalNorm(ImgTransform):
    def __init__(self, mean: Optional[Union[Tuple[float, float, float], float]] = None,
                 std: Optional[Union[Tuple[float, float, float], float]] = None):

        self.__user_defined = (std is not None) and (mean is not None)

        assert (std is not None) ^ (mean is not None), f"Hybrid mode not supported yet"

        self.__mean = th.tensor([mean] * 3) if isinstance(mean, float) else mean
        self.__std = th.tensor([std] * 3) if isinstance(std, float) else std

        self.__apply_fun: Callable[[th.Tensor], th.Tensor] = self.__apply_user if self.__user_defined else self.__apply_data

    def __apply_user(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__mean) / self.__std

    def __apply_data(self, x: th.Tensor) -> th.Tensor:
        mean = x.view(3, -1).mean(dim=0).view(3, 1, 1)
        std = x.view(3, -1).std(dim=0).view(3, 1, 1)

        return (x - mean) / std

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return self.__apply_fun(x)

    def __repr__(self):
        return self.__class__.__name__ + f"(mean = {str(self.__mean)}, std = {str(self.__std)})"


class MinMaxNorm(ImgTransform):
    def __init__(self, min_value: Optional[Union[Tuple[float, float, float], float]] = None,
                 max_value: Optional[Union[Tuple[float, float, float], float]] = None):

        self.__user_defined = (min_value is not None) and (max_value is not None)

        self.__min = th.tensor([min_value] * 3) if isinstance(min_value, float) else min_value
        self.__max = th.tensor([max_value] * 3) if isinstance(max_value, float) else max_value

        assert (min_value is not None) ^ (max_value is not None) == self.__user_defined, f"Hybrid mode not supported yet"

        self.__apply_fun: Callable[[th.Tensor], th.Tensor] = self.__apply_user if self.__user_defined else self.__apply_data

    def __apply_user(self, x: th.Tensor) -> th.Tensor:
        return (x - self.__min) / (self.__max - self.__min)

    def __apply_data(self, x: th.Tensor) -> th.Tensor:
        max_v = x.view(3, -1).max(dim=-1)[0].view(3, 1, 1)
        min_v = x.view(3, -1).min(dim=-1)[0].view(3, 1, 1)

        return (x - min_v) / (max_v - min_v)

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return self.__apply_fun(x)

    def __repr__(self):
        return self.__class__.__name__ + f"(min_value = {self.__min}, max_value = {self.__max})"

