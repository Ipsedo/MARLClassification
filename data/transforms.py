import torch as th

from typing import Tuple, Optional, Union

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

        self.__user_defined_mean = mean is not None
        self.__user_defined_std = std is not None

        self.__mean = th.tensor([mean] * 3) if isinstance(mean, float) else mean
        self.__std = th.tensor([std] * 3) if isinstance(std, float) else std

    def __call__(self, img_data: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        x, y = img_data

        mean = self.__mean if self.__user_defined_mean else x.view(3, -1).mean(dim=-1)
        std = self.__std if self.__user_defined_std else x.view(3, -1).std(dim=-1)

        return (x - mean) / std, y

    def __repr__(self):
        return self.__class__.__name__ + f"(mean = {str(self.__mean)}, std = {str(self.__std)})"


class MinMaxNorm(ImgTransform):
    def __init__(self, min_value: Optional[Union[Tuple[float, float, float], float]] = None,
                 max_value: Optional[Union[Tuple[float, float, float], float]] = None):

        self.__user_defined_max = max_value is not None
        self.__user_defined_min = min_value is not None

        self.__min = th.tensor([min_value] * 3) if isinstance(min_value, float) else min_value
        self.__max = th.tensor([max_value] * 3) if isinstance(max_value, float) else max_value

    def __call__(self, img_data: Tuple[th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        x, y = img_data

        max_value = self.__max if self.__user_defined_max else x.view(3, -1).max(dim=-1)[0]
        min_value = self.__min if self.__user_defined_min else x.view(3, -1).min(dim=-1)[0]

        return (x - min_value) / (max_value - min_value), y

    def __repr__(self):
        return self.__class__.__name__ + f"(min_value = {self.__min}, max_value = {self.__max})"

