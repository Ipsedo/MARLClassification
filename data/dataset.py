import torch as th
from torch.utils.data import TensorDataset

from typing import Mapping, Any

from .loader import load_mnist, load_resisc45, DATASET_CHOICES


class ImgDataset(TensorDataset):
    def __init__(self, dataset: str) -> None:
        assert dataset in DATASET_CHOICES, f"Dataset choice not recognized ! Choose between {DATASET_CHOICES}"

        if dataset == "mnist":
            (x, y), class_map = load_mnist()
        elif dataset == "resisc45":
            (x, y), class_map = load_resisc45()
        else:
            (x, y), class_map = (th.rand(1, 3, 256, 256), th.zeros(1, 1)), {0: 0}

        super().__init__(x, y)

        self.__class_map = class_map
    
    @property
    def class_map(self) -> Mapping[Any, int]:
        return self.__class_map
