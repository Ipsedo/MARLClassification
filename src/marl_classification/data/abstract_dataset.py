from abc import ABC, abstractmethod
from typing import Dict

import torch as th
from torch.utils.data.dataset import Dataset


class AbstractDataset(Dataset, ABC):
    class_to_idx: Dict[str, int]

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[th.Tensor, th.Tensor]: ...

    @abstractmethod
    def __len__(self) -> int: ...
