import torch as th
from torch.utils.data import Dataset, ConcatDataset
from typing import Tuple


class MainDataset(Dataset):
    def __init__(self, shuffle: bool, seed: int,
                 img_size: int, channel_size: int) -> None:
        super().__init__()

        self._shuffle = shuffle
        self._seed = seed

        self._img_size = img_size
        self._channel_size = channel_size


class MNISTDataset(MainDataset):
    def __init__(self, shuffle: bool = False,
                 seed: int = 314159) -> None:
        super().__init__(shuffle, seed, 28, 1)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other: Tuple[th.Tensor, th.Tensor]) -> ConcatDataset[Tuple[th.Tensor, th.Tensor]]:
        return super().__add__(other)


class RESISCDataset(MainDataset):
    def __init__(self, shuffle: bool = False,
                 seed: int = 314159,
                 verbose: bool = True) -> None:
        super().__init__(shuffle, seed, 256, 3)

        self.__verbose = True

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other: Tuple[th.Tensor, th.Tensor]) -> ConcatDataset[Tuple[th.Tensor, th.Tensor]]:
        return super().__add__(other)
