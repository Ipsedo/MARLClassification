from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataset import T_co


class MNISTDataset(Dataset):
    def __init__(self, shuffle: bool = False,
                 seed: int = 314159) -> None:
        super().__init__()

        self.__img_size = 28
        self.__channel_size = 1

    def __getitem__(self, index: int) -> T_co:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other: T_co) -> ConcatDataset[T_co]:
        return super().__add__(other)


class RESISCDataset(Dataset):
    def __init__(self, shuffle: bool = False,
                 seed: int = 314159) -> None:
        super().__init__()

        self.__img_size = 256
        self.__channel_size = 3

    def __getitem__(self, index: int) -> T_co:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def __add__(self, other: T_co) -> ConcatDataset[T_co]:
        return super().__add__(other)
