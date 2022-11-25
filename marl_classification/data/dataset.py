from abc import ABC, abstractmethod
import pickle as pkl
from os.path import exists, isdir, join
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as fun
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def my_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    f = open(path, 'rb')
    img = Image.open(f)
    return img.convert('RGB')


class MNISTDataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        mnist_root_path = join(res_path, "downloaded", "mnist_png", "all_png")

        assert exists(mnist_root_path) and isdir(mnist_root_path), \
            f"{mnist_root_path} does not exist or is not a directory"

        super().__init__(mnist_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class RESISC45Dataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        resisc_root_path = join(res_path, "downloaded", "NWPU-RESISC45")

        assert exists(resisc_root_path) and isdir(resisc_root_path), \
            f"{resisc_root_path} does not exist or is not a directory"

        super().__init__(resisc_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class AIDDataset(ImageFolder):
    def __init__(self, res_path: str, img_transform: Any) -> None:
        aid_root_path = join(res_path, "downloaded", "AID")

        assert exists(aid_root_path) and isdir(aid_root_path), \
            f"{aid_root_path} does not exist or is not a directory"

        super().__init__(aid_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class KneeMRIDataset(Dataset):
    def __init__(self, res_path: str, img_transform: Any):
        super().__init__()

        self.__knee_mri_root_path = join(res_path, "downloaded", "knee_mri")

        self.__img_transform = img_transform

        metadata_csv = pd.read_csv(
            join(self.__knee_mri_root_path, "metadata.csv"), sep=","
        )

        tqdm.tqdm.pandas()

        self.__max_depth = -1
        self.__max_width = -1
        self.__max_height = -1
        self.__nb_img = 0

        def __open_pickle_size(fn: str) -> None:
            f = open(join(self.__knee_mri_root_path, "extracted", fn), "rb")
            x = pkl.load(f)
            f.close()
            self.__max_depth = max(self.__max_depth, x.shape[0])
            self.__max_width = max(self.__max_width, x.shape[1])
            self.__max_height = max(self.__max_height, x.shape[2])
            self.__nb_img += 1

        metadata_csv["volumeFilename"].progress_map(__open_pickle_size)

        self.__dataset = [
            (str(fn), lbl)
            for fn, lbl in zip(
                metadata_csv["volumeFilename"].tolist(),
                metadata_csv["aclDiagnosis"].tolist()
            )
        ]

        self.class_to_idx = {
            "healthy": 0,
            "partially injured": 1,
            "completely ruptured": 2
        }

    def __open_img(self, fn: str) -> th.Tensor:
        f = open(join(self.__knee_mri_root_path, "extracted", fn), "rb")
        x = pkl.load(f)
        f.close()

        x = th.from_numpy(x).to(th.float)

        # depth
        curr_depth = x.size(0)

        to_pad = self.__max_depth - curr_depth
        pad_1 = to_pad // 2 + to_pad % 2
        pad_2 = to_pad // 2

        # width
        curr_width = x.size(1)

        to_pad = self.__max_width - curr_width
        pad_3 = to_pad // 2 + to_pad % 2
        pad_4 = to_pad // 2

        # height
        curr_height = x.size(2)

        to_pad = self.__max_height - curr_height
        pad_5 = to_pad // 2 + to_pad % 2
        pad_6 = to_pad // 2

        return fun.pad(
            x, [pad_6, pad_5, pad_4, pad_3, pad_2, pad_1], value=0
        )

    def __getitem__(self, index) -> Tuple[th.Tensor, th.Tensor]:
        fn = self.__dataset[index][0]

        label = self.__dataset[index][1]
        img = self.__open_img(fn).unsqueeze(0)

        return img, th.tensor(label)

    def __len__(self) -> int:
        return self.__nb_img
