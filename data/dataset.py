import torch as th
import torch.nn.functional as fun
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

import numpy as np

from PIL import Image

import pickle as pkl

import pandas as pd

from os.path import exists, isdir

from typing import Any

import tqdm

DATASET_CHOICES = ["mnist", "resisc45"]


def my_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    f = open(path, 'rb')
    img = Image.open(f)
    return img.convert('RGB')


class MNISTDataset(ImageFolder):
    def __init__(self, img_transform: Any) -> None:
        mnist_root_path = "./res/downloaded/mnist_png/all_png"

        assert exists(mnist_root_path) and isdir(mnist_root_path), \
            f"{mnist_root_path} does not exist or is not a directory"

        super().__init__(mnist_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class RESISC45Dataset(ImageFolder):
    def __init__(self, img_transform: Any) -> None:
        resisc_root_path = "./res/downloaded/NWPU-RESISC45"

        assert exists(resisc_root_path) and isdir(resisc_root_path), \
            f"{resisc_root_path} does not exist or is not a directory"

        super().__init__(resisc_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader,
                         is_valid_file=None)


class KneeMRIDataset(TensorDataset):
    def __init__(self, img_transform: Any):
        knee_mri_root_path = "./res/downloaded/knee_mri"

        metadata_csv = pd.read_csv(
            knee_mri_root_path + "/" + "metadata.csv", sep=","
        )

        tqdm.tqdm.pandas()

        self.__max_depth = -1
        self.__max_width = -1
        self.__max_height = -1

        def __open_pickle(fn: str) -> th.Tensor:
            f = open(knee_mri_root_path + "/extracted/" + fn, "rb")
            x = pkl.load(f)
            f.close()
            self.__max_depth = max(self.__max_depth, x.shape[0])
            self.__max_width = max(self.__max_width, x.shape[1])
            self.__max_height = max(self.__max_height, x.shape[2])
            return th.from_numpy(x.astype(np.float)).to(th.float)

        metadata_csv["x"] = \
            metadata_csv["volumeFilename"].progress_map(__open_pickle)

        def __pad_img(x: th.Tensor) -> th.Tensor:
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

        metadata_csv["x"] = metadata_csv["x"].progress_map(__pad_img)

        metadata_csv["depth"] = metadata_csv["x"].progress_map(lambda x: x.size(0))

        self.class_to_idx = {
            "healthy": 0,
            "partially injured": 1,
            "completely ruptured": 2
        }

        data = th.stack(metadata_csv["x"].tolist(), dim=0).unsqueeze(1)
        labels = th.tensor(metadata_csv["aclDiagnosis"].tolist())

        super().__init__(data, labels)
