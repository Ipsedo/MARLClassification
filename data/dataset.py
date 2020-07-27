from torchvision.datasets import ImageFolder

from PIL import Image

from os.path import exists, isdir

from typing import Any


DATASET_CHOICES = ["mnist", "resisc45"]


def my_pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MNISTDataset(ImageFolder):
    def __init__(self, img_transform: Any) -> None:
        mnist_root_path = "./res/downloaded/mnist_png/all_png"

        assert exists(mnist_root_path) and isdir(mnist_root_path),\
            f"{mnist_root_path} does not exist or is not a directory"

        super().__init__(mnist_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader, is_valid_file=None)


class RESISC45Dataset(ImageFolder):
    def __init__(self, img_transform: Any) -> None:
        resisc_root_path = "./res/downloaded/NWPU-RESISC45"

        assert exists(resisc_root_path) and isdir(resisc_root_path),\
            f"{resisc_root_path} does not exist or is not a directory"

        super().__init__(resisc_root_path, transform=img_transform,
                         target_transform=None, loader=my_pil_loader, is_valid_file=None)
