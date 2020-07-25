import torch as th

import numpy as np

import sys
from os import listdir
from os.path import exists, isdir, join, isfile, splitext

import random

from PIL import Image

from typing import Tuple, Mapping, Any

from tqdm import tqdm


def load_resisc45() -> Tuple[Tuple[th.Tensor, th.Tensor], Mapping[Any, int]]:
    resisc45_root_path = './res/downloaded/NWPU-RESISC45'

    assert exists(resisc45_root_path),\
        "You must download RESISC45 dataset via download_resisc45_helper.sh script"
    assert isdir(resisc45_root_path),\
        f"\"{resisc45_root_path}\" is not a directory"

    resisc45_class_map = {
        c: i for i, c in enumerate(sorted(
            [d for d in listdir(resisc45_root_path)
             if isdir(join(resisc45_root_path, d))]
        ))
    }

    nb_class = len(resisc45_class_map)
    img_per_class = 700
    img_side = 256
    channels = 3

    images = th.empty(nb_class * img_per_class, channels, img_side, img_side)
    images_class = th.empty(nb_class * img_per_class, dtype=th.long)

    images_file_path = [join(resisc45_root_path, class_folder, img_file)
                        for class_folder in resisc45_class_map
                        for img_file in listdir(join(resisc45_root_path, class_folder))
                        if isfile(join(resisc45_root_path, class_folder, img_file))
                        and splitext(img_file)[-1] == ".jpg"]

    tqdm_bar = tqdm(images_file_path)
    for i, img_path in enumerate(tqdm_bar):
        curr_class = splitext(img_path)[-2].split("/")[-2]

        pil_img = Image.open(img_path, "r")
        img = np.asarray(pil_img).copy() / 255.
        img = th.from_numpy(img).permute(2, 0, 1)
        pil_img.close()

        images[i, :, :, :] = img
        images_class[i] = resisc45_class_map[curr_class]

        tqdm_bar.set_description(f"RESISCS45 - {i} / {nb_class * img_per_class} images loaded")

    tqdm_bar = tqdm(range(images.size(0) - 1))
    for i in tqdm_bar:
        j = i + random.randint(0, sys.maxsize) // (sys.maxsize // (images.size(0) - i) + 1)

        images[i, :, :, :], images[j, :, :, :] = images[j, :, :, :], images[i, :, :, :]
        images_class[i], images_class[j] = images_class[j], images_class[i]

        tqdm_bar.set_description(f"RESISCS45 - {i} / {nb_class * img_per_class} images shuffled")

    return (images, images_class), resisc45_class_map
