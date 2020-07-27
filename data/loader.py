import torch as th

import numpy as np

from os import listdir
from os.path import exists, isdir, join, isfile, splitext

from PIL import Image
import gzip
import pickle

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
        img = th.from_numpy(np.asarray(pil_img).copy()).permute(2, 0, 1)
        pil_img.close()

        images[i, :, :, :] = img
        images_class[i] = resisc45_class_map[curr_class]

        tqdm_bar.set_description(f"RESISCS45 - {i} / {nb_class * img_per_class} images loaded")

    return (images, images_class), resisc45_class_map


def load_mnist() -> Tuple[Tuple[th.Tensor, th.Tensor], Mapping[Any, int]]:
    """

    :return:
    :rtype:
    """

    file = './res/downloaded/mnist.pkl.gz'

    assert exists(file), "You must download mnist dataset via download_mnist.sh script !"

    f = gzip.open(file, 'rb')

    u = pickle._Unpickler(f)
    u.encoding = 'latin1'

    train_set, valid_set, test_set = u.load()

    f.close()

    x_train = th.from_numpy(train_set[0]).view(-1, 1, 28, 28)
    y_train = th.from_numpy(train_set[1])

    x_valid = th.from_numpy(valid_set[0]).view(-1, 1, 28, 28)
    y_valid = th.from_numpy(valid_set[1])

    x_test = th.from_numpy(test_set[0]).view(-1, 1, 28, 28)
    y_test = th.from_numpy(test_set[1])

    x = th.cat([x_train, x_valid, x_test])
    y = th.cat([y_train, y_valid, y_test])

    return (x, y), {i: i for i in range(10)}
