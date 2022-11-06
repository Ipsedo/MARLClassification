import glob
import json
from os import mkdir
from os.path import join, exists, isfile

import torch as th
import torchvision.transforms as tr
from tqdm import tqdm

from .data import transforms as custom_tr
from .data.dataset import (
    my_pil_loader
)
from .environment import (
    MultiAgent,
    obs_generic,
    trans_generic
)
from .networks.models import ModelsWrapper
from .utils import (
    MainOptions,
    InferOptions,
    visualize_steps
)


def infer(
        main_options: MainOptions,
        infer_options: InferOptions
) -> None:
    images_path = infer_options.images_path
    output_dir = infer_options.output_dir
    state_dict_path = infer_options.state_dict_path
    json_path = infer_options.json_path

    assert exists(json_path), \
        f"JSON path \"{json_path}\" does not exist"
    assert isfile(json_path), \
        f"\"{json_path}\" is not a file"

    assert exists(state_dict_path), \
        f"State dict path {state_dict_path} does not exist"
    assert isfile(state_dict_path), \
        f"{state_dict_path} is not a file"

    json_f = open(infer_options.class_to_idx, "r")
    class_to_idx = json.load(json_f)
    json_f.close()

    nn_models = ModelsWrapper.from_json(json_path)
    nn_models.load_state_dict(th.load(state_dict_path))

    marl_m = MultiAgent.load_from(
        json_path,
        main_options.nb_agent,
        nn_models,
        obs_generic,
        trans_generic
    )

    img_ori_pipeline = tr.Compose([
        tr.ToTensor()
    ])

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    images = tqdm([
        img for img_path in images_path
        for img in glob.glob(img_path, recursive=True)
    ])

    for img_path in images:
        img = my_pil_loader(img_path)
        x_ori = img_ori_pipeline(img)
        x = img_pipeline(img)

        curr_img_path = join(output_dir, img_path.split("/")[-1])

        if not exists(curr_img_path):
            mkdir(curr_img_path)

        info_f = open(join(curr_img_path, "info.txt"), "w")
        info_f.writelines(
            [f"{img_path}\n",
             f"{infer_options.json_path}\n",
             f"{infer_options.state_dict_path}\n"]
        )
        info_f.close()

        visualize_steps(
            marl_m, x, x_ori,
            main_options.step,
            nn_models.f,
            curr_img_path,
            nn_models.nb_class,
            device_str,
            class_to_idx
        )
