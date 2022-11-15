import glob
import json
from os import mkdir
from os.path import join, exists, isfile
from typing import Mapping, Any

import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as th_fun
import torchvision.transforms as tr
from tqdm import tqdm

from .data import transforms as custom_tr
from .data.dataset import (
    my_pil_loader
)
from .environment import (
    MultiAgent,
    obs_generic,
    trans_generic,
    detailed_episode
)
from .networks.models import ModelsWrapper
from .options import MainOptions, InferOptions


def visualize_steps(
        agents: MultiAgent, img: th.Tensor, img_ori: th.Tensor,
        max_it: int, f: int, output_dir: str,
        nb_class: int, device_str: str,
        class_map: Mapping[Any, int]
) -> None:

    idx_to_class = {class_map[k]: k for k in class_map}

    color_map = None

    preds, _, pos = detailed_episode(
        agents, img.unsqueeze(0), 0.,
        max_it, device_str, nb_class
    )
    preds, pos = preds.cpu(), pos.cpu()
    img_ori = img_ori.permute(1, 2, 0).cpu()

    h, w, c = img_ori.size()

    if c == 1:
        # grey scale case
        img_ori = img_ori.repeat(1, 1, 3)

    img_idx = 0

    fig = plt.figure()
    plt.imshow(img_ori, cmap=color_map)
    plt.title("Original")
    plt.savefig(join(output_dir, f"pred_original.png"))
    plt.close(fig)

    curr_img = th.zeros(h, w, 4)
    for t in range(max_it):

        for i in range(len(agents)):
            # Color
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
            pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :3] = \
                img_ori[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :]
            # Alpha
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
            pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, 3] = 1

        fig = plt.figure()
        plt.imshow(curr_img, cmap=color_map)
        pred_softmax = th_fun.softmax(preds[t][img_idx], dim=-1)
        pred_max = pred_softmax.argmax(dim=-1).item()
        pred_proba = pred_softmax[pred_max].item()
        plt.title(
            f"Step = {t}, step_pred_class = "
            f"{idx_to_class[pred_max]} ({pred_proba * 100.:.1f}%)"
        )

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))
        plt.close(fig)


def infer(
        main_options: MainOptions,
        infer_options: InferOptions
) -> None:

    assert exists(infer_options.json_path), \
        f"JSON path \"{infer_options.json_path}\" does not exist"
    assert isfile(infer_options.json_path), \
        f"\"{infer_options.json_path}\" is not a file"

    assert exists(infer_options.state_dict_path), \
        f"State dict path {infer_options.state_dict_path} does not exist"
    assert isfile(infer_options.state_dict_path), \
        f"{infer_options.state_dict_path} is not a file"

    json_f = open(infer_options.class_to_idx, "r")
    class_to_idx = json.load(json_f)
    json_f.close()

    nn_models = ModelsWrapper.from_json(infer_options.json_path)
    nn_models.load_state_dict(th.load(infer_options.state_dict_path))

    marl_m = MultiAgent.load_from(
        infer_options.json_path,
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
        img for img_path in infer_options.images_path
        for img in glob.glob(img_path, recursive=True)
    ])

    for img_path in images:
        img = my_pil_loader(img_path)
        x_ori = img_ori_pipeline(img)
        x = img_pipeline(img)

        curr_img_path = join(infer_options.output_dir, img_path.split("/")[-1])

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
