import glob
import json
from datetime import datetime
from os import mkdir
from os.path import exists, getmtime, isfile, join, split

import torch as th
from tqdm import tqdm

from .config import InferConfig, MainConfig, ModelConfig
from .core import EpisodeSampler
from .data.datasets import my_pil_loader
from .registry import default_image_pipeline
from .visualization import visualize_steps


def infer_main(main_config: MainConfig, infer_config: InferConfig) -> None:
    assert exists(
        infer_config.json_path
    ), f'JSON path "{infer_config.json_path}" does not exist'
    assert isfile(
        infer_config.json_path
    ), f'"{infer_config.json_path}" is not a file'

    assert exists(
        infer_config.state_dict_path
    ), f"State dict path {infer_config.state_dict_path} does not exist"
    assert isfile(
        infer_config.state_dict_path
    ), f"{infer_config.state_dict_path} is not a file"

    print(
        "Will use :\n"
        "- JSON of : "
        f"{datetime.fromtimestamp(getmtime(infer_config.json_path))}\n"
        "- state_dict of : "
        f"{datetime.fromtimestamp(getmtime(infer_config.state_dict_path))}\n"
        "- class_to_idx of : "
        f"{datetime.fromtimestamp(getmtime(infer_config.class_to_idx))}"
    )

    with open(infer_config.class_to_idx, "r", encoding="utf-8") as json_f:
        class_to_idx = json.load(json_f)

    marl_config = ModelConfig.load_marl_config(infer_config.json_path)

    nn_models, marl_m, env = marl_config.build_marl(main_config.nb_agent)

    nn_models.load_state_dict(th.load(infer_config.state_dict_path))
    nn_models.eval()

    episode_sampler = EpisodeSampler(marl_m, env)

    img_pipeline = default_image_pipeline()

    device = th.device("cuda" if main_config.cuda else "cpu")
    nn_models.to(device)

    images = tqdm(
        [
            img
            for img_path in infer_config.images_path
            for img in glob.glob(img_path, recursive=True)
        ]
    )

    for img_path in images:
        img = my_pil_loader(img_path)
        x_ori = img_pipeline(img)
        x = img_pipeline(img)

        curr_img_path = join(infer_config.output_dir, split(img_path)[-1])

        if not exists(curr_img_path):
            mkdir(curr_img_path)

        with open(
            join(curr_img_path, "info.txt"), "w", encoding="utf-8"
        ) as info_f:
            info_f.writelines(
                [
                    f"{img_path}\n",
                    f"{infer_config.json_path}\n",
                    f"{infer_config.state_dict_path}\n",
                ]
            )

        visualize_steps(
            episode_sampler,
            x,
            x_ori,
            marl_config.window_size,
            main_config.step,
            curr_img_path,
            class_to_idx,
        )
