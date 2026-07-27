from os.path import join
from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import torch as th
import torch.nn.functional as th_fun
from PIL import Image

from .core import EpisodeSampler


def visualize_steps(
    episode_sampler: EpisodeSampler,
    img: th.Tensor,
    img_ori: th.Tensor,
    window_size: int,
    max_it: int,
    output_dir: str,
    class_map: Mapping[Any, int],
) -> None:
    idx_to_class = {class_map[k]: k for k in class_map}

    color_map = None

    output = episode_sampler.run_episode(
        img.unsqueeze(0),
        max_it,
    )

    nb_agents = output.step_preds.size(1)

    # mean over agents
    preds, pos = output.step_preds.mean(dim=1).cpu(), output.step_pos.cpu()
    img_ori = img_ori.permute(1, 2, 0).cpu()

    h, w, c = img_ori.size()

    if c == 1:
        # grey scale case
        img_ori = img_ori.repeat(1, 1, 3)

    img_idx = 0

    frames: List[Image.Image] = []

    fig = plt.figure()
    plt.imshow(img_ori, cmap=color_map)
    plt.title("Original")
    frame_file_name = join(output_dir, "pred_original.png")
    plt.savefig(frame_file_name)
    plt.close(fig)

    # for GIF : 5 * 200ms -> 1s
    for _ in range(5):
        frames.append(Image.open(frame_file_name))

    curr_img = th.zeros(h, w, 4)
    for t in range(max_it):
        for i in range(nb_agents):
            # agent coordinates
            x = int(pos[t][i][img_idx][0].item())
            y = int(pos[t][i][img_idx][1].item())

            # fmt : off

            # Color
            curr_img[x : x + window_size, y : y + window_size, :3] = img_ori[
                x : x + window_size, y : y + window_size, :
            ]

            # Alpha
            curr_img[x : x + window_size, y : y + window_size, 3] = 1

            # fmt : on

        fig = plt.figure()
        plt.imshow(curr_img, cmap=color_map)

        pred_softmax = th_fun.softmax(preds[t][img_idx], dim=-1)
        pred_max = int(pred_softmax.argmax(dim=-1).item())
        pred_proba = pred_softmax[pred_max].item()

        plt.title(
            f"Step = {t}, step_pred_class = "
            f"{idx_to_class[pred_max]} ({pred_proba * 100.:.1f}%)"
        )

        frame_file_name = join(output_dir, f"pred_step_{t}.png")
        plt.savefig(frame_file_name)
        plt.close(fig)

        frames.append(Image.open(frame_file_name))

    frames[0].save(
        join(output_dir, "animated_gif.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
