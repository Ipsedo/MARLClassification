from argparse import ArgumentParser, Namespace, Action

from environment.agent import MultiAgent
from environment.core import detailed_episode

import torch as th
from torchnet.meter import ConfusionMeter

import numpy as np

import matplotlib.pyplot as plt

from os.path import join

from collections import Counter

import typing as typ

MainOptions = typ.NamedTuple(
    "MainOptions",
    [("step", int),
     ("cuda", bool),
     ("nb_agent", int)]
)

TrainOptions = typ.NamedTuple(
    "TrainOptions",
    [("hidden_size", int),
     ("hidden_size_linear", int),
     ("hidden_size_msg", int),
     ("hidden_size_state", int),
     ("dim", int),
     ("window_size", int),
     ("img_size", int),
     ("nb_class", int),
     ("nb_action", int),
     ("nb_epoch", int),
     ("learning_rate", float),
     ("retry_number", int),
     ("epsilon", float),
     ("batch_size", int),
     ("output_dir", str),
     ("frozen_modules", typ.List[str]),
     ("ft_extr_str", str)]
)

TestOptions = typ.NamedTuple(
    "TestOptions",
    [("img_size", int),
     ("batch_size", int),
     ("json_path", str),
     ("state_dict_path", str),
     ("image_root", str),
     ("output_dir", str)]
)

InferOptions = typ.NamedTuple(
    "InferOptions",
    [("json_path", str),
     ("state_dict_path", str),
     ("images_path", typ.List[str]),
     ("output_dir", str),
     ("class_to_idx_json", str)]
)


def visualize_steps(
        agents: MultiAgent, img: th.Tensor, img_ori: th.Tensor,
        max_it: int, f: int, output_dir: str,
        nb_class: int, device_str: str,
        class_map: typ.Mapping[typ.Any, int]
) -> None:
    """

    :param agents:
    :type agents:
    :param img:
    :type img:
    :param img_ori:
    :type img_ori:
    :param max_it:
    :type max_it:
    :param f:
    :type f:
    :param output_dir:
    :type output_dir:
    :param nb_class:
    :type nb_class:
    :param device_str:
    :type device_str:
    :param class_map:
    :type class_map:
    :return:
    :rtype:
    """

    idx_to_class = {class_map[k]: k for k in class_map}

    color_map = None

    preds, _, pos = detailed_episode(agents, img.unsqueeze(0), 0.,
                                     max_it, device_str, nb_class)
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
        prediction = preds[t][img_idx].argmax(dim=-1).item()
        pred_proba = th.nn.functional.softmax(preds[t], dim=-1) \
            [img_idx][prediction].item()
        plt.title(f"Step = {t}, step_pred_class = "
                  f"{idx_to_class[prediction]} ({pred_proba * 100.:.1f}%)")

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))
        plt.close(fig)


def prec_rec(conf_meter: ConfusionMeter) \
        -> typ.Tuple[np.ndarray, np.ndarray]:
    conf_mat = conf_meter.value()

    precs_sum = [conf_mat[:, i].sum() for i in range(conf_mat.shape[1])]
    precs = np.array(
        [conf_mat[i, i] / precs_sum[i] if precs_sum[i] != 0. else 0.
         for i in range(conf_mat.shape[1])]
    )

    recs_sum = [conf_mat[i, :].sum() for i in range(conf_mat.shape[1])]
    recs = np.array(
        [(conf_mat[i, i] / recs_sum[i] if recs_sum[i] != 0. else 0.)
         for i in range(conf_mat.shape[0])]
    )

    return precs, recs


def format_metric(metric: np.ndarray,
                  class_map: typ.Mapping[typ.Any, int]) -> str:
    idx_to_class = {class_map[k]: k for k in class_map}
    return ", ".join(
        [f'\"{idx_to_class[curr_cls]}\" : {metric[curr_cls] * 100.:.1f}%'
         for curr_cls in range(metric.shape[0])]
    )


class SetAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: typ.Union[typ.Text, typ.Sequence[typ.Any], None],
                 option_string: typ.Optional[typ.Text] = ...) -> None:
        unique_values = set(values)

        if len(unique_values) != len(values):
            dupl_values = [
                item
                for item, count in Counter(values).items()
                if count > 1
            ]

            error_msg = f"duplicates value(s) found for " \
                        f"\"{self.option_strings[-1]}\": " \
                        f"{dupl_values}"

            parser.error(error_msg)
            exit(1)

        setattr(namespace, self.dest, list(unique_values))
