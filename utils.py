from environment.agent import MultiAgent
from environment.core import detailed_episode

import torch as th
from torchnet.meter import ConfusionMeter

import numpy as np

import matplotlib.pyplot as plt

from os.path import join

from typing import NamedTuple, Tuple

MAOptions = NamedTuple("MAOption",
                       [("nb_agent", int),
                        ("dim", int),
                        ("window_size", int),
                        ("img_size", int),
                        ("nb_class", int),
                        ("nb_action", int)])

RLOptions = NamedTuple("RLOptions",
                       [("nb_step", int),
                        ("hidden_size", int),
                        ("hidden_size_msg", int),
                        ("cuda", bool)])

TrainOptions = NamedTuple("TrainOptions",
                          [("nb_epoch", int),
                           ("learning_rate", float),
                           ("batch_size", int),
                           ("output_model_path", str)])

TestOptions = NamedTuple("TestOptions",
                         [("json_path", str),
                          ("state_dict_path", str),
                          ("output_img_path", str),
                          ("nb_test_img", int)])


def visualize_steps(agents: MultiAgent, one_img: th.Tensor,
                    max_it: int, f: int, output_dir: str) -> None:
    """

    :param agents:
    :type agents:
    :param one_img:
    :type one_img:
    :param max_it:
    :type max_it:
    :param f:
    :type f:
    :param output_dir:
    :type output_dir:
    :return:
    :rtype:
    """

    color_map = None

    preds, _, pos = detailed_episode(agents, one_img.unsqueeze(0).cuda(),
                                     max_it, True, 10)

    one_img = one_img.permute(1, 2, 0)

    h, w, c = one_img.size()

    img_idx = 0

    plt.figure()
    plt.imshow(one_img.repeat(1, 1, 3), cmap=color_map)
    plt.savefig(join(output_dir, f"pred_original.png"))

    curr_img = th.zeros(h, w, 4)
    for t in range(max_it):

        for i in range(len(agents)):
            # Color
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                     pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :3] = \
                one_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                        pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, :]
            # Alpha
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                     pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f, 3] = 1

        plt.figure()
        plt.imshow(curr_img, cmap=color_map)
        prediction = preds[t].mean(dim=0)[img_idx].argmax(dim=-1)
        pred_proba = preds[t].mean(dim=0)[img_idx][prediction]
        plt.title(f"Step = {t}, step_pred_class = {prediction} ({pred_proba * 100.:.1f}%)")

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))


def prec_rec(conf_meter: ConfusionMeter, smoothing: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
    conf_mat = conf_meter.value()

    precs = np.array([(conf_mat[i, i] / (conf_mat[:, i].sum() + smoothing))
                      for i in range(conf_mat.shape[1])])

    recs = np.array([(conf_mat[i, i] / (conf_mat[i, :].sum() + smoothing))
                     for i in range(conf_mat.shape[0])])

    return precs, recs


def format_metric(metric: np.ndarray) -> str:
    return ", ".join([f'\"{curr_cls}\" : {metric[curr_cls] * 100.:.1f}%' for curr_cls in range(metric.shape[0])])
