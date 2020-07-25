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

MAOptions = typ.NamedTuple("MAOption",
                           [("nb_agent", int),
                            ("dim", int),
                            ("window_size", int),
                            ("img_size", int),
                            ("nb_class", int),
                            ("nb_action", int)])

RLOptions = typ.NamedTuple("RLOptions",
                           [("nb_step", int),
                            ("hidden_size", int),
                            ("hidden_size_msg", int),
                            ("cuda", bool)])

TrainOptions = typ.NamedTuple("TrainOptions",
                              [("nb_epoch", int),
                               ("learning_rate", float),
                               ("batch_size", int),
                               ("output_model_path", str),
                               ("frozen_modules", typ.List[str]),
                               ("data_set", str)])

TestOptions = typ.NamedTuple("TestOptions",
                             [("json_path", str),
                              ("state_dict_path", str),
                              ("output_img_path", str),
                              ("nb_test_img", int)])


def visualize_steps(agents: MultiAgent, one_img: th.Tensor,
                    max_it: int, f: int, output_dir: str,
                    nb_class: int, cuda: bool,
                    class_map: typ.Mapping[typ.Any, int]) -> None:
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
    :param nb_class:
    :type nb_class:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    idx_to_class = {class_map[k]: k for k in class_map}

    color_map = None

    preds, _, pos = detailed_episode(agents, one_img.unsqueeze(0),
                                     max_it, cuda, nb_class)

    one_img = one_img.permute(1, 2, 0)

    h, w, c = one_img.size()

    if c == 1:
        # grey scale case
        one_img = one_img.repeat(1, 1, 3)

    img_idx = 0

    plt.figure()
    plt.imshow(one_img, cmap=color_map)
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
        pred_proba = th.nn.functional.softmax(preds[t].mean(dim=0), dim=-1)[img_idx][prediction]
        plt.title(f"Step = {t}, step_pred_class = {idx_to_class[prediction]} ({pred_proba * 100.:.1f}%)")

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))


def prec_rec(conf_meter: ConfusionMeter) -> typ.Tuple[np.ndarray, np.ndarray]:
    conf_mat = conf_meter.value()

    precs_sum = [conf_mat[:, i].sum() for i in range(conf_mat.shape[1])]
    precs = np.array([conf_mat[i, i] / precs_sum[i] if precs_sum[i] != 0. else 0.
                      for i in range(conf_mat.shape[1])])

    recs_sum = [conf_mat[i, :].sum() for i in range(conf_mat.shape[1])]
    recs = np.array([(conf_mat[i, i] / recs_sum[i] if recs_sum[i] != 0. else 0.)
                     for i in range(conf_mat.shape[0])])

    return precs, recs


def format_metric(metric: np.ndarray, class_map: typ.Mapping[typ.Any, int]) -> str:
    idx_to_class = {class_map[k]: k for k in class_map}
    return ", ".join([f'\"{idx_to_class[curr_cls]}\" : {metric[curr_cls] * 100.:.1f}%'
                      for curr_cls in range(metric.shape[0])])


class SetAppendAction(Action):
    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: typ.Union[typ.Text, typ.Sequence[typ.Any], None],
                 option_string: typ.Optional[typ.Text] = ...) -> None:
        unique_values = set(values)

        if len(unique_values) != len(values):
            error_msg = f"duplicates value(s) found for \"{self.option_strings[-1]}\": " \
                        f"{[item for item, count in Counter(values).items() if count > 1]}"
            parser.error(error_msg)
            exit(1)

        setattr(namespace, self.dest, list(unique_values))
