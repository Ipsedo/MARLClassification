from argparse import ArgumentParser, Namespace, Action
from collections import Counter
from os.path import join
from typing import Mapping, Any, Union, Sequence, Tuple, Text, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch as th
from torchnet.meter import ConfusionMeter

from .environment.agent import MultiAgent
from .environment.episode import detailed_episode


def visualize_steps(
        agents: MultiAgent, img: th.Tensor, img_ori: th.Tensor,
        max_it: int, f: int, output_dir: str,
        nb_class: int, device_str: str,
        class_map: Mapping[Any, int]
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

    mlflow.log_artifact(join(output_dir, f"pred_original.png"))

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
        pred_proba = preds[t][img_idx][prediction].item()
        plt.title(
            f"Step = {t}, step_pred_class = "
            f"{idx_to_class[prediction]} ({pred_proba * 100.:.1f}%)"
        )

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))
        plt.close(fig)

        mlflow.log_artifact(join(output_dir, f"pred_step_{t}.png"))


def prec_rec(conf_meter: ConfusionMeter) \
        -> Tuple[np.ndarray, np.ndarray]:
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
                  class_map: Mapping[Any, int]) -> str:
    idx_to_class = {class_map[k]: k for k in class_map}
    return ", ".join(
        [f'\"{idx_to_class[curr_cls]}\" : {metric[curr_cls] * 100.:.1f}%'
         for curr_cls in range(metric.shape[0])]
    )


def save_conf_matrix(
        conf_meter: ConfusionMeter, epoch: int,
        output_dir: str, stage: str
) -> None:
    plt.matshow(conf_meter.value().tolist())
    plt.title(f"confusion matrix epoch {epoch} - {stage}")
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(
        join(output_dir, f"confusion_matrix_epoch_{epoch}_{stage}.png")
    )
    plt.close()


class SetAppendAction(Action):
    def __call__(
            self, parser: ArgumentParser, namespace: Namespace,
            values: Union[Text, Sequence[Any], None],
            option_string: Optional[Text] = ...
    ) -> None:
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
