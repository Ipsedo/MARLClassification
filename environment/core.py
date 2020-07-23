import torch as th
import torch.nn as nn

from .agent import MultiAgent
from typing import Tuple


def episode(agents: MultiAgent, img_batch: th.Tensor, max_it: int, eps: float) -> \
        Tuple[th.Tensor, th.Tensor]:
    """
    TODO

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param cuda:
    :type cuda:
    :param eps:
    :type eps:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

    agents.new_episode(img_batch.size(0))

    for t in range(max_it):
        agents.step(img_batch, eps)

    q, probas = agents.predict()

    return nn.functional.softmax(q, dim=-1), probas


def detailed_episode(agents: MultiAgent, img_batch: th.Tensor, max_it: int,
                     cuda: bool, nb_class: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    TODO

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param cuda:
    :type cuda:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

    agents.new_episode(img_batch.size(0))

    pos = th.zeros(max_it, *agents.pos.size(), dtype=th.long)

    q = th.zeros(max_it, len(agents), img_batch.size(0), nb_class,
                 device=img_batch.device)

    all_probas = th.zeros(max_it, len(agents), img_batch.size(0),
                          device=img_batch.device)

    for t in range(max_it):
        agents.step(img_batch, 0.)

        pos[t, :, :, :] = agents.pos

        preds, probas = agents.predict()

        q[t, :, :] = preds
        all_probas[t, :, :] = probas

    return nn.functional.softmax(q, dim=-1), all_probas, pos
