import torch as th

from .agent import MultiAgent
from typing import Tuple


def episode(agents: MultiAgent, img_batch: th.Tensor, max_it: int) -> \
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
        agents.step(img_batch)

    q, probas = agents.predict()

    return q, probas


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

    device_str = "cuda" if cuda else "cpu"

    img_batch = img_batch.to(th.device(device_str))

    step_pos = th.zeros(max_it, *agents.pos.size(), dtype=th.long,
                        device=th.device(device_str))

    step_preds = th.zeros(max_it, len(agents), img_batch.size(0), nb_class,
                          device=th.device(device_str))

    step_probas = th.zeros(max_it, len(agents), img_batch.size(0),
                           device=th.device(device_str))

    agents.new_episode(img_batch.size(0))

    for t in range(max_it):
        agents.step(img_batch)

        step_pos[t, :, :, :] = agents.pos

        preds, probas = agents.predict()

        step_preds[t, :, :, :] = preds
        step_probas[t, :, :] = probas

    return step_preds, step_probas, step_pos
