import torch as th

from .agent import MultiAgent
from typing import Tuple


def episode(
        agents: MultiAgent,
        img_batch: th.Tensor,
        max_it: int
) -> Tuple[th.Tensor, th.Tensor]:
    """

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :return:
    :rtype:
    """

    agents.new_episode(img_batch.size(0), img_batch.size(-1))

    for t in range(max_it):
        agents.step(img_batch)

    q, probas = agents.predict()

    return q, probas


def detailed_episode(
        agents: MultiAgent, img_batch: th.Tensor,
        max_it: int, device_str: str, nb_class: int
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param device_str:
    :type device_str:
    :param nb_class:
    :type nb_class:
    :return:
    :rtype:
    """

    agents.new_episode(img_batch.size(0), img_batch.size(-1))

    img_batch = img_batch.to(th.device(device_str))

    step_pos = th.zeros(max_it, *agents.pos.size(), dtype=th.long,
                        device=th.device(device_str))

    step_preds = th.zeros(max_it, img_batch.size(0), nb_class,
                          device=th.device(device_str))

    step_probas = th.zeros(max_it, img_batch.size(0),
                           device=th.device(device_str))

    for t in range(max_it):
        agents.step(img_batch)

        step_pos[t, :, :, :] = agents.pos

        preds, probas = agents.predict()

        step_preds[t, :, :] = preds
        step_probas[t, :] = probas

    return step_preds, step_probas, step_pos


def episode_retry(
        agents: MultiAgent, img_batch: th.Tensor,
        max_it: int, max_retry: int, nb_class: int,
        device_str: str
) -> Tuple[th.Tensor, th.Tensor]:
    """

    :param agents:
    :type agents:
    :param img_batch:
    :type img_batch:
    :param max_it:
    :type max_it:
    :param max_retry:
    :type max_retry:
    :param nb_class:
    :type nb_class:
    :param device_str:
    :type device_str:
    :return:
    :rtype:
    """

    img_batch = img_batch.to(th.device(device_str))

    retry_pred = th.zeros(max_retry, img_batch.size(0), nb_class,
                          device=th.device(device_str))

    retry_prob = th.zeros(max_retry, img_batch.size(0),
                          device=th.device(device_str))

    for r in range(max_retry):
        pred, prob = episode(agents, img_batch, max_it)

        retry_pred[r, :, :] = pred
        retry_prob[r, :] = prob

    return retry_pred, retry_prob
