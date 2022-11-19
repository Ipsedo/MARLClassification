from typing import Tuple

import torch as th

from .agent import MultiAgent


def episode(
        agents: MultiAgent,
        img_batch: th.Tensor,
        max_it: int
) -> Tuple[th.Tensor, th.Tensor]:

    img_sizes = [s for s in img_batch.size()[2:]]
    agents.new_episode(img_batch.size(0), img_sizes)

    for t in range(max_it):
        agents.step(img_batch)

    q, probas, _ = agents.predict()

    return q, probas


def detailed_episode(
        agents: MultiAgent, img_batch: th.Tensor,
        max_it: int, device_str: str, nb_class: int
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

    img_sizes = [s for s in img_batch.size()[2:]]
    batch_size = img_batch.size(0)

    agents.new_episode(batch_size, img_sizes)

    img_batch = img_batch.to(th.device(device_str))

    step_pos = th.zeros(
        max_it, *agents.pos.size(), dtype=th.long,
        device=th.device(device_str)
    )

    step_preds = th.zeros(
        max_it, len(agents), batch_size, nb_class,
        device=th.device(device_str)
    )

    step_probas = th.zeros(
        max_it, len(agents), batch_size,
        device=th.device(device_str)
    )

    step_values = th.zeros(
        max_it, len(agents), batch_size,
        device=th.device(device_str)
    )

    for t in range(max_it):
        agents.step(img_batch)

        step_pos[t, :, :, :] = agents.pos

        preds, probas, values = agents.predict()

        step_preds[t, :, :, :] = preds
        step_probas[t, :, :] = probas
        step_values[t, :, :] = values

    return step_preds, step_probas, step_values, step_pos


def episode_retry(
        agents: MultiAgent, img_batch: th.Tensor,
        max_it: int, max_retry: int, nb_class: int,
        device_str: str
) -> Tuple[th.Tensor, th.Tensor]:

    img_batch = img_batch.to(th.device(device_str))

    retry_pred = th.zeros(
        max_retry, max_it, img_batch.size(0), nb_class,
        device=th.device(device_str)
    )

    retry_prob = th.zeros(
        max_retry, max_it, img_batch.size(0),
        device=th.device(device_str)
    )

    for r in range(max_retry):
        pred, prob, _, _ = detailed_episode(
            agents, img_batch, max_it, device_str, nb_class
        )

        retry_pred[r, :, :, :] = pred
        retry_prob[r, :, :] = prob

    return retry_pred, retry_prob
