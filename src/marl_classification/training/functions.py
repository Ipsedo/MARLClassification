from math import log

import torch as th
import torch.nn.functional as th_fun


def classification_rewards(
    step_preds: th.Tensor, targets: th.Tensor
) -> th.Tensor:
    """
    Reward per step, agent and batch element: positive if the prediction
    is better than a random guess, negative otherwise.

    step_preds: [Ns, Na, Nb, Nc], targets: [Nb] -> [Ns, Na, Nb]
    """
    nb_step, nb_agent, _, nb_class = step_preds.size()

    # [Nb] -> [Nb, Ns, Na]
    tmp_targets = targets[:, None, None].repeat(1, nb_step, nb_agent)

    # [Ns, Na, Nb, Nc] -> [Nb, Nc, Ns, Na]
    tmp_preds = step_preds.permute(2, 3, 0, 1)

    # random prediction error
    random_error = log(nb_class)

    # [Nb, Ns, Na] -> [Ns, Na, Nb]
    error = th_fun.cross_entropy(
        tmp_preds, tmp_targets, reduction="none"
    ).permute(1, 2, 0)

    return (random_error - error) / random_error


def discounted_returns(rewards: th.Tensor, gamma: float) -> th.Tensor:
    """
    Discounted sum of future rewards along the first (step) dimension.

    rewards: [Ns, ...] -> [Ns, ...]
    """
    shape = [rewards.size(0)] + [1] * (rewards.dim() - 1)

    t_steps = (
        th.arange(rewards.size(0), device=rewards.device)
        .view(*shape)
        .to(th.float)
    )

    returns = rewards * gamma**t_steps

    return returns.flip(dims=(0,)).cumsum(0).flip(dims=(0,)) / gamma**t_steps


def standardize(values: th.Tensor, eps: float = 1e-8) -> th.Tensor:
    return (values - values.mean()) / (values.std() + eps)
