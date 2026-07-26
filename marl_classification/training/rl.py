from dataclasses import dataclass
from math import log

import torch as th
import torch.nn.functional as th_fun


@dataclass
class A2CLoss:
    path_loss: th.Tensor
    policy_loss: th.Tensor
    critic_loss: th.Tensor


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


def a2c_loss(
    log_probs: th.Tensor,
    values: th.Tensor,
    returns: th.Tensor,
    error: th.Tensor,
) -> A2CLoss:
    """
    Advantage actor-critic losses.

    log_probs, values, returns: [Ns, Na, Nb]
    error: classification error [Nb], broadcast over steps and agents
    """
    # actor advantage
    advantage = returns - values

    # actor loss, maximize(log_proba * advantage)
    path_loss = -log_probs * advantage.detach()

    # add agent's votes -> train classifier
    policy_loss = path_loss + error

    # critic loss : difference between values and returns
    critic_loss = th_fun.smooth_l1_loss(
        values,
        returns.detach(),
        reduction="none",
    )

    return A2CLoss(path_loss, policy_loss, critic_loss)
