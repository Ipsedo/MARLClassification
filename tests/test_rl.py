# -*- coding: utf-8 -*-
import torch as th

from marl_classification.training import (
    a2c_loss,
    classification_rewards,
    discounted_returns,
    standardize,
)


def test_classification_rewards_shape_and_sign() -> None:
    nb_step, nb_agent, batch_size, nb_class = 5, 3, 7, 10

    targets = th.randint(nb_class, (batch_size,))

    # perfect prediction -> positive reward
    perfect_logits = th.full((nb_step, nb_agent, batch_size, nb_class), -20.0)
    for b in range(batch_size):
        perfect_logits[:, :, b, targets[b]] = 20.0

    rewards = classification_rewards(perfect_logits, targets)

    assert (nb_step, nb_agent, batch_size) == rewards.size()
    assert bool((rewards > 0).all())

    # worst prediction -> negative reward
    worst_logits = -perfect_logits
    rewards = classification_rewards(worst_logits, targets)

    assert bool((rewards < 0).all())


def test_discounted_returns_gamma_one() -> None:
    # with gamma = 1, returns are reversed cumulative sums
    rewards = th.tensor([[1.0], [2.0], [3.0]])

    returns = discounted_returns(rewards, gamma=1.0)

    expected = th.tensor([[6.0], [5.0], [3.0]])
    assert th.allclose(returns, expected)


def test_discounted_returns_discounting() -> None:
    rewards = th.tensor([[1.0], [1.0], [1.0]])
    gamma = 0.5

    returns = discounted_returns(rewards, gamma)

    # R_t = r_t + gamma * R_{t+1}
    expected = th.tensor([[1.75], [1.5], [1.0]])
    assert th.allclose(returns, expected)


def test_standardize() -> None:
    x = th.randn(4, 5, 6)

    standardized = standardize(x)

    assert th.allclose(standardized.mean(), th.tensor(0.0), atol=1e-6)
    assert th.allclose(standardized.std(), th.tensor(1.0), atol=1e-4)


def test_a2c_loss_shapes() -> None:
    nb_step, nb_agent, batch_size = 5, 3, 7

    log_probs = th.rand(nb_step, nb_agent, batch_size).log()
    values = th.randn(nb_step, nb_agent, batch_size)
    returns = th.randn(nb_step, nb_agent, batch_size)
    error = th.rand(batch_size)

    losses = a2c_loss(log_probs, values, returns, error)

    expected_size = (nb_step, nb_agent, batch_size)
    assert expected_size == losses.path_loss.size()
    assert expected_size == losses.policy_loss.size()
    assert expected_size == losses.critic_loss.size()

    # critic loss is a distance -> positive
    assert bool((losses.critic_loss >= 0).all())
