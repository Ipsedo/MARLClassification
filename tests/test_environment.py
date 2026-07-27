from typing import Tuple

import torch as th

from marl_classification.core import Environment


def test_reset_positions_within_bounds(
    env: Environment,
    batch_size: int,
    nb_agent: int,
    window_size: int,
    height_width: Tuple[int, int],
) -> None:
    x = th.randn(batch_size, 1, *height_width)

    obs = env.reset(x, nb_agent)

    assert (nb_agent, batch_size, 2) == env.positions.size()
    assert bool((env.positions >= 0).all())

    for d, size in enumerate(height_width):
        assert bool((env.positions[:, :, d] + window_size <= size).all())

    assert (
        nb_agent,
        batch_size,
        1,
        window_size,
        window_size,
    ) == obs.size()


def test_step_keeps_positions_within_bounds(
    env: Environment,
    batch_size: int,
    nb_agent: int,
    window_size: int,
    height_width: Tuple[int, int],
) -> None:
    x = th.randn(batch_size, 1, *height_width)

    env.reset(x, nb_agent)

    for _ in range(50):
        action_indices = th.randint(env.nb_actions, (nb_agent, batch_size))
        obs = env.step(action_indices)

        assert bool((env.positions >= 0).all())
        for d, size in enumerate(height_width):
            assert bool((env.positions[:, :, d] + window_size <= size).all())

        assert (
            nb_agent,
            batch_size,
            1,
            window_size,
            window_size,
        ) == obs.size()


def test_normalized_positions(
    env: Environment,
    batch_size: int,
    nb_agent: int,
    height_width: Tuple[int, int],
) -> None:
    x = th.randn(batch_size, 1, *height_width)

    env.reset(x, nb_agent)

    norm_pos = env.normalized_positions

    assert env.positions.size() == norm_pos.size()
    assert bool((norm_pos >= 0).all())
    assert bool((norm_pos < 1).all())
