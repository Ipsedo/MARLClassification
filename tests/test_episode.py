from typing import Tuple

import torch as th

from marl_classification.core import Environment, EpisodeSampler, MultiAgent


def test_episode(
    batch_size: int,
    marl_m: MultiAgent,
    env: Environment,
    step: int,
    nb_class: int,
    nb_agent: int,
    height_width: Tuple[int, int],
) -> None:
    x = th.randn(batch_size, 1, *height_width)
    episode_sampler = EpisodeSampler(marl_m, env)

    output = episode_sampler.run_episode_get_last_step(x, step)

    assert 3 == len(output.prediction.size())
    assert nb_agent == output.prediction.size()[0]
    assert batch_size == output.prediction.size()[1]
    assert nb_class == output.prediction.size()[2]

    assert 2 == len(output.actions_log_probs.size())
    assert nb_agent == output.actions_log_probs.size()[0]
    assert batch_size == output.actions_log_probs.size()[1]


def test_detailed_episode(
    batch_size: int,
    marl_m: MultiAgent,
    env: Environment,
    step: int,
    nb_class: int,
    nb_agent: int,
    dim: int,
    height_width: Tuple[int, int],
) -> None:
    x = th.randn(batch_size, 1, *height_width)
    episode_sampler = EpisodeSampler(marl_m, env)

    output = episode_sampler.run_episode(x, step)

    assert 4 == len(output.step_preds.size())
    assert step == output.step_preds.size()[0]
    assert nb_agent == output.step_preds.size()[1]
    assert batch_size == output.step_preds.size()[2]
    assert nb_class == output.step_preds.size()[3]

    assert 3 == len(output.step_log_probas.size())
    assert step == output.step_log_probas.size()[0]
    assert nb_agent == output.step_log_probas.size()[1]
    assert batch_size == output.step_log_probas.size()[2]

    assert 3 == len(output.step_values.size())
    assert step == output.step_values.size()[0]
    assert nb_agent == output.step_values.size()[1]
    assert batch_size == output.step_values.size()[2]

    assert 4 == len(output.step_pos.size())
    assert step == output.step_pos.size()[0]
    assert nb_agent == output.step_pos.size()[1]
    assert batch_size == output.step_pos.size()[2]
    assert dim == output.step_pos.size()[3]
