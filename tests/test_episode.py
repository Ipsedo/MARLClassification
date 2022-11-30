import torch as th

from marl_classification.environment import MultiAgent, detailed_episode


def test_episode(
    batch_size: int,
    marl_m: MultiAgent,
    step: int,
    nb_class: int,
    nb_agent: int,
    dim: int,
) -> None:
    x = th.randn(batch_size, 1, 28, 28)

    pred, log_proba, values, pos = detailed_episode(
        marl_m,
        x,
        step,
        "cpu",
        nb_class,
    )

    assert 4 == len(pred.size())
    assert step == pred.size()[0]
    assert nb_agent == pred.size()[1]
    assert batch_size == pred.size()[2]
    assert nb_class == pred.size()[3]

    assert 3 == len(log_proba.size())
    assert step == log_proba.size()[0]
    assert nb_agent == log_proba.size()[1]
    assert batch_size == log_proba.size()[2]

    assert 3 == len(values.size())
    assert step == values.size()[0]
    assert nb_agent == values.size()[1]
    assert batch_size == values.size()[2]

    assert 4 == len(pos.size())
    assert step == pos.size()[0]
    assert nb_agent == pos.size()[1]
    assert batch_size == pos.size()[2]
    assert dim == pos.size()[3]
