import unittest

import torch as th

from marl_classification.environment import detailed_episode, obs_generic, trans_generic
from marl_classification.environment import MultiAgent
from marl_classification.networks import ModelsWrapper


class TestEpisode(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.__batch_size = 19
        self.__nb_agent = 5
        self.__nb_class = 10
        self.__step = 7
        self.__dim = 2

        action = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ]

        n_b = 23
        n_a = 22
        n_m = 21

        model_wrapper = ModelsWrapper(
            "mnist", 12, n_b, n_a, n_m, 20, self.__dim,
            action, self.__nb_class, 24, 25
        )

        self.__marl = MultiAgent(
            self.__nb_agent, model_wrapper,
            n_b, n_a, 12, n_m, action,
            obs_generic, trans_generic
        )

    def test_episode(self):
        x = th.randn(
            self.__batch_size, 1,
            28, 28
        )

        pred, log_proba, values, pos = detailed_episode(
            self.__marl, x, self.__step, "cpu", self.__nb_class
        )

        assert pred.size()[0] == self.__step
        assert pred.size()[1] == self.__nb_agent
        assert pred.size()[2] == self.__batch_size
        assert pred.size()[3] == self.__nb_class

        assert log_proba.size()[0] == self.__step
        assert log_proba.size()[1] == self.__nb_agent
        assert log_proba.size()[2] == self.__batch_size

        assert values.size()[0] == self.__step
        assert values.size()[1] == self.__nb_agent
        assert values.size()[2] == self.__batch_size

        assert pos.size()[0] == self.__step
        assert pos.size()[1] == self.__nb_agent
        assert pos.size()[2] == self.__batch_size
        assert pos.size()[3] == self.__dim
