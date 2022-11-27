import unittest

import torch as th

from marl_classification.environment import detailed_episode
from tests.get_model import GetModel


class TestEpisode(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.__model = GetModel()

    def test_episode(self):
        x = th.randn(
            self.__model.batch_size, 1,
            28, 28
        )

        pred, log_proba, values, pos = detailed_episode(
            self.__model.marl,
            x,
            self.__model.step,
            "cpu",
            self.__model.nb_class
        )

        assert len(pred.size()) == 4
        assert pred.size()[0] == self.__model.step
        assert pred.size()[1] == self.__model.nb_agent
        assert pred.size()[2] == self.__model.batch_size
        assert pred.size()[3] == self.__model.nb_class

        assert len(log_proba.size()) == 3
        assert log_proba.size()[0] == self.__model.step
        assert log_proba.size()[1] == self.__model.nb_agent
        assert log_proba.size()[2] == self.__model.batch_size

        assert len(values.size()) == 3
        assert values.size()[0] == self.__model.step
        assert values.size()[1] == self.__model.nb_agent
        assert values.size()[2] == self.__model.batch_size

        assert len(pos.size()) == 4
        assert pos.size()[0] == self.__model.step
        assert pos.size()[1] == self.__model.nb_agent
        assert pos.size()[2] == self.__model.batch_size
        assert pos.size()[3] == self.__model.dim
