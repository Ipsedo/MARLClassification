import unittest

import torch as th

from marl_classification.environment import detailed_episode
from tests.get_model import GetModel


class TestEpisode(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.__model = GetModel()

    def test_episode(self):
        x = th.randn(self.__model.batch_size, 1, 28, 28)

        pred, log_proba, values, pos = detailed_episode(
            self.__model.marl,
            x,
            self.__model.step,
            "cpu",
            self.__model.nb_class,
        )

        self.assertEqual(4, len(pred.size()))
        self.assertEqual(self.__model.step, pred.size()[0])
        self.assertEqual(self.__model.nb_agent, pred.size()[1])
        self.assertEqual(self.__model.batch_size, pred.size()[2])
        self.assertEqual(self.__model.nb_class, pred.size()[3])

        self.assertEqual(3, len(log_proba.size()))
        self.assertEqual(self.__model.step, log_proba.size()[0])
        self.assertEqual(self.__model.nb_agent, log_proba.size()[1])
        self.assertEqual(self.__model.batch_size, log_proba.size()[2])

        self.assertEqual(3, len(values.size()))
        self.assertEqual(self.__model.step, values.size()[0])
        self.assertEqual(self.__model.nb_agent, values.size()[1])
        self.assertEqual(self.__model.batch_size, values.size()[2])

        self.assertEqual(4, len(pos.size()))
        self.assertEqual(self.__model.step, pos.size()[0])
        self.assertEqual(self.__model.nb_agent, pos.size()[1])
        self.assertEqual(self.__model.batch_size, pos.size()[2])
        self.assertEqual(self.__model.dim, pos.size()[3])
