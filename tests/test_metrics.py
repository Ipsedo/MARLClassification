import shutil
import unittest
from os import mkdir
from os.path import abspath, exists, isdir, join

import torch as th

from marl_classification.metrics import ConfusionMeter, LossMeter


class TestMetrics(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.__tmp_path = abspath(join(__file__, "..", "tmp"))

        if not exists(self.__tmp_path):
            mkdir(self.__tmp_path)
        elif not isdir(self.__tmp_path):
            self.fail(f"\"{self.__tmp_path}\" is not a directory")

    def tearDown(self) -> None:
        super().tearDown()

        shutil.rmtree(self.__tmp_path)

    def test_confusion(self) -> None:
        nb_class = int(th.randint(2, 32, (1,)).item())
        y_pred = th.eye(nb_class).to(th.float)

        # error at first index of batch
        y_pred[0, 0] = 0.
        y_pred[0, 1] = 1.

        y_true = th.arange(0, nb_class)

        conf_meter = ConfusionMeter(nb_class, None)

        conf_meter.add(y_pred, y_true)

        conf_mat = conf_meter.conf_mat()

        self.assertEqual(0, conf_mat[0, 0])
        self.assertEqual(1, conf_mat[0, 1])

        self.assertTrue((th.diag(conf_mat)[1:] == 1).all())
        self.assertEqual(nb_class, conf_mat.sum())

        try:
            conf_meter.save_conf_matrix(0, self.__tmp_path, "unittest")
        except Exception as e:
            self.fail(str(e))

    def test_loss(self):
        loss_meter = LossMeter(None)

        for v in [0.5, 0.25, 0.75, 0.5]:
            loss_meter.add(v)

        self.assertEqual(0.5, loss_meter.loss())
