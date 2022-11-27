import unittest

import torch as th

from marl_classification.metrics import ConfusionMeter, LossMeter


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_confusion(self) -> None:
        nb_class = int(th.randint(2, 32, (1,)).item())
        y_pred = th.eye(nb_class).to(th.float)

        # error at first batch
        y_pred[0, 0] = 0.
        y_pred[0, 1] = 1.

        y_true = th.arange(0, nb_class)

        conf_meter = ConfusionMeter(nb_class, None)

        conf_meter.add(y_pred, y_true)

        conf_mat = conf_meter.conf_mat()

        assert conf_mat[0, 0] == 0
        assert conf_mat[0, 1] == 1

        assert (th.diag(conf_mat)[1:] == 1).all()
        assert conf_mat.sum() == nb_class

    def test_loss(self):
        loss_meter = LossMeter(None)

        for v in [0.5, 0.25, 0.75, 0.5]:
            loss_meter.add(v)

        assert loss_meter.loss() == 0.5
