import pytest
import torch as th

from marl_classification.metrics import ConfusionMeter, LossMeter


def test_confusion(tmp_path: str) -> None:
    nb_class = int(th.randint(2, 32, (1,)).item())
    y_pred = th.eye(nb_class).to(th.float)

    # error at first index of batch
    y_pred[0, 0] = 0.0
    y_pred[0, 1] = 1.0

    y_true = th.arange(0, nb_class)

    conf_meter = ConfusionMeter(nb_class, None)

    conf_meter.add(y_pred, y_true)

    conf_mat = conf_meter.conf_mat()

    assert 0 == conf_mat[0, 0]
    assert 1 == conf_mat[0, 1]

    assert (th.diag(conf_mat)[1:] == 1).all()
    assert nb_class == conf_mat.sum()

    try:
        conf_meter.save_conf_matrix(0, tmp_path, "unittest")
    except Exception as e:
        pytest.fail(str(e))


def test_loss():
    loss_meter = LossMeter(None)

    for v in [0.5, 0.25, 0.75, 0.5]:
        loss_meter.add(v)

    assert 0.5 == loss_meter.loss()
