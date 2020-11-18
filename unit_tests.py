import argparse

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as tr
from torchnet.meter import ConfusionMeter

from tqdm import tqdm

from data.dataset import RESISC45Dataset
from environment.agent import MultiAgent
from environment.core import episode
from environment.observation import obs_2d_img
from environment.transition import trans_2d_img
from networks.ft_extractor import TestRESISC45
from networks.models import MNISTModelWrapper

import data.transforms as custom_tr
from utils import prec_rec


def test_mnist_transition():
    """
    TODO

    :return:
    :rtype:
    """
    a_1 = th.tensor([[[1., 0.]]])
    a_2 = th.tensor([[[0., 1.]]])
    a_3 = th.tensor([[[-1., 0.]]])
    a_4 = th.tensor([[[0., -1.]]])

    print("First test :")
    pos_1 = th.tensor([[[0., 0.]]])
    print(trans_2d_img(pos_1, a_1, 5, 28))
    print(trans_2d_img(pos_1, a_2, 5, 28))
    print(trans_2d_img(pos_1, a_3, 5, 28))
    print(trans_2d_img(pos_1, a_4, 5, 28))
    print()

    print("Snd test")
    pos_2 = th.tensor([[[22., 22.]]])
    print(trans_2d_img(pos_2, a_1, 5, 28))
    print(trans_2d_img(pos_2, a_2, 5, 28))
    print(trans_2d_img(pos_2, a_3, 5, 28))
    print(trans_2d_img(pos_2, a_4, 5, 28))


def test_mnist_obs():
    """
    TODO

    :return:
    :rtype:
    """
    img = th.arange(0, 28).view(1, 1, 28).repeat(2, 28, 1).unsqueeze(1).cuda()

    pos = th.tensor(
        [[[0, 0], [0, 0]],
         [[0, 4], [0, 4]],
         [[4, 0], [4, 0]],
         [[4, 4], [4, 4]],
         [[0 + 1, 0 + 1], [0 + 1, 0 + 1]],
         [[0 + 1, 4 + 1], [0 + 1, 4 + 1]],
         [[4 + 1, 0 + 1], [4 + 1, 0 + 1]],
         [[4 + 1, 4 + 1], [4 + 1, 4 + 1]]]
    ).cuda()

    print(obs_2d_img(img, pos, 6))
    print(obs_2d_img(img.permute(0, 1, 3, 2), pos, 6))
    print()

    for p in [[[0, 0]], [[0, 27]], [[27, 0]], [[27, 27]]]:
        try:
            print(obs_2d_img(img, th.tensor([p]).cuda(), 4))
            print(f"Test failed with pos = {p}")
        except Exception as e:
            print(e)


def test_agent_step():
    """
    TODO

    :return:
    :rtype:
    """

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    # d = 2
    action_size = 2
    batch_size = 1

    m = MNISTModelWrapper(f, n, n_m, 64)

    marl_m = MultiAgent(
        3, m, n, f, n_m,
        action_size,
        obs_2d_img, trans_2d_img
    )

    m.cuda()
    marl_m.cuda()

    img = th.rand(batch_size, 28, 28, device=th.device("cuda"))

    marl_m.new_episode(batch_size, 28)

    print("First step")
    print(marl_m.pos)
    print(marl_m.msg[0])

    marl_m.step(img)

    print("Second step")
    print(marl_m.pos)
    print(marl_m.msg[1])

    marl_m.step(img)

    print("Third step")
    print(marl_m.pos)
    print(marl_m.msg[2])


def test_core_step():
    """
    TODO

    :return:
    :rtype:
    """

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    d = 2
    action_size = 4

    batch_size = 2

    m = MNISTModelWrapper(f, n, n_m, 64)
    m.cuda()
    marl_m = MultiAgent(
        3, m, n, f, n_m,
        action_size,
        obs_2d_img, trans_2d_img
    )
    marl_m.cuda()

    img = th.rand(batch_size, 28, 28)
    c = th.zeros(batch_size, 10)
    c[:, 5] = 1

    optim = th.optim.SGD(m.parameters(), lr=1e-4)

    nb_epoch = 10

    for e in range(nb_epoch):
        pred, proba = episode(marl_m, img, 5)

        r = -(pred - c) ** 2
        r = r.mean(dim=-1)

        loss = th.log(proba) * r.detach() + r
        loss = -loss.sum() / batch_size

        optim.zero_grad()
        loss.backward()
        optim.step()

    for n in m.get_networks():
        if hasattr(n, 'seq_lin'):
            if n.seq_lin[0].weight.grad is None:
                print(n)
            else:
                print(n.seq_lin[0].weight.grad)
        elif hasattr(n, "lstm"):
            if n.lstm.weight_hh_l0.grad is None:
                print(n)
            else:
                print(n.lstm.weight_hh_l0.grad)


######################
# CNN test
######################

def test_cnn():
    """
    TODO

    :return:
    :rtype:
    """

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.MinMaxNorm()
    ])
    dataset = RESISC45Dataset(img_transform=img_pipeline)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    m = TestRESISC45()
    mse = nn.MSELoss()

    m.cuda()
    mse.cuda()

    optim = th.optim.Adam(m.parameters(), lr=1e-4)

    nb_epoch = 10

    i = 0
    for e in range(nb_epoch):
        sum_loss = 0

        m.train()

        conf_meter = ConfusionMeter(45)
        th.autograd.set_detect_anomaly(True)

        tqdm_bar = tqdm(dataloader)
        for x, y in tqdm_bar:
            x, y = x.cuda(), y.cuda()

            pred = m(x)

            y = y.view(-1, 1).repeat(1, 16 * 16).view(-1)
            conf_meter.add(pred.detach(), y)
            precs, recs = prec_rec(conf_meter)

            losses = th.pow(pred - th.eye(45)[y].cuda(), 2.).mean(dim=-1)

            optim.zero_grad()

            losses.mean().backward(retain_graph=True)

            # print("CNN_el = %d, grad_norm = %f" % (m.seq_lin[0].weight.grad.nelement(), m.seq_lin[0].weight.grad.norm()))

            sum_loss += losses.detach()[0].item()

            i += 1

            tqdm_bar.set_description(
                f"Epoch {e}, loss = {sum_loss / i:.4f}, "
                f"prec = {precs.mean():.4f}, "
                f"rec = {recs.mean():.4f}"
            )

            optim.step()

    return m.seq_conv


def main() -> None:
    parser = argparse.ArgumentParser("Unit test")

    choice_aux_subparser = parser.add_subparsers()
    choice_aux_subparser.required = True
    choice_aux_subparser.dest = "aux_choice"

    # aux parsers
    unit_test_parser = choice_aux_subparser.add_parser("unit")
    test_cnn_parser = choice_aux_subparser.add_parser("cnn")

    ##################
    # Unit test args
    ##################
    unit_test_choices = [
        "transition", "observation",
        "agent-step", "core-step"
    ]
    unit_test_parser.add_argument(
        "-t", "--test-id", type=str, choices=unit_test_choices,
        default=unit_test_choices[0], dest="test_id"
    )

    ##################
    # CNN test args
    ##################

    # TODO

    # parse args
    args = parser.parse_args()

    if args.aux_choice == "unit":
        # Choose between one of basic test
        if args.test_id == "transition":
            test_mnist_transition()
        elif args.test_id == "observation":
            test_mnist_obs()
        elif args.test_id == "agent-step":
            test_agent_step()
        elif args.test_id == "core-step":
            test_core_step()
        else:
            parser.error(f"Unrecognized unit test ID : "
                         f"\"{args.test_id}\", choices = {unit_test_choices}.")
    if args.aux_choice == "cnn":
        print(test_cnn())
        pass
    else:
        parser.error(
            f"Unrecognized mode : \"{args.mode}\" type == {type(args.mode)}.")


if __name__ == '__main__':
    main()
