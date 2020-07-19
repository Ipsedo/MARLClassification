from environment.observation import obs_MNIST
from environment.transition import trans_MNIST
from environment.agent import Agent
from environment.core import step, detailled_step

from networks.models import ModelsUnion
from networks.ft_extractor import TestCNN

from data.mnist import load_mnist

import torch as th
from torch.nn import Softmax, MSELoss, NLLLoss

from math import ceil
from random import randint

import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse


def test_mnist_transition():
    """
    TODO

    :return:
    :rtype:
    """
    a_1 = th.tensor([1., 0.])
    a_2 = th.tensor([0., 1.])
    a_3 = th.tensor([-1., 0.])
    a_4 = th.tensor([0., -1.])

    print("First test :")
    pos_1 = th.tensor([[0., 0.]])
    print(trans_MNIST(pos_1, a_1, 5, 28))
    print(trans_MNIST(pos_1, a_2, 5, 28))
    print(trans_MNIST(pos_1, a_3, 5, 28))
    print(trans_MNIST(pos_1, a_4, 5, 28))
    print()

    print("Snd test")
    pos_2 = th.tensor([[22., 22.]])
    print(trans_MNIST(pos_2, a_1, 5, 28))
    print(trans_MNIST(pos_2, a_2, 5, 28))
    print(trans_MNIST(pos_2, a_3, 5, 28))
    print(trans_MNIST(pos_2, a_4, 5, 28))


def test_mnist_obs():
    """
    TODO

    :return:
    :rtype:
    """
    img = th.arange(0, 28 * 28).view(1, 28, 28)

    print(img)
    print()

    pos = th.tensor([[2, 2]])

    print(obs_MNIST(img, pos, 4))
    print()

    try:
        pos_fail = th.tensor([[-1., -1.]])
        print(obs_MNIST(img, pos_fail, 4))
    except Exception as e:
        print(e)

    try:
        pos_fail = th.tensor([[24., 26.]])
        print(obs_MNIST(img, pos_fail, 4))
    except Exception as e:
        print(e)


def test_agent_step():
    """
    TODO

    :return:
    :rtype:
    """
    ag = []

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    d = 2
    action_size = 2
    batch_size = 1

    m = ModelsUnion(n, f, n_m, d, action_size, nb_class)

    a1 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    img = th.rand(batch_size, 28, 28)

    for a in ag:
        a.step(img, True)
    for a in ag:
        a.step_finished()

    print(a1.get_t_msg())


def test_core_step():
    """
    TODO

    :return:
    :rtype:
    """
    ag = []

    nb_class = 10
    img_size = 28
    n = 16
    f = 5
    n_m = 8
    d = 2
    action_size = 4

    batch_size = 2

    m = ModelsUnion(n, f, n_m, d, action_size, nb_class)

    a1 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, m, n, f, n_m, img_size, action_size, batch_size, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    img = th.rand(batch_size, 28, 28)
    c = th.zeros(batch_size, 10)
    c[:, 5] = 1

    sm = Softmax(dim=1)

    mse = MSELoss()

    params = []
    for n in m.get_networks():
        params += n.parameters()
    optim = th.optim.SGD(params, lr=1e-4)

    nb_epoch = 10

    for e in range(nb_epoch):
        optim.zero_grad()
        pred, proba = step(ag, img, 5, sm, False, False, 10)
        r = mse(pred, c)

        Nr = 1
        loss = (th.log(proba) * r.detach() + r) / Nr
        loss = loss.sum() / batch_size

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

def test_mnist():
    """
    TODO

    :return:
    :rtype:
    """
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()

    m = TestCNN(10)
    mse = MSELoss()

    m.cuda()
    mse.cuda()

    optim = th.optim.SGD(m.parameters(), lr=1e-2)

    batch_size = 64
    nb_epoch = 10

    nb_batch = ceil(x_train.size(0) / batch_size)

    for e in range(nb_epoch):
        sum_loss = 0

        m.train()

        for i in tqdm(range(nb_batch)):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < x_train.size(0) else x_train.size(0)
            x, y = x_train[i_min:i_max, :, :].cuda(), y_train[i_min:i_max].cuda()

            pred = m(x)

            loss = mse(pred, th.eye(10)[y].cuda())

            optim.zero_grad()
            loss.backward()

            #print("CNN_el = %d, grad_norm = %f" % (m.seq_lin[0].weight.grad.nelement(), m.seq_lin[0].weight.grad.norm()))

            optim.step()

            sum_loss += loss.item()
        print("Epoch %d, loss = %f" % (e, sum_loss / nb_batch))

        with th.no_grad():
            nb_batch_valid = ceil(x_valid.size(0) / batch_size)
            nb_correct = 0
            for i in tqdm(range(nb_batch_valid)):
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_valid.size(0) else x_valid.size(0)

                x, y = x_valid[i_min:i_max, :, :].cuda(), y_valid[i_min:i_max].cuda()

                pred = m(x)

                nb_correct += (pred.argmax(dim=1) == y).sum().item()

            nb_correct /= x_valid.size(0)
            print("Epoch %d, accuracy = %f" % (e, nb_correct))
    return m.seq_conv


######################
# Train - Main
######################

def train_mnist(nb_class: int, img_size: int,
                n: int, f: int, n_m: int,
                d: int, nb_action: int, batch_size: int, t: int,
                nr: int,
                nb_epoch: int, cuda: bool):
    """
    TODO

    :param nb_class:
    :type nb_class:
    :param img_size:
    :type img_size:
    :param n:
    :type n:
    :param f:
    :type f:
    :param n_m:
    :type n_m:
    :param d:
    :type d:
    :param nb_action:
    :type nb_action:
    :param batch_size:
    :type batch_size:
    :param t:
    :type t:
    :param nr:
    :type nr:
    :param nb_epoch:
    :type nb_epoch:
    :param cuda:
    :type cuda:
    :return:
    :rtype:
    """

    ag = []

    #m = ModelsUnion(n, f, n_m, d, nb_action, nb_class, test_mnist())
    m = ModelsUnion(n, f, n_m, d, nb_action, nb_class)

    a1 = Agent(ag, m, n, f, n_m, img_size, nb_action, batch_size, obs_MNIST, trans_MNIST)
    a2 = Agent(ag, m, n, f, n_m, img_size, nb_action, batch_size, obs_MNIST, trans_MNIST)
    a3 = Agent(ag, m, n, f, n_m, img_size, nb_action, batch_size, obs_MNIST, trans_MNIST)

    ag.append(a1)
    ag.append(a2)
    ag.append(a3)

    # for agents hidden tensors (belief etc.)
    if cuda:
        for a in ag:
            a.cuda()

    sm = Softmax(dim=-1)

    criterion = MSELoss()
    if cuda:
        criterion.cuda()

    # for RL agent models parameters
    params = []
    for net in m.get_networks():
        if cuda:
            net.cuda()
        params += list(net.parameters())

    optim = th.optim.Adam(params, lr=1e-3)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    x_train, y_train = x_train[:10000], y_train[:10000]

    nb_batch = ceil(x_train.size(0) / batch_size)

    loss_v = []
    acc = []

    for e in range(nb_epoch):
        sum_loss = 0

        for net in m.get_networks():
            net.train()

        grad_norm_cnn = []
        grad_norm_pred = []

        random_walk = False

        tqdm_bar = tqdm(range(nb_batch))
        for i in tqdm_bar:
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < x_train.size(0) else x_train.size(0)

            losses = []

            for k in range(nr):

                x, y = x_train[i_min:i_max, :, :], y_train[i_min:i_max]

                if cuda:
                    x, y = x.cuda(), y.cuda()

                pred, log_probas = step(ag, x, t, sm, cuda, random_walk, nb_class)

                # Sum on agent dimension
                proba_per_image = log_probas.sum(dim=0)

                y_eye = th.eye(nb_class)[y]
                if cuda:
                    y_eye = y_eye.cuda()

                r = -criterion(pred, y_eye)

                # Mean on image batch
                l = (proba_per_image * r.detach() + r).mean(dim=0).view(-1)

                losses.append(l)

            loss = -th.cat(losses).sum() / nr

            optim.zero_grad()
            loss.backward()
            optim.step()

            sum_loss += loss.item()

            grad_norm_cnn.append(m.get_networks()[0].seq_lin[0].weight.grad.norm())
            grad_norm_pred.append(m.get_networks()[-1].seq_lin[0].weight.grad.norm())

            tqdm_bar.set_description(f"Epoch {e}, Loss = {sum_loss / (i + 1):.4f}, "
                                     f"grad_cnn_norm_mean = {sum(grad_norm_cnn) / len(grad_norm_cnn):.6f}, "
                                     f"grad_pred_norm_mean = {sum(grad_norm_pred) / len(grad_norm_pred):.6f}, "
                                     f"CNN_el = {m.get_networks()[0].seq_lin[0].weight.grad.nelement()}, "
                                     f"Pred_el = {m.get_networks()[-1].seq_lin[0].weight.grad.nelement()}")

        sum_loss /= nb_batch

        nb_correct = 0

        nb_batch_valid = ceil(x_valid.size(0) / batch_size)

        for net in m.get_networks():
            net.eval()

        with th.no_grad():
            tqdm_bar = tqdm(range(nb_batch_valid))
            for i in tqdm_bar:
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_valid.size(0) else x_valid.size(0)

                x, y = x_valid[i_min:i_max, :, :].cuda(), y_valid[i_min:i_max].cuda()

                pred, proba = step(ag, x, t, sm, cuda, random_walk, nb_class)

                nb_correct += (pred.argmax(dim=1) == y).sum().item()

                tqdm_bar.set_description(f"Epoch {e}, accuracy = {nb_correct / i_max}")

        acc.append(nb_correct)
        loss_v.append(sum_loss)

    plt.plot(acc, "b", label="accuracy")
    plt.plot(loss_v, "r", label="criterion value")
    plt.xlabel("Epoch")
    plt.title("MARL Classification f=%d, n=%d, n_m=%d, d=%d, T=%d" % (f, n, n_m, d, t))
    plt.legend()
    plt.show()

    viz(ag, x_test[randint(0, x_test.size(0)-1)], t, sm, f)


def viz(agents: list, one_img: th.Tensor, max_it: int, softmax: Softmax, f: int) -> None:
    """
    TODO

    :param agents:
    :type agents:
    :param one_img:
    :type one_img:
    :param max_it:
    :type max_it:
    :param softmax:
    :type softmax:
    :param f:
    :type f:
    :return:
    :rtype:
    """
    pred, pos = detailled_step(agents, one_img.unsqueeze(0).cuda(), max_it, softmax, True, 10)

    plt.imshow(one_img)
    plt.show()

    print(pos)

    tmp = th.zeros(28, 28) - 1
    for t in range(max_it):

        for i in range(len(agents)):
            tmp[pos[i][t][0]:pos[i][t][0]+f, pos[i][t][1]:pos[i][t][1]+f] = \
                one_img[pos[i][t][0]:pos[i][t][0]+f, pos[i][t][1]:pos[i][t][1]+f]

        plt.imshow(tmp, cmap='gray_r')
        plt.title("Step = %d" % t)
        plt.show()


def main() -> None:
    """
    TODO

    :return:
    :rtype:
    """
    parser = argparse.ArgumentParser("Multi agent reinforcement learning for image classification - Main")

    sub_parser = parser.add_subparsers()
    sub_parser.dest = "mode"
    sub_parser.required = True

    unit_test_parser = sub_parser.add_parser("unit")
    train_parser = sub_parser.add_parser("train")
    test_parser = sub_parser.add_parser("test")
    test_cnn_parser = sub_parser.add_parser("cnn")

    ##################
    # Unit test args
    ##################
    unit_test_choices = ["transition", "observation", "agent-step", "core-step"]
    unit_test_parser.add_argument("-t", "--test-id", type=str, choices=unit_test_choices,
                                  default=unit_test_choices[0], dest="test_id")

    ##################
    # Train args
    ##################

    # MNIST default params

    # Image / data set arguments
    train_parser.add_argument("--nb-class", type=int, default=10, dest="nb_class")
    train_parser.add_argument("--img-size", type=int, default=28, dest="img_size")

    # Algorithm arguments
    train_parser.add_argument("--n", type=int, default=8)
    train_parser.add_argument("--f", type=int, default=7)
    train_parser.add_argument("--nm", type=int, default=2, dest="n_m")

    # Environment arguments
    train_parser.add_argument("-d", "--dim", type=int, default=2)
    train_parser.add_argument("--nb-action", type=int, default=4, dest="nb_action")

    # Training arguments
    train_parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    train_parser.add_argument("--step", type=int, default=7)
    train_parser.add_argument("--nb-epoch", type=int, default=10, dest="nb_epoch")
    train_parser.add_argument("--cuda", action="store_true", dest="cuda")

    train_parser.add_argument("--nr", type=int, default=7)

    ##################
    # Test args
    ##################

    # TODO

    ##################
    # CNN test args
    ##################

    # TODO

    ###################################
    # Main - start different mods
    ###################################

    args = parser.parse_args()

    # Unit tests main
    if args.mode == "unit":
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
            parser.error(f"Unrecognized unit test ID : \"{args.test_id}\", choices = {unit_test_choices}.")
    # Train main
    elif args.mode == "train":
        train_mnist(args.nb_class, args.img_size,
                    args.n, args.f, args.n_m,
                    args.dim, args.nb_action,
                    args.batch_size, args.step,
                    args.nr,
                    args.nb_epoch, args.cuda)
    # Test main
    elif args.mode == "test":
        pass
    # CNN test main
    elif args.mode == "cnn":
        print(test_mnist())
        pass
    else:
        parser.error(f"Unrecognized mode : \"{args.mode}\" type == {type(args.mode)}.")


if __name__ == "__main__":
    main()
