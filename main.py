from environment.observation import obs_MNIST
from environment.transition import trans_MNIST
from environment.agent import MultiAgent
from environment.core import episode, detailled_step

from networks.models import ModelsWrapper
from networks.ft_extractor import TestCNN

from data.mnist import load_mnist

import torch as th
from torch.nn import MSELoss
from torchnet.meter import ConfusionMeter

from math import ceil
from random import randint

import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import NamedTuple, AnyStr, Optional

from os import mkdir
from os.path import join, exists, isdir

import argparse


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
    print(trans_MNIST(pos_1, a_1, 5, 28))
    print(trans_MNIST(pos_1, a_2, 5, 28))
    print(trans_MNIST(pos_1, a_3, 5, 28))
    print(trans_MNIST(pos_1, a_4, 5, 28))
    print()

    print("Snd test")
    pos_2 = th.tensor([[[22., 22.]]])
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
    img = th.arange(0, 28).view(1, 1, 28).repeat(2, 28, 1).cuda()

    pos = th.tensor([[[0, 0]], [[0, 4]], [[4, 0]], [[4, 4]], [[0 + 1, 0 + 1]], [[0 + 1, 4 + 1]], [[4 + 1, 0 + 1]], [[4 + 1, 4 + 1]]]).cuda()

    print(obs_MNIST(img, pos, 6))
    print(obs_MNIST(img.permute(0, 2, 1), pos, 6))
    print()

    for p in [[[0, 0]], [[0, 27]], [[27, 0]], [[27, 27]]]:
        try:
            print(obs_MNIST(img, th.tensor([p]).cuda(), 4))
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
    d = 2
    action_size = 2
    batch_size = 1

    m = ModelsWrapper(n, f, n_m, d, action_size, nb_class)

    marl_m = MultiAgent(3, m, n, f, n_m, img_size, action_size, obs_MNIST, trans_MNIST)

    m.cuda()
    marl_m.cuda()

    img = th.rand(batch_size, 28, 28, device=th.device("cuda"))

    marl_m.new_episode(batch_size)

    print("First step")
    print(marl_m.pos)
    print(marl_m.msg[0])

    marl_m.step(img, 0.5)

    print("Second step")
    print(marl_m.pos)
    print(marl_m.msg[1])

    marl_m.step(img, 0.5)

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

    m = ModelsWrapper(n, f, n_m, d, action_size, nb_class)
    m.cuda()
    marl_m = MultiAgent(3, m, n, f, n_m, img_size, action_size, obs_MNIST, trans_MNIST)
    marl_m.cuda()

    img = th.rand(batch_size, 28, 28)
    c = th.zeros(batch_size, 10)
    c[:, 5] = 1

    optim = th.optim.SGD(m.parameters(), lr=1e-4)

    nb_epoch = 10

    for e in range(nb_epoch):
        pred, proba = episode(marl_m, img, 5, True, 0.5, 10)

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

MAOptions = NamedTuple("MAOption",
                       [("nb_agent", int),
                        ("dim", int),
                        ("window_size", int),
                        ("img_size", int),
                        ("nb_class", int),
                        ("nb_action", int)])

RLOptions = NamedTuple("RLOptions",
                       [("eps", float),
                        ("eps_decay", float),
                        ("nb_step", int),
                        ("nb_epoch", int),
                        ("learning_rate", float),
                        ("hidden_size", int),
                        ("hidden_size_msg", int),
                        ("batch_size", int),
                        ("cuda", bool)])


def train_mnist(ma_options: MAOptions, rl_option: RLOptions, output_dir: AnyStr) -> None:
    """

    :param ma_options:
    :type ma_options:
    :param rl_option:
    :type rl_option:
    :param output_dir:
    :type output_dir:
    :return:
    :rtype:
    """

    model_dir = "models"
    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\" is not a directory.")

    nn_models = ModelsWrapper(rl_option.hidden_size,
                              ma_options.window_size,
                              rl_option.hidden_size_msg,
                              ma_options.dim,
                              ma_options.nb_action,
                              ma_options.nb_class)

    marl_m = MultiAgent(ma_options.nb_agent,
                        nn_models,
                        rl_option.hidden_size,
                        ma_options.window_size,
                        rl_option.hidden_size_msg,
                        ma_options.img_size,
                        ma_options.nb_action,
                        obs_MNIST, trans_MNIST)

    cuda = rl_option.cuda

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()

    # for RL agent models parameters
    optim = th.optim.Adam(nn_models.parameters(), lr=rl_option.learning_rate)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()

    nb_batch = ceil(x_train.size(0) / rl_option.batch_size)

    loss_v = []
    prec_epoch = []
    recall_epoch = []

    eps = rl_option.eps
    eps_decay = rl_option.eps_decay

    for e in range(rl_option.nb_epoch):
        sum_loss = 0

        nn_models.train()

        conf_meter = ConfusionMeter(ma_options.nb_class)

        tqdm_bar = tqdm(range(nb_batch))
        for i in tqdm_bar:
            i_min = i * rl_option.batch_size
            i_max = (i + 1) * rl_option.batch_size
            i_max = i_max if i_max < x_train.size(0) else x_train.size(0)

            x, y = x_train[i_min:i_max, :, :].to(th.device("cuda") if cuda else th.device("cpu")),\
                   y_train[i_min:i_max].to(th.device("cuda") if cuda else th.device("cpu"))

            # get predictions and probabilities
            preds, log_probas = episode(marl_m, x, rl_option.nb_step,
                                        cuda, eps, ma_options.nb_class)

            # Class one hot encoding
            y_eye = th.eye(ma_options.nb_class,
                           device=th.device("cuda") if cuda else th.device("cpu"))[y]\
                .repeat(preds.size(0), 1, 1)

            # Update confusion meter
            conf_meter.add(preds.mean(dim=0).view(-1, ma_options.nb_class).detach(),
                           y.detach())

            # SE Loss
            r = -(preds - y_eye) ** 2

            # Mean on one hot encoding
            r = r.mean(dim=-1)

            # Keep loss for future update
            losses = log_probas * r.detach() + r

            eps *= eps_decay

            # Losses mean on agents and batch
            loss = -losses.mean()

            # Reset gradient
            optim.zero_grad()

            # Backward on compute graph
            loss.backward()

            # Update weights
            optim.step()

            # Update epoch loss sum
            sum_loss += loss.item()

            # Compute score
            conf_mat = conf_meter.value()

            lissage = 1e-30  # 10^(-38) -> float64

            prec = th.tensor(
                [(conf_mat[i, i] / (conf_mat[:, i].sum() + lissage)).item()
                 for i in range(ma_options.nb_class)]).mean()

            rec = th.tensor(
                [(conf_mat[i, i] / (conf_mat[i, :].sum() + lissage)).item()
                 for i in range(ma_options.nb_class)]).mean()

            tqdm_bar.set_description(f"Epoch {e} eps({eps:.5f}), "
                                     f"Loss = {sum_loss / (i + 1):.4f}, "
                                     f"train_prec = {prec:.4f}, "
                                     f"train_rec = {rec:.4f}")

        sum_loss /= nb_batch

        nb_batch_valid = ceil(x_valid.size(0) / rl_option.batch_size)

        nn_models.eval()
        conf_meter.reset()

        with th.no_grad():
            tqdm_bar = tqdm(range(nb_batch_valid))
            for i in tqdm_bar:
                i_min = i * rl_option.batch_size
                i_max = (i + 1) * rl_option.batch_size
                i_max = i_max if i_max < x_valid.size(0) else x_valid.size(0)

                x, y = x_valid[i_min:i_max, :, :].to(th.device("cuda") if cuda else th.device("cpu")),\
                       y_valid[i_min:i_max].to(th.device("cuda") if cuda else th.device("cpu"))

                preds, proba = episode(marl_m, x, rl_option.nb_step, cuda, eps, ma_options.nb_class)

                conf_meter.add(preds.mean(dim=0).view(-1, ma_options.nb_class).detach(),
                               y.detach())

                # Compute score
                conf_mat = conf_meter.value()
                lissage = 1e-30  # 10^(-38) -> float64
                prec = th.tensor([(conf_mat[i, i] / (conf_mat[:, i].sum() + lissage)).item()
                                  for i in range(ma_options.nb_class)]).mean()
                rec = th.tensor([(conf_mat[i, i] / (conf_mat[i, :].sum() + lissage)).item()
                                 for i in range(ma_options.nb_class)]).mean()

                tqdm_bar.set_description(f"Epoch {e}, eval_prec = {prec:.4f}, eval_rec = {rec:.4f}")

        # Compute score
        conf_mat = conf_meter.value()
        prec = th.tensor([conf_mat[i, i] / conf_mat[:, i].sum()
                          for i in range(ma_options.nb_class)]).mean()
        rec = th.tensor([conf_mat[i, i] / conf_mat[i, :].sum()
                         for i in range(ma_options.nb_class)]).mean()

        prec_epoch.append(prec)
        recall_epoch.append(rec)
        loss_v.append(sum_loss)

        marl_m.params_to_json(join(output_dir, model_dir, f"marl_epoch_{e}.json"))
        th.save(nn_models.state_dict(), join(output_dir, model_dir, f"nn_models_epoch_{e}.pt"))
        th.save(optim.state_dict(), join(output_dir, model_dir, f"optim_epoch_{e}.pt"))

    plt.figure()
    plt.plot(prec_epoch, "b", label="precision - mean (eval)")
    plt.plot(recall_epoch, "r", label="recall - mean (eval)")
    plt.plot(loss_v, "g", label="criterion value")
    plt.xlabel("Epoch")
    plt.title(f"MARL Classification f={ma_options.window_size}, "
              f"n={rl_option.hidden_size}, n_m={rl_option.hidden_size_msg}, "
              f"d={ma_options.dim}, T={rl_option.nb_step}")

    plt.legend()
    plt.savefig(join(output_dir, "train_graph.png"))

    viz(marl_m, x_test[randint(0, x_test.size(0) - 1)],
        rl_option.nb_step, ma_options.window_size,
        output_dir)


def viz(agents: MultiAgent, one_img: th.Tensor,
        max_it: int, f: int, output_dir: AnyStr) -> None:
    """

    :param agents:
    :type agents:
    :param one_img:
    :type one_img:
    :param max_it:
    :type max_it:
    :param f:
    :type f:
    :param output_dir:
    :type output_dir:
    :return:
    :rtype:
    """

    preds, _, pos = detailled_step(agents, one_img.unsqueeze(0).cuda(),
                                   max_it, True, 10)

    img_idx = 0

    plt.figure()
    plt.imshow(one_img, cmap='gray_r')
    plt.savefig(join(output_dir, f"pred_original.png"))

    curr_img = th.zeros(28, 28) - 1
    for t in range(max_it):

        for i in range(len(agents)):
            curr_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                     pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f] = \
                one_img[pos[t][i][img_idx][0]:pos[t][i][img_idx][0] + f,
                        pos[t][i][img_idx][1]:pos[t][i][img_idx][1] + f]

        plt.figure()
        plt.imshow(curr_img, cmap='gray_r')
        prediction = preds[t].mean(dim=0)[img_idx].argmax(dim=-1)
        pred_proba = preds[t].mean(dim=0)[img_idx][prediction]
        plt.title(f"Step = {t}, step_pred_class = {prediction} ({pred_proba * 100.:.1f}%)")

        plt.savefig(join(output_dir, f"pred_step_{t}.png"))


#######################
# Main script function
#######################

def main() -> None:
    """
    TODO

    :return:
    :rtype:
    """
    parser = argparse.ArgumentParser("Multi agent reinforcement learning "
                                     "for image classification - Main")

    sub_parser = parser.add_subparsers()
    sub_parser.dest = "mode"
    sub_parser.required = True

    unit_test_parser = sub_parser.add_parser("unit")
    train_parser = sub_parser.add_parser("train")
    test_parser = sub_parser.add_parser("infer")
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
    train_parser.add_argument("-o", "--output-dir", type=str, required=True, dest="output_dir",
                              help="The output dir containing res and models per epoch. "
                                   "Created if needed.")

    # Image / data set arguments
    train_parser.add_argument("--nb-class", type=int, default=10, dest="nb_class",
                              help="Image dataset number of class")
    train_parser.add_argument("--img-size", type=int, default=28, dest="img_size",
                              help="Image side size, assume all image are squared")

    # Algorithm arguments
    train_parser.add_argument("-a", "--agents", type=int, default=3, dest="agents",
                              help="Number of agents")
    train_parser.add_argument("--n", type=int, default=8)
    train_parser.add_argument("--f", type=int, default=7)
    train_parser.add_argument("--nm", type=int, default=2, dest="n_m")

    # Environment arguments
    train_parser.add_argument("-d", "--dim", type=int, default=2,
                              help="State dimension (eg. 2 -> move on a plan)")
    train_parser.add_argument("--nb-action", type=int, default=4, dest="nb_action",
                              help="Number of discrete actions")

    # Training arguments
    train_parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate",
                              help="")
    train_parser.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                              help="Image batch size for training and evaluation")
    train_parser.add_argument("--step", type=int, default=7,
                              help="Step number of RL episode")
    train_parser.add_argument("--nb-epoch", type=int, default=10, dest="nb_epoch",
                              help="Number of training epochs")
    train_parser.add_argument("--cuda", action="store_true", dest="cuda",
                              help="Train NNs with CUDA")
    train_parser.add_argument("--nr", type=int, default=7,
                              help="Number of retry")
    train_parser.add_argument("--eps", type=float, default=0.7, dest="eps",
                              help="Epsilon value at training beginning")
    train_parser.add_argument("--eps-decay", type=float, default=1.0 - 4e-5, dest="eps_decay",
                              help="Epsilon decay, update epsilon after each episode : "
                                   "eps = eps * eps_decay")

    ##################
    # Infer args
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
            parser.error(f"Unrecognized unit test ID : "
                         f"\"{args.test_id}\", choices = {unit_test_choices}.")
    # Train main
    elif args.mode == "train":
        rl_options = RLOptions(args.eps, args.eps_decay,
                               args.step, args.nb_epoch, args.learning_rate,
                               args.n, args.n_m,
                               args.batch_size, args.cuda)

        ma_options = MAOptions(args.agents, args.dim, args.f, args.img_size,
                               args.nb_class, args.nb_action)

        if not exists(args.output_dir):
            mkdir(args.output_dir)
        if exists(args.output_dir) and not isdir(args.output_dir):
            raise Exception(f"\"{args.output_dir}\" is not a directory.")

        train_mnist(ma_options, rl_options, args.output_dir)

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
