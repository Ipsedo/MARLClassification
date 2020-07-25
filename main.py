from environment.observation import obs_img
from environment.transition import trans_img
from environment.agent import MultiAgent
from environment.core import episode

from networks.models import ModelsWrapper, MNISTModelWrapper, RESISC45ModelsWrapper
from networks.ft_extractor import TestCNN

from data.mnist import load_mnist
from data.resisc45 import load_resisc45

from utils import RLOptions, MAOptions, TrainOptions, TestOptions,\
    visualize_steps, prec_rec, format_metric, SetAppendAction

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchnet.meter import ConfusionMeter

from math import ceil
from random import randint

import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from os import mkdir, makedirs
from os.path import join, exists, isdir

import sys

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
    print(trans_img(pos_1, a_1, 5, 28))
    print(trans_img(pos_1, a_2, 5, 28))
    print(trans_img(pos_1, a_3, 5, 28))
    print(trans_img(pos_1, a_4, 5, 28))
    print()

    print("Snd test")
    pos_2 = th.tensor([[[22., 22.]]])
    print(trans_img(pos_2, a_1, 5, 28))
    print(trans_img(pos_2, a_2, 5, 28))
    print(trans_img(pos_2, a_3, 5, 28))
    print(trans_img(pos_2, a_4, 5, 28))


def test_mnist_obs():
    """
    TODO

    :return:
    :rtype:
    """
    img = th.arange(0, 28).view(1, 1, 28).repeat(2, 28, 1).unsqueeze(1).cuda()

    pos = th.tensor([[[0, 0], [0, 0]],
                     [[0, 4], [0, 4]],
                     [[4, 0], [4, 0]],
                     [[4, 4], [4, 4]],
                     [[0 + 1, 0 + 1], [0 + 1, 0 + 1]],
                     [[0 + 1, 4 + 1], [0 + 1, 4 + 1]],
                     [[4 + 1, 0 + 1], [4 + 1, 0 + 1]],
                     [[4 + 1, 4 + 1], [4 + 1, 4 + 1]]]).cuda()

    print(obs_img(img, pos, 6))
    print(obs_img(img.permute(0, 1, 3, 2), pos, 6))
    print()

    for p in [[[0, 0]], [[0, 27]], [[27, 0]], [[27, 27]]]:
        try:
            print(obs_img(img, th.tensor([p]).cuda(), 4))
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

    m = MNISTModelWrapper(f, n, n_m)

    marl_m = MultiAgent(3, m, n, f, n_m, img_size, action_size, obs_img, trans_img)

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

    m = MNISTModelWrapper(f, n, n_m)
    m.cuda()
    marl_m = MultiAgent(3, m, n, f, n_m, img_size, action_size, obs_img, trans_img)
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

def test_mnist():
    """
    TODO

    :return:
    :rtype:
    """
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()

    m = TestCNN(10)
    mse = nn.MSELoss()

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

def train_mnist(ma_options: MAOptions, rl_option: RLOptions, train_options: TrainOptions) -> None:
    """

    :param ma_options:
    :type ma_options:
    :param rl_option:
    :type rl_option:
    :param train_options:
    :type train_options:
    :return:
    :rtype:
    """

    output_dir = train_options.output_model_path

    model_dir = "models"
    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\" is not a directory.")

    logs_file = open(join(output_dir, "train.log"), "w")
    args_str = " ".join([a for a in sys.argv])
    logs_file.write(args_str + "\n\n")
    logs_file.flush()

    ops_to_skip = train_options.frozen_modules

    (x, y), class_map = (th.zeros(1), th.tensor(1)), {}

    nn_models = None

    if train_options.data_set == "mnist":
        (x, y), class_map = load_mnist()

        nn_models = MNISTModelWrapper(ma_options.window_size,
                                      rl_option.hidden_size,
                                      rl_option.hidden_size_msg)
    elif train_options.data_set == "resisc45":
        (x, y), class_map = load_resisc45()

        nn_models = RESISC45ModelsWrapper(ma_options.window_size,
                                          rl_option.hidden_size,
                                          rl_option.hidden_size_msg)
    else:
        print(f"Unrecognized data set \"{train_options.data_set}\"")
        exit(1)

    marl_m = MultiAgent(ma_options.nb_agent,
                        nn_models,
                        rl_option.hidden_size,
                        ma_options.window_size,
                        rl_option.hidden_size_msg,
                        ma_options.img_size,
                        ma_options.nb_action,
                        obs_img, trans_img)

    criterion = nn.CrossEntropyLoss(reduction='none')

    cuda = rl_option.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    # for RL agent models parameters
    optim = th.optim.Adam(nn_models.parameters(), lr=train_options.learning_rate)

    limit_train = int(0.7 * x.size(0))
    limit_eval = int(0.85 * x.size(0))

    x_train, y_train = x[:limit_train, :, :, :], y[:limit_train]
    x_valid, y_valid = x[limit_train:limit_eval, :, :, :], y[limit_train:limit_eval]
    x_test, y_test = x[limit_eval:, :, :, :], y[limit_eval:]

    batch_size = train_options.batch_size
    nb_batch = ceil(x_train.size(0) / batch_size)

    loss_v = []
    prec_epoch = []
    recall_epoch = []

    for e in range(train_options.nb_epoch):
        train_ep_st = datetime.datetime.now()

        sum_loss = 0

        nn_models.train()

        conf_meter = ConfusionMeter(ma_options.nb_class)

        tqdm_bar = tqdm(range(nb_batch))
        for i in tqdm_bar:
            # Get real batch indexes
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < x_train.size(0) else x_train.size(0)

            # Select batch images & class
            x, y = x_train[i_min:i_max, :, :].to(th.device(device_str)),\
                   y_train[i_min:i_max].to(th.device(device_str))

            # get predictions and probabilities
            preds, log_prob = episode(marl_m, x, rl_option.nb_step)

            # Class one hot encoding
            y_eye = th.eye(ma_options.nb_class,
                           device=th.device(device_str))[y]

            # Mean on all agents
            # then pass to class proba (softmax)
            preds = F.softmax(preds.mean(dim=0), dim=-1)

            # Update confusion meter
            conf_meter.add(preds.detach(), y.detach())

            # L2 Loss - Classification error / reward
            # reward = 1 - error(y_pred, y_true).mean(class_dim)
            r = 1. - th.pow(y_eye - preds, 2.).mean(dim=-1)

            # Compute loss
            losses = log_prob * r.detach() + r

            # Losses mean image batch
            # maximize(E[reward]) -> minimize(-E[reward])
            loss = -losses.mean()

            # Reset gradient
            optim.zero_grad()

            # Backward on compute graph
            loss.backward()

            # Erase gradient - test
            nn_models.erase_grad(ops_to_skip)

            # Update weights
            optim.step()

            # Update epoch loss sum
            sum_loss += loss.item()

            # Compute score
            precs, recs = prec_rec(conf_meter)

            # verify some parameters are un-optimized - test
            param_list = nn_models.get_params(ops_to_skip)
            mean_param_list_str = ", ".join([f'{p.mean():.4f}' for p in param_list])

            tqdm_bar.set_description(f"Epoch {e} - Train, "
                                     f"loss = {sum_loss / (i + 1):.4f}, "
                                     f"train_prec = {precs.mean():.4f}, "
                                     f"train_rec = {recs.mean():.4f}, "
                                     f"param_mean_dummy = [{mean_param_list_str}]")

        precs, recs = prec_rec(conf_meter)

        precs_str = format_metric(precs, class_map)
        recs_str = format_metric(recs, class_map)

        sum_loss /= nb_batch
        elapsed_time = datetime.datetime.now() - train_ep_st
        logs_file.write(f"#############################################\n"
                        f"Epoch {e} - Train - Loss = {sum_loss:.4f}\n"
                        f"train_prec = mean([{precs_str}]) = {precs.mean() * 100.:.1f}\n"
                        f"train_rec = mean([{recs_str}]) = {recs.mean() * 100.:.1f}\n"
                        f"elapsed_time = {elapsed_time.seconds // 60 // 60}h "
                        f"{elapsed_time.seconds // 60 % 60}min "
                        f"{elapsed_time.seconds % 60}s\n")
        logs_file.flush()

        nb_batch_valid = ceil(x_valid.size(0) / batch_size)

        nn_models.eval()
        conf_meter.reset()

        train_ep_st = datetime.datetime.now()

        with th.no_grad():
            tqdm_bar = tqdm(range(nb_batch_valid))
            for i in tqdm_bar:
                i_min = i * batch_size
                i_max = (i + 1) * batch_size
                i_max = i_max if i_max < x_valid.size(0) else x_valid.size(0)

                x, y = x_valid[i_min:i_max, :, :].to(th.device(device_str)),\
                       y_valid[i_min:i_max].to(th.device(device_str))

                preds, proba = episode(marl_m, x, rl_option.nb_step)

                preds = F.softmax(preds.mean(dim=0), dim=-1)

                conf_meter.add(preds.detach(), y.detach())

                # Compute score
                precs, recs = prec_rec(conf_meter)

                tqdm_bar.set_description(f"Epoch {e} - Eval, "
                                         f"eval_prec = {precs.mean():.4f}, "
                                         f"eval_rec = {recs.mean():.4f}")

        # Compute score
        precs, recs = prec_rec(conf_meter)

        precs_str = format_metric(precs, class_map)
        recs_str = format_metric(recs, class_map)

        elapsed_time = datetime.datetime.now() - train_ep_st
        logs_file.write(f"#############################################\n"
                        f"Epoch {e} - Eval\n"
                        f"eval_prec = mean([{precs_str}]) = {precs.mean() * 100.:.1f}\n"
                        f"eval_rec = mean([{recs_str}]) = {recs.mean() * 100.:.1f}\n"
                        f"elapsed_time = {elapsed_time.seconds // 60 // 60}h "
                        f"{elapsed_time.seconds // 60 % 60}min "
                        f"{elapsed_time.seconds % 60}s\n\n")
        logs_file.flush()

        prec_epoch.append(precs.mean())
        recall_epoch.append(recs.mean())
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

    visualize_steps(marl_m, x_test[randint(0, x_test.size(0) - 1)],
                    rl_option.nb_step, ma_options.window_size,
                    output_dir, ma_options.nb_class, cuda, class_map)

    logs_file.close()


#######################
# Test - Main
#######################

def test(ma_options: MAOptions, rl_options: RLOptions, test_options: TestOptions) -> None:
    pass


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

    # root subparser
    sub_parser = parser.add_subparsers()
    sub_parser.dest = "prgm"
    sub_parser.required = True

    # prgm parser
    main_parser = sub_parser.add_parser("main")
    aux_parser = sub_parser.add_parser("aux")

    # prgm subparsers creation
    # main subparser
    choice_main_subparser = main_parser.add_subparsers()
    choice_main_subparser.dest = "main_choice"
    choice_main_subparser.required = True

    # aux subparser
    choice_aux_subparser = aux_parser.add_subparsers()
    choice_aux_subparser.default = "unit"
    choice_aux_subparser.dest = "aux_choice"

    # main parsers
    train_parser = choice_main_subparser.add_parser("train")
    test_parser = choice_main_subparser.add_parser("infer")

    # aux parsers
    unit_test_parser = choice_aux_subparser.add_parser("unit")
    test_cnn_parser = choice_aux_subparser.add_parser("cnn")

    ##################
    # Unit test args
    ##################
    unit_test_choices = ["transition", "observation", "agent-step", "core-step"]
    unit_test_parser.add_argument("-t", "--test-id", type=str, choices=unit_test_choices,
                                  default=unit_test_choices[0], dest="test_id")

    ##################
    # Main args
    ##################

    # Algorithm arguments
    main_parser.add_argument("-a", "--agents", type=int, default=3, dest="agents",
                             help="Number of agents")
    main_parser.add_argument("-d", "--dim", type=int, default=2,
                             help="State dimension (eg. 2 -> move on a plan)")
    main_parser.add_argument("--f", type=int, default=7)

    # Image / data set arguments
    main_parser.add_argument("--nb-class", type=int, default=10, dest="nb_class",
                             help="Image dataset number of class")
    main_parser.add_argument("--nb-action", type=int, default=4, dest="nb_action",
                             help="Number of discrete actions")
    main_parser.add_argument("--img-size", type=int, default=28, dest="img_size",
                             help="Image side size, assume all image are squared")

    # RL Options
    main_parser.add_argument("--step", type=int, default=7,
                             help="Step number of RL episode")
    main_parser.add_argument("--n", type=int, default=8,
                             help="Hidden size for NNs")
    main_parser.add_argument("--nm", type=int, default=2, dest="n_m",
                             help="Message size for NNs")
    main_parser.add_argument("--cuda", action="store_true", dest="cuda",
                             help="Train NNs with CUDA")

    ##################
    # Train args
    ##################

    # Training arguments
    train_parser.add_argument("--dataset", type=str, choices=["mnist", "resisc45"], default="mnist",
                              dest="dataset", help="Choose the training data set")
    train_parser.add_argument("-o", "--output-dir", type=str, required=True, dest="output_dir",
                              help="The output dir containing res and models per epoch. "
                                   "Created if needed.")
    train_parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate",
                              help="")
    train_parser.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                              help="Image batch size for training and evaluation")
    train_parser.add_argument("--nb-epoch", type=int, default=10, dest="nb_epoch",
                              help="Number of training epochs")
    train_parser.add_argument("--nr", type=int, default=7,
                              help="Number of retry - unused")
    train_parser.add_argument("--freeze", type=str, default=[], nargs="+",
                              dest="frozen_modules", action=SetAppendAction,
                              choices=[ModelsWrapper.map_obs, ModelsWrapper.map_pos,
                                       ModelsWrapper.evaluate_msg, ModelsWrapper.decode_msg,
                                       ModelsWrapper.belief_unit, ModelsWrapper.action_unit,
                                       ModelsWrapper.predict, ModelsWrapper.policy],
                              help="Choose module(s) to be frozen during training")

    ##################
    # Infer args
    ##################
    test_parser.add_argument("-i", "--images", type=str, required=True, nargs="+", dest="images",
                             help="Input images for inference")
    test_parser.add_argument("--json-path", type=str, required=True, dest="json_path",
                             help="JSON multi agent metadata path")
    test_parser.add_argument("--state-dict", type=str, required=True, dest="state_dict",
                             help="networks.models.ModelsWrapper PyTorch state dict file")
    test_parser.add_argument("-o", "--output-image-dir", type=str, required=True, dest="output_image_dir",
                             help="The directory where the model outputs will be saved. Created if needed")
    test_parser.add_argument("--nb-test-img", type=str, default=10, dest="nb_test_img",
                             help="The number of test image to infer and output")

    ##################
    # CNN test args
    ##################

    # TODO

    ###################################
    # Main - start different mods
    ###################################

    args = parser.parse_args()

    # Unit tests main
    if args.prgm == "aux" and args.aux_choice == "unit":
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
    elif args.prgm == "main" and args.main_choice == "train":
        # Create Options
        rl_options = RLOptions(args.step, args.n, args.n_m, args.cuda)

        ma_options = MAOptions(args.agents, args.dim, args.f, args.img_size,
                               args.nb_class, args.nb_action)

        train_options = TrainOptions(args.nb_epoch, args.learning_rate,
                                     args.batch_size, args.output_dir,
                                     args.frozen_modules, args.dataset)

        if not exists(args.output_dir):
            makedirs(args.output_dir)
        if exists(args.output_dir) and not isdir(args.output_dir):
            raise Exception(f"\"{args.output_dir}\" is not a directory.")

        train_mnist(ma_options, rl_options, train_options)

    # Test main
    elif args.prgm == "main" and args.main_choice == "test":
        rl_options = RLOptions(args.step, args.n, args.n_m, args.cuda)

        ma_options = MAOptions(args.agents, args.dim, args.f, args.img_size,
                               args.nb_class, args.nb_action)

        test_options = TestOptions(args.json_path, args.state_dict,
                                   args.output_image_dir, args.nb_img_test)

        test(ma_options, rl_options, test_options)

    # CNN test main
    elif args.prgm == "aux" and args.mode == "cnn":
        print(test_mnist())
        pass
    else:
        parser.error(f"Unrecognized mode : \"{args.mode}\" type == {type(args.mode)}.")


if __name__ == "__main__":
    main()
