from networks.models import RESISC45ModelsWrapper
from environment.agent import MultiAgent
from environment.core import episode
from utils import MAOptions, RLOptions, viz

import torch as th
from torchnet.meter import ConfusionMeter

from os import mkdir
from os.path import join, exists, isdir

from typing import AnyStr

from math import ceil
from random import randint

import matplotlib.pyplot as plt

from tqdm import tqdm


# TODO transition, observation and data loading
# TODO tester genericité du code x)
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

    nn_models = RESISC45ModelsWrapper(ma_options.window_size,
                                  rl_option.hidden_size,
                                  rl_option.hidden_size_msg)

    marl_m = MultiAgent(ma_options.nb_agent,
                        nn_models,
                        rl_option.hidden_size,
                        ma_options.window_size,
                        rl_option.hidden_size_msg,
                        ma_options.img_size,
                        ma_options.nb_action,
                        None, None)

    cuda = rl_option.cuda

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()

    # for RL agent models parameters
    optim = th.optim.Adam(nn_models.parameters(), lr=rl_option.learning_rate)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = None

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