from environment.observation import obs_img
from environment.transition import trans_img
from environment.agent import MultiAgent
from environment.core import episode, episode_retry

from networks.models import ModelsWrapper, MNISTModelWrapper, \
    RESISC45ModelsWrapper

from data.dataset import ImageFolder, MNISTDataset, RESISC45Dataset, \
    DATASET_CHOICES
import data.transforms as custom_tr

from utils import RLOptions, MAOptions, TrainOptions, TestOptions, \
    visualize_steps, prec_rec, format_metric, SetAppendAction

import torch as th
import torch.nn.functional as fun
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as tr

from torchnet.meter import ConfusionMeter

from random import randint, shuffle

import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

from os import mkdir, makedirs
from os.path import join, exists, isdir, isfile

import sys

import argparse


######################
# Train - Main
######################

def train(
        ma_options: MAOptions,
        rl_option: RLOptions,
        train_options: TrainOptions
) -> None:
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
    if exists(join(output_dir, model_dir)) \
            and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\""
                        f"is not a directory.")

    logs_file = open(join(output_dir, "train.log"), "w")
    args_str = " ".join([a for a in sys.argv])
    logs_file.write(args_str + "\n\n")
    logs_file.flush()

    ops_to_skip = train_options.frozen_modules

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    dataset = None
    nn_models = None

    if train_options.data_set == "mnist":
        dataset = MNISTDataset(img_pipeline)

        nn_models = MNISTModelWrapper(
            ma_options.window_size,
            rl_option.hidden_size,
            rl_option.hidden_size_msg,
            rl_option.hidden_size_linear,
            rl_option.hidden_size_state
        )

    elif train_options.data_set == "resisc45":
        dataset = RESISC45Dataset(img_pipeline)

        nn_models = RESISC45ModelsWrapper(
            ma_options.window_size,
            rl_option.hidden_size,
            rl_option.hidden_size_msg,
            rl_option.hidden_size_linear,
            rl_option.hidden_size_state
        )

    else:
        print(f"Unrecognized data set \"{train_options.data_set}\"")
        exit(1)

    marl_m = MultiAgent(
        ma_options.nb_agent, nn_models, rl_option.hidden_size,
        ma_options.window_size, rl_option.hidden_size_msg,
        ma_options.nb_action, obs_img, trans_img
    )

    cuda = rl_option.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    module_to_train = ModelsWrapper.module_list \
        .difference(train_options.frozen_modules)
    # for RL agent models parameters
    optim = th.optim.Adam(nn_models.get_params(list(module_to_train)),
                          lr=train_options.learning_rate)

    idx = list(range(len(dataset)))
    shuffle(idx)
    idx_train = idx[:int(0.85 * len(idx))]
    idx_test = idx[int(0.85 * len(idx)):]

    train_dataset = Subset(dataset, idx_train)
    test_dataset = Subset(dataset, idx_test)

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=8, drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=8, drop_last=False
    )

    loss_v = []
    prec_epoch = []
    recall_epoch = []

    for e in range(train_options.nb_epoch):
        train_ep_st = datetime.datetime.now()

        sum_loss = 0

        nn_models.train()

        conf_meter = ConfusionMeter(ma_options.nb_class)

        i = 0
        tqdm_bar = tqdm(train_dataloader)
        for x_train, y_train in tqdm_bar:
            x_train, y_train = x_train.to(th.device(device_str)), \
                               y_train.to(th.device(device_str))

            # pred = [Nr, Nb, Nc]
            # prob = [Nr, Nb]
            retry_pred, retry_prob = episode_retry(
                marl_m, x_train, rl_option.nb_step,
                train_options.retry_number,
                ma_options.nb_class, device_str
            )

            # Class one hot encoding
            y_eye = th.eye(
                ma_options.nb_class,
                device=th.device(device_str)
            )[y_train.unsqueeze(0)]

            # pass to class proba (softmax)
            retry_pred = fun.softmax(retry_pred, dim=-1)

            # Update confusion meter
            # mean between trials
            conf_meter.add(
                fun.softmax(retry_pred.detach().mean(dim=0), dim=-1),
                y_train
            )

            # L2 Loss - Classification error / reward
            # reward = -error(y_true, y_step_pred).mean(class_dim)
            r = -th.pow(y_eye - retry_pred, 2.).mean(dim=-1)

            # Compute loss
            losses = retry_prob * r.detach() + r

            # Losses mean on images batch and trials
            # maximize(E[reward]) -> minimize(-E[reward])
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
            precs, recs = prec_rec(conf_meter)

            # verify some parameters are un-optimized - test
            param_list = nn_models.get_params(ops_to_skip)
            mean_param_list_str = ", ".join(
                [f'{p.norm():.0e}' for p in param_list]
            )

            tqdm_bar.set_description(
                f"Epoch {e} - Train, "
                f"loss = {sum_loss / (i + 1):.4f}, "
                f"train_prec = {precs.mean():.4f}, "
                f"train_rec = {recs.mean():.4f}, "
                f"frozen_params = [{mean_param_list_str}]"
            )

            i += 1

        precs, recs = prec_rec(conf_meter)

        precs_str = format_metric(precs, dataset.class_to_idx)
        recs_str = format_metric(recs, dataset.class_to_idx)

        sum_loss /= len(train_dataloader)

        elapsed_time = datetime.datetime.now() - train_ep_st
        logs_file.write(
            f"#############################################\n"
            f"Epoch {e} - Train - Loss = {sum_loss:.4f}\n"
            f"train_prec = mean([{precs_str}]) = {precs.mean() * 100.:.1f}%\n"
            f"train_rec = mean([{recs_str}]) = {recs.mean() * 100.:.1f}%\n"
            f"elapsed_time = {elapsed_time.seconds // 60 // 60}h "
            f"{elapsed_time.seconds // 60 % 60}min "
            f"{elapsed_time.seconds % 60}s\n"
        )
        logs_file.flush()

        nn_models.eval()
        conf_meter.reset()

        train_ep_st = datetime.datetime.now()

        with th.no_grad():
            tqdm_bar = tqdm(test_dataloader)
            for x_test, y_test in tqdm_bar:
                x_test, y_test = x_test.to(th.device(device_str)), \
                                 y_test.to(th.device(device_str))

                preds, _ = episode(marl_m, x_test, rl_option.nb_step)

                preds = fun.softmax(preds, dim=-1)

                conf_meter.add(preds.detach(), y_test)

                # Compute score
                precs, recs = prec_rec(conf_meter)

                tqdm_bar.set_description(
                    f"Epoch {e} - Eval, "
                    f"eval_prec = {precs.mean():.4f}, "
                    f"eval_rec = {recs.mean():.4f}"
                )

        # Compute score
        precs, recs = prec_rec(conf_meter)

        precs_str = format_metric(precs, dataset.class_to_idx)
        recs_str = format_metric(recs, dataset.class_to_idx)

        elapsed_time = datetime.datetime.now() - train_ep_st
        logs_file.write(
            f"#############################################\n"
            f"Epoch {e} - Eval\n"
            f"eval_prec = mean([{precs_str}]) = {precs.mean() * 100.:.1f}%\n"
            f"eval_rec = mean([{recs_str}]) = {recs.mean() * 100.:.1f}%\n"
            f"elapsed_time = {elapsed_time.seconds // 60 // 60}h "
            f"{elapsed_time.seconds // 60 % 60}min "
            f"{elapsed_time.seconds % 60}s\n\n"
        )
        logs_file.flush()

        prec_epoch.append(precs.mean())
        recall_epoch.append(recs.mean())
        loss_v.append(sum_loss)

        nn_models.json_args(
            join(output_dir,
                 model_dir,
                 f"marl_epoch_{e}.json")
        )
        th.save(nn_models.state_dict(),
                join(output_dir, model_dir, f"nn_models_epoch_{e}.pt"))
        th.save(optim.state_dict(),
                join(output_dir, model_dir, f"optim_epoch_{e}.pt"))

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

    empty_pipe = tr.Compose([
        tr.ToTensor()
    ])

    dataset_tmp = None

    if train_options.data_set == "mnist":
        dataset_tmp = MNISTDataset(empty_pipe)

    elif train_options.data_set == "resisc45":
        dataset_tmp = RESISC45Dataset(empty_pipe)

    else:
        print(f"Unrecognized data set \"{train_options.data_set}\"")
        exit(1)

    test_dataloader_ori = Subset(dataset_tmp, idx_test)
    test_dataloader = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataloader_ori))

    visualize_steps(marl_m, test_dataloader[test_idx][0],
                    test_dataloader_ori[test_idx][0],
                    rl_option.nb_step, ma_options.window_size,
                    output_dir, ma_options.nb_class, device_str,
                    dataset.class_to_idx)

    logs_file.close()


#######################
# Test - Main
#######################

def test(ma_options: MAOptions,
         rl_options: RLOptions,
         test_options: TestOptions) -> None:
    steps = rl_options.nb_step

    json_path = test_options.json_path
    state_dict_path = test_options.state_dict_path
    image_path = test_options.image_path
    output_img_path = test_options.output_img_path
    nb_test_img = test_options.nb_test_img

    assert exists(json_path), \
        f"JSON path \"{json_path}\" does not exist"
    assert isfile(json_path), \
        f"\"{json_path}\" is not a file"

    assert exists(state_dict_path), \
        f"State dict path \"{state_dict_path}\" does not exist"
    assert isfile(state_dict_path), \
        f"\"{state_dict_path}\" is not a file"

    if exists(output_img_path) and isdir(output_img_path):
        print(f"File in {output_img_path} will be overwrite")
    elif exists(output_img_path) and not isdir(output_img_path):
        raise Exception(f"\"{output_img_path}\" is not a directory")
    else:
        print(f"Create \"{output_img_path}\"")
        mkdir(output_img_path)

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    img_dataset = ImageFolder(image_path, transform=img_pipeline)

    idx = list(range(len(img_dataset)))
    shuffle(idx)
    idx_test = idx[int(0.85 * len(idx)):]

    test_dataset = Subset(img_dataset, idx_test)

    nn_models = ModelsWrapper.from_json(json_path)
    nn_models.load_state_dict(th.load(state_dict_path))
    marl_m = MultiAgent.load_from(
        json_path, ma_options.nb_agent,
        nn_models, obs_img, trans_img
    )

    data_loader = DataLoader(
        test_dataset, batch_size=test_options.batch_size,
        shuffle=True, num_workers=8, drop_last=False
    )

    cuda = rl_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    conf_meter = ConfusionMeter(nn_models.nb_class)

    for x, y in tqdm(data_loader):
        x, y = x.to(th.device(device_str)), y.to(th.device(device_str))

        preds, probas = episode(marl_m, x, steps)

        preds = fun.softmax(preds, dim=-1)

        conf_meter.add(preds.detach(), y)

    print(conf_meter.value())

    precs, recs = prec_rec(conf_meter)

    precs_str = format_metric(precs, img_dataset.class_to_idx)
    recs_str = format_metric(recs, img_dataset.class_to_idx)

    print(f"Precision : {precs_str}")
    print(f"Precision mean = {precs.mean()}")
    print(f"Recall : {recs_str}")
    print(f"Recall mean : {recs.mean()}")

    img_ori_pipeline = tr.Compose([
        tr.ToTensor()
    ])

    img_dataset_ori = ImageFolder(image_path, transform=img_ori_pipeline)
    test_dataset_ori = Subset(img_dataset_ori, idx_test)

    rand_idx = list(range(len(test_dataset_ori)))
    shuffle(rand_idx)
    rand_idx = rand_idx[:nb_test_img]

    idx_to_class = {img_dataset.class_to_idx[k]: k
                    for k in img_dataset.class_to_idx}

    for i in tqdm(rand_idx):
        x, y = test_dataset[i]
        x_ori, y_ori = test_dataset_ori[i]

        x, x_ori = x.to(th.device(device_str)), \
                   x_ori.to(th.device(device_str))

        curr_img_path = join(output_img_path, f"img_{i}_{idx_to_class[y]}")

        if not exists(curr_img_path):
            mkdir(curr_img_path)

        visualize_steps(
            marl_m, x, x_ori, steps, nn_models.f,
            curr_img_path,
            nn_models.nb_class, device_str,
            img_dataset_ori.class_to_idx
        )


#######################
# Main script function
#######################

def main() -> None:
    """
    TODO

    :return:
    :rtype:
    """
    main_parser = argparse.ArgumentParser(
        "Multi agent reinforcement learning "
        "for image classification - Main"
    )

    # prgm subparsers creation
    # main subparser
    choice_main_subparser = main_parser.add_subparsers()
    choice_main_subparser.dest = "main_choice"
    choice_main_subparser.required = True

    # main parsers
    train_parser = choice_main_subparser.add_parser("train")
    test_parser = choice_main_subparser.add_parser("infer")

    ##################
    # Main args
    ##################

    # Algorithm arguments
    main_parser.add_argument(
        "-a", "--agents", type=int, default=3, dest="agents",
        help="Number of agents"
    )

    # data option
    main_parser.add_argument(
        "--dataset", type=str, choices=DATASET_CHOICES, default="mnist",
        dest="dataset", help="Choose the training data set"
    )

    # Image / data set arguments
    main_parser.add_argument(
        "--nb-class", type=int, default=10, dest="nb_class",
        help="Image dataset number of class"
    )
    main_parser.add_argument(
        "--img-size", type=int, default=28, dest="img_size",
        help="Image side size, assume all image are squared"
    )

    # RL Options
    main_parser.add_argument(
        "--batch-size", type=int, default=8, dest="batch_size",
        help="Image batch size for training and evaluation"
    )
    main_parser.add_argument(
        "--step", type=int, default=7,
        help="Step number of RL episode"
    )
    main_parser.add_argument(
        "--cuda", action="store_true", dest="cuda",
        help="Train NNs with CUDA"
    )

    ##################
    # Train args
    ##################

    # Data options
    train_parser.add_argument(
        "--nb-action", type=int, default=4, dest="nb_action",
        help="Number of discrete actions"
    )

    # Algorithm arguments
    train_parser.add_argument(
        "-d", "--dim", type=int, default=2,
        help="State dimension (eg. 2 -> move on a plan)"
    )
    train_parser.add_argument(
        "--f", type=int, default=7,
        help="Window size"
    )

    # RL Options
    train_parser.add_argument(
        "--n", type=int, default=64,
        help="Hidden size for NNs"
    )
    train_parser.add_argument(
        "--nm", type=int, default=16, dest="n_m",
        help="Message size for NNs"
    )
    train_parser.add_argument(
        "--nd", type=int, default=4, dest="n_d",
        help="State hidden size"
    )
    train_parser.add_argument(
        "--nl", type=int, default=128, dest="n_l",
        help="Network internal hidden size for linear projections"
    )

    # Training arguments
    train_parser.add_argument(
        "-o", "--output-dir", type=str, required=True, dest="output_dir",
        help="The output dir containing res and models per epoch. "
             "Created if needed."
    )
    train_parser.add_argument(
        "--lr", "--learning-rate", type=float, default=1e-3,
        dest="learning_rate",
        help=""
    )
    train_parser.add_argument(
        "--nb-epoch", type=int, default=10, dest="nb_epoch",
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--nr", "--number-retry", type=int, default=7, dest="number_retry",
        help="Number of retry to estimate expectation."
    )
    train_parser.add_argument(
        "--freeze", type=str, default=[], nargs="+",
        dest="frozen_modules", action=SetAppendAction,
        choices=[
            ModelsWrapper.map_obs, ModelsWrapper.map_pos,
            ModelsWrapper.evaluate_msg,  # ModelsWrapper.decode_msg,
            ModelsWrapper.belief_unit, ModelsWrapper.action_unit,
            ModelsWrapper.predict, ModelsWrapper.policy],
        help="Choose module(s) to be frozen during training"
    )

    ##################
    # Infer args
    ##################
    test_parser.add_argument(
        "-i", "--image-path", type=str, required=True, dest="image_path",
        help="Input image path for inference"
    )
    test_parser.add_argument(
        "--json-path", type=str, required=True, dest="json_path",
        help="JSON multi agent metadata path"
    )
    test_parser.add_argument(
        "--state-dict", type=str, required=True, dest="state_dict",
        help="networks.models.ModelsWrapper PyTorch state dict file"
    )
    test_parser.add_argument(
        "-o", "--output-image-dir", type=str, required=True,
        dest="output_image_dir",
        help="The directory where the model outputs will be saved. "
             "Created if needed"
    )
    test_parser.add_argument(
        "--nb-test-img", type=int, default=10, dest="nb_test_img",
        help="The number of test image to infer and output"
    )

    ###################################
    # Main - start different mods
    ###################################

    args = main_parser.parse_args()

    # Unit tests main
    if args.main_choice == "train":
        # Create Options
        rl_options = RLOptions(
            args.step,
            args.n,
            args.n_l,
            args.n_m,
            args.n_d,
            args.cuda
        )

        ma_options = MAOptions(
            args.agents,
            args.dim,
            args.f,
            args.img_size,
            args.nb_class,
            args.nb_action
        )

        train_options = TrainOptions(
            args.nb_epoch,
            args.learning_rate,
            args.number_retry,
            args.batch_size,
            args.output_dir,
            args.frozen_modules,
            args.dataset
        )

        if not exists(args.output_dir):
            makedirs(args.output_dir)
        if exists(args.output_dir) and not isdir(args.output_dir):
            raise Exception(f"\"{args.output_dir}\" is not a directory.")

        train(ma_options, rl_options, train_options)

    # Test main
    elif args.prgm == "main" and args.main_choice == "infer":
        rl_options = RLOptions(
            args.step, -1,
            -1, -1, -1,
            args.cuda
        )

        ma_options = MAOptions(
            args.agents, -1, -1,
            args.img_size, -1, -1
        )

        test_options = TestOptions(
            args.json_path,
            args.state_dict,
            args.image_path,
            args.output_image_dir,
            args.nb_test_img,
            args.batch_size
        )

        test(ma_options, rl_options, test_options)

    else:
        main_parser.error(
            f"Unrecognized mode : \"{args.mode}\""
            f"type == {type(args.mode)}."
        )


if __name__ == "__main__":
    main()
