from environment import MultiAgent, \
    episode, episode_retry, \
    obs_img, trans_img

from networks.models import ModelsWrapper

from data.dataset import ImageFolder, MNISTDataset, RESISC45Dataset, \
    my_pil_loader
import data.transforms as custom_tr

from utils import MainOptions, TrainOptions, TestOptions, InferOptions, \
    visualize_steps, prec_rec, SetAppendAction, \
    format_metric, save_conf_matrix

import torch as th
import torch.nn.functional as fun
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as tr

from torchnet.meter import ConfusionMeter

import mlflow

from random import randint, shuffle

from tqdm import tqdm

import json

from os import mkdir, makedirs
from os.path import join, exists, isdir, isfile
import glob

import argparse


######################
# Train - Main
######################

def train(
        main_options: MainOptions,
        train_options: TrainOptions
) -> None:
    assert train_options.dim == 2, \
        "Only 2D is supported at the moment " \
        "for data loading and observation / transition. " \
        "See torchvision.datasets.ImageFolder"

    output_dir = train_options.output_dir

    model_dir = "models"
    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) \
            and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\""
                        f"is not a directory.")

    exp_name = "MARLClassification"
    mlflow.set_experiment(exp_name)

    mlflow.start_run(run_name="train")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    dataset_constructor = RESISC45Dataset \
        if train_options.ft_extr_str.startswith("resisc") \
        else MNISTDataset

    nn_models = ModelsWrapper(
        train_options.ft_extr_str,
        train_options.window_size,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.hidden_size_msg,
        train_options.hidden_size_state,
        train_options.dim,
        train_options.nb_action,
        train_options.nb_class,
        train_options.hidden_size_linear_belief,
        train_options.hidden_size_linear_action
    )

    dataset = dataset_constructor(img_pipeline)

    marl_m = MultiAgent(
        main_options.nb_agent,
        nn_models,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.window_size,
        train_options.hidden_size_msg,
        train_options.nb_action,
        obs_img,
        trans_img
    )

    mlflow.log_params({
        "ft_extractor": train_options.ft_extr_str,
        "window_size": train_options.window_size,
        "hidden_size_belief": train_options.hidden_size_belief,
        "hidden_size_action": train_options.hidden_size_action,
        "hidden_size_msg": train_options.hidden_size_msg,
        "hidden_size_state": train_options.hidden_size_state,
        "dim": train_options.dim,
        "nb_action": train_options.nb_action,
        "nb_class": train_options.nb_class,
        "hidden_size_linear_belief":
            train_options.hidden_size_linear_belief,
        "hidden_size_linear_action":
            train_options.hidden_size_linear_action,
        "nb_agent": main_options.nb_agent,
        "frozen_modules": train_options.frozen_modules,
        "epsilon": train_options.epsilon,
        "epsilon_decay": train_options.epsilon_decay,
        "nb_epoch": train_options.nb_epoch,
        "learning_rate": train_options.learning_rate,
        "img_size": train_options.img_size,
        "retry_number": train_options.retry_number,
        "step": main_options.step,
        "batch_size": train_options.batch_size
    })

    json_f = open(join(output_dir, "class_to_idx.json"), "w")
    json.dump(dataset.class_to_idx, json_f)
    json_f.close()
    mlflow.log_artifact(join(output_dir, "class_to_idx.json"))

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    mlflow.log_param("device", device_str)

    module_to_train = ModelsWrapper.module_list \
        .difference(train_options.frozen_modules)

    # for RL agent models parameters
    optim = th.optim.Adam(
        nn_models.get_params(list(module_to_train)),
        lr=train_options.learning_rate
    )

    idx = th.randperm(len(dataset))
    idx_train = idx[:int(0.85 * idx.size(0))]
    idx_test = idx[int(0.85 * idx.size(0)):]

    train_dataset = Subset(dataset, idx_train)
    test_dataset = Subset(dataset, idx_test)

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=3, drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=3, drop_last=False
    )

    epsilon = train_options.epsilon

    curr_step = 0

    for e in range(train_options.nb_epoch):
        nn_models.train()

        sum_loss = 0.
        i = 0

        conf_meter = ConfusionMeter(train_options.nb_class)

        tqdm_bar = tqdm(train_dataloader)
        for x_train, y_train in tqdm_bar:
            x_train, y_train = x_train.to(th.device(device_str)), \
                               y_train.to(th.device(device_str))

            # pred = [Nr, Ns, Nb, Nc]
            # prob = [Nr, Ns, Nb]
            retry_pred, retry_prob = episode_retry(
                marl_m, x_train, epsilon,
                main_options.step,
                train_options.retry_number,
                train_options.nb_class, device_str
            )

            # Class one hot encoding
            y_eye = th.eye(
                train_options.nb_class,
                device=th.device(device_str)
            )[y_train.unsqueeze(0)].unsqueeze(1).repeat(
                1, main_options.step, 1, 1)

            # Update confusion meter
            # mean between trials
            conf_meter.add(
                retry_pred.detach()[:, -1, :, :].mean(dim=0),
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

            # Compute global score
            precs, recs = prec_rec(conf_meter)

            if curr_step % 100 == 0:
                mlflow.log_metrics(
                    {"loss": loss.item(),
                     "train_prec": precs.mean().item(),
                     "train_rec": recs.mean().item(),
                     "epsilon": epsilon},
                    step=curr_step
                )

            tqdm_bar.set_description(
                f"Epoch {e} - Train, "
                f"loss = {sum_loss / (i + 1):.4f}, "
                f"eps = {epsilon:.4f}, "
                f"train_prec = {precs.mean():.3f}, "
                f"train_rec = {recs.mean():.3f}"
            )

            epsilon *= train_options.epsilon_decay
            epsilon = max(epsilon, 0.)

            i += 1
            curr_step += 1

        sum_loss /= len(train_dataloader)

        save_conf_matrix(conf_meter, e, output_dir, "train")

        mlflow.log_artifact(
            join(output_dir, f"confusion_matrix_epoch_{e}_train.png")
        )

        nn_models.eval()
        conf_meter.reset()

        with th.no_grad():
            tqdm_bar = tqdm(test_dataloader)
            for x_test, y_test in tqdm_bar:
                x_test, y_test = x_test.to(th.device(device_str)), \
                                 y_test.to(th.device(device_str))

                preds, _ = episode(marl_m, x_test, 0., main_options.step)

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

        save_conf_matrix(conf_meter, e, output_dir, "eval")

        mlflow.log_metrics(
            {"eval_prec": precs.mean(),
             "eval_recs": recs.mean()},
            step=curr_step
        )

        nn_models.json_args(
            join(output_dir,
                 model_dir,
                 f"marl_epoch_{e}.json")
        )
        th.save(
            nn_models.state_dict(),
            join(output_dir, model_dir,
                 f"nn_models_epoch_{e}.pt")
        )

        mlflow.log_artifact(
            join(output_dir,
                 model_dir,
                 f"marl_epoch_{e}.json")
        )
        mlflow.log_artifact(
            join(output_dir, model_dir,
                 f"nn_models_epoch_{e}.pt")
        )
        mlflow.log_artifact(
            join(output_dir,
                 f"confusion_matrix_epoch_{e}_eval.png")
        )

    empty_pipe = tr.Compose([
        tr.ToTensor()
    ])

    dataset_tmp = dataset_constructor(empty_pipe)

    test_dataloader_ori = Subset(dataset_tmp, idx_test)
    test_dataloader = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataloader_ori))

    visualize_steps(
        marl_m, test_dataloader[test_idx][0],
        test_dataloader_ori[test_idx][0],
        main_options.step, train_options.window_size,
        output_dir, train_options.nb_class, device_str,
        dataset.class_to_idx
    )

    mlflow.end_run()


#######################
# Test - Main
#######################

def test(
        main_options: MainOptions,
        test_options: TestOptions
) -> None:
    steps = main_options.step

    json_path = test_options.json_path
    state_dict_path = test_options.state_dict_path
    image_root = test_options.image_root
    output_dir = test_options.output_dir

    assert exists(json_path), \
        f"JSON path \"{json_path}\" does not exist"
    assert isfile(json_path), \
        f"\"{json_path}\" is not a file"

    assert exists(state_dict_path), \
        f"State dict path {state_dict_path} does not exist"
    assert isfile(state_dict_path), \
        f"{state_dict_path} is not a file"

    if exists(output_dir) and isdir(output_dir):
        print(f"File in {output_dir} will be overwritten")
    elif exists(output_dir) and not isdir(output_dir):
        raise Exception(f"\"{output_dir}\" is not a directory")
    else:
        print(f"Create \"{output_dir}\"")
        mkdir(output_dir)

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    img_dataset = ImageFolder(image_root, transform=img_pipeline)

    idx = list(range(len(img_dataset)))
    shuffle(idx)
    idx_test = idx[int(0.85 * len(idx)):]

    test_dataset = Subset(img_dataset, idx_test)

    nn_models = ModelsWrapper.from_json(json_path)
    nn_models.load_state_dict(th.load(state_dict_path))
    marl_m = MultiAgent.load_from(
        json_path, main_options.nb_agent,
        nn_models, obs_img, trans_img
    )

    data_loader = DataLoader(
        test_dataset, batch_size=test_options.batch_size,
        shuffle=True, num_workers=8, drop_last=False
    )

    cuda = main_options.cuda
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

        preds, probas = episode(marl_m, x, 0., steps)

        conf_meter.add(preds.detach(), y)

    print(conf_meter.value())

    precs, recs = prec_rec(conf_meter)

    precs_str = format_metric(precs, img_dataset.class_to_idx)
    recs_str = format_metric(recs, img_dataset.class_to_idx)

    print(f"Precision : {precs_str}")
    print(f"Precision mean = {precs.mean()}")
    print(f"Recall : {recs_str}")
    print(f"Recall mean : {recs.mean()}")


def infer(
        main_options: MainOptions,
        infer_options: InferOptions
) -> None:
    images_path = infer_options.images_path
    output_dir = infer_options.output_dir
    state_dict_path = infer_options.state_dict_path
    json_path = infer_options.json_path

    assert exists(json_path), \
        f"JSON path \"{json_path}\" does not exist"
    assert isfile(json_path), \
        f"\"{json_path}\" is not a file"

    assert exists(state_dict_path), \
        f"State dict path {state_dict_path} does not exist"
    assert isfile(state_dict_path), \
        f"{state_dict_path} is not a file"

    json_f = open(infer_options.class_to_idx, "r")
    class_to_idx = json.load(json_f)
    json_f.close()

    nn_models = ModelsWrapper.from_json(json_path)
    nn_models.load_state_dict(th.load(state_dict_path))

    marl_m = MultiAgent.load_from(
        json_path,
        main_options.nb_agent,
        nn_models,
        obs_img,
        trans_img
    )

    img_ori_pipeline = tr.Compose([
        tr.ToTensor()
    ])

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    images = tqdm(
        [img for img_path in images_path
         for img in glob.glob(img_path, recursive=True)]
    )

    for img_path in images:
        img = my_pil_loader(img_path)
        x_ori = img_ori_pipeline(img)
        x = img_pipeline(img)

        curr_img_path = join(output_dir, img_path.split("/")[-1])

        if not exists(curr_img_path):
            mkdir(curr_img_path)

        info_f = open(join(curr_img_path, "info.txt"), "w")
        info_f.writelines(
            [f"{img_path}\n",
             f"{infer_options.json_path}\n",
             f"{infer_options.state_dict_path}\n"]
        )
        info_f.close()

        visualize_steps(
            marl_m, x, x_ori,
            main_options.step,
            nn_models.f,
            curr_img_path,
            nn_models.nb_class,
            device_str,
            class_to_idx
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

    # main subparser
    choice_main_subparser = main_parser.add_subparsers()
    choice_main_subparser.dest = "main_choice"
    choice_main_subparser.required = True

    # main parsers
    train_parser = choice_main_subparser.add_parser("train")
    test_parser = choice_main_subparser.add_parser("test")
    infer_parser = choice_main_subparser.add_parser("infer")

    ##################
    # Main args
    ##################

    # Algorithm arguments
    main_parser.add_argument(
        "-a", "--agents", type=int, default=3, dest="agents",
        help="Number of agents"
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
    train_parser.add_argument(
        "--img-size", type=int, default=28, dest="img_size",
        help="Image side size, assume all image are squared"
    )
    train_parser.add_argument(
        "--nb-class", type=int, default=10, dest="nb_class",
        help="Image dataset number of class"
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
        "--ft-extr", type=str,
        choices=[
            ModelsWrapper.mnist,
            ModelsWrapper.resisc
        ],
        default="mnist", dest="ft_extractor",
        help="Choose features extractor (CNN)"
    )
    train_parser.add_argument(
        "--nb", type=int, default=64, dest="n_b",
        help="Hidden size for belief LSTM"
    )
    train_parser.add_argument(
        "--na", type=int, default=16, dest="n_a",
        help="Hidden size for action LSTM"
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
        "--nlb", type=int, default=128, dest="n_l_b",
        help="Network internal hidden size "
             "for linear projections (belief unit)"
    )
    train_parser.add_argument(
        "--nla", type=int, default=128, dest="n_l_a",
        help="Network internal hidden size for "
             "linear projections (action unit)"
    )

    # Training arguments
    train_parser.add_argument(
        "--batch-size", type=int,
        default=8, dest="batch_size",
        help="Image batch size for training and evaluation"
    )
    train_parser.add_argument(
        "-o", "--output-dir", type=str,
        required=True, dest="output_dir",
        help="The output dir containing res "
             "and models per epoch. Created if needed."
    )
    train_parser.add_argument(
        "--lr", "--learning-rate",
        type=float, default=1e-3,
        dest="learning_rate",
        help=""
    )
    train_parser.add_argument(
        "--nb-epoch", type=int,
        default=10, dest="nb_epoch",
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--nr", "--number-retry", type=int,
        default=7, dest="number_retry",
        help="Number of retry to estimate expectation."
    )
    train_parser.add_argument(
        "--eps", type=float, default=0.,
        dest="epsilon_greedy",
        help="Threshold from which apply "
             "greedy policy (random otherwise)"
    )
    train_parser.add_argument(
        "--eps-dec", type=float, default=0.,
        dest="epsilon_decay",
        help="Epsilon decay, at each forward "
             "eps <- eps * eps_decay"
    )
    train_parser.add_argument(
        "--freeze", type=str, default=[], nargs="+",
        dest="frozen_modules", action=SetAppendAction,
        choices=[
            ModelsWrapper.map_obs,
            ModelsWrapper.map_pos,
            ModelsWrapper.evaluate_msg,
            ModelsWrapper.belief_unit,
            ModelsWrapper.action_unit,
            ModelsWrapper.predict,
            ModelsWrapper.policy],
        help="Choose module(s) to be frozen during training"
    )

    ##################
    # Test args
    ##################
    test_parser.add_argument(
        "--batch-size", type=int,
        default=8, dest="batch_size",
        help="Image batch size for training and evaluation"
    )
    test_parser.add_argument(
        "--image-path", type=str,
        required=True, dest="image_path",
        help="Input image path for inference"
    )
    test_parser.add_argument(
        "--img-size", type=int,
        default=28, dest="img_size",
        help="Image side size, assume all image are squared"
    )
    test_parser.add_argument(
        "--json-path", type=str,
        required=True, dest="json_path",
        help="JSON multi agent metadata path"
    )
    test_parser.add_argument(
        "--state-dict-path", type=str,
        required=True, dest="state_dict_path",
        help="ModelsWrapper state dict path"
    )
    test_parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        dest="output_dir",
        help="The directory where the model outputs "
             "will be saved. Created if needed"
    )

    ##################
    # Infer args
    ##################
    infer_parser.add_argument(
        "--images", type=str, nargs="+",
        required=True, dest="infer_images",
        help="Path of images used for inference"
    )
    infer_parser.add_argument(
        "--json-path", type=str,
        required=True, dest="json_path",
        help="JSON multi agent metadata path"
    )
    infer_parser.add_argument(
        "--state-dict-path", type=str,
        required=True, dest="state_dict_path",
        help="ModelsWrapper state dict path"
    )
    infer_parser.add_argument(
        "--class2idx", type=str, required=True,
        dest="class_to_idx",
        help="Class to index JSON file"
    )
    infer_parser.add_argument(
        "-o", "--output-image-dir", type=str,
        required=True, dest="output_image_dir",
        help="The directory where the model outputs "
             "will be saved. Created if needed"
    )

    ###################################
    # Main - start different mods
    ###################################

    args = main_parser.parse_args()

    # Unit tests main
    if args.main_choice == "train":
        # Create Options
        main_options = MainOptions(
            args.step, args.cuda, args.agents
        )

        train_options = TrainOptions(
            args.n_b,
            args.n_l_b,
            args.n_l_a,
            args.n_m,
            args.n_d,
            args.n_a,
            args.dim,
            args.f,
            args.img_size,
            args.nb_class,
            args.nb_action,
            args.nb_epoch,
            args.learning_rate,
            args.number_retry,
            args.epsilon_greedy,
            args.epsilon_decay,
            args.batch_size,
            args.output_dir,
            args.frozen_modules,
            args.ft_extractor
        )

        if not exists(args.output_dir):
            makedirs(args.output_dir)
        if exists(args.output_dir) and not isdir(args.output_dir):
            raise Exception(f"\"{args.output_dir}\" is not a directory.")

        train(main_options, train_options)

    # Test main
    elif args.main_choice == "test":
        main_options = MainOptions(
            args.step, args.cuda, args.agents
        )

        test_options = TestOptions(
            args.img_size,
            args.state_dict_path,
            args.batch_size,
            args.json_path,
            args.image_path,
            args.output_dir
        )

        if not exists(args.output_dir):
            makedirs(args.output_dir)
        if exists(args.output_dir) and not isdir(args.output_dir):
            raise Exception(f"\"{args.output_dir}\" is not a directory.")

        test(main_options, test_options)

    elif args.main_choice == "infer":
        main_options = MainOptions(
            args.step, args.cuda, args.agents
        )

        infer_options = InferOptions(
            args.state_dict_path,
            args.json_path,
            args.infer_images,
            args.output_image_dir,
            args.class_to_idx
        )

        if not exists(args.output_image_dir):
            makedirs(args.output_image_dir)
        if exists(args.output_image_dir) and not isdir(args.output_image_dir):
            raise Exception(f"\"{args.output_image_dir}\" is not a directory.")

        infer(main_options, infer_options)

    else:
        main_parser.error(
            f"Unrecognized mode : \"{args.mode}\""
            f"type == {type(args.mode)}."
        )


if __name__ == "__main__":
    main()
