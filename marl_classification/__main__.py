import argparse
import re
from os import makedirs
from os.path import isdir, exists

from .eval import evaluation
from .infer import infer
from .networks import ModelsWrapper
from .options import (
    TrainOptions,
    EvalOptions,
    MainOptions,
    InferOptions
)
from .train import train
from .utils import SetAppendAction


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

    main_parser.add_argument(
        "--run-id", type=str, required=True,
        dest="run_id", help="MLFlow run id"
    )

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
        "--action", type=str,
        default="[[1, 0], [-1, 0], [0, 1], [0, -1]]",
        dest="action",
        help="Discrete actions"
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
            ModelsWrapper.resisc,
            ModelsWrapper.knee_mri
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
        help="The output directory containing results "
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

    match args.main_choice:
        case "train":
            # Create Options
            main_options = MainOptions(
                args.step, args.run_id, args.cuda, args.agents
            )

            reg_action = re.compile(r"] *, *\[")
            action = reg_action.split(args.action[2:-2])
            action = [[int(i) for i in act.split(",")] for act in action]

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
                action,
                args.nb_epoch,
                args.learning_rate,
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
                raise NotADirectoryError(f"\"{args.output_dir}\" is not a directory.")

            train(main_options, train_options)

        # Test main
        case "test":
            main_options = MainOptions(
                args.step, args.run_id, args.cuda, args.agents
            )

            eval_options = EvalOptions(
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
                raise NotADirectoryError(f"\"{args.output_dir}\" is not a directory.")

            evaluation(main_options, eval_options)

        case "infer":
            main_options = MainOptions(
                args.step, args.run_id, args.cuda, args.agents
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
                raise NotADirectoryError(f"\"{args.output_image_dir}\" is not a directory.")

            infer(main_options, infer_options)

        case _:
            main_parser.error(
                f"Unrecognized mode : \"{args.mode}\""
                f"type == {type(args.mode)}."
            )


if __name__ == "__main__":
    main()
