import json
from os import mkdir
from os.path import exists, isdir, join
from random import randint
from typing import Dict

import mlflow
import torch as th
from torch.utils.data import DataLoader, Subset

from .options import MainOptions, TrainOptions
from .registry import default_image_pipeline, get_dataset_spec
from .serde import MarlConfig
from .training import Trainer
from .visualization import visualize_steps


def train_main(main_options: MainOptions, train_options: TrainOptions) -> None:
    assert train_options.dim in (2, 3), (
        "Only 2D or 3D is supported at the moment "
        "for data loading and observation / transition. "
        "See torchvision.datasets.ImageFolder"
    )

    output_dir = train_options.output_dir

    model_dir = "models"
    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) and not isdir(
        join(output_dir, model_dir)
    ):
        raise NotADirectoryError(
            f'"{join(output_dir, model_dir)}"' f"is not a directory."
        )

    exp_name = "MARLClassification"
    mlflow.set_experiment(exp_name)

    mlflow.start_run(run_name=f"train_{main_options.run_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = default_image_pipeline()

    dataset_spec = get_dataset_spec(train_options.ft_extr_str)

    marl_config = MarlConfig(
        ft_extr_str=train_options.ft_extr_str,
        window_size=train_options.window_size,
        hidden_size_belief=train_options.hidden_size_belief,
        hidden_size_action=train_options.hidden_size_action,
        hidden_size_msg=train_options.hidden_size_msg,
        hidden_size_state=train_options.hidden_size_state,
        state_dim=train_options.dim,
        actions=train_options.action,
        nb_class=train_options.nb_class,
        hidden_size_linear_belief=train_options.hidden_size_linear_belief,
        hidden_size_linear_action=train_options.hidden_size_linear_action,
    )

    nn_models, marl_m, env = marl_config.build_marl(main_options.nb_agent)

    dataset = dataset_spec.dataset_constructor(
        train_options.resources_dir,
        img_pipeline,
    )

    mlflow.log_params(
        {
            "ft_extractor": train_options.ft_extr_str,
            "window_size": train_options.window_size,
            "hidden_size_belief": train_options.hidden_size_belief,
            "hidden_size_action": train_options.hidden_size_action,
            "hidden_size_msg": train_options.hidden_size_msg,
            "hidden_size_state": train_options.hidden_size_state,
            "dim": train_options.dim,
            "action": train_options.action,
            "nb_class": train_options.nb_class,
            "hidden_size_linear_belief": train_options.hidden_size_linear_belief,
            "hidden_size_linear_action": train_options.hidden_size_linear_action,
            "nb_agent": main_options.nb_agent,
            "nb_epoch": train_options.nb_epoch,
            "learning_rate": train_options.learning_rate,
            "img_size": train_options.img_size,
            "step": main_options.step,
            "batch_size": train_options.batch_size,
        }
    )

    marl_config.save_marl_config(join(output_dir, "marl.json"))

    with open(
        join(output_dir, "class_to_idx.json"), "w", encoding="utf-8"
    ) as json_f:
        json.dump(dataset.class_to_idx, json_f)

    device = th.device("cuda" if main_options.cuda else "cpu")
    nn_models.to(device)

    mlflow.log_param("device", device.type)

    # for RL agent models parameters
    optim = th.optim.Adam(
        nn_models.parameters(), lr=train_options.learning_rate
    )

    ratio_eval = 0.85
    idx = th.randperm(len(dataset))
    # fmt: off
    idx_train = idx[:int(ratio_eval * idx.size()[0])].tolist()
    idx_test = idx[int(ratio_eval * idx.size()[0]):].tolist()
    # fmt: on

    train_dataset = Subset(dataset, idx_train)
    test_dataset = Subset(dataset, idx_test)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_options.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_options.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    def mlflow_metric_logger(step: int, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(step=step, metrics=metrics)

    trainer = Trainer(
        marl_m,
        env,
        nn_models,
        optim,
        episode_steps=main_options.step,
        gamma=train_options.gamma,
        metric_logger=mlflow_metric_logger,
    )

    for e in range(train_options.nb_epoch):
        trainer.train_epoch(train_dataloader, e)

        conf_meter_eval = trainer.eval_epoch(test_dataloader, e)

        precs, recs = (
            conf_meter_eval.precision(),
            conf_meter_eval.recall(),
        )

        conf_meter_eval.save_conf_matrix(e, output_dir, "eval")

        mlflow.log_metrics(
            step=trainer.curr_step,
            metrics={
                "eval_prec": precs.mean().item(),
                "eval_recs": recs.mean().item(),
            },
        )

        th.save(
            nn_models.state_dict(),
            join(output_dir, model_dir, f"nn_models_epoch_{e}.pt"),
        )

    dataset_tmp = dataset_spec.dataset_constructor(
        train_options.resources_dir,
        img_pipeline,
    )

    test_dataset_ori = Subset(dataset_tmp, idx_test)
    test_dataset = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataset_ori))

    visualize_steps(
        marl_m,
        env,
        test_dataset[test_idx][0],
        test_dataset_ori[test_idx][0],
        main_options.step,
        output_dir,
        dataset.class_to_idx,
    )

    mlflow.end_run()
