import json
from os import mkdir
from os.path import exists, isdir, join
from random import randint
from typing import Dict

import mlflow
import torch as th
from torch.utils.data import DataLoader, Subset

from .config import MainConfig, ModelConfig, TrainConfig
from .core import EpisodeSampler
from .registry import default_image_pipeline, get_dataset_spec
from .training import Trainer
from .visualization import visualize_steps


def train_main(
    main_config: MainConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> None:
    assert model_config.state_dim in (2, 3), (
        "Only 2D or 3D is supported at the moment "
        "for data loading and observation / transition. "
        "See torchvision.datasets.ImageFolder"
    )

    output_dir = train_config.output_dir

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

    mlflow.start_run(run_name=f"train_{main_config.run_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = default_image_pipeline()

    dataset_spec = get_dataset_spec(model_config.ft_extr_str)

    nn_models, marl_m, env = model_config.build_marl(main_config.nb_agent)

    dataset = dataset_spec.dataset_constructor(
        train_config.resources_dir,
        img_pipeline,
    )

    mlflow.log_params(
        {
            **dict(main_config),
            **dict(model_config),
            **dict(train_config),
        }
    )

    model_config.save_marl_config(join(output_dir, "marl.json"))

    with open(
        join(output_dir, "class_to_idx.json"), "w", encoding="utf-8"
    ) as json_f:
        json.dump(dataset.class_to_idx, json_f)

    device = th.device("cuda" if main_config.cuda else "cpu")
    nn_models.to(device)

    mlflow.log_param("device", device.type)

    # for RL agent models parameters
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
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=False,
        pin_memory=True,
    )

    def mlflow_metric_logger(step: int, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(step=step, metrics=metrics)

    episode_sampler = EpisodeSampler(marl_m, env)

    trainer = Trainer(
        nn_models,
        marl_m.nb_class,
        train_config.learning_rate,
        main_config.step,
        train_config.gamma,
        metric_logger=mlflow_metric_logger,
    )

    for e in range(train_config.nb_epoch):
        trainer.train_epoch(train_dataloader, e, episode_sampler)

        conf_meter_eval = trainer.eval_epoch(
            test_dataloader, e, episode_sampler
        )

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
        train_config.resources_dir,
        img_pipeline,
    )

    test_dataset_ori = Subset(dataset_tmp, idx_test)
    test_dataset = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataset_ori))

    visualize_steps(
        episode_sampler,
        test_dataset[test_idx][0],
        test_dataset_ori[test_idx][0],
        model_config.window_size,
        main_config.step,
        output_dir,
        dataset.class_to_idx,
    )

    mlflow.end_run()
