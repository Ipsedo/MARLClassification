import json
from math import log
from os import mkdir
from os.path import exists, isdir, join
from random import randint
from typing import Callable, Dict

import mlflow
import torch as th
import torch.nn.functional as th_fun
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .core import (
    MultiAgent,
    detailed_episode,
    episode,
    obs_generic,
    trans_generic,
)
from .data import (
    AIDDataset,
    KneeMRIDataset,
    MNISTDataset,
    RESISC45Dataset,
    SkinCancerDataset,
    WorldStratDataset,
)
from .infer import visualize_steps
from .metrics import ConfusionMeter, LossMeter
from .networks import ModelsWrapper
from .options import MainOptions, TrainOptions


def train(main_options: MainOptions, train_options: TrainOptions) -> None:
    assert train_options.dim == 2 or train_options.dim == 3, (
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
        raise Exception(
            f'"{join(output_dir, model_dir)}"' f"is not a directory."
        )

    exp_name = "MARLClassification"
    mlflow.set_experiment(exp_name)

    mlflow.start_run(run_name=f"train_{main_options.run_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = tr.Compose([tr.ToTensor()])

    dataset_constructors: Dict[
        str, Callable[[str, tr.Compose], ImageFolder]
    ] = {
        ModelsWrapper.mnist: MNISTDataset,
        ModelsWrapper.resisc: RESISC45Dataset,
        ModelsWrapper.knee_mri: KneeMRIDataset,
        ModelsWrapper.aid: AIDDataset,
        ModelsWrapper.world_strat: WorldStratDataset,
        ModelsWrapper.skin_cancer: SkinCancerDataset,
    }

    dataset_constructor = dataset_constructors[train_options.ft_extr_str]

    nn_models = ModelsWrapper(
        train_options.ft_extr_str,
        train_options.window_size,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.hidden_size_msg,
        train_options.hidden_size_state,
        train_options.dim,
        train_options.action,
        train_options.nb_class,
        train_options.hidden_size_linear_belief,
        train_options.hidden_size_linear_action,
    )

    dataset = dataset_constructor(
        train_options.resources_dir,
        img_pipeline,
    )

    marl_m = MultiAgent(
        main_options.nb_agent,
        nn_models,
        train_options.hidden_size_belief,
        train_options.hidden_size_action,
        train_options.window_size,
        train_options.hidden_size_msg,
        train_options.action,
        obs_generic,
        trans_generic,
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
            "frozen_modules": train_options.frozen_modules,
            "nb_epoch": train_options.nb_epoch,
            "learning_rate": train_options.learning_rate,
            "img_size": train_options.img_size,
            "step": main_options.step,
            "batch_size": train_options.batch_size,
        }
    )

    nn_models.json_args(join(output_dir, "marl.json"))

    with open(
        join(output_dir, "class_to_idx.json"), "w", encoding="utf-8"
    ) as json_f:
        json.dump(dataset.class_to_idx, json_f)

    cuda = main_options.cuda
    device_str = "cpu"

    # Pass pytorch stuff to GPU
    # for agents hidden tensors (belief etc.)
    if cuda:
        nn_models.cuda()
        marl_m.cuda()
        device_str = "cuda"

    mlflow.log_param("device", device_str)

    module_to_train = ModelsWrapper.module_list.difference(
        train_options.frozen_modules
    )

    # for RL agent models parameters
    optim = th.optim.Adam(
        nn_models.get_params(list(module_to_train)),
        lr=train_options.learning_rate,
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

    curr_step = 0

    conf_meter_train = ConfusionMeter(
        train_options.nb_class,
        window_size=64,
    )

    path_loss_meter = LossMeter(window_size=64)
    error_meter = LossMeter(window_size=64)

    policy_loss_meter = LossMeter(window_size=64)
    critic_loss_meter = LossMeter(window_size=64)

    for e in range(train_options.nb_epoch):
        nn_models.train()

        tqdm_bar = tqdm(train_dataloader)
        for x_train, y_train in tqdm_bar:
            x_train, y_train = (
                x_train.to(th.device(device_str)),
                y_train.to(th.device(device_str)),
            )

            # pred = [Ns, Na, Nb, Nc]
            # prob = [Ns, Na, Nb]
            # values = [Ns, Na, Nb]
            pred, log_proba, values, _ = detailed_episode(
                marl_m,
                x_train,
                main_options.step,
                device_str,
                train_options.nb_class,
            )

            # compute error : last step prediction and mean over agents
            error = th_fun.cross_entropy(
                pred[-1].mean(dim=0),
                y_train,
                reduction="none",
            )

            # fmt: off

            # [Nb] -> [Nb, 1, 1] -> [Nb, Ns, Na]
            tmp_y_train = (
                y_train[:, None, None]
                .repeat(1, main_options.step, len(marl_m))
            )
            # [Ns, Na, Nb, Nc] -> [Nb, Nc, Ns, Na]
            tmp_pred = pred.permute(2, 3, 0, 1)

            # random prediction error
            # make reward positive if better than random, else negative
            random_error = log(train_options.nb_class)

            # reward = (random_error - error) / random_error
            # reward per agent and step
            # [Nb, Ns, Na] -> [Ns, Na, Nb]
            rewards = (
                random_error - th_fun.cross_entropy(
                    tmp_pred, tmp_y_train,
                    reduction="none"
                ).permute(1, 2, 0)
            ) / random_error

            # [Ns, 1, 1]
            t_steps = (
                th.arange(
                    rewards.size(0),
                    device=th.device(device_str)
                )[:, None, None]
                .to(th.float)
            )

            # discounting reward
            returns = rewards * train_options.gamma ** t_steps
            returns = (
                returns.flip(dims=(0,))
                .cumsum(0)
                .flip(dims=(0,)) /
                train_options.gamma ** t_steps
            )

            # fmt: on

            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # actor advantage
            advantage = returns - values
            # actor loss, maximize(log_proba * advantage)
            path_loss = -log_proba * advantage.detach()

            # add agent's votes -> train classifier
            policy_loss = path_loss + error

            # critic loss : difference between values and rewards
            critic_loss = th_fun.smooth_l1_loss(
                values,
                returns.detach(),
                reduction="none",
            )

            # sum over steps, mean over agents and batch
            loss = (policy_loss + critic_loss).sum(dim=0).mean()

            # backward and update weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update meters
            path_loss_item = path_loss.sum(dim=0).mean().item()
            error_item = error.mean().item()
            policy_loss_item = policy_loss.sum(dim=0).mean().item()
            critic_loss_item = critic_loss.sum(dim=0).mean().item()

            conf_meter_train.add(
                # select last step, mean over agents
                pred[-1].mean(dim=0).detach(),
                y_train,
            )
            path_loss_meter.add(path_loss_item)
            error_meter.add(error_item)
            policy_loss_meter.add(policy_loss_item)
            critic_loss_meter.add(critic_loss_item)

            # Compute global score
            precs, recs = (
                conf_meter_train.precision(),
                conf_meter_train.recall(),
            )

            # log metrics to mlflow
            if curr_step % 100 == 0:
                mlflow.log_metrics(
                    step=curr_step,
                    metrics={
                        "error": error_item,
                        "path_loss": path_loss_item,
                        "loss": loss.item(),
                        "train_prec": precs.mean().item(),
                        "train_rec": recs.mean().item(),
                        "critic_loss": critic_loss_item,
                        "actor_loss": policy_loss_item,
                    },
                )

            # update tqdm bar wit metrics
            tqdm_bar.set_description(
                f"Epoch {e} - Train, "
                f"train_prec = {precs.mean().item():.3f}, "
                f"train_rec = {recs.mean().item():.3f}, "
                f"c_loss = {critic_loss_meter.loss():.4f}, "
                f"a_loss = {policy_loss_meter.loss():.4f}, "
                f"error = {error_meter.loss():.4f}, "
                f"path = {path_loss_meter.loss():.4f}"
            )

            curr_step += 1

        nn_models.eval()
        conf_meter_eval = ConfusionMeter(train_options.nb_class, None)

        with th.no_grad():
            tqdm_bar = tqdm(test_dataloader)
            for x_test, y_test in tqdm_bar:
                x_test, y_test = (
                    x_test.to(th.device(device_str)),
                    y_test.to(th.device(device_str)),
                )

                pred, _ = episode(marl_m, x_test, main_options.step)

                # mean over agents
                conf_meter_eval.add(pred.mean(dim=0).detach(), y_test)

                # Compute score
                precs, recs = (
                    conf_meter_eval.precision(),
                    conf_meter_eval.recall(),
                )

                tqdm_bar.set_description(
                    f"Epoch {e} - Eval, "
                    f"eval_prec = {precs.mean().item():.4f}, "
                    f"eval_rec = {recs.mean().item():.4f}"
                )

        # Compute score
        precs, recs = (
            conf_meter_eval.precision(),
            conf_meter_eval.recall(),
        )

        conf_meter_eval.save_conf_matrix(e, output_dir, "eval")

        mlflow.log_metrics(
            step=curr_step,
            metrics={
                "eval_prec": precs.mean().item(),
                "eval_recs": recs.mean().item(),
            },
        )

        th.save(
            nn_models.state_dict(),
            join(output_dir, model_dir, f"nn_models_epoch_{e}.pt"),
        )

    dataset_tmp = dataset_constructor(
        train_options.resources_dir,
        img_pipeline,
    )

    test_dataset_ori = Subset(dataset_tmp, idx_test)
    test_dataset = Subset(dataset, idx_test)

    test_idx = randint(0, len(test_dataset_ori))

    visualize_steps(
        marl_m,
        test_dataset[test_idx][0],
        test_dataset_ori[test_idx][0],
        main_options.step,
        train_options.window_size,
        output_dir,
        train_options.nb_class,
        device_str,
        dataset.class_to_idx,
    )

    mlflow.end_run()
