import json
from os import mkdir
from os.path import exists, isdir, join
from random import randint

import mlflow
import torch as th
import torchvision.transforms as tr
from torch.utils.data import Subset, DataLoader
from torchnet.meter import ConfusionMeter
from tqdm import tqdm

from .data import KneeMRIDataset, MNISTDataset, RESISC45Dataset
from .data import transforms as custom_tr
from .environment import (
    MultiAgent,
    obs_generic,
    trans_generic,
    episode_retry,
    episode
)
from .networks import ModelsWrapper
from .utils import MainOptions, TrainOptions
from .utils import prec_rec, save_conf_matrix, visualize_steps


def train(
        main_options: MainOptions,
        train_options: TrainOptions
) -> None:
    assert train_options.dim == 2 or train_options.dim == 3, \
        "Only 2D or 3D is supported at the moment " \
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

    mlflow.start_run(run_name=f"train_{main_options.run_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    img_pipeline = tr.Compose([
        tr.ToTensor(),
        custom_tr.NormalNorm()
    ])

    if train_options.ft_extr_str.startswith("resisc"):
        dataset_constructor = RESISC45Dataset
    elif train_options.ft_extr_str.startswith("mnist"):
        dataset_constructor = MNISTDataset
    else:
        dataset_constructor = KneeMRIDataset

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
        train_options.action,
        obs_generic,
        trans_generic
    )

    mlflow.log_params({
        "ft_extractor": train_options.ft_extr_str,
        "window_size": train_options.window_size,
        "hidden_size_belief": train_options.hidden_size_belief,
        "hidden_size_action": train_options.hidden_size_action,
        "hidden_size_msg": train_options.hidden_size_msg,
        "hidden_size_state": train_options.hidden_size_state,
        "dim": train_options.dim,
        "action": train_options.action,
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
    idx_train = idx[:int(0.85 * idx.size()[0])]
    idx_test = idx[int(0.85 * idx.size()[0]):]

    train_dataset = Subset(dataset, idx_train)
    test_dataset = Subset(dataset, idx_test)

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=train_options.batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
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
            # select last step
            r = -th.pow(y_eye - retry_pred, 2.).mean(dim=-1)[:, -1, :]

            # Compute loss
            # sum log proba (on steps), then pass to exponential
            losses = retry_prob.sum(dim=1).exp() * r.detach() + r

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
                mlflow.log_metrics(step=curr_step, metrics={
                    "loss": loss.item(),
                    "train_prec": precs.mean().item(),
                    "train_rec": recs.mean().item(),
                    "epsilon": epsilon
                })

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

        mlflow.log_metrics(step=curr_step, metrics={
            "eval_prec": precs.mean(),
            "eval_recs": recs.mean()
        })

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
