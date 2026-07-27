from typing import Callable, Dict, Optional

import torch as th
import torch.nn.functional as th_fun
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core import EpisodeSampler
from ..metrics import ConfusionMeter, LossMeter
from ..networks import ModelsWrapper
from .functions import classification_rewards, discounted_returns, standardize

MetricLogger = Callable[[int, Dict[str, float]], None]


class Trainer:
    """
    A2C training loop for the MARL classification game. Metric logging
    is injected so the trainer does not depend on MLflow.
    """

    def __init__(
        self,
        model: ModelsWrapper,
        nb_class: int,
        learning_rate: float,
        gamma: float,
        metric_logger: Optional[MetricLogger] = None,
        log_interval: int = 100,
        meter_window_size: int = 64,
    ) -> None:
        self.__model = model
        self.__optim = th.optim.Adam(model.parameters(), lr=learning_rate)

        self.__nb_class = nb_class
        self.__gamma = gamma

        self.__metric_logger = metric_logger
        self.__log_interval = log_interval

        self.__curr_step = 0

        self.__conf_meter = ConfusionMeter(
            self.__nb_class, window_size=meter_window_size
        )
        self.__path_loss_meter = LossMeter(window_size=meter_window_size)
        self.__error_meter = LossMeter(window_size=meter_window_size)
        self.__actor_loss_meter = LossMeter(window_size=meter_window_size)
        self.__critic_loss_meter = LossMeter(window_size=meter_window_size)

    @property
    def curr_step(self) -> int:
        return self.__curr_step

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch_index: int,
        episode_sampler: EpisodeSampler,
    ) -> None:
        self.__model.train()

        device = self.__model.device

        tqdm_bar = tqdm(dataloader)
        for x_train, y_train in tqdm_bar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # pred = [Ns, Na, Nb, Nc]
            # prob = [Ns, Na, Nb]
            # values = [Ns, Na, Nb]
            output = episode_sampler.run_episode(x_train)

            # compute error : per step prediction and mean over agents
            batch_size = y_train.size(0)
            nb_steps = output.step_preds.size(0)
            predictions = output.step_preds.mean(dim=1).flatten(0, 1)
            targets = y_train.unsqueeze(0).repeat(nb_steps, 1).flatten(0, 1)

            # output shape (steps, 1, batch)
            # 1 => vote over agents
            error = th_fun.cross_entropy(
                predictions,
                targets,
                reduction="none",
            ).unflatten(0, (nb_steps, 1, batch_size))

            rewards = classification_rewards(output.step_preds, y_train)
            returns = discounted_returns(rewards, self.__gamma)

            advantages = returns - output.step_values
            normalized_advantages = standardize(advantages)

            # actor loss, maximize(log_proba * advantage)
            path_loss = (
                -output.step_log_probas * normalized_advantages.detach()
            )

            # add agent's votes -> train classifier
            actor_loss = path_loss + error

            # critic loss
            critic_loss = th_fun.smooth_l1_loss(
                output.step_values,
                returns.detach(),
                reduction="none",
            )

            # sum over steps, mean over agents and batch
            loss = th.sum(actor_loss + critic_loss, 0).mean()

            # backward and update weights
            self.__optim.zero_grad()
            loss.backward()
            self.__optim.step()

            # Update meters
            path_loss_item = path_loss.sum(dim=0).mean().item()
            error_item = error.mean().item()
            actor_loss_item = actor_loss.sum(dim=0).mean().item()
            critic_loss_item = critic_loss.sum(dim=0).mean().item()

            self.__conf_meter.add(
                # select last step, mean over agents
                output.step_preds[-1].mean(dim=0).detach(),
                y_train,
            )
            self.__path_loss_meter.add(path_loss_item)
            self.__error_meter.add(error_item)
            self.__actor_loss_meter.add(actor_loss_item)
            self.__critic_loss_meter.add(critic_loss_item)

            precs = self.__conf_meter.precision()
            recs = self.__conf_meter.recall()

            if (
                self.__metric_logger is not None
                and self.__curr_step % self.__log_interval == 0
            ):
                self.__metric_logger(
                    self.__curr_step,
                    {
                        "error": error_item,
                        "path_loss": path_loss_item,
                        "loss": loss.item(),
                        "train_prec": precs.mean().item(),
                        "train_rec": recs.mean().item(),
                    },
                )

            tqdm_bar.set_description(
                f"Epoch {epoch_index} - Train, "
                f"train_prec = {precs.mean().item():.3f}, "
                f"train_rec = {recs.mean().item():.3f}, "
                f"error = {self.__error_meter.loss():.4f}, "
                f"path = {self.__path_loss_meter.loss():.4f}, "
                f"actor = {self.__actor_loss_meter.loss():.4f}, "
                f"critic = {self.__critic_loss_meter.loss():.4f}"
            )

            self.__curr_step += 1

    def eval_epoch(
        self,
        dataloader: DataLoader,
        epoch_index: int,
        episode_sampler: EpisodeSampler,
    ) -> ConfusionMeter:
        self.__model.eval()

        device = self.__model.device

        conf_meter = ConfusionMeter(self.__nb_class, None)

        with th.no_grad():
            tqdm_bar = tqdm(dataloader)
            for x_test, y_test in tqdm_bar:
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                output = episode_sampler.run_episode_get_last_step(x_test)

                # mean over agents
                conf_meter.add(output.prediction.mean(dim=0).detach(), y_test)

                precs = conf_meter.precision()
                recs = conf_meter.recall()

                tqdm_bar.set_description(
                    f"Epoch {epoch_index} - Eval, "
                    f"eval_prec = {precs.mean().item():.4f}, "
                    f"eval_rec = {recs.mean().item():.4f}"
                )

        return conf_meter
