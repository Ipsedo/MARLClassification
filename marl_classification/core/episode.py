from dataclasses import dataclass

import torch as th

from .agent import MultiAgent
from .environment import Environment


@dataclass
class EpisodeOutput:
    prediction: th.Tensor
    actions_log_probs: th.Tensor


@dataclass
class EpisodeDetailedOutput:
    step_preds: th.Tensor
    step_log_probas: th.Tensor
    step_values: th.Tensor
    step_pos: th.Tensor


class EpisodeSampler:
    def __init__(self, agents: MultiAgent, env: Environment):
        self.__agents = agents
        self.__env = env

    def __episode_impl(
        self, img_batch: th.Tensor, max_it: int
    ) -> EpisodeDetailedOutput:
        device = self.__agents.device
        img_batch = img_batch.to(device)

        batch_size = img_batch.size(0)

        obs = self.__env.reset(img_batch, len(self.__agents))
        self.__agents.reset(batch_size)

        step_pos = th.zeros(
            max_it,
            *self.__env.positions.size(),
            dtype=th.long,
            device=device,
        )

        step_preds = th.zeros(
            max_it,
            len(self.__agents),
            batch_size,
            self.__agents.nb_class,
            device=device,
        )

        step_probas = th.zeros(
            max_it,
            len(self.__agents),
            batch_size,
            device=device,
        )

        step_values = th.zeros(
            max_it,
            len(self.__agents),
            batch_size,
            device=device,
        )

        for t in range(max_it):
            output = self.__agents.act(obs, self.__env.normalized_positions)
            obs = self.__env.step(output.actions)

            step_pos[t] = self.__env.positions

            step_preds[t] = output.predictions
            step_probas[t] = output.actions_log_probs
            step_values[t] = output.values

        return EpisodeDetailedOutput(
            step_preds, step_probas, step_values, step_pos
        )

    def run_episode(
        self, img_batch: th.Tensor, max_it: int
    ) -> EpisodeDetailedOutput:
        return self.__episode_impl(img_batch, max_it)

    def run_episode_get_last_step(
        self, img_batch: th.Tensor, max_it: int
    ) -> EpisodeOutput:
        detailed_output = self.__episode_impl(img_batch, max_it)

        return EpisodeOutput(
            prediction=detailed_output.step_preds[-1],
            actions_log_probs=detailed_output.step_log_probas[-1],
        )
