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


def episode(
    agents: MultiAgent,
    env: Environment,
    img_batch: th.Tensor,
    max_it: int,
) -> EpisodeOutput:
    img_batch = img_batch.to(agents.device)

    obs = env.reset(img_batch, len(agents))
    agents.reset(img_batch.size(0))

    for _ in range(max_it):
        output = agents.act(obs, env.normalized_positions)
        obs = env.step(output.actions)

    output = agents.act(obs, env.normalized_positions)

    return EpisodeOutput(output.predictions, output.actions_log_probs)


def detailed_episode(
    agents: MultiAgent,
    env: Environment,
    img_batch: th.Tensor,
    max_it: int,
) -> EpisodeDetailedOutput:
    device = agents.device
    img_batch = img_batch.to(device)

    batch_size = img_batch.size(0)

    obs = env.reset(img_batch, len(agents))
    agents.reset(batch_size)

    step_pos = th.zeros(
        max_it,
        *env.positions.size(),
        dtype=th.long,
        device=device,
    )

    step_preds = th.zeros(
        max_it,
        len(agents),
        batch_size,
        agents.nb_class,
        device=device,
    )

    step_probas = th.zeros(
        max_it,
        len(agents),
        batch_size,
        device=device,
    )

    step_values = th.zeros(
        max_it,
        len(agents),
        batch_size,
        device=device,
    )

    for t in range(max_it):
        output = agents.act(obs, env.normalized_positions)
        obs = env.step(output.actions)

        step_pos[t] = env.positions

        step_preds[t] = output.predictions
        step_probas[t] = output.actions_log_probs
        step_values[t] = output.values

    return EpisodeDetailedOutput(
        step_preds, step_probas, step_values, step_pos
    )
