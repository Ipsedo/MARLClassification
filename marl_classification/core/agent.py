from dataclasses import dataclass

import torch as th

from ..networks.models import ModelsWrapper


@dataclass
class AgentOutput:
    # chosen action indices in the environment action set, [Na, Nb]
    actions: th.Tensor
    actions_log_probs: th.Tensor
    predictions: th.Tensor
    values: th.Tensor


class MultiAgent:
    """
    The agents: neural networks, recurrent internal state and policy
    sampling. Knows nothing about the image, the positions or the
    transition dynamics — it only maps observations to actions.
    """

    def __init__(self, nb_agents: int, model: ModelsWrapper) -> None:
        self.__nb_agents = nb_agents
        self.__model = model

        fake_batch_size = 1

        self.__hidden = model.random_first_state(len(self), fake_batch_size)
        self.__last_msg = model.zero_first_message(len(self), fake_batch_size)

    def reset(self, batch_size: int) -> None:
        self.__hidden = self.__model.random_first_state(len(self), batch_size)

        self.__last_msg = self.__model.zero_first_message(
            len(self), batch_size
        )

    def act(self, observation: th.Tensor, norm_pos: th.Tensor) -> AgentOutput:
        output, hidden = self.__model(
            observation,
            self.__last_msg,
            norm_pos,
            self.__hidden,
        )

        self.__hidden = hidden
        self.__last_msg = output.messages

        probs = output.actions_probabilities

        action_indices = th.multinomial(
            probs.flatten(0, 1), num_samples=1, replacement=True
        ).view(self.__nb_agents, -1)

        log_probs = (
            th.gather(probs, -1, action_indices.unsqueeze(-1))
            .squeeze(-1)
            .log()
        )

        return AgentOutput(
            actions=action_indices,
            actions_log_probs=log_probs,
            predictions=output.predictions,
            values=output.values,
        )

    @property
    def nb_class(self) -> int:
        return self.__model.nb_class

    @property
    def device(self) -> th.device:
        return self.__model.device

    def __len__(self) -> int:
        return self.__nb_agents
