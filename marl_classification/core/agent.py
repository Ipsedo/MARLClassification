from dataclasses import dataclass
from typing import Protocol

import torch as th

from ..networks.models import ModelOutput, RecurrentOutput


class AgentModel(Protocol):
    """Minimal interface MultiAgent needs from its neural networks."""

    @property
    def n_b(self) -> int: ...

    @property
    def n_a(self) -> int: ...

    @property
    def n_m(self) -> int: ...

    @property
    def nb_class(self) -> int: ...

    @property
    def device(self) -> th.device: ...

    def __call__(
        self,
        img_patch: th.Tensor,
        msg_t: th.Tensor,
        norm_pos: th.Tensor,
        recurrent_hidden: RecurrentOutput,
    ) -> tuple[ModelOutput, RecurrentOutput]: ...


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

    def __init__(self, nb_agents: int, model: AgentModel) -> None:
        self.__nb_agents = nb_agents
        self.__model = model

        self.__hidden: RecurrentOutput | None = None
        self.__last_msg: th.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        device = self.__model.device

        def rand_hidden(size: int) -> th.Tensor:
            return th.randn(self.__nb_agents, batch_size, size, device=device)

        self.__hidden = RecurrentOutput(
            h=rand_hidden(self.__model.n_b),
            c=rand_hidden(self.__model.n_b),
            h_caret=rand_hidden(self.__model.n_a),
            c_caret=rand_hidden(self.__model.n_a),
        )

        self.__last_msg = th.zeros(
            self.__nb_agents, batch_size, self.__model.n_m, device=device
        )

    def act(self, observation: th.Tensor, norm_pos: th.Tensor) -> AgentOutput:
        assert (
            self.__hidden is not None and self.__last_msg is not None
        ), "reset() must be called before act()"

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
