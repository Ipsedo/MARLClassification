from typing import Protocol

import torch as th

from .observation import obs_generic
from .transition import trans_generic


class ObservationFn(Protocol):
    def __call__(self, x: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor: ...


class TransitionFn(Protocol):
    def __call__(
        self,
        pos: th.Tensor,
        a_t_next: th.Tensor,
        f: int,
        img_size: list[int],
    ) -> th.Tensor: ...


class Environment:
    """
    Owns the world: the image batch, the agent positions and the action
    semantics. Agents never see the full image, only the partial
    observations served by this class.
    """

    def __init__(
        self,
        actions: list[list[int]],
        window_size: int,
        obs: ObservationFn = obs_generic,
        trans: TransitionFn = trans_generic,
    ) -> None:
        self.__actions = actions
        self.__f = window_size
        self.__obs = obs
        self.__trans = trans

        self.__img_batch: th.Tensor | None = None
        self.__img_sizes: list[int] = []
        self.__pos = th.empty(0)
        self.__actions_table = th.empty(0)

    def reset(self, img_batch: th.Tensor, nb_agents: int) -> th.Tensor:
        """Place agents randomly on a new image batch, return o_0."""
        device = img_batch.device
        batch_size = img_batch.size(0)

        self.__img_batch = img_batch
        self.__img_sizes = list(img_batch.size()[2:])

        self.__actions_table = th.tensor(self.__actions, device=device)

        self.__pos = th.stack(
            [
                th.randint(
                    i_s - self.__f,
                    (nb_agents, batch_size),
                    device=device,
                )
                for i_s in self.__img_sizes
            ],
            dim=-1,
        )

        return self.observe()

    def observe(self) -> th.Tensor:
        assert (
            self.__img_batch is not None
        ), "reset() must be called before observe()"

        return self.__obs(self.__img_batch, self.__pos, self.__f)

    def step(self, action_indices: th.Tensor) -> th.Tensor:
        """Apply the chosen actions (indices in the action set) and
        return the new observations."""
        movements = self.__actions_table[action_indices]

        self.__pos = self.__trans(
            self.__pos.to(th.float),
            movements,
            self.__f,
            self.__img_sizes,
        ).to(th.long)

        return self.observe()

    @property
    def positions(self) -> th.Tensor:
        return self.__pos

    @property
    def normalized_positions(self) -> th.Tensor:
        sizes = th.tensor(
            [[self.__img_sizes]],
            dtype=th.float,
            device=self.__pos.device,
        )
        return self.__pos.to(th.float) / sizes

    @property
    def window_size(self) -> int:
        return self.__f

    @property
    def actions(self) -> list[list[int]]:
        return self.__actions

    @property
    def nb_actions(self) -> int:
        return len(self.__actions)
