import operator as op
from functools import reduce

import torch as th


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
    ) -> None:
        self.__actions = actions
        self.__f = window_size
        self.__obs = Environment.__obs_generic
        self.__trans = Environment.__trans_generic

        self.__img_batch: th.Tensor = th.empty([1])  # fake img batch
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

    @staticmethod
    def __obs_generic(x: th.Tensor, pos: th.Tensor, f: int) -> th.Tensor:
        x_sizes = x.size()
        b_img, c = x_sizes[0], x_sizes[1]
        sizes = list(x_sizes[2:])

        nb_a, _, _ = pos.size()

        pos_min = pos
        pos_max = pos_min + f

        masks = []

        for d, s in enumerate(sizes):
            values = th.arange(0, s, device=pos.device)

            mask = (pos_min[:, :, d, None] <= values.view(1, 1, s)) & (
                values.view(1, 1, s) < pos_max[:, :, d, None]
            )

            for n_unsq in range(len(sizes) - 1):
                mask = mask.unsqueeze(-2) if n_unsq < d else mask.unsqueeze(-1)

            masks.append(mask)
        mask = reduce(op.and_, masks)
        mask = mask.unsqueeze(2)

        return (
            x.unsqueeze(0)
            .masked_select(mask)
            .view(nb_a, b_img, c, *[f for _ in range(len(sizes))])
        )

    @staticmethod
    def __trans_generic(
        pos: th.Tensor,
        a_t_next: th.Tensor,
        f: int,
        img_size: list[int],
    ) -> th.Tensor:
        new_pos = pos.clone()
        dim = new_pos.size(-1)

        idxs = []
        for d in range(dim):
            curr_idx = (new_pos[:, :, d] + a_t_next[:, :, d] >= 0) * (
                new_pos[:, :, d] + a_t_next[:, :, d] + f < img_size[d]
            )
            idxs.append(curr_idx)

        idx = reduce(op.mul, idxs)
        idx = idx.unsqueeze(2).to(th.float)

        new_pos = idx * (new_pos + a_t_next) + (1 - idx) * new_pos

        return new_pos
