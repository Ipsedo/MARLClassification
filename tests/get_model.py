from marl_classification.environment import (
    MultiAgent,
    obs_generic,
    trans_generic
)
from marl_classification.networks import ModelsWrapper


class GetModel:
    def __init__(self):
        self._batch_size = 19
        self._nb_agent = 5
        self._nb_class = 10
        self._step = 7
        self._dim = 2

        action = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ]

        n_b = 23
        n_a = 22
        n_m = 21

        f = 12

        model_wrapper = ModelsWrapper(
            "mnist", f, n_b, n_a, n_m, 20, self._dim,
            action, self._nb_class, 24, 25
        )

        self._marl = MultiAgent(
            self._nb_agent, model_wrapper,
            n_b, n_a, f, n_m, action,
            obs_generic, trans_generic
        )

    @property
    def marl(self) -> MultiAgent:
        return self._marl

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def nb_agent(self) -> int:
        return self._nb_agent

    @property
    def nb_class(self) -> int:
        return self._nb_class

    @property
    def step(self) -> int:
        return self._step

    @property
    def dim(self) -> int:
        return self._dim
