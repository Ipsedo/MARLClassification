import json
from typing import Callable, List, Tuple

import torch as th

from ..networks.models import ModelsWrapper


class MultiAgent:
    def __init__(
        self,
        nb_agents: int,
        model_wrapper: ModelsWrapper,
        n_b: int,
        n_a: int,
        f: int,
        n_m: int,
        action: List[List[int]],
        obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
        trans: Callable[[th.Tensor, th.Tensor, int, List[int]], th.Tensor],
    ) -> None:

        # Agent info
        self.__nb_agents = nb_agents

        self.__n_b = n_b
        self.__n_a = n_a
        self.__f = f
        self.__n_m = n_m

        self.__actions: List[List[int]] = action
        self.__nb_action = len(self.__actions)
        self.__dim = len(self.__actions[0])
        self.__batch_size = 1

        # Env info
        self.__obs = obs
        self.__trans = trans

        # NNs wrapper
        self.__networks = model_wrapper

        # initial state
        self.__pos = th.zeros(
            nb_agents, self.__batch_size, *list(range(self.__dim))
        )
        self.__t = 0

        # Hidden vectors
        self.__h: List[th.Tensor] = []
        self.__c: List[th.Tensor] = []

        self.__h_caret: List[th.Tensor] = []
        self.__c_caret: List[th.Tensor] = []

        self.__msg: List[th.Tensor] = []

        self.__action_probas: List[th.Tensor] = []

        # CPU vs GPU
        self.__is_cuda = False
        self.__device_str = "cpu"

    def new_episode(self, batch_size: int, img_size: List[int]) -> None:

        self.__batch_size = batch_size

        self.__t = 0

        self.__h = [
            th.randn(
                self.__nb_agents,
                batch_size,
                self.__n_b,
                device=th.device(self.__device_str),
            )
        ]
        self.__c = [
            th.randn(
                self.__nb_agents,
                batch_size,
                self.__n_b,
                device=th.device(self.__device_str),
            )
        ]

        self.__h_caret = [
            th.randn(
                self.__nb_agents,
                batch_size,
                self.__n_a,
                device=th.device(self.__device_str),
            )
        ]
        self.__c_caret = [
            th.randn(
                self.__nb_agents,
                batch_size,
                self.__n_a,
                device=th.device(self.__device_str),
            )
        ]

        self.__msg = [
            th.zeros(
                self.__nb_agents,
                batch_size,
                self.__n_m,
                device=th.device(self.__device_str),
            )
        ]

        self.__action_probas = [
            th.ones(
                self.__nb_agents,
                batch_size,
                device=th.device(self.__device_str),
            )
            / self.__nb_action
        ]

        self.__pos = th.stack(
            [
                th.randint(
                    i_s - self.__f,
                    (self.__nb_agents, batch_size),
                    device=th.device(self.__device_str),
                )
                for i_s in img_size
            ],
            dim=-1,
        )

    def step(self, img: th.Tensor) -> None:

        img_sizes = list(img.size()[2:])
        nb_agent = len(self)

        # Observation
        o_t = self.__obs(img, self.pos, self.__f)

        # Feature space
        # CNN need (N, C, S1, S2, ..., Sd)
        # got (Na, Nb, C, S1, S2, ..., Sd)
        # => flatten agent and batch dims
        b_t = self.__networks(
            self.__networks.map_obs,
            o_t.flatten(0, 1),
        ).view(nb_agent, self.__batch_size, -1)

        # Get messages
        d_bar_t_tmp = self.__msg[self.__t]
        # sum on agent
        d_bar_t_sum = d_bar_t_tmp.sum(dim=0)
        d_bar_t = (d_bar_t_sum - d_bar_t_tmp) / (nb_agent - 1)

        # Map pos in feature space
        norm_pos = self.pos.to(th.float) / th.tensor(
            [[img_sizes]], device=th.device(self.__device_str)
        )

        lambda_t = self.__networks(
            self.__networks.map_pos,
            norm_pos,
        )

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=2)

        # Belief LSTM
        h_t_next, c_t_next = self.__networks(
            self.__networks.belief_unit,
            self.__h[self.__t],
            self.__c[self.__t],
            u_t,
        )

        # Append new h and c (t + 1 step)
        self.__h.append(h_t_next)
        self.__c.append(c_t_next)

        # Evaluate message
        self.__msg.append(
            self.__networks(
                self.__networks.evaluate_msg,
                self.__h[self.__t + 1],
            )
        )

        # Action unit LSTM
        h_caret_t_next, c_caret_t_next = self.__networks(
            self.__networks.action_unit,
            self.__h_caret[self.__t],
            self.__c_caret[self.__t],
            u_t,
        )

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_next)
        self.__c_caret.append(c_caret_t_next)

        # Get action probabilities
        action_scores = self.__networks(
            self.__networks.policy,
            self.__h_caret[self.__t + 1],
        )

        # Create actions tensor
        actions = th.tensor(
            self.__actions,
            device=th.device(self.__device_str),
        )

        # Greedy policy
        prob, policy_actions = action_scores.max(dim=-1)
        a_t_next = actions[policy_actions]

        # Append probability
        self.__action_probas.append(prob)

        # Apply action / Upgrade agent state
        self.__pos = self.__trans(
            self.pos.to(th.float),
            a_t_next,
            self.__f,
            img_sizes,
        ).to(th.long)

        self.__t += 1

    def predict(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return (
            self.__networks(self.__networks.predict, self.__h[-1]),
            self.__action_probas[-1].log(),
            self.__networks(self.__networks.critic, self.__h_caret[-1]),
        )

    @property
    def is_cuda(self) -> bool:
        return self.__is_cuda

    def cuda(self) -> None:
        self.__is_cuda = True
        self.__device_str = "cuda"

    def cpu(self) -> None:
        self.__is_cuda = False
        self.__device_str = "cpu"

    @property
    def pos(self) -> th.Tensor:
        return self.__pos

    def __len__(self) -> int:
        return self.__nb_agents

    @classmethod
    def load_from(
        cls,
        models_wrapper_json_file: str,
        nb_agent: int,
        model_wrapper: ModelsWrapper,
        obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
        trans: Callable[[th.Tensor, th.Tensor, int, List[int]], th.Tensor],
    ) -> "MultiAgent":

        with open(models_wrapper_json_file, "r") as f_json:
            j_obj = json.load(f_json)
            try:
                return cls(
                    nb_agent,
                    model_wrapper,
                    j_obj["hidden_size_belief"],
                    j_obj["hidden_size_action"],
                    j_obj["window_size"],
                    j_obj["hidden_size_msg"],
                    j_obj["actions"],
                    obs,
                    trans,
                )
            except Exception as e:
                raise Exception(
                    "Exception during loading MultiAgent "
                    f'from file "{models_wrapper_json_file}"'
                ) from e
