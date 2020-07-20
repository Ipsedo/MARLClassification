import torch as th
from networks.models import ModelsWrapper

from typing import Callable, Tuple, AnyStr

import json


class MultiAgent:
    def __init__(self, nb_agents: int, model_wrapper: ModelsWrapper,
                 n: int, f: int, n_m: int,
                 size: int, nb_action: int,
                 obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
                 trans: Callable[[th.Tensor, th.Tensor, int, int], th.Tensor]) -> None:
        """

        :param nb_agents:
        :type nb_agents:
        :param model_wrapper:
        :type model_wrapper:
        :param n:
        :type n:
        :param f:
        :type f:
        :param n_m:
        :type n_m:
        :param size:
        :type size:
        :param nb_action:
        :type nb_action:
        :param obs:
        :type obs:
        :param trans:
        :type trans:
        """

        # Agent info
        self.__nb_agents = nb_agents

        self.__n = n
        self.__f = f
        self.__n_m = n_m

        self.__size = size
        self.__nb_action = nb_action
        self.__batch_size = None

        # Env info
        self.__obs = obs
        self.__trans = trans

        # NNs wrapper
        self.__networks = model_wrapper

        # initial state
        self.pos = None
        self.__t = 0

        # Hidden vectors
        self.__h = None
        self.__c = None

        self.__h_caret = None
        self.__c_caret = None

        self.msg = None

        self.__log_probas = None

        # CPU vs GPU
        self.is_cuda = False

    def new_episode(self, batch_size: int) -> None:
        """

        :param batch_size:
        :type batch_size:
        :return:
        :rtype:
        """

        self.__batch_size = batch_size

        self.__t = 0

        self.__h = [th.zeros(self.__nb_agents, batch_size, self.__n,
                             device=th.device("cuda")
                             if self.is_cuda else th.device("cpu"))]
        self.__c = [th.zeros(self.__nb_agents, batch_size, self.__n,
                             device=th.device("cuda")
                             if self.is_cuda else th.device("cpu"))]

        self.__h_caret = [th.zeros(self.__nb_agents, batch_size, self.__n,
                                   device=th.device("cuda")
                                   if self.is_cuda else th.device("cpu"))]
        self.__c_caret = [th.zeros(self.__nb_agents, batch_size, self.__n,
                                   device=th.device("cuda")
                                   if self.is_cuda else th.device("cpu"))]

        self.msg = [th.zeros(self.__nb_agents, batch_size, self.__n_m,
                             device=th.device("cuda")
                             if self.is_cuda else th.device("cpu"))]

        self.__log_probas = [
            th.log(th.ones(self.__nb_agents,
                           device=th.device("cuda")
                           if self.is_cuda else th.device("cpu"))) / self.__nb_action
        ]

        self.pos = th.randint(self.__size - self.__f,
                              (self.__nb_agents, batch_size, 2),
                              device=th.device("cuda")
                              if self.is_cuda else th.device("cpu"))

    def step(self, img: th.Tensor, eps: float) -> None:
        """

        :param img:
        :type img:
        :param eps:
        :type eps:
        :return:
        :rtype:
        """

        # Observation
        o_t = self.__obs(img, self.pos, self.__f)

        # Feature space
        # CNN need (N, C, W, H) not (N1, ..., N18, C, W, H)
        b_t = self.__networks(self.__networks.map_obs, o_t.flatten(0, 1))\
            .view(len(self), self.__batch_size, self.__n)

        # Get messages
        d_bar_t_tmp = self.__networks(self.__networks.decode_msg, self.msg[self.__t])
        # Mean on agent
        d_bar_t_mean = d_bar_t_tmp.mean(dim=0)
        d_bar_t = ((d_bar_t_mean * self.__nb_agents) - d_bar_t_tmp) / (self.__nb_agents - 1)

        # Map pos in feature space
        lambda_t = self.__networks(self.__networks.map_pos, self.pos.to(th.float))

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=2)

        # Belief LSTM
        h_t_next, c_t_next = \
            self.__networks(self.__networks.belief_unit,
                            self.__h[self.__t],
                            self.__c[self.__t],
                            u_t)

        # Append new h and c (t + 1 step)
        self.__h.append(h_t_next)
        self.__c.append(c_t_next)

        # Evaluate message
        self.msg.append(
            self.__networks(self.__networks.evaluate_msg,
                            self.__h[self.__t + 1].squeeze(0)))

        # Action unit LSTM
        h_caret_t_next, c_caret_t_next = \
            self.__networks(self.__networks.action_unit,
                            self.__h_caret[self.__t],
                            self.__c_caret[self.__t],
                            u_t)

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_next)
        self.__c_caret.append(c_caret_t_next)

        # Define possible actions
        # TODO generic actions
        actions = th.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
                            device=th.device("cuda")
                            if self.is_cuda else th.device("cpu"))\
            .repeat(self.__nb_agents, self.__batch_size, 1, 1)

        # Get action probabilities
        action_scores = self.__networks(self.__networks.policy,
                                        self.__h_caret[self.__t + 1].squeeze(0))

        # If random walk : pick one action with uniform probability
        # Else : greedy policy
        random_walk = (th.rand(self.__nb_agents, img.size(0),
                               device=th.device("cuda")
                               if self.is_cuda else th.device("cpu"))
                       < eps).to(th.float)

        # Greedy policy
        policy_actions = action_scores.argmax(dim=-1)

        # Random choice
        # uniform -> real pb ?
        random_actions = th.randint(0, self.__nb_action,
                                    (self.__nb_agents, img.size(0),),
                                    device=th.device("cuda")
                                    if self.is_cuda else th.device("cpu"))

        # Get final actions of epsilon greedy policy
        idx = (random_walk * random_actions +
               (1. - random_walk) * policy_actions).to(th.long)

        # Get a(t + 1) for each batch image
        idx_one_hot = th.nn.functional.one_hot(idx.view(-1, 1)).view(-1)
        a_t_next = actions.flatten(0, 2)[idx_one_hot.nonzero(), :]\
            .view(len(self), self.__batch_size, 2)

        # Get chosen action probabilities (one per batch image)
        prob = action_scores.flatten(0, 2)[idx_one_hot.nonzero()]\
            .view(len(self), self.__batch_size)

        # Append log probability
        self.__log_probas.append(th.log(prob))

        # Apply action / Upgrade agent state
        self.pos = self.__trans(self.pos.to(th.float),
                                a_t_next, self.__f,
                                self.__size).to(th.long)

        self.__t += 1

    def predict(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        TODO

        :return: tuple <prediction, proba>
        """

        return self.__networks(self.__networks.predict, self.__c[self.__t].squeeze(0)),\
               self.__log_probas[self.__t]

    def cuda(self) -> None:
        """

        :return:
        :rtype:
        """
        self.is_cuda = True

        # TODO pass model union to cuda here ? better real separation NNs wrapper and rl/ma agent

    def __len__(self):
        """

        :return:
        :rtype:
        """
        return self.__nb_agents

    def params_to_json(self, out_json_path: AnyStr) -> None:
        with open(out_json_path, mode="w") as f_json:

            json_raw_txt = "{" \
                           "    \"nb_agent\": " + str(self.__nb_agents) + ",\n" \
                           "    \"hidden_size\": " + str(self.__n) + ",\n" \
                           "    \"window_size\": " + str(self.__f) + ",\n" \
                           "    \"hidden_size_msg\": " + str(self.__n_m) + ",\n" \
                           "    \"size\": " + str(self.__size) + ",\n" \
                           "    \"nb_action\": " + str(self.__nb_action) + "\n" \
                           "}\n"

            f_json.write(json_raw_txt)
            f_json.close()

    def load_from(self, json_file: AnyStr, model_wrapper: ModelsWrapper,
                  obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
                  trans: Callable[[th.Tensor, th.Tensor, int, int], th.Tensor]) -> None:

        with open(json_file, "r") as f_json:
            j_obj = json.load(f_json)

            self.__nb_agents = j_obj["nb_agent"]
            self.__n = j_obj["hidden_size"]
            self.__f = j_obj["window_size"]
            self.__n_m = j_obj["hidden_size_msg"]
            self.__size = j_obj["size"]
            self.__nb_action = j_obj["nb_action"]

            self.__networks = model_wrapper
            self.__obs = obs
            self.__trans = trans

