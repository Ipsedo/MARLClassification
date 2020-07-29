import torch as th
from networks.models import ModelsWrapper

from typing import Callable, Tuple

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

        self.__action_probas = None

        # CPU vs GPU
        self.__is_cuda = False
        self.__device_str = "cpu"

    def new_episode(self, batch_size: int) -> None:
        """

        :param batch_size:
        :type batch_size:
        :return:
        :rtype:
        """

        self.__batch_size = batch_size

        self.__t = 0

        self.__h = [
            th.zeros(self.__nb_agents, batch_size, self.__n,
                     device=th.device(self.__device_str))
        ]
        self.__c = [
            th.zeros(self.__nb_agents, batch_size, self.__n,
                     device=th.device(self.__device_str))
        ]

        self.__h_caret = [
            th.zeros(self.__nb_agents, batch_size, self.__n,
                     device=th.device(self.__device_str))
        ]
        self.__c_caret = [
            th.zeros(self.__nb_agents, batch_size, self.__n,
                     device=th.device(self.__device_str))
        ]

        self.msg = [
            th.zeros(self.__nb_agents, batch_size, self.__n_m,
                     device=th.device(self.__device_str))
        ]

        self.__action_probas = [
            th.ones(self.__nb_agents, batch_size,
                    device=th.device(self.__device_str)) / self.__nb_action
        ]

        self.pos = th.randint(self.__size - self.__f,
                              (self.__nb_agents, batch_size, 2),
                              device=th.device(self.__device_str))

    def step(self, img: th.Tensor) -> None:
        """

        :param img:
        :type img:
        :return:
        :rtype:
        """

        # Observation
        o_t = self.__obs(img, self.pos, self.__f)

        # Feature space
        # CNN need (N, C, W, H) not (N1, ..., N18, C, W, H)
        b_t = self.__networks(self.__networks.map_obs, o_t.flatten(0, -4))\
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
                            device=th.device(self.__device_str))

        # Get action probabilities
        action_scores = self.__networks(self.__networks.policy,
                                        self.__h_caret[self.__t + 1].squeeze(0))

        # Greedy policy
        prob, policy_actions = action_scores.max(dim=-1)

        # Create next action mask
        actions_mask = th.arange(0, action_scores.size(-1),
                                 device=th.device(self.__device_str))
        actions_mask = actions_mask.view(-1, 1)\
            .repeat(1, actions.size(-1))\
            .view(1, 1, action_scores.size(-1), actions.size(-1))
        actions_mask = actions_mask == policy_actions.view(*policy_actions.size(), 1, 1)

        a_t_next = actions.masked_select(actions_mask)\
            .view(self.__nb_agents, self.__batch_size, actions.size(-1))

        # Append log probability
        self.__action_probas.append(prob)

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

        # Pour proba -> big pb stabilité numérique (explosion / disparition gradient)
        #
        # Idée 1
        # prob [0; 1]         -> prod on step -> prod on agent
        # log prob [-inf, 0]  -> sum          -> sum            : le plus censé
        #
        # Idée 2
        # chemin par step (5 step pour un agent par exemple) :
        #   pas un trirage indépendant ie pas de p(p_5, p_4, ..., p_0) - LSTM / Recurrent
        #   plus p(p_5 | p_4 | ... | p_0) <-> p(A /\ B) -> flemme donc on prend le dernier ie p_5
        # chemin par agent équiprobable -> moyenne des probas

        # mean on agent at last step
        return self.__networks(self.__networks.predict, self.__c[-1].squeeze(0)).mean(dim=0),\
               self.__action_probas[-1].log().mean(dim=0)

    @property
    def is_cuda(self) -> bool:
        return self.__is_cuda

    def cuda(self) -> None:
        """

        :return:
        :rtype:
        """
        self.__is_cuda = True
        self.__device_str = "cuda"

    def cpu(self) -> None:
        self.__is_cuda = False
        self.__device_str = "cpu"

    def __len__(self):
        """

        :return:
        :rtype:
        """
        return self.__nb_agents

    def params_to_json(self, out_json_path: str) -> None:
        with open(out_json_path, mode="w") as f_json:

            json_raw_txt = "{\n" \
                           "    \"nb_agent\": " + str(self.__nb_agents) + ",\n" \
                           "    \"hidden_size\": " + str(self.__n) + ",\n" \
                           "    \"window_size\": " + str(self.__f) + ",\n" \
                           "    \"hidden_size_msg\": " + str(self.__n_m) + ",\n" \
                           "    \"size\": " + str(self.__size) + ",\n" \
                           "    \"nb_action\": " + str(self.__nb_action) + "\n" \
                           "}\n"

            f_json.write(json_raw_txt)
            f_json.close()

    @classmethod
    def load_from(cls, json_file: str, model_wrapper: ModelsWrapper,
                  obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
                  trans: Callable[[th.Tensor, th.Tensor, int, int], th.Tensor]) -> 'MultiAgent':

        with open(json_file, "r") as f_json:
            j_obj = json.load(f_json)

            try:
                nb_agents = j_obj["nb_agent"]
                n = j_obj["hidden_size"]
                f = j_obj["window_size"]
                n_m = j_obj["hidden_size_msg"]
                size = j_obj["size"]
                nb_action = j_obj["nb_action"]

                obs = obs
                trans = trans

                return cls(nb_agents, model_wrapper, n, f, n_m, size, nb_action, obs, trans)
            except Exception as e:
                print(f"Exception during loading MultiAgent from file !\nRaised Exception :")
                raise e

