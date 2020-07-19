import torch as th
from networks.models import ModelsUnion
from numpy.random import choice
from random import random

from typing import Callable, List, Tuple


class Agent:
    def __init__(self, neighbours: List['Agent'], model_union: ModelsUnion,
                 n: int, f: int, n_m: int,
                 size: int, action_size: int, batch_size: int,
                 obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor],
                 trans: Callable[[th.Tensor, th.Tensor, int, int], th.Tensor]) -> None:
        """
        TODO

        Multi agent reinforcement learning for image classification.
        Constructor method of agent.
        One agent shares the same neural networks (models) with all the other agent.

        :param neighbours: The list of other agents
        :type neighbours: List[Agent]
        :param model_union: The class (wrapper) containing all the neural networks and their call
        :type model_union: ModelsUnion
        :param n: The latent space size
        :type n: int
        :param f:
        :type f: int
        :param n_m:
        :type n_m: int
        :param size: The image side size (squared images)
        :type size: int
        :param action_size: The action space dimension
        :type action_size: int
        :param batch_size: The image batch size
        :type batch_size: int
        :param obs: The function which permits the observation recover
        :type obs: Callable[[th.Tensor, th.Tensor, int], th.Tensor]
        :param trans: The function which permits the transition recover
        :type trans: Callable[[th.Tensor, th.Tensor, int, int], th.Tensor]
        """

        self.__neighbours = neighbours
        self.__batch_size = batch_size
        self.__size = size
        self.__f = f
        self.p = th.randint(self.__size - self.__f, (self.__batch_size, 2))
        self.__n = n
        self.__n_m = n_m
        self.__action_size = action_size
        self.__t = 0

        self.__obs = obs
        self.__trans = trans

        self.__networks = model_union

        self.__h = None
        self.__c = None

        self.__h_caret = None
        self.__c_caret = None

        self.__m = None

        self.__log_probas = None

        self.is_cuda = False

        self.new_img(self.__batch_size)

    def new_img(self, batch_size: int) -> None:
        """
        TODO

        :param batch_size:
        :type batch_size:
        :return:
        :rtype:
        """

        self.__t = 0

        self.__h = [th.zeros(batch_size, self.__n)]
        self.__c = [th.zeros(batch_size, self.__n)]

        self.__h_caret = [th.zeros(batch_size, self.__n)]
        self.__c_caret = [th.zeros(batch_size, self.__n)]

        self.__m = [th.zeros(batch_size, self.__n_m)]

        self.__log_probas = [th.log(th.tensor([1. / 4.] * batch_size))]

        self.p = th.randint(self.__size - self.__f, (batch_size, 2))

        if self.is_cuda:
            self.cuda()

    def get_t_msg(self) -> th.Tensor:
        """
        TODO

        :return:
        :rtype:
        """

        return self.__m[self.__t]

    def step(self, img: th.Tensor, eps: float) -> None:
        """
        TODO

        :param img:
        :type img:
        :param eps:
        :type eps:
        :return:
        :rtype:
        """

        # Observation
        o_t = self.__obs(img, self.p, self.__f)

        # Feature space
        b_t = self.__networks.map_obs(o_t)

        d_bar_t = th.zeros(img.size(0), self.__n,
                           device=th.device("cuda") if self.is_cuda else th.device("cpu"))

        # Get messages
        for ag in self.__neighbours:
            msg = ag.get_t_msg()
            d_bar_t += self.__networks.decode_msg(msg)

        # Compute average message
        d_bar_t /= len(self.__neighbours)

        # Map pos in feature space
        lambda_t = self.__networks.map_pos(self.p.to(th.float))

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=1)

        # Belief LSTM
        h_t_next, c_t_next = \
            self.__networks.belief_unit(self.__h[self.__t], self.__c[self.__t], u_t)

        # Append new h and c (t + 1 step)
        self.__h.append(h_t_next)
        self.__c.append(c_t_next)

        # Evaluate message
        self.__m.append(self.__networks.evaluate_msg(self.__h[self.__t + 1]))

        # Action unit LSTM
        h_caret_t_next, c_caret_t_next = \
            self.__networks.action_unit(self.__h_caret[self.__t], self.__c_caret[self.__t], u_t)

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_next)
        self.__c_caret.append(c_caret_t_next)

        # Define possible actions
        actions = th.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
                            device=th.device("cuda") if self.is_cuda else th.device("cpu"))

        # Get action probabilities
        action_scores = self.__networks.policy(self.__h_caret[self.__t + 1])

        # If random walk : pick one action with uniform probability
        # Else : greedy policy
        random_walk = (th.rand(img.size(0), device=th.device("cuda") if self.is_cuda else th.device("cpu"))
                       < eps).to(th.float)

        # Greedy policy
        policy_actions = action_scores.argmax(dim=-1)

        # Random choice
        # uniform -> real pb ?
        # random_actions[i] = th.tensor(choice(range(4), p=action_scores[i].cpu().detach().numpy()))
        random_actions = th.randint(0, actions.size(0), (img.size(0),),
                                    device=th.device("cuda") if self.is_cuda else th.device("cpu"))

        # Get final actions of epsilon greedy policy
        idx = (random_walk * random_actions + (1. - random_walk) * policy_actions).to(th.long)

        # Get a(t + 1) for each batch image
        a_t_next = actions[idx]

        # Get chosen action probabilities (one per batch image)
        prob = action_scores.gather(1, idx.view(-1, 1)).squeeze(1)

        # Append log probability
        self.__log_probas.append(th.log(prob))

        # Apply action / Upgrade agent state
        self.p = self.__trans(self.p.to(th.float),
                              a_t_next, self.__f,
                              self.__size).to(th.long)

    def step_finished(self) -> None:
        """
        TODO

        :return:
        :rtype:
        """

        self.__t += 1

    def predict(self) -> Tuple[th.Tensor, th.Tensor]:
        """
        TODO

        :return: tuple <prediction, proba>
        """

        #return self.__networks.predict(self.__c[self.__t]), th.cat(self.__log_probas).sum(dim=0)
        return self.__networks.predict(self.__c[self.__t]), self.__log_probas[self.__t]

    def cuda(self) -> None:
        """
        TODO

        :return:
        :rtype:
        """

        self.is_cuda = True

        self.__h = [h.cuda() for h in self.__h]

        self.__c = [c.cuda() for c in self.__c]

        self.__h_caret = [h.cuda() for h in self.__h_caret]

        self.__c_caret = [c.cuda() for c in self.__c_caret]

        self.__m = [m.cuda() for m in self.__m]

        self.__log_probas = [p.cuda() for p in self.__log_probas]

        self.p = self.p.cuda()
