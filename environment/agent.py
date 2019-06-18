import torch as th
from networks.ft_extractor import CNN_MNIST, StateToFeatures
from networks.messages import MessageReceiver, MessageSender
from networks.recurrents import BeliefUnit, ActionUnit
from networks.policy import Policy
from networks.prediction import Prediction


class Agent:
    def __init__(self, neighbours: list,
                 n: int, f: int, n_m: int, d: int, size: int, action_size: int, nb_class: int, batch_size: int,
                 obs: callable, trans: callable) -> None:
        self.__neighbours = neighbours
        self.__batch_size = batch_size
        self.__size = size
        self.__f = f
        self.__p = th.randint(self.__size - self.__f, (self.__batch_size, 2))
        self.__n = n
        self.__n_m = n_m
        self.__action_size = action_size
        self.__t = 0

        self.__obs = obs
        self.__trans = trans

        self.__b_theta_5 = CNN_MNIST(self.__f, self.__n)
        self.__d_theta_6 = MessageReceiver(self.__n_m, self.__n)
        self.__lambda_theta_7 = StateToFeatures(d, self.__n)
        self.__belief_unit = BeliefUnit(self.__n)
        self.__m_theta_4 = MessageSender(self.__n, self.__n_m)
        self.__action_unit = ActionUnit(self.__n)
        self.__pi_theta_3 = Policy(self.__action_size, self.__n)
        self.__q_theta_8 = Prediction(self.__n, nb_class)

        self.__h = [th.zeros(1, self.__batch_size, self.__n)]
        self.__c = [th.zeros(1, self.__batch_size, self.__n)]

        self.__h_caret = [th.zeros(1, self.__batch_size, self.__n)]
        self.__c_caret = [th.zeros(1, self.__batch_size, self.__n)]

        self.__m = [th.zeros(self.__batch_size, self.__n_m)]

        self.__probas = th.ones(self.__batch_size)

        self.is_cuda = False

    def new_img(self, batch_size):
        self.__t = 0

        self.__h = [th.zeros(1, batch_size, self.__n)]
        self.__c = [th.zeros(1, batch_size, self.__n)]

        self.__h_caret = [th.zeros(1, batch_size, self.__n)]
        self.__c_caret = [th.zeros(1, batch_size, self.__n)]

        self.__m = [th.zeros(batch_size, self.__n_m)]

        self.__probas = th.ones(batch_size)

        self.__p = th.randint(self.__size - self.__f, (batch_size, 2))

        if self.is_cuda:
            self.cuda()

    def get_t_msg(self) -> th.Tensor:
        return self.__m[self.__t]

    def step(self, img: th.Tensor, random_walk: bool) -> None:
        # Observation
        o_t = self.__obs(img, self.__p, self.__f)

        # Feature space
        b_t = self.__b_theta_5(o_t)

        d_bar_t = th.zeros(img.size(0), self.__n)
        if self.is_cuda:
            d_bar_t = d_bar_t.cuda()

        # Get messages
        for ag in self.__neighbours:
            msg = ag.get_t_msg()
            d_bar_t += self.__d_theta_6(msg)

        d_bar_t /= th.tensor(len(self.__neighbours))

        # Map pos in feature space
        lambda_t = self.__lambda_theta_7(self.__p.to(th.float))

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=1)

        # Belief LSTM
        h_t_p_one, c_t_p_one = self.__belief_unit(self.__h[self.__t],
                                                  self.__c[self.__t],
                                                  u_t.unsqueeze(1))
        # Append new h et c (t + 1 step)
        self.__h.append(h_t_p_one)
        self.__c.append(c_t_p_one)

        # Evaluate message
        self.__m.append(self.__m_theta_4(self.__h[self.__t + 1]).squeeze(0))

        # Action unit LSTM
        h_caret_t_p_one, c_caret_t_p_one = self.__action_unit(self.__h_caret[self.__t],
                                                              self.__c_caret[self.__t],
                                                              u_t.unsqueeze(1))

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_p_one)
        self.__c_caret.append(c_caret_t_p_one)

        actions = th.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        if self.is_cuda:
            actions = actions.cuda()

        action_scores = self.__pi_theta_3(actions, self.__h_caret[self.__t + 1])

        if random_walk:
            idx = th.randint(4, (img.size(0),))
        else:
            idx = action_scores.argmax(dim=1)

        a_t_next = th.zeros(img.size(0), self.__action_size)
        if self.is_cuda:
            a_t_next = a_t_next.cuda()

        for i in range(img.size(0)):
            a_t_next[i, :] = actions[idx[i]]

        tmp = th.zeros(img.size(0))
        if self.is_cuda:
            tmp = tmp.cuda()

        for i in range(img.size(0)):
            tmp[i] = action_scores[i, idx[i]]
        self.__probas *= tmp

        self.__p = self.__trans(self.__p.to(th.float), a_t_next, self.__f, self.__size).to(th.long)

    def step_finished(self):
        self.__t += 1

    def predict(self) -> tuple:
        """
        :return: <prediction, proba>
        """
        return self.__q_theta_8(self.__c[self.__t]).squeeze(0), self.__probas

    def get_networks(self):
        return [self.__b_theta_5, self.__d_theta_6, self.__lambda_theta_7, self.__belief_unit,
                self.__m_theta_4, self.__action_unit, self.__pi_theta_3, self.__q_theta_8]

    def get_probas(self):
        return self.__probas

    def cuda(self):
        self.is_cuda = True

        for n in self.get_networks():
            n.cuda()

        self.__h = [h.cuda() for h in self.__h]

        self.__c = [c.cuda() for c in self.__c]

        self.__h_caret = [h.cuda() for h in self.__h_caret]

        self.__c_caret = [c.cuda() for c in self.__c_caret]

        self.__m = [m.cuda() for m in self.__m]

        self.__probas = self.__probas.cuda()

        self.__p = self.__p.cuda()
