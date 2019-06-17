import torch as th
from networks.ft_extractor import CNN_MNIST, StateToFeatures
from networks.messages import MessageReceiver, MessageSender
from networks.recurrents import BeliefUnit, ActionUnit
from networks.policy import Policy
from networks.prediction import Prediction


class Agent:
    def __init__(self, neighbours: list, init_pos: th.Tensor,
                 n: int, f: int, n_m: int, d: int, size: int, action_size: int, nb_class: int,
                 obs: callable, trans: callable) -> None:
        self.__neighbours = neighbours
        self.__p = init_pos
        self.__f = f
        self.__n = n
        self.__n_m = n_m
        self.__size = size
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

        self.__h = [th.zeros(self.__n)]
        self.__c = [th.zeros(self.__n)]

        self.__h_caret = [th.zeros(self.__n)]
        self.__c_caret = [th.zeros(self.__n)]

        self.__m = [th.zeros(self.__n_m)]

        self.__probas = th.tensor([1.0])

    def new_img(self):
        self.__t = 0

        self.__h = [th.zeros(self.__n)]
        self.__c = [th.zeros(self.__n)]

        self.__h_caret = [th.zeros(self.__n)]
        self.__c_caret = [th.zeros(self.__n)]

        self.__m = [th.zeros(self.__n_m)]

        self.__probas = th.tensor([1.0])

    def get_t_msg(self) -> th.Tensor:
        return self.__m[self.__t]

    def step(self, img: th.Tensor) -> None:
        # Observation
        o_t = self.__obs(img, self.__p, self.__f)

        # Feature space
        b_t = self.__b_theta_5(o_t.unsqueeze(0)).squeeze(0)

        # Get messages
        d_bar_t = th.zeros(self.__n)
        for ag in self.__neighbours:
            d_bar_t += self.__d_theta_6(ag.get_t_msg())

        d_bar_t /= th.tensor(len(self.__neighbours))

        # Map pos in feature space
        lambda_t = self.__lambda_theta_7(self.__p.to(th.float))

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t))

        # Belief LSTM
        h_t_p_one, c_t_p_one = self.__belief_unit(self.__h[self.__t].unsqueeze(0).unsqueeze(0),
                                                  self.__c[self.__t].unsqueeze(0).unsqueeze(0),
                                                  u_t.unsqueeze(0).unsqueeze(0))
        # Append new h et c (t + 1 step)
        self.__h.append(h_t_p_one.squeeze(0).squeeze(0))
        self.__c.append(c_t_p_one.squeeze(0).squeeze(0))

        # Evaluate message
        self.__m.append(self.__m_theta_4(self.__h[self.__t + 1]))

        # Action unit LSTM
        h_caret_t_p_one, c_caret_t_p_one = self.__action_unit(self.__h_caret[self.__t].unsqueeze(0).unsqueeze(0),
                                                              self.__c_caret[self.__t].unsqueeze(0).unsqueeze(0),
                                                              u_t.unsqueeze(0).unsqueeze(0))

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_p_one.squeeze(0).squeeze(0))
        self.__c_caret.append(c_caret_t_p_one.squeeze(0).squeeze(0))

        actions = th.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        action_scores = self.__pi_theta_3(actions, self.__h_caret[self.__t + 1])
        idx = action_scores.argmax()
        a_t_p_one = actions[idx].squeeze(0)

        self.__probas *= action_scores[idx]

        self.__p = self.__trans(self.__p.to(th.float), a_t_p_one, self.__f, self.__size).to(th.long)

    def step_finished(self):
        self.__t += 1

    def predict(self) -> tuple:
        """
        :return: <prediction, proba>
        """
        return self.__q_theta_8(self.__c[self.__t]), self.__probas

    def get_networks(self):
        return [self.__b_theta_5, self.__d_theta_6, self.__lambda_theta_7, self.__belief_unit,
                self.__m_theta_4, self.__action_unit, self.__pi_theta_3, self.__q_theta_8]

    def get_probas(self):
        return self.__probas
