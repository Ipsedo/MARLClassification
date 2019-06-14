import torch as th
from networks.ft_extractor import CNN_MNIST, StateToFeatures
from networks.messages import MessageReceiver, MessageSender
from networks.recurrents import BeliefUnit, ActionUnit


class Agent:
    def __init__(self, neighboors: list, init_pos: th.Tensor,
                 n: int, f: int, n_m: int, d: int, size: int,
                 obs: callable, trans: callable) -> None:
        self.__neighboors = neighboors
        self.__p = init_pos
        self.__f = f
        self.__n = n
        self.__n_m = n_m
        self.__size = size

        self.__obs = obs
        self.__trans = trans

        self.__b_theta_5 = CNN_MNIST(self.__f, self.__n)
        self.__d_theta_6 = MessageReceiver(self.__n_m, self.__n)
        self.__lambda_theta_7 = StateToFeatures(d, self.__n)
        self.__belief_unit = BeliefUnit(self.__n)
        self.__m_theta_4 = MessageSender(self.__n, self.__n_m)
        self.__action_unit = ActionUnit(self.__n)

        self.__h_t = th.zeros(self.__n)
        self.__c_t = th.zeros(self.__n)

        self.__h_caret_t = th.zeros(self.__n)
        self.__c_caret_t = th.zeros(self.__n)

        self.__m_t = th.zeros(self.__n_m)
        self.__m_t_p_one = th.zeros(self.__n_m)

    def new_img(self):
        self.__h_t[:] = 0
        self.__c_t[:] = 0
        self.__m_t[:] = 0
        self.__m_t_p_one[:] = 0

    def act(self, random=False) -> th.Tensor:
        pass

    def get_t_msg(self) -> th.Tensor:
        return self.__m_t

    def step(self, img: th.Tensor) -> None:
        o_t = self.__obs(img, self.__p, self.__f)

        b_t = self.__b_theta_5(o_t.unsqueeze(0)).squeeze(0)

        d_bar_t = th.zeros(self.__n)
        for ag in self.__neighboors:
            d_bar_t += self.__d_theta_6(ag.get_t_msg())

        d_bar_t /= th.tensor(len(self.__neighboors))

        lambda_t = self.__lambda_theta_7(self.__p.to(th.float))

        u_t = th.cat((b_t, d_bar_t, lambda_t))

        h_t_p_one, c_t_p_one = self.__belief_unit(self.__h_t.unsqueeze(0).unsqueeze(0),
                                                  self.__c_t.unsqueeze(0).unsqueeze(0),
                                                  u_t.unsqueeze(0).unsqueeze(0))

        h_t_p_one = h_t_p_one.squeeze(0).squeeze(0)
        c_t_p_one = c_t_p_one.squeeze(0).squeeze(0)

        self.__m_t_p_one = self.__m_theta_4(h_t_p_one)

        h_caret_t_p_one, c_caret_t_p_one = self.__action_unit(self.__h_caret_t.unsqueeze(0).unsqueeze(0),
                                                              self.__c_caret_t.unsqueeze(0).unsqueeze(0),
                                                              u_t.unsqueeze(0).unsqueeze(0))

        h_caret_t_p_one = h_caret_t_p_one.squeeze(0).squeeze(0)
        c_caret_t_p_one = c_caret_t_p_one.squeeze(0).squeeze(0)

        # TODO update policy distribution
        # TODO sample action

        a_t_p_one = th.tensor([1, 0, 0, 0])

        self.__p = self.__trans(self.__p, a_t_p_one, self.__size)

        self.__h_t = h_t_p_one.clone()
        self.__c_t = c_t_p_one.clone()

        self.__h_caret_t = h_caret_t_p_one.clone()
        self.__c_caret_t = c_caret_t_p_one.clone()

        """
        print(self.__h_t.size())
        print(self.__c_t.size())
        print(self.__h_caret_t.size())
        print(self.__c_caret_t.size())
        print(self.__p.size())
        print(self.__m_t.size())
        print(self.__m_t_p_one.size())
        """

    def step_finished(self):
        self.__m_t = self.__m_t_p_one.clone()
        self.__m_t_p_one[:] = 0

    def predict(self) -> th.Tensor:
        """
        :return: th.Tensor scalar class prediction
        """
        pass