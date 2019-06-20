import torch as th
from networks.models import ModelsUnion


class Agent:
    def __init__(self, neighbours: list, model_union: ModelsUnion,
                 n: int, f: int, n_m: int, d: int,
                 size: int, action_size: int, nb_class: int, batch_size: int,
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

        self.__networks = model_union

        self.__h = [th.zeros(1, self.__batch_size, self.__n)]
        self.__c = [th.zeros(1, self.__batch_size, self.__n)]

        self.__h_caret = [th.zeros(1, self.__batch_size, self.__n)]
        self.__c_caret = [th.zeros(1, self.__batch_size, self.__n)]

        self.__m = [th.zeros(self.__batch_size, self.__n_m)]

        self.__log_probas = [th.zeros(self.__batch_size)]

        self.is_cuda = False

    def new_img(self, batch_size):
        self.__t = 0

        self.__h = [th.zeros(1, batch_size, self.__n)]
        self.__c = [th.zeros(1, batch_size, self.__n)]

        self.__h_caret = [th.zeros(1, batch_size, self.__n)]
        self.__c_caret = [th.zeros(1, batch_size, self.__n)]

        self.__m = [th.zeros(batch_size, self.__n_m)]

        self.__log_probas = [th.zeros(batch_size)]

        self.__p = th.randint(self.__size - self.__f, (batch_size, 2))

        if self.is_cuda:
            self.cuda()

    def get_t_msg(self) -> th.Tensor:
        return self.__m[self.__t]

    def step(self, img: th.Tensor, random_walk: bool) -> None:
        # Observation
        o_t = self.__obs(img, self.__p, self.__f)

        # Feature space
        b_t = self.__networks.map_obs(o_t)

        d_bar_t = th.zeros(img.size(0), self.__n)
        if self.is_cuda:
            d_bar_t = d_bar_t.cuda()

        # Get messages
        for ag in self.__neighbours:
            msg = ag.get_t_msg()
            d_bar_t += self.__networks.decode_msg(msg)

        d_bar_t /= th.tensor(len(self.__neighbours))

        # Map pos in feature space
        lambda_t = self.__networks.map_pos(self.__p.to(th.float))

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=1).unsqueeze(1)

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
            self.__networks.action_unit(self.__h_caret[self.__t],
                                        self.__c_caret[self.__t],
                                        u_t)

        # Append ĥ et ĉ (t + 1 step)
        self.__h_caret.append(h_caret_t_next)
        self.__c_caret.append(c_caret_t_next)

        actions = th.tensor([[1., 0.],
                             [-1., 0.],
                             [0., 1.],
                             [0., -1.]])

        if self.is_cuda:
            actions = actions.cuda()

        action_scores = self.__networks.policy(self.__h_caret[self.__t + 1])

        if random_walk:
            idx = th.randint(4, (img.size(0),))
        else:
            idx = action_scores.argmax(dim=1)

        a_t_next = actions[idx]

        if self.is_cuda:
            a_t_next = a_t_next.cuda()
            idx = idx.cuda()

        prob = action_scores.gather(1, idx.view(-1, 1)).squeeze(1)
        if self.is_cuda:
            prob = prob.cuda()

        self.__log_probas.append(th.log(prob))

        if th.isnan(self.__h_caret[self.__t + 1]).sum().item() != 0:
            print(self.__h_caret[self.__t + 1])

        self.__p = self.__trans(self.__p.to(th.float),
                                a_t_next, self.__f,
                                self.__size).to(th.long)

    def step_finished(self):
        self.__t += 1

    def predict(self) -> tuple:
        """
        :return: <prediction, proba>
        """
        proba = th.zeros(self.__log_probas[0].size(0))

        if self.is_cuda:
            proba = proba.cuda()

        for p in self.__log_probas:
            proba += p

        return self.__networks.predict(self.__c[self.__t]), proba

    def cuda(self):
        self.is_cuda = True

        self.__h = [h.cuda() for h in self.__h]

        self.__c = [c.cuda() for c in self.__c]

        self.__h_caret = [h.cuda() for h in self.__h_caret]

        self.__c_caret = [c.cuda() for c in self.__c_caret]

        self.__m = [m.cuda() for m in self.__m]

        self.__log_probas = [p.cuda() for p in self.__log_probas]

        self.__p = self.__p.cuda()
