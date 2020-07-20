from networks.ft_extractor import CNN_MNIST, CNN_MNIST_2, StateToFeatures
from networks.messages import MessageReceiver, MessageSender
from networks.recurrents import BeliefUnit, ActionUnit
from networks.policy import Policy
from networks.prediction import Prediction
import torch.nn as nn

from typing import Optional, Dict


class ModelsWrapper(nn.Module):
    map_obs: int = 0
    map_pos: int = 1

    decode_msg: int = 2
    evaluate_msg: int = 3

    belief_unit: int = 4
    action_unit: int = 5

    policy: int = 6
    predict: int = 7

    def __init__(self, n: int, f: int, n_m: int, d: int,
                 nb_action: int, nb_class: int,
                 pretrained_seq_conv_cnn: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.__b_theta_5 = CNN_MNIST(f, n)

        if pretrained_seq_conv_cnn is not None:
            self.__b_theta_5.seq_conv = pretrained_seq_conv_cnn

        # self.__b_theta_5 = CNN_MNIST_2(f, n)
        self.__d_theta_6 = MessageReceiver(n_m, n)
        self.__lambda_theta_7 = StateToFeatures(d, n)
        self.__belief_unit = BeliefUnit(n)
        self.__m_theta_4 = MessageSender(n, n_m)
        self.__action_unit = ActionUnit(n)
        self.__pi_theta_3 = Policy(nb_action, n)
        self.__q_theta_8 = Prediction(n, nb_class)

        self.__op_to_module: Dict[int, nn.Module] = {
            0: self.__b_theta_5,
            1: self.__lambda_theta_7,
            2: self.__d_theta_6,
            3: self.__m_theta_4,
            4: self.__belief_unit,
            5: self.__action_unit,
            6: self.__pi_theta_3,
            7: self.__q_theta_8
        }

    def forward(self, op: int, *args):
        return self.__op_to_module[op](*args)



class ModelsUnion:
    def __init__(self, n: int, f: int, n_m: int, d: int, nb_action: int, nb_class: int, pretrained_seq_conv_cnn: nn.Sequential=None):
        self.__b_theta_5 = CNN_MNIST(f, n)

        if pretrained_seq_conv_cnn is not None:
            self.__b_theta_5.seq_conv = pretrained_seq_conv_cnn

        #self.__b_theta_5 = CNN_MNIST_2(f, n)
        self.__d_theta_6 = MessageReceiver(n_m, n)
        self.__lambda_theta_7 = StateToFeatures(d, n)
        self.__belief_unit = BeliefUnit(n)
        self.__m_theta_4 = MessageSender(n, n_m)
        self.__action_unit = ActionUnit(n)
        self.__pi_theta_3 = Policy(nb_action, n)
        self.__q_theta_8 = Prediction(n, nb_class)

    def map_obs(self, o_t):
        return self.__b_theta_5(o_t)

    def decode_msg(self, m_t):
        return self.__d_theta_6(m_t)

    def map_pos(self, p_t):
        return self.__lambda_theta_7(p_t)

    def belief_unit(self, h_t, c_t, u_t):
        return self.__belief_unit(h_t, c_t, u_t)

    def evaluate_msg(self, h_t_next):
        return self.__m_theta_4(h_t_next.squeeze(0))

    def action_unit(self, h_caret_t, c_caret_t, u_t):
        return self.__action_unit(h_caret_t, c_caret_t, u_t)

    def policy(self, h_caret_t_next):
        return self.__pi_theta_3(h_caret_t_next.squeeze(0))

    def predict(self, c_t):
        return self.__q_theta_8(c_t.squeeze(0))

    def get_networks(self):
        return [self.__b_theta_5, self.__d_theta_6, self.__lambda_theta_7, self.__belief_unit,
                self.__m_theta_4, self.__action_unit, self.__pi_theta_3, self.__q_theta_8]

    def parameters(self):
        return [p for net in self.get_networks() for p in net.parameters()]

    def cuda(self):
        for n in self.get_networks():
            n.cuda()
