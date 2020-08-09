from abc import ABC

from networks.ft_extractor import \
    MNISTCnn, RESISC45CnnSmall, StateToFeatures
from networks.messages import MessageReceiver, MessageSender
from networks.recurrents import LSTMCellWrapper
from networks.policy import Policy
from networks.prediction import Prediction

from data.dataset import DATASET_CHOICES

import torch as th
import torch.nn as nn

import json

from os.path import exists, isfile

from typing import List


#####################
# Base class test
#####################
class ModelsWrapper(nn.Module, ABC):
    map_obs: str = "b_theta_5"
    map_pos: str = "lambda_theta_7"

    decode_msg: str = "d_theta_6"
    evaluate_msg: str = "m_theta_4"

    belief_unit: str = "belief_unit"
    action_unit: str = "action_unit"

    policy: str = "pi_theta_3"
    predict: str = "q_theta_8"

    def __init__(self, dataset: str, f: int,
                 n: int, n_m: int, d: int,
                 nb_action: int, nb_class: int,
                 hidden_size: int) -> None:
        super().__init__()

        assert dataset in DATASET_CHOICES, \
            f"\"{dataset}\" not in {DATASET_CHOICES}"

        # TODO trouver moyen plus propre que ce branchement
        map_obs_module = None
        if dataset == "mnist":
            map_obs_module = MNISTCnn(f, n)
        elif dataset == "resisc45":
            map_obs_module = RESISC45CnnSmall(f, n)
        else:
            raise Exception(f"Unvalid dataset \"{dataset}\"")

        self._networks_dict = nn.ModuleDict({
            self.map_obs: map_obs_module,
            self.map_pos: StateToFeatures(d, n),
            self.decode_msg: MessageReceiver(n_m, n),
            self.evaluate_msg: MessageSender(n, n_m, hidden_size),
            self.belief_unit: LSTMCellWrapper(n),
            self.action_unit: LSTMCellWrapper(n),
            self.policy: Policy(nb_action, n, hidden_size),
            self.predict: Prediction(n, nb_class, hidden_size)
        })

        self.__dataset = dataset

        self.__f = f
        self.__n = n
        self.__n_l = hidden_size
        self.__n_m = n_m

        self.__d = d
        self.__nb_action = nb_action
        self.__nb_class = nb_class

    def forward(self, op: str, *args):
        return self._networks_dict[op](*args)

    def erase_grad(self, ops: List[str]) -> None:
        """
        Erase gradients from module(s) in op
        :param ops:
        :type ops:
        :return:
        :rtype:
        """

        for op in ops:
            for p in self._networks_dict[op].parameters():
                p.grad = th.zeros_like(p.grad)

    @property
    def nb_class(self) -> int:
        return self.__nb_class

    @property
    def f(self) -> int:
        return self.__f

    def get_params(self, ops: List[str]) -> List[th.Tensor]:
        return [
            p for op in ops
            for p in self._networks_dict[op].parameters()
        ]

    def json_args(self, out_json_path: str) -> None:
        json_f = open(out_json_path, "w")

        args_d = {
            "dataset": self.__dataset,
            "window_size": self.__f,
            "hidden_size": self.__n,
            "hidden_size_msg": self.__n_m,
            "state_dim": self.__d,
            "action_dim": self.__nb_action,
            "class_number": self.__nb_class,
            "hidden_size_linear": self.__n_l
        }

        json.dump(args_d, json_f)

        json_f.close()

    @classmethod
    def from_json(cls, json_path: str) -> 'ModelsWrapper':
        assert exists(json_path) and isfile(json_path), \
            f"\"{json_path}\" does not exist or is not a file"

        json_f = open(json_path, "r")
        args_d = json.load(json_f)
        json_f.close()

        try:
            return cls(
                args_d["dataset"], args_d["window_size"],
                args_d["hidden_size"], args_d["hidden_size_msg"],
                args_d["state_dim"], args_d["action_dim"],
                args_d["class_number"], args_d["hidden_size_linear"]
            )
        except Exception as e:
            print(f"Error while parsing {json_path} "
                  f"and creating {cls.__name__}")
            raise e


#####################
# MNIST version
#####################
class MNISTModelWrapper(ModelsWrapper):
    def __init__(self, f: int, n: int, n_m: int, n_l: int) -> None:
        super().__init__("mnist", f, n, n_m, 2, 4, 10, n_l)


#####################
# RESISC45 version
#####################
class RESISC45ModelsWrapper(ModelsWrapper):
    def __init__(self, f: int, n: int, n_m: int, n_l: int) -> None:
        super().__init__("resisc45", f, n, n_m, 2, 4, 45, n_l)
