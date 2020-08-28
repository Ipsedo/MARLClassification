from networks.ft_extractor import \
    MNISTCnn, RESISC45Cnn, RESISC45CnnSmall, \
    StateToFeatures, CNNFtExtract
from networks.messages import MessageSender
from networks.recurrents import LSTMCellWrapper
from networks.policy import Policy
from networks.prediction import Prediction

from data.dataset import DATASET_CHOICES

import torch as th
import torch.nn as nn

import json

from os.path import exists, isfile

from typing import List, Set, Dict, Callable


#####################
# Base class test
#####################
class ModelsWrapper(nn.Module):
    # Modules
    map_obs: str = "b_theta_5"
    map_pos: str = "lambda_theta_7"

    evaluate_msg: str = "m_theta_4"

    belief_unit: str = "belief_unit"
    action_unit: str = "action_unit"

    policy: str = "pi_theta_3"
    predict: str = "q_theta_8"

    module_list: Set[str] = {
        map_obs, map_pos,
        evaluate_msg,
        belief_unit, action_unit,
        policy, predict
    }

    # Features extractors - CNN

    mnist: str = "mnist"
    resisc_small: str = "resisc45_small"
    resisc: str = "resisc45"

    ft_extractors: Dict[str, Callable[[int], CNNFtExtract]] = {
        mnist: MNISTCnn,
        resisc_small: RESISC45CnnSmall,
        resisc: RESISC45Cnn
    }

    def __init__(self, ft_extr_str: str, f: int,
                 n: int, n_m: int, n_d: int, d: int,
                 nb_action: int, nb_class: int,
                 hidden_size: int) -> None:
        super().__init__()

        map_obs_module = self.ft_extractors[ft_extr_str](f)

        self._networks_dict = nn.ModuleDict({
            self.map_obs: map_obs_module,
            self.map_pos: StateToFeatures(d, n_d),
            self.evaluate_msg: MessageSender(n, n_m, hidden_size),
            self.belief_unit: LSTMCellWrapper(
                map_obs_module.out_size + n_d + n_m, n),
            self.action_unit: LSTMCellWrapper(
                map_obs_module.out_size + n_d + n_m, n),
            self.policy: Policy(nb_action, n, hidden_size),
            self.predict: Prediction(n, nb_class, hidden_size)
        })

        self.__ft_extr_str = ft_extr_str

        self.__f = f
        self.__n = n
        self.__n_l = hidden_size
        self.__n_m = n_m
        self.__n_d = n_d

        self.__d = d
        self.__nb_action = nb_action
        self.__nb_class = nb_class

    def forward(self, op: str, *args):
        return self._networks_dict[op](*args)

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
            "ft_extr_str": self.__ft_extr_str,
            "window_size": self.__f,
            "hidden_size": self.__n,
            "hidden_size_msg": self.__n_m,
            "hidden_size_state": self.__n_d,
            "state_dim": self.__d,
            "nb_action": self.__nb_action,
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
                args_d["ft_extr_str"],
                args_d["window_size"],
                args_d["hidden_size"],
                args_d["hidden_size_msg"],
                args_d["hidden_size_state"],
                args_d["state_dim"],
                args_d["nb_action"],
                args_d["class_number"],
                args_d["hidden_size_linear"]
            )
        except Exception as e:
            print(f"Error while parsing {json_path} "
                  f"and creating {cls.__name__}")
            raise e
