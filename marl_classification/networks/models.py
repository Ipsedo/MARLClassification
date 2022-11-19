import json
from os.path import exists, isfile
from typing import List, Set, Dict, Callable

import torch as th
import torch.nn as nn

from .ft_extractor import (
    MNISTCnn,
    RESISC45Cnn,
    StateToFeatures,
    CNNFtExtract,
    KneeMRICnn,
    AIDCnn
)
from .messages import MessageSender
from .policy import Policy, Critic
from .prediction import Prediction
from .units import Unit


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

    critic: str = "critic"

    module_list: Set[str] = {
        map_obs,
        map_pos,
        evaluate_msg,
        belief_unit,
        action_unit,
        policy,
        critic,
        predict
    }

    # Features extractors - CNN

    mnist: str = "mnist"
    resisc: str = "resisc45"
    knee_mri: str = "kneemri"
    aid: str = "aid"

    ft_extractors: Dict[str, Callable[[int], CNNFtExtract]] = {
        mnist: MNISTCnn,
        resisc: RESISC45Cnn,
        knee_mri: KneeMRICnn,
        aid: AIDCnn
    }

    def __init__(self, ft_extr_str: str, f: int,
                 n_b: int, n_a: int, n_m: int, n_d: int, d: int,
                 actions: List[List[int]], nb_class: int,
                 hidden_size_belief: int,
                 hidden_size_action: int) -> None:
        super().__init__()

        map_obs_module = self.ft_extractors[ft_extr_str](f)

        self.__networks_dict = nn.ModuleDict({
            self.map_obs: map_obs_module,
            self.map_pos: StateToFeatures(d, n_d),
            self.evaluate_msg: MessageSender(n_b, n_m, hidden_size_belief),
            self.belief_unit: Unit(
                map_obs_module.out_size + n_d + n_m, n_b
            ),
            self.action_unit: Unit(
                map_obs_module.out_size + n_d + n_m, n_a
            ),
            self.policy: Policy(len(actions), n_a, hidden_size_action),
            self.critic: Critic(n_a, hidden_size_action),
            self.predict: Prediction(n_b, nb_class, hidden_size_belief)
        })

        self.__ft_extr_str = ft_extr_str

        self.__f = f
        self.__n = n_b
        self.__n_a = n_a
        self.__n_l_b = hidden_size_belief
        self.__n_l_a = hidden_size_action
        self.__n_m = n_m
        self.__n_d = n_d

        self.__d = d
        self.__actions = actions
        self.__nb_class = nb_class

        def __init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(__init_weights)

    def forward(self, op: str, *args):
        return self.__networks_dict[op](*args)

    @property
    def nb_class(self) -> int:
        return self.__nb_class

    @property
    def f(self) -> int:
        return self.__f

    def get_params(self, ops: List[str]) -> List[th.Tensor]:
        return [
            p for op in ops
            for p in self.__networks_dict[op].parameters()
        ]

    def json_args(self, out_json_path: str) -> None:
        json_f = open(out_json_path, "w")

        args_d = {
            "ft_extr_str": self.__ft_extr_str,
            "window_size": self.__f,
            "hidden_size_belief": self.__n,
            "hidden_size_action": self.__n_a,
            "hidden_size_msg": self.__n_m,
            "hidden_size_state": self.__n_d,
            "state_dim": self.__d,
            "actions": self.__actions,
            "class_number": self.__nb_class,
            "hidden_size_linear_belief": self.__n_l_b,
            "hidden_size_linear_action": self.__n_l_a,
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
                args_d["hidden_size_belief"],
                args_d["hidden_size_action"],
                args_d["hidden_size_msg"],
                args_d["hidden_size_state"],
                args_d["state_dim"],
                args_d["actions"],
                args_d["class_number"],
                args_d["hidden_size_linear_belief"],
                args_d["hidden_size_linear_action"]
            )
        except Exception as e:
            print(f"Error while parsing {json_path} "
                  f"and creating {cls.__name__}")
            raise e
