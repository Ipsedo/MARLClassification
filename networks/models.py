from networks.ft_extractor import MNISTCnn, RESISC45Cnn, StateToFeatures
from networks.messages import MessageReceiver, MessageSender, DummyMessageReceiver, DummyMessageSender
from networks.recurrents import BeliefUnit, ActionUnit
from networks.policy import Policy
from networks.prediction import Prediction
import torch.nn as nn


#####################
# Base class test
#####################
class ModelsWrapper(nn.Module):
    map_obs: str = "b_theta_5"
    map_pos: str = "lambda_theta_7"

    decode_msg: str = "d_theta_6"
    evaluate_msg: str = "m_theta_4"

    belief_unit: str = "belief_unit"
    action_unit: str = "action_unit"

    policy: str = "pi_theta_3"
    predict: str = "q_theta_8"

    def __init__(self, map_obs_module: nn.Module,
                 n: int, n_m: int, d: int,
                 nb_action: int, nb_class: int) -> None:
        super().__init__()

        self._networks_dict = nn.ModuleDict({
            self.map_obs: map_obs_module,
            self.map_pos: StateToFeatures(d, n),
            self.decode_msg: MessageReceiver(n_m, n),
            self.evaluate_msg: MessageSender(n, n_m),
            self.belief_unit: BeliefUnit(n),
            self.action_unit: ActionUnit(n),
            self.policy: Policy(nb_action, n),
            self.predict: Prediction(n, nb_class)
        })

    def forward(self, op: str, *args):
        return self._networks_dict[op](*args)


#####################
# MNIST version
#####################
class MNISTModelWrapper(ModelsWrapper):
    def __init__(self, f: int, n: int, n_m: int) -> None:
        super().__init__(MNISTCnn(f, n), n, n_m, 2, 4, 10)


class MNISTModelsWrapperMsgLess(MNISTModelWrapper):
    def __init__(self, f: int, n: int, n_m: int) -> None:
        super().__init__(f, n, n_m)

        """for p in self._networks_dict[self.decode_msg].parameters():
            p.requires_grad = False

        for p in self._networks_dict[self.evaluate_msg].parameters():
            p.requires_grad = False"""
        self._networks_dict.update({
            self.decode_msg: DummyMessageReceiver(n, n_m),
            self.evaluate_msg: DummyMessageSender(n, n_m)
        })


#####################
# RESISC45 version
#####################
class RESISC45ModelsWrapper(ModelsWrapper):
    def __init__(self, f: int, n: int, n_m: int) -> None:
        super().__init__(RESISC45Cnn(f, n), n, n_m, 2, 4, 45)

