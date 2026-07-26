from dataclasses import dataclass

import torch as th
from torch import nn

from .ft_extractor import CNNFtExtract, StateToFeatures
from .message import MessageSender, aggregate_messages
from .policy import Critic, Policy
from .prediction import Prediction
from .recurrent import LSTMCellWrapper


@dataclass
class ModelOutput:
    actions_probabilities: th.Tensor
    values: th.Tensor
    predictions: th.Tensor
    messages: th.Tensor


@dataclass
class RecurrentOutput:
    h: th.Tensor
    c: th.Tensor
    h_caret: th.Tensor
    c_caret: th.Tensor


class ModelsWrapper(nn.Module):
    """
    Bundles every agent network. The features extractor is injected, so
    this class has no knowledge of the underlying dataset.
    """

    def __init__(
        self,
        ft_extractor: CNNFtExtract,
        n_b: int,
        n_a: int,
        n_m: int,
        n_d: int,
        d: int,
        nb_action: int,
        nb_class: int,
        hidden_size_belief: int,
        hidden_size_action: int,
    ) -> None:
        super().__init__()

        self.__map_obs = ft_extractor
        self.__map_pos = StateToFeatures(d, n_d)
        self.__evaluate_msg = MessageSender(n_b, n_m, hidden_size_belief)
        self.__belief_unit = LSTMCellWrapper(
            ft_extractor.out_size + n_d + n_m, n_b
        )
        self.__action_unit = LSTMCellWrapper(
            ft_extractor.out_size + n_d + n_m, n_a
        )
        self.__policy = Policy(nb_action, n_a, hidden_size_action)
        self.__critic = Critic(n_a, hidden_size_action)
        self.__predict = Prediction(n_b, nb_class, hidden_size_belief)

        self.__n_b = n_b
        self.__n_a = n_a
        self.__n_m = n_m
        self.__nb_action = nb_action
        self.__nb_class = nb_class

        def __init_weights(m: nn.Module) -> None:
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(__init_weights)

    def forward(
        self,
        img_patch: th.Tensor,
        msg_t: th.Tensor,
        norm_pos: th.Tensor,
        recurrent_hidden: RecurrentOutput,
    ) -> tuple[ModelOutput, RecurrentOutput]:
        nb_agent = img_patch.size(0)
        batch_size = img_patch.size(1)

        # Feature space
        # CNN need (N, C, S1, S2, ..., Sd)
        # got (Na, Nb, C, S1, S2, ..., Sd)
        # => flatten agent and batch dims
        b_t = self.__map_obs(
            img_patch.flatten(0, 1),
        ).view(nb_agent, batch_size, -1)

        # Mean of the messages sent by the other agents
        d_bar_t = aggregate_messages(msg_t)

        # Map pos in feature space
        lambda_t = self.__map_pos(norm_pos)

        # LSTMs input
        u_t = th.cat((b_t, d_bar_t, lambda_t), dim=2)

        # Belief LSTM
        h_t_next, c_t_next = self.__belief_unit(
            recurrent_hidden.h,
            recurrent_hidden.c,
            u_t,
        )

        # Evaluate message
        new_msg = self.__evaluate_msg(
            h_t_next,
        )

        # Action unit LSTM
        h_caret_t_next, c_caret_t_next = self.__action_unit(
            recurrent_hidden.h_caret,
            recurrent_hidden.c_caret,
            u_t,
        )

        # Get action probabilities
        action_scores = self.__policy(
            h_caret_t_next,
        )

        # values
        values = self.__critic(h_caret_t_next)

        # predictions
        predictions = self.__predict(h_t_next)

        return ModelOutput(
            action_scores, values, predictions, new_msg
        ), RecurrentOutput(h_t_next, c_t_next, h_caret_t_next, c_caret_t_next)

    @property
    def n_b(self) -> int:
        return self.__n_b

    @property
    def n_a(self) -> int:
        return self.__n_a

    @property
    def n_m(self) -> int:
        return self.__n_m

    @property
    def nb_action(self) -> int:
        return self.__nb_action

    @property
    def nb_class(self) -> int:
        return self.__nb_class

    @property
    def device(self) -> th.device:
        return next(self.parameters()).device
