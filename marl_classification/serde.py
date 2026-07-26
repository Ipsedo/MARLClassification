import json
from dataclasses import dataclass
from os.path import exists, isfile

from .core import Environment, MultiAgent
from .networks import ModelsWrapper
from .registry import get_dataset_spec


@dataclass(frozen=True)
class MarlConfig:
    """Serializable description of a trained MARL setup."""

    ft_extr_str: str
    window_size: int
    hidden_size_belief: int
    hidden_size_action: int
    hidden_size_msg: int
    hidden_size_state: int
    state_dim: int
    actions: list[list[int]]
    nb_class: int
    hidden_size_linear_belief: int
    hidden_size_linear_action: int

    def save_marl_config(self, out_json_path: str) -> None:
        # legacy "marl.json" keys, kept for backward compatibility
        args_d = {
            "ft_extr_str": self.ft_extr_str,
            "window_size": self.window_size,
            "hidden_size_belief": self.hidden_size_belief,
            "hidden_size_action": self.hidden_size_action,
            "hidden_size_msg": self.hidden_size_msg,
            "hidden_size_state": self.hidden_size_state,
            "state_dim": self.state_dim,
            "actions": self.actions,
            "class_number": self.nb_class,
            "hidden_size_linear_belief": self.hidden_size_linear_belief,
            "hidden_size_linear_action": self.hidden_size_linear_action,
        }

        with open(out_json_path, "w", encoding="utf-8") as json_f:
            json.dump(args_d, json_f)

    @classmethod
    def load_marl_config(cls, json_path: str) -> "MarlConfig":
        assert exists(json_path) and isfile(
            json_path
        ), f'"{json_path}" does not exist or is not a file'

        with open(json_path, "r", encoding="utf-8") as json_f:
            args_d = json.load(json_f)

        return cls(
            ft_extr_str=args_d["ft_extr_str"],
            window_size=args_d["window_size"],
            hidden_size_belief=args_d["hidden_size_belief"],
            hidden_size_action=args_d["hidden_size_action"],
            hidden_size_msg=args_d["hidden_size_msg"],
            hidden_size_state=args_d["hidden_size_state"],
            state_dim=args_d["state_dim"],
            actions=args_d["actions"],
            nb_class=args_d["class_number"],
            hidden_size_linear_belief=args_d["hidden_size_linear_belief"],
            hidden_size_linear_action=args_d["hidden_size_linear_action"],
        )

    def build_networks(self) -> ModelsWrapper:
        spec = get_dataset_spec(self.ft_extr_str)

        return ModelsWrapper(
            spec.cnn_constructor(self.window_size),
            self.hidden_size_belief,
            self.hidden_size_action,
            self.hidden_size_msg,
            self.hidden_size_state,
            self.state_dim,
            len(self.actions),
            self.nb_class,
            self.hidden_size_linear_belief,
            self.hidden_size_linear_action,
        )

    def build_environment(self) -> Environment:
        return Environment(self.actions, self.window_size)

    def build_marl(
        self, nb_agents: int
    ) -> tuple[ModelsWrapper, MultiAgent, Environment]:
        networks = self.build_networks()

        return (
            networks,
            MultiAgent(nb_agents, networks),
            self.build_environment(),
        )
