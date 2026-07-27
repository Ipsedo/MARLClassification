import shutil
from os import mkdir
from os.path import abspath, exists, isdir, join
from typing import Tuple

import pytest
from pytest import Session

from marl_classification.core import Environment, MultiAgent
from marl_classification.networks import ModelsWrapper
from marl_classification.networks.vision import MnistCnn

__TMP_PATH = abspath(join(__file__, "..", "tmp"))


@pytest.fixture(scope="session", name="batch_size")
def get_batch_size() -> int:
    return 19


@pytest.fixture(scope="session", name="nb_agent")
def get_nb_agent() -> int:
    return 5


@pytest.fixture(scope="session", name="nb_class")
def get_nb_class() -> int:
    return 10


@pytest.fixture(scope="session", name="step")
def get_step() -> int:
    return 7


@pytest.fixture(scope="session", name="dim")
def get_dim() -> int:
    return 2


@pytest.fixture(scope="session", name="window_size")
def get_window_size() -> int:
    return 12


@pytest.fixture(scope="session", name="actions")
def get_actions() -> list[list[int]]:
    return [[1, 0], [-1, 0], [0, 1], [0, -1]]


@pytest.fixture(scope="session", name="height_width")
def get_height_width() -> Tuple[int, int]:
    return 28, 28


@pytest.fixture(scope="session", name="model_wrapper")
def get_model_wrapper(
    dim: int,
    nb_class: int,
    window_size: int,
    actions: list[list[int]],
) -> ModelsWrapper:
    n_b = 23
    n_a = 22
    n_m = 21

    return ModelsWrapper(
        MnistCnn(window_size),
        n_b,
        n_a,
        n_m,
        20,
        19,
        dim,
        len(actions),
        nb_class,
        24,
        25,
    )


@pytest.fixture(scope="session", name="marl_m")
def get_marl_m(nb_agent: int, model_wrapper: ModelsWrapper) -> MultiAgent:
    return MultiAgent(nb_agent, model_wrapper)


@pytest.fixture(name="env")
def get_env(actions: list[list[int]], window_size: int) -> Environment:
    return Environment(actions, window_size)


@pytest.fixture(scope="module", name="tmp_path")
def get_tmp_path() -> str:
    return __TMP_PATH


# pylint: disable=(unused-argument)
def pytest_sessionstart(session: Session) -> None:
    if not exists(__TMP_PATH):
        mkdir(__TMP_PATH)
    elif not isdir(__TMP_PATH):
        pytest.fail(f'"{__TMP_PATH}" is not a directory')


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    shutil.rmtree(__TMP_PATH)


# pylint: enable=(unused-argument)
