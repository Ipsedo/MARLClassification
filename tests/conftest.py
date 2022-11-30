import shutil
from os import mkdir
from os.path import abspath, exists, isdir, join

import pytest
from pytest import Session

from marl_classification.environment import (
    MultiAgent,
    obs_generic,
    trans_generic,
)
from marl_classification.networks import ModelsWrapper

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


@pytest.fixture(scope="session", name="marl_m")
def get_marl_m(dim: int, nb_class: int, nb_agent: int) -> MultiAgent:

    action = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    n_b = 23
    n_a = 22
    n_m = 21

    f = 12

    model_wrapper = ModelsWrapper(
        "mnist",
        f,
        n_b,
        n_a,
        n_m,
        20,
        dim,
        action,
        nb_class,
        24,
        25,
    )

    return MultiAgent(
        nb_agent,
        model_wrapper,
        n_b,
        n_a,
        f,
        n_m,
        action,
        obs_generic,
        trans_generic,
    )


@pytest.fixture(scope="module", name="tmp_path")
def get_tmp_path() -> str:
    return __TMP_PATH


def pytest_sessionstart(session: Session) -> None:
    if not exists(__TMP_PATH):
        mkdir(__TMP_PATH)
    elif not isdir(__TMP_PATH):
        pytest.fail(f'"{__TMP_PATH}" is not a directory')


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    shutil.rmtree(__TMP_PATH)
