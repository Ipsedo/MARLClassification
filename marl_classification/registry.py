from dataclasses import dataclass
from typing import Any, Callable

import torch as th
import torchvision.transforms as tr

from .data import (
    AbstractDataset,
    AidDataset,
    KneeMriDataset,
    MnistDataset,
    Resisc45Dataset,
    SkinCancerDataset,
    WorldStratDataset,
)
from .networks.vision import (
    AidCnn,
    KneeMriCnn,
    MnistCnn,
    Resisc45Cnn,
    SkinCancerCnn,
    VisionCnnModule,
    WorldStratCnn,
)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to support a dataset. Adding a dataset means
    adding one entry to DATASET_REGISTRY, nothing else."""

    dataset_constructor: Callable[
        [str, Callable[[Any], th.Tensor]], AbstractDataset
    ]
    cnn_constructor: Callable[[int], VisionCnnModule]


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(MnistDataset, MnistCnn),
    "resisc45": DatasetSpec(Resisc45Dataset, Resisc45Cnn),
    "kneemri": DatasetSpec(KneeMriDataset, KneeMriCnn),
    "aid": DatasetSpec(AidDataset, AidCnn),
    "worldstrat": DatasetSpec(WorldStratDataset, WorldStratCnn),
    "skin_cancer": DatasetSpec(SkinCancerDataset, SkinCancerCnn),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    assert (
        name in DATASET_REGISTRY
    ), f'Unknown dataset "{name}", expected one of {sorted(DATASET_REGISTRY)}'

    return DATASET_REGISTRY[name]


def default_image_pipeline() -> tr.Compose:
    return tr.Compose([tr.ToTensor()])
