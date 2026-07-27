from dataclasses import dataclass
from typing import Any, Callable

import torch as th
import torchvision.transforms as tr

from .data import (
    AIDDataset,
    KneeMRIDataset,
    MNISTDataset,
    RESISC45Dataset,
    SkinCancerDataset,
    WorldStratDataset,
)
from .data.abstract_dataset import AbstractDataset
from .networks.vision import (
    AIDCnn,
    KneeMRICnn,
    MNISTCnn,
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
    "mnist": DatasetSpec(MNISTDataset, MNISTCnn),
    "resisc45": DatasetSpec(RESISC45Dataset, Resisc45Cnn),
    "kneemri": DatasetSpec(KneeMRIDataset, KneeMRICnn),
    "aid": DatasetSpec(AIDDataset, AIDCnn),
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
