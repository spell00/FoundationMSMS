"""Dataset definitions and loaders."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass
class VoxelSample:
    coords: np.ndarray
    vals: np.ndarray
    path: Path


def iter_voxel_npz(paths: Iterable[Path]) -> Iterator[VoxelSample]:
    for p in paths:
        data = np.load(p)
        yield VoxelSample(coords=data["coords"], vals=data["vals"], path=p)
