"""Inference skeleton."""

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def predict(model, samples: Iterable[np.ndarray], out_dir: Optional[Path] = None):
    outputs = []
    for s in samples:
        outputs.append(np.asarray(s))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    return outputs
