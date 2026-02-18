"""Training loop skeleton."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..logging.logger import get_logger


@dataclass
class TrainConfig:
    epochs: int = 10
    seed: int = 42
    log_every: int = 50


def train(
    model,
    dataset: Iterable,
    out_dir: Path,
    config: Optional[TrainConfig] = None,
) -> None:
    cfg = config or TrainConfig()
    logger = get_logger("training")
    rng = np.random.default_rng(cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        for step, batch in enumerate(dataset):
            _ = rng.random()
            if step % cfg.log_every == 0:
                logger.info("epoch=%s step=%s", epoch, step)
