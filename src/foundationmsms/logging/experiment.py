"""Experiment logger interfaces for TensorBoard, Comet, and Pluto."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    log_dir: str = "logs"
    project: str = "foundationmsms"
    use_tensorboard: bool = True
    use_comet: bool = False
    use_pluto: bool = False


def init_experiment(config: ExperimentConfig):
    writers: Dict[str, Any] = {}

    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            SummaryWriter = None
        if SummaryWriter is not None:
            writers["tensorboard"] = SummaryWriter(log_dir=config.log_dir)

    if config.use_comet:
        writers["comet"] = None

    if config.use_pluto:
        writers["pluto"] = None

    return writers
