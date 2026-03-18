"""Experiment logger interfaces for TensorBoard, Comet, and Pluto."""

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    log_dir: str = "logs"
    run_name: Optional[str] = None
    auto_run_subdir: bool = True
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
            if config.auto_run_subdir:
                run_name = config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
                tb_log_dir = Path(config.log_dir) / run_name
            else:
                tb_log_dir = Path(config.log_dir)
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            writers["tensorboard"] = SummaryWriter(log_dir=str(tb_log_dir))
            writers["tensorboard_log_dir"] = str(tb_log_dir)

    if config.use_comet:
        writers["comet"] = None

    if config.use_pluto:
        writers["pluto"] = None

    return writers


# Unified metric logging function at module level
def log_metric(writers: Dict[str, Any], name: str, value: float, step: int = None, head: str = None):
    """
    Log a metric to all enabled loggers. Optionally per classification head.
    """
    tag = f"{name}/{head}" if head else name
    # TensorBoard
    tb = writers.get("tensorboard")
    # if a dict we register each value separately with the key as part of the tag
    if isinstance(value, dict):
        for k, v in value.items():
            sub_tag = f"{tag}/{k}"
            if tb is not None:
                tb.add_scalar(sub_tag, v, step)
            comet_exp = writers.get("comet")
            if comet_exp is not None:
                comet_exp.log_metric(sub_tag, v, step=step)
            pluto_exp = writers.get("pluto")
            if pluto_exp is not None:
                pluto_exp.log_metric(sub_tag, v, step=step)
    elif tb is not None:
        tb.add_scalar(tag, value, step)
        # Comet
        comet_exp = writers.get("comet")
        if comet_exp is not None:
            comet_exp.log_metric(tag, value, step=step)
        # Pluto
        pluto_exp = writers.get("pluto")
        if pluto_exp is not None:
            pluto_exp.log_metric(tag, value, step=step)


def log_hparams(writers: Dict[str, Any], hparam_dict: dict, metric_dict: dict) -> None:
    """
    Log hyperparameters + final metrics for the TensorBoard HParams plugin.
    Creates the parallel-coordinates and scatter-matrix views across runs.
    Each call must include all hparam keys and at least one metric key.
    """
    tb = writers.get("tensorboard")
    if tb is not None:
        # TensorBoard requires metric values to be plain Python floats
        safe_metrics = {k: float(v) for k, v in metric_dict.items()}
        tb.add_hparams(hparam_dict, safe_metrics)
    comet_exp = writers.get("comet")
    if comet_exp is not None:
        for k, v in hparam_dict.items():
            comet_exp.log_parameter(k, v)


def save_code_artifacts(writers: Dict[str, Any], code_paths=None, model_path=None, config_path=None):
    """
    Save code, model, and config artifacts to all enabled loggers.
    code_paths: list of source code files or folders to log
    model_path: path to model checkpoint
    config_path: path to config file
    """
    # Comet
    comet_exp = writers.get("comet")
    if comet_exp is not None:
        if code_paths:
            for path in code_paths:
                comet_exp.log_code(path)
        if model_path:
            comet_exp.log_model("model", model_path)
        if config_path:
            comet_exp.log_asset(config_path, file_name="config.yaml")
    # Pluto
    pluto_exp = writers.get("pluto")
    if pluto_exp is not None:
        if code_paths:
            for path in code_paths:
                pluto_exp.log_code(path)
        if model_path:
            pluto_exp.log_model(model_path)
        if config_path:
            pluto_exp.log_asset(config_path)
    # TensorBoard (limited support)
    tb = writers.get("tensorboard")
    if tb is not None and model_path:
        try:
            tb.add_text("model_checkpoint", str(model_path))
        except Exception:
            pass
