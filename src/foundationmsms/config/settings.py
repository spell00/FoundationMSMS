"""Project settings and configuration defaults."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path("/app")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    experiments_dir: Path = Path("experiments")
    logs_dir: Path = Path("logs")
    models_dir: Path = Path("models")
