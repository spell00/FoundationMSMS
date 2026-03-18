"""Unified logger for console and experiment tracking."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None, log_level: str = "info") -> logging.Logger:
    logger = logging.getLogger(name or "foundationmsms")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set log level
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger.setLevel(level_map.get(log_level.lower(), logging.INFO))
    return logger
