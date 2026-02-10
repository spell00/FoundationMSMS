"""Model registry and factory methods."""

from typing import Callable, Dict


_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def _wrap(fn: Callable):
        _REGISTRY[name] = fn
        return fn

    return _wrap


def create(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return _REGISTRY[name](**kwargs)
