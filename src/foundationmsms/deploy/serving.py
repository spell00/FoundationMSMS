"""Model serving skeleton."""

from typing import Any, Dict


def healthcheck() -> Dict[str, Any]:
    return {"status": "ok"}
