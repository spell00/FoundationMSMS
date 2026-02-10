"""Shell helpers."""

import subprocess
from typing import List


def run(cmd: List[str], check: bool = True) -> None:
    p = subprocess.run(cmd, check=check)
    if p.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
