"""Shell helpers."""

import subprocess
from typing import List


def run(cmd: List[str], check: bool = True) -> None:
    # Don't capture output - let it go directly to terminal for progress bars
    p = subprocess.run(cmd, check=check)
    if p.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
