#!/usr/bin/env python3
"""Legacy wrapper for preprocessing CLI.

This file is kept for backward compatibility. Prefer:
    python -m foundationmsms.preprocessing <command>
"""

from foundationmsms.preprocessing.download_and_voxelize import main


if __name__ == "__main__":
    main()
