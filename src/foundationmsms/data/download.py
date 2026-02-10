"""Download utilities for PRIDE/MassIVE and other sources."""

from pathlib import Path
from typing import Optional

from foundationmsms.utils.shell import run


def download_pride(pxd: str, out_dir: Path, protocol: str = "s3") -> None:
    cmd = [
        "pridepy",
        "download-all-public-raw-files",
        "-a",
        pxd,
        "-o",
        str(out_dir),
        "-p",
        protocol,
        "--skip-if-downloaded-already",
    ]
    run(cmd)


def download_massive(msv: str, out_dir: Path, host: str = "massive-ftp.ucsd.edu") -> None:
    cmd = [
        "lftp",
        "-e",
        f"set net:max-retries 2; set net:timeout 20; set ftp:ssl-allow no; "
        f"open -u {msv},anonymous {host}; mirror --continue --verbose / {out_dir}; bye",
    ]
    run(cmd)
