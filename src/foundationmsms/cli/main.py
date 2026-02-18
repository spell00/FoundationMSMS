"""Command line interface entrypoint."""

import argparse
from pathlib import Path

from ..data.download import download_massive, download_pride, download_priority_datasets, download_priority_missing
from ..preprocessing.windowing import make_rt_windows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("foundationmsms")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pride-download")
    p1.add_argument("--pxd", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--protocol", default="s3")

    p2 = sub.add_parser("massive-download")
    p2.add_argument("--msv", required=True)
    p2.add_argument("--out", required=True)

    p3 = sub.add_parser("window")
    p3.add_argument("--voxel-dir", required=True)
    p3.add_argument("--window-sec", type=int, default=30)
    p3.add_argument("--stride-sec", type=int, default=15)

    p4 = sub.add_parser("download-priority")
    p4.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    p4.add_argument("--out", type=Path, default=Path("data/raw"))
    p4.add_argument("--priority", type=int, default=1)
    p4.add_argument("--protocol", type=str, default="s3", help="PRIDE download protocol (s3 or ftp)")
    p4.add_argument("--massive-host", type=str, default="massive-ftp.ucsd.edu")
    p4.add_argument("--mode", type=str, default="DIA", help="Acquisition mode (subfolder under raw)")

    p5 = sub.add_parser("download-priority-debug")
    p5.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    p5.add_argument("--out", type=Path, default=Path("data/raw"))
    p5.add_argument("--priority", type=int, default=1)
    p5.add_argument("--protocol", type=str, default="s3", help="PRIDE download protocol (s3 or ftp)")
    p5.add_argument("--massive-host", type=str, default="massive-ftp.ucsd.edu")
    p5.add_argument("--mode", type=str, default="DIA", help="Acquisition mode (subfolder under raw)")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "pride-download":
        download_pride(args.pxd, Path(args.out), protocol=args.protocol)
    elif args.cmd == "massive-download":
        download_massive(args.msv, Path(args.out))
    elif args.cmd == "window":
        make_rt_windows(Path(args.voxel_dir), window_sec=args.window_sec, stride_sec=args.stride_sec)
    elif args.cmd == "download-priority":
        download_priority_datasets(
            Path(args.config),
            Path(args.out) / args.mode,
            priority=args.priority,
            protocol=args.protocol,
            massive_host=args.massive_host
        )
    elif args.cmd == "download-priority-debug":
        download_priority_missing(
            Path(args.config),
            Path(args.out) / args.mode,
            priority=args.priority,
            protocol=args.protocol,
            massive_host=args.massive_host
        )


if __name__ == "__main__":
    main()
