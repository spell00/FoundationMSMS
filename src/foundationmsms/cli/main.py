"""Command line interface entrypoint."""

import argparse
from pathlib import Path

from foundationmsms.data.download import download_massive, download_pride
from foundationmsms.preprocessing.windowing import make_rt_windows


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


if __name__ == "__main__":
    main()
