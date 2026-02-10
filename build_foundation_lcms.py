import argparse
import os
import re
import subprocess
import time
import selectors
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm

# -----------------------------
# Helpers: safe shell execution
# -----------------------------
def _dir_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    file_count = 0
    total_bytes = 0
    for p in path.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    return file_count, total_bytes


def run(
    cmd: list[str],
    check: bool = True,
    progress_dir: Optional[Path] = None,
    progress_label: str = "progress",
    progress_interval_sec: int = 0,
) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    start_time = time.time()
    last_output_time = start_time
    next_tick = start_time + progress_interval_sec

    while True:
        if process.poll() is not None:
            break

        events = selector.select(timeout=1.0)
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                last_output_time = time.time()
                print(line, end="", flush=True)

        now = time.time()
        if progress_dir is not None and progress_interval_sec > 0 and now >= next_tick:
            files, total_bytes = _dir_stats(progress_dir)
            mb = total_bytes / (1024 * 1024)
            elapsed = int(now - start_time)
            quiet = int(now - last_output_time)
            print(
                f"[{progress_label}] files={files} size={mb:.2f}MB elapsed={elapsed}s no_output_for={quiet}s",
                flush=True,
            )
            next_tick = now + progress_interval_sec

    for line in process.stdout:
        print(line, end="", flush=True)

    if process.returncode != 0 and check:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
def which(name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(name)

# -----------------------------
# Downloaders
# -----------------------------
def pride_download_raw(project: str, out_dir: Path, protocol: str = "ftp") -> None:
    """
    Uses pridepy to download all public RAW files for a PRIDE project.
    pridepy supports ftp/aspera/globus/s3 depending on your environment. :contentReference[oaicite:4]{index=4}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if which("pridepy") is None:
        raise RuntimeError("pridepy not found. Install with: pip install -U pridepy")

    def _run_with(proto: str) -> None:
        cmd = [
            "pridepy",
            "download-all-public-raw-files",
            "-a",
            project,
            "-o",
            str(out_dir),
            "-p",
            proto,
            "--skip-if-downloaded-already",
        ]
        run(cmd, progress_dir=out_dir, progress_label=f"pride:{project}", progress_interval_sec=0)

    if protocol == "auto":
        try:
            _run_with("s3")
        except RuntimeError:
            _run_with("ftp")
    else:
        _run_with(protocol)


def massive_download_dataset(msv: str, out_dir: Path) -> None:
    """
    MassIVE commonly exposes dataset files via massive-ftp with a dataset-specific username (MSV accession). :contentReference[oaicite:5]{index=5}
    This is a best-effort FTP mirror download. Exact folder layouts vary by dataset.
    """
    if not re.match(r"^MSV\d{9}$", msv) or msv == "MSV000000000":
        raise RuntimeError(
            f"Invalid MassIVE accession: {msv}. Use a real ID like MSV000012345."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if which("lftp") is None:
        raise RuntimeError("lftp not found. Install with: sudo apt-get install -y lftp")

    host = "massive-ftp.ucsd.edu"
    user = msv
    password = "anonymous"

    cmd = [
        "lftp",
        "-e",
        f"set net:max-retries 2; set net:timeout 20; set ftp:ssl-allow no; "
        f"open -u {user},{password} {host}; "
        f"mirror --continue --verbose / {out_dir}; "
        f"bye",
    ]
    try:
        run(cmd, progress_dir=out_dir, progress_label=f"massive:{msv}", progress_interval_sec=0)
    except RuntimeError as exc:
        raise RuntimeError(
            f"MassIVE download failed for {msv}. Verify the accession exists and "
            f"that massive-ftp.ucsd.edu is reachable. Original error: {exc}"
        ) from exc

# -----------------------------
# Conversion: RAW -> mzML
# -----------------------------
def msconvert_to_mzml(raw_path: Path, mzml_path: Path, centroid: bool = True) -> None:
    """
    Requires ProteoWizard msconvert on PATH.
    """
    if which("msconvert") is None:
        raise RuntimeError("msconvert not found. Install ProteoWizard and ensure msconvert is on PATH.")

    mzml_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "msconvert",
        str(raw_path),
        "--mzML",
        "-o",
        str(mzml_path.parent),
    ]
    if centroid:
        # Common choice; you can remove if you want profile
        cmd += ["--filter", "peakPicking true 1-"]

    run(cmd)

    # msconvert writes into output dir with same base name; normalize path
    produced = mzml_path.parent / (raw_path.stem + ".mzML")
    if produced != mzml_path and produced.exists():
        produced.rename(mzml_path)

# -----------------------------
# mzML -> voxel (mz_parent, mz, RT) sparse
# -----------------------------
def _bin(x: float, lo: float, step: float) -> int:
    return int(np.floor((x - lo) / step))

def mzml_to_voxel_npz(
    mzml_file: Path,
    out_npz: Path,
    mz_bin: float = 1.0,
    mz_parent_bin: float = 1.0,
    rt_bin_sec: float = 1.0,
    mz_range: Tuple[float, float] = (100.0, 2000.0),
    mz_parent_range: Tuple[float, float] = (300.0, 2000.0),
    rt_range_sec: Optional[Tuple[float, float]] = None,
    ms2_only: bool = True,
    intensity_transform: str = "log1p",
) -> None:
    """
    Creates a sparse COO-style dump:
      coords: (N, 3) int32 columns [parent_bin, frag_bin, rt_bin]
      vals:   (N,) float32 intensities (optionally transformed)
    """
    import pymzml

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    lo_mz, hi_mz = mz_range
    lo_p, hi_p = mz_parent_range

    coords = []
    vals = []

    reader = pymzml.run.Reader(str(mzml_file), build_index_from_scratch=True)

    for spec in reader:
        ms_level = spec.ms_level
        if ms2_only and ms_level != 2:
            continue
        if ms_level == 2:
            # precursor m/z and RT
            try:
                prec = spec.selected_precursors[0]["mz"]
            except Exception:
                continue
            if not (lo_p <= prec <= hi_p):
                continue

            rt_min = spec.scan_time_in_minutes()
            if rt_min is None:
                continue
            rt_sec = float(rt_min) * 60.0

            if rt_range_sec is not None:
                if not (rt_range_sec[0] <= rt_sec <= rt_range_sec[1]):
                    continue

            pbin = _bin(float(prec), lo_p, mz_parent_bin)
            tbin = _bin(rt_sec, rt_range_sec[0] if rt_range_sec else 0.0, rt_bin_sec)

            peaks = spec.peaks("raw")
            if not peaks:
                continue

            for frag_mz, inten in peaks:
                if frag_mz is None or inten is None:
                    continue
                if not (lo_mz <= frag_mz <= hi_mz):
                    continue
                if inten <= 0:
                    continue

                fbin = _bin(float(frag_mz), lo_mz, mz_bin)

                if intensity_transform == "log1p":
                    v = np.log1p(float(inten))
                elif intensity_transform == "sqrt":
                    v = np.sqrt(float(inten))
                else:
                    v = float(inten)

                coords.append((pbin, fbin, tbin))
                vals.append(v)

    if len(vals) == 0:
        # Still write a file so your pipeline can skip it cleanly
        np.savez_compressed(out_npz, coords=np.zeros((0, 3), np.int32), vals=np.zeros((0,), np.float32))
        return

    coords = np.asarray(coords, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float32)

    # Optional: combine duplicates (same voxel summed)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    coords = coords[order]
    vals = vals[order]

    uniq = np.ones(len(vals), dtype=bool)
    uniq[1:] = np.any(coords[1:] != coords[:-1], axis=1)

    if not np.all(uniq):
        idx = np.flatnonzero(uniq)
        sums = np.add.reduceat(vals, idx)
        coords = coords[idx]
        vals = sums.astype(np.float32)

    np.savez_compressed(out_npz, coords=coords, vals=vals)

# -----------------------------
# Windowing utility
# -----------------------------
def make_rt_windows(
    out_dir: Path,
    window_sec: int = 30,
    stride_sec: int = 15,
) -> None:
    """
    Takes voxel npz files and creates windowed npz files (same format) by slicing rt_bin.
    This assumes rt_bin_sec=1.0 and rt starts at 0.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for npz in tqdm(list(out_dir.glob("*.npz")), desc="Windowing"):
        data = np.load(npz)
        coords = data["coords"]
        vals = data["vals"]
        if coords.shape[0] == 0:
            continue
        rt_bins = coords[:, 2]
        max_rt = int(rt_bins.max())

        base = npz.stem
        w = int(window_sec)
        s = int(stride_sec)

        for start in range(0, max_rt + 1, s):
            end = start + w
            m = (rt_bins >= start) & (rt_bins < end)
            if not np.any(m):
                continue
            c = coords[m].copy()
            c[:, 2] -= start  # normalize within-window time
            v = vals[m]
            out = npz.parent / f"{base}__t{start:05d}_{end:05d}.npz"
            np.savez_compressed(out, coords=c.astype(np.int32), vals=v.astype(np.float32))

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pride-download")
    p1.add_argument("--pxd", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--protocol", default="ftp", choices=["s3", "ftp", "aspera", "globus", "auto"])

    p2 = sub.add_parser("massive-download")
    p2.add_argument("--msv", required=True)
    p2.add_argument("--out", required=True)

    p3 = sub.add_parser("convert-raw")
    p3.add_argument("--raw-dir", required=True)
    p3.add_argument("--mzml-dir", required=True)

    p4 = sub.add_parser("mzml-to-voxel")
    p4.add_argument("--mzml-dir", required=True)
    p4.add_argument("--voxel-dir", required=True)
    p4.add_argument("--mz-bin", type=float, default=1.0)
    p4.add_argument("--mz-parent-bin", type=float, default=1.0)
    p4.add_argument("--rt-bin-sec", type=float, default=1.0)

    p5 = sub.add_parser("window")
    p5.add_argument("--voxel-dir", required=True)
    p5.add_argument("--window-sec", type=int, default=30)
    p5.add_argument("--stride-sec", type=int, default=15)

    args = ap.parse_args()

    if args.cmd == "pride-download":
        pride_download_raw(args.pxd, Path(args.out), protocol=args.protocol)

    elif args.cmd == "massive-download":
        massive_download_dataset(args.msv, Path(args.out))

    elif args.cmd == "convert-raw":
        raw_dir = Path(args.raw_dir)
        mzml_dir = Path(args.mzml_dir)
        mzml_dir.mkdir(parents=True, exist_ok=True)
        raws = list(raw_dir.rglob("*.raw")) + list(raw_dir.rglob("*.RAW"))
        for r in tqdm(raws, desc="RAW->mzML"):
            msconvert_to_mzml(r, mzml_dir / (r.stem + ".mzML"), centroid=True)

    elif args.cmd == "mzml-to-voxel":
        mzml_dir = Path(args.mzml_dir)
        voxel_dir = Path(args.voxel_dir)
        voxel_dir.mkdir(parents=True, exist_ok=True)
        files = list(mzml_dir.rglob("*.mzML"))
        for f in tqdm(files, desc="mzML->voxel"):
            out = voxel_dir / (f.stem + ".npz")
            mzml_to_voxel_npz(
                f,
                out,
                mz_bin=args.mz_bin,
                mz_parent_bin=args.mz_parent_bin,
                rt_bin_sec=args.rt_bin_sec,
                rt_range_sec=None,  # leave None to keep full run
                ms2_only=True,
                intensity_transform="log1p",
            )

    elif args.cmd == "window":
        make_rt_windows(Path(args.voxel_dir), window_sec=args.window_sec, stride_sec=args.stride_sec)

if __name__ == "__main__":
    main()
