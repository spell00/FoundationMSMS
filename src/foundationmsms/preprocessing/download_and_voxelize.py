import argparse
import os
import re
import subprocess
import time
import selectors
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import sys
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

def convert_one(f, voxel_dir, mz_bin, mz_parent_bin, rt_bin_sec):
    import os
    out = voxel_dir / (f.stem + ".npz")
    worker_id = os.getpid()
    if out.exists():
        return f"⏭️  Skipped {f.name} - voxel already exists"
    print(f"[Worker {worker_id}] Voxelizing {f} -> {out}")
    try:
        mzml_to_voxel_npz(
            f,
            out,
            mz_bin=mz_bin,
            mz_parent_bin=mz_parent_bin,
            rt_bin_sec=rt_bin_sec,
            rt_range_sec=None,  # leave None to keep full run
            ms2_only=True,
            intensity_transform="log1p",
        )
        return f"✅ {f.name}"
    except Exception as e:
        return f"❌ {f.name}: {e}"

def run(
    cmd: list[str],
    check: bool = True,
    progress_dir: Optional[Path] = None,
    progress_label: str = "progress",
    progress_interval_sec: int = 0,
) -> None:
    # If no progress monitoring needed, run directly without capturing output
    # This preserves terminal behavior for progress bars
    if progress_dir is None or progress_interval_sec <= 0:
        # Set PYTHONUNBUFFERED to ensure progress bars work correctly
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.run(cmd, check=False, env=env)
        if process.returncode != 0 and check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return
    
    # Otherwise, capture output for monitoring
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


def _purge_empty_raw_files(out_dir: Path) -> int:
    removed = 0
    for p in out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".raw":
            try:
                if p.stat().st_size == 0:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
    return removed
def which(name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(name)

# -----------------------------
# Downloaders
# -----------------------------
def pride_download_raw(
    project: str,
    out_dir: Path,
    protocol: str = "ftp",
    retries: int = 3,
    retry_delay_sec: int = 10,
) -> None:
    """
    Uses pridepy to download all public RAW files for a PRIDE project.
    pridepy supports ftp/aspera/globus/s3 depending on your environment. :contentReference[oaicite:4]{index=4}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pridepy_cli = which("pridepy")
    if pridepy_cli is None:
        try:
            import pridepy  # noqa: F401
        except Exception as exc:
            raise RuntimeError("pridepy not found. Install with: pip install -U pridepy") from exc

    def _run_with(proto: str) -> None:
        removed = _purge_empty_raw_files(out_dir)
        if removed:
            print(f"[pride:{project}] Removed {removed} empty .raw placeholders before download")
        if pridepy_cli is None:
            cmd = [
                sys.executable,
                "-m",
                "pridepy.pridepy",
            ]
        else:
            cmd = [pridepy_cli]

        cmd += [
            "download-all-public-raw-files",
            "-a",
            project,
            "-o",
            str(out_dir),
            "-p",
            proto,
            "--skip-if-downloaded-already",
        ]
        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                run(cmd, progress_dir=out_dir, progress_label=f"pride:{project}", progress_interval_sec=0)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(retry_delay_sec)
        if last_exc is not None:
            raise last_exc

    if protocol == "auto":
        try:
            _run_with("s3")
        except Exception:
            _run_with("ftp")
    else:
        _run_with(protocol)


def massive_download_dataset(msv: str, out_dir: Path, voxel_dir: Path = None, mz_bin: float = 1.0, mz_parent_bin: float = 1.0, rt_bin_sec: float = 1.0, async_voxel: bool = False, type_: str = None) -> None:
    """
    MassIVE exposes public datasets via anonymous FTP. Correct path is /{msv}/ccms_data/{version} or /{msv}/ccms_peak/peak/mzml, etc.
    """
    if not re.match(r"^MSV\d{9}$", msv) or msv == "MSV000000000":
        raise RuntimeError(
            f"Invalid MassIVE accession: {msv}. Use a real ID like MSV000012345."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if which("lftp") is None:
        raise RuntimeError("lftp not found. Install with: sudo apt-get install -y lftp")


    versions = ["v12","v11","v10","v09","v08","v07","v06","v05","v04","v03","v02","v01","x01","z01"]
    remote_dirs = []

    if type_ is not None and type_.lower() == "raw":
        # Only look in ccms_data root, not subfolders
        product = "raw"
        for v in versions:
            remote_dirs.append(f"/{v}/{msv}/{product}")
        for v in versions:
            remote_dirs.append(f"/{msv}/{product}/{v}")
            remote_dirs.append(f"/{msv}/{product}")
        # last resort
        remote_dirs.append(f"/{msv}")
    else:
        # Default: ccms_peak or other types, keep all subfolder logic
        product = "ccms_peak" if (type_ is not None and type_.lower() == "ccms_peak") else "ccms_data"
        for v in versions:
            remote_dirs.append(f"/{v}/{msv}/{product}")
            remote_dirs.append(f"/{v}/{msv}/{product}/peak")
            remote_dirs.append(f"/{v}/{msv}/{product}/peak/mzml")
            remote_dirs.append(f"/{v}/{msv}/{product}/mzml")
        for v in versions:
            remote_dirs.append(f"/{msv}/{product}/{v}")
            remote_dirs.append(f"/{msv}/{product}")
            remote_dirs.append(f"/{msv}/{product}/peak")
            remote_dirs.append(f"/{msv}/{product}/peak/mzml")
            remote_dirs.append(f"/{msv}/{product}/mzml")
        remote_dirs.append(f"/{msv}")

    # Parallelism options for lftp (default: 1/1, but can be overridden by CLI or env)
    parallel_files = int(os.environ.get("LFTP_PARALLEL", "1"))
    pget_n = int(os.environ.get("LFTP_PGET", "1"))
    # Allow override by function attributes (set by CLI)
    if hasattr(massive_download_dataset, "_cli_parallel"):
        parallel_files = massive_download_dataset._cli_parallel
    if hasattr(massive_download_dataset, "_cli_pget"):
        pget_n = massive_download_dataset._cli_pget

    mirror_opts = f"--continue --verbose --parallel={parallel_files}"
    if pget_n > 1:
        mirror_opts += f" --use-pget-n={pget_n}"

    hosts = ["ccms-ftp.ucsd.edu", "massive-ftp.ucsd.edu"]
    user = "anonymous"
    password = "anonymous"
    last_exc: Optional[Exception] = None
    for host in hosts:
        for remote_dir in remote_dirs:
            print(f"[massive] Trying FTP: ftp://{host}{remote_dir} (user={user})")
            cmd = [
                "lftp",
                "-e",
                (
                    "set net:max-retries 2; "
                    "set net:timeout 20; "
                    "set ftp:ssl-allow no; "
                    # optional but often helps stability under parallelism
                    f"set net:connection-limit {parallel_files * max(1, pget_n)}; "
                    f"open -u {user},{password} {host}; "
                    f"mirror {mirror_opts} {remote_dir} {out_dir}; "
                    "bye"
                ),
            ]
            try:
                run(cmd, progress_dir=out_dir, progress_label=f"massive:{msv}", progress_interval_sec=5)
                # Async voxel conversion and cleanup
                if async_voxel and voxel_dir is not None:
                    from threading import Thread
                    def convert_and_cleanup():
                        import time
                        from pathlib import Path
                        from tqdm import tqdm
                        # Wait for all mzML files to finish downloading
                        time.sleep(2)
                        files = list(Path(out_dir).rglob("*.mzML"))
                        for f in tqdm(files, desc="mzML->voxel (async)"):
                            out = Path(voxel_dir) / (f.stem + ".npz")
                            try:
                                mzml_to_voxel_npz(
                                    f,
                                    out,
                                    mz_bin=mz_bin,
                                    mz_parent_bin=mz_parent_bin,
                                    rt_bin_sec=rt_bin_sec,
                                    rt_range_sec=None,
                                    ms2_only=True,
                                    intensity_transform="log1p",
                                )
                                f.unlink()
                            except Exception as e:
                                print(f"[async-voxel] Failed {f}: {e}")
                    Thread(target=convert_and_cleanup, daemon=True).start()
                return
            except RuntimeError as exc:
                last_exc = exc

    # If all anonymous attempts fail, try private (user=msv, password from env)
    env_user = os.environ.get("MASSIVE_FTP_USER")
    env_pass = os.environ.get("MASSIVE_FTP_PASS")
    if env_user and env_pass:
        for host in hosts:
            for remote_dir in remote_dirs:
                print(f"[massive] Trying FTP (private): ftp://{host}{remote_dir} (user={env_user})")
                cmd = [
                    "lftp",
                    "-e",
                    f"set net:max-retries 2; set net:timeout 20; set ftp:ssl-allow no; "
                    f"open -u {env_user},{env_pass} {host}; "
                    f"mirror --continue --verbose {remote_dir} {out_dir}; "
                    f"bye",
                ]
                try:
                    run(cmd, progress_dir=out_dir, progress_label=f"massive:{msv}", progress_interval_sec=5)
                    if async_voxel and voxel_dir is not None:
                        from threading import Thread
                        def convert_and_cleanup():
                            import time
                            from pathlib import Path
                            from tqdm import tqdm
                            time.sleep(2)
                            files = list(Path(out_dir).rglob("*.mzML"))
                            for f in tqdm(files, desc="mzML->voxel (async)"):
                                out = Path(voxel_dir) / (f.stem + ".npz")
                                try:
                                    mzml_to_voxel_npz(
                                        f,
                                        out,
                                        mz_bin=mz_bin,
                                        mz_parent_bin=mz_parent_bin,
                                        rt_bin_sec=rt_bin_sec,
                                        rt_range_sec=None,
                                        ms2_only=True,
                                        intensity_transform="log1p",
                                    )
                                    f.unlink()
                                except Exception as e:
                                    print(f"[async-voxel] Failed {f}: {e}")
                        Thread(target=convert_and_cleanup, daemon=True).start()
                    return
                except RuntimeError as exc:
                    last_exc = exc

    raise RuntimeError(
        f"MassIVE download failed for {msv}. Verify the accession exists and "
        f"that massive-ftp.ucsd.edu or ccms-ftp.ucsd.edu is reachable. Original error: {last_exc}"
    )

def detect_acquisition_mode(mzml_file: Path) -> str:
    """
    Detect acquisition mode (DIA/DDA/OTHER) from mzML file using pymzml.
    Returns: 'DIA', 'DDA', or 'OTHER'
    """
    import pymzml
    try:
        reader = pymzml.run.Reader(str(mzml_file), build_index_from_scratch=True)
        ms2_count = 0
        dia_count = 0
        dda_count = 0
        for spec in reader:
            if getattr(spec, 'ms_level', None) == 2:
                ms2_count += 1
                # DIA: isolation window is wide, precursor is None or 0, or has 'window' in param
                # DDA: isolation window is narrow, precursor m/z present
                precs = getattr(spec, 'selected_precursors', [])
                if not precs or precs[0].get('mz', 0) in (0, None):
                    dia_count += 1
                else:
                    # Check for window width
                    width = precs[0].get('windowWideness', None)
                    if width is not None and width > 10:
                        dia_count += 1
                    else:
                        dda_count += 1
        if ms2_count == 0:
            return 'OTHER'
        if dia_count > dda_count:
            return 'DIA'
        if dda_count > 0:
            return 'DDA'
        return 'OTHER'
    except Exception:
        return 'OTHER'



# -----------------------------
# Conversion: RAW -> mzML
# -----------------------------
def msconvert_to_mzml(raw_path: Path, mzml_path: Path, centroid: bool = True) -> None:
    """
    Requires ProteoWizard msconvert on PATH.
    """
    msconvert_bin = os.environ.get("MSCONVERT_BIN") or which("msconvert")
    if msconvert_bin is None:
        candidate = Path(sys.executable).resolve().parent / "msconvert"
        if candidate.exists():
            msconvert_bin = str(candidate)
    use_docker = False
    if msconvert_bin is None:
        if which("docker") is not None:
            use_docker = True
        else:
            raise RuntimeError("msconvert not found. Install ProteoWizard or ensure Docker is available.")

    mzml_path.parent.mkdir(parents=True, exist_ok=True)

    if use_docker:
        image = os.environ.get("MSCONVERT_IMAGE", "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses")
        raw_dir = raw_path.parent.resolve()
        out_dir = mzml_path.parent.resolve()
        raw_win_path = f"Z:\\data\\{raw_path.name}"
        out_win_dir = "Z:\\out"
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{raw_dir}:/data",
            "-v",
            f"{out_dir}:/out",
            image,
            "wine",
            "msconvert",
            raw_win_path,
            "--mzML",
            "-o",
            out_win_dir,
        ]
    else:
        cmd = [
            msconvert_bin,
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
            if peaks is None or len(peaks) == 0:
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
    window_dir: Optional[Path] = None,
    window_sec: int = 30,
    stride_sec: int = 15,
) -> None:
    """
    Takes voxel npz files and creates windowed npz files (same format) by slicing rt_bin.
    This assumes rt_bin_sec=1.0 and rt starts at 0.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if window_dir is None:
        window_dir = out_dir
    window_dir.mkdir(parents=True, exist_ok=True)

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
            sample_dir = window_dir / base
            sample_dir.mkdir(parents=True, exist_ok=True)
            out = sample_dir / f"{base}__t{start:05d}_{end:05d}.npz"
            np.savez_compressed(out, coords=c.astype(np.int32), vals=v.astype(np.float32))

# -----------------------------
# Batch download
# -----------------------------
def batch_download_datasets(
    config_file: Path,
    base_dir: Path,
    skip_existing: bool = True,
    reload: bool = False,
    pride_protocol: str = "ftp",
    sources: Optional[list[str]] = None,
    dataset_ids: Optional[list[str]] = None,
    max_priority: Optional[int] = None,
    parallel: int = 1,
    retries: int = 3,
    retry_delay_sec: int = 10,
) -> None:
    """Download multiple datasets from a config file.
    
    Args:
        config_file: Path to datasets.yaml
        base_dir: Base directory for downloads (e.g., data/)
        skip_existing: Skip datasets that already exist
        reload: Delete and re-download existing datasets
        pride_protocol: Protocol for PRIDE downloads
        sources: Filter by source type (pride, massive). None = all
        dataset_ids: Filter by specific dataset IDs. None = all
        max_priority: Only download datasets with priority <= this value (1=highest)
        parallel: Number of parallel downloads (default=1)
    """
    import yaml
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    datasets = []
    if sources is None or "pride" in sources:
        for ds in config.get("pride_datasets", []):
            datasets.append(("pride", ds))
    if sources is None or "massive" in sources:
        for ds in config.get("massive_datasets", []):
            datasets.append(("massive", ds))
    
    # Filter by dataset IDs if specified
    if dataset_ids:
        datasets = [(src, ds) for src, ds in datasets if ds["id"] in dataset_ids]
    
    # Filter by priority
    if max_priority is not None:
        datasets = [(src, ds) for src, ds in datasets if ds.get("priority", 999) <= max_priority]
    
    # Sort by priority (lower number = higher priority)
    datasets = sorted(datasets, key=lambda x: x[1].get("priority", 999))
    
    print(f"Found {len(datasets)} datasets to process with {parallel} parallel downloads")
    
    def download_one(source, ds):
        dataset_id = ds["id"]
        desc = ds.get("description", "")
        size = ds.get("size", "unknown")
        samples = ds.get("samples", "?")
        priority = ds.get("priority", "?")
        
        if source == "pride":
            out_dir = base_dir / "raw" / "pride" / dataset_id
        else:
            out_dir = base_dir / "raw" / "massive" / dataset_id
        
        # Check if exists
        exists = out_dir.exists() and any(out_dir.iterdir())
        
        if exists and skip_existing and not reload:
            return f"⏭️  Skipped {dataset_id} - already exists"
        
        if exists and reload:
            import shutil
            shutil.rmtree(out_dir)
        
        print(f"\n📥 Downloading {dataset_id} [Priority {priority}]")
        print(f"   Description: {desc}")
        print(f"   Samples: {samples}")
        print(f"   Size: {size}")
        print(f"   Output: {out_dir}")
        
        try:
            if source == "pride":
                pride_download_raw(
                    dataset_id,
                    out_dir,
                    protocol=pride_protocol,
                    retries=retries,
                    retry_delay_sec=retry_delay_sec,
                )
            else:
                massive_download_dataset(dataset_id, out_dir)
            return f"✅ Completed {dataset_id}"
        except Exception as e:
            return f"❌ Failed {dataset_id}: {e}"
    
    if parallel <= 1:
        # Sequential
        for source, ds in datasets:
            result = download_one(source, ds)
            print(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_one, src, ds): ds["id"] for src, ds in datasets}
            for future in as_completed(futures):
                result = future.result()
                print(f"\n{result}")

# -----------------------------
# Pipeline: helper functions (module level for pickling)
# -----------------------------
def _convert_raw_file(args):
    """Convert single RAW file to mzML (module-level for multiprocessing)"""
    r, mzml_dir, cleanup_raw, force = args
    mzml_out = mzml_dir / (r.stem + ".mzML")
    if mzml_out.exists() and not force:
        return f"⏭️  Skipped {r.name} - exists"
    try:
        if mzml_out.exists() and force:
            mzml_out.unlink()
        msconvert_to_mzml(r, mzml_out, centroid=True)
        if cleanup_raw:
            r.unlink()
            return f"✅ {r.name} -> 🗑️  Deleted RAW"
        return f"✅ {r.name}"
    except Exception as e:
        return f"❌ {r.name}: {e}"

def _convert_mzml_file(args):
    """Convert single mzML file to voxel (module-level for multiprocessing), with mode-based folder split"""
    f, voxel_dir, cleanup_mzml, mz_bin, mz_parent_bin, rt_bin_sec = args
    mode = detect_acquisition_mode(f)
    mode_folder = mode if mode in ("DIA", "DDA") else "OTHERS"
    out_dir = voxel_dir / mode_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_out = out_dir / (f.stem + ".npz")
    if voxel_out.exists():
        return f"⏭️  Skipped {f.name} - exists ({mode_folder})"
    try:
        mzml_to_voxel_npz(
            f,
            voxel_out,
            mz_bin=mz_bin,
            mz_parent_bin=mz_parent_bin,
            rt_bin_sec=rt_bin_sec,
        )
        if cleanup_mzml:
            f.unlink()
            return f"✅ {f.name} -> 🗑️  Deleted mzML ({mode_folder})"
        return f"✅ {f.name} ({mode_folder})"
    except Exception as e:
        return f"❌ {f.name}: {e} ({mode_folder})"

# -----------------------------
# Pipeline: full processing
# -----------------------------
def run_pipeline(
    dataset_dir: Path,
    output_base: Path,
    cleanup_raw: bool = False,
    cleanup_mzml: bool = False,
    force_raw: bool = False,
    mz_bin: float = 1.0,
    mz_parent_bin: float = 10.,
    rt_bin_sec: float = 10.,
    window_sec: int = 30,
    stride_sec: int = 15,
    workers: int = 12,
) -> None:
    """Run full pipeline: raw -> mzml -> voxel -> windows
    
    Args:
        dataset_dir: Directory containing .raw files
        output_base: Base output directory
        cleanup_raw: Delete .raw files after converting to mzML
        cleanup_mzml: Delete .mzML files after converting to voxel
        mz_bin: m/z binning for fragments
        mz_parent_bin: m/z binning for precursors
        rt_bin_sec: RT binning in seconds
        window_sec: Window size for temporal slicing
        stride_sec: Stride for temporal slicing
        workers: Number of parallel workers for file conversion (default=1)
    """
    import shutil
    
    # Determine source and dataset ID from dataset_dir path
    # e.g., data/raw/pride/PXD012353 -> pride/PXD012353
    parts = dataset_dir.parts
    if "raw" in parts:
        idx = parts.index("raw")
        if idx + 2 < len(parts):
            source = parts[idx + 1]  # pride or massive
            dataset_id = parts[idx + 2]  # PXD012353
        else:
            source = "unknown"
            dataset_id = dataset_dir.name
    else:
        source = "unknown"
        dataset_id = dataset_dir.name
    
    mzml_dir = output_base / "mzml" / source / dataset_id
    voxel_dir = output_base / "voxel" / source / dataset_id
    window_dir = output_base / "windows" / source / dataset_id
    
    # Step 1: RAW -> mzML (or copy existing mzMLs for MassIVE peak/mzml)
    print(f"\n🔄 Step 1/3: Converting RAW to mzML (workers={workers})")
    mzml_dir.mkdir(parents=True, exist_ok=True)
    raws = list(dataset_dir.rglob("*.raw")) + list(dataset_dir.rglob("*.RAW"))
    mzml_source_dir = dataset_dir / "peak" / "mzml"
    has_mzml_source = mzml_source_dir.exists()
    print(f"Found {len(raws)} RAW files")

    if len(raws) == 0 and has_mzml_source:
        mzmls = list(mzml_source_dir.rglob("*.mzML")) + list(mzml_source_dir.rglob("*.mzml"))
        print(f"Found {len(mzmls)} mzML files in {mzml_source_dir} (copying)")
        for f in tqdm(mzmls, desc="Copy mzML"):
            dest = mzml_dir / f.name
            if dest.exists() and not force_raw:
                continue
            if dest.exists() and force_raw:
                dest.unlink()
            shutil.copy2(f, dest)
    else:
        if workers <= 1:
            for r in tqdm(raws, desc="RAW->mzML"):
                result = _convert_raw_file((r, mzml_dir, cleanup_raw, force_raw))
                if "❌" in result or "🗑️" in result:
                    print(f"   {result}")
        else:
            args_list = [(r, mzml_dir, cleanup_raw, force_raw) for r in raws]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_convert_raw_file, args) for args in args_list]
                for future in tqdm(as_completed(futures), total=len(raws), desc="RAW->mzML"):
                    result = future.result()
                    if "❌" in result or "🗑️" in result:
                        print(f"   {result}")
    
    # Step 2: mzML -> voxel
    print(f"\n🔄 Step 2/3: Converting mzML to voxels (workers={workers})")
    voxel_dir.mkdir(parents=True, exist_ok=True)
    mzmls = list(mzml_dir.rglob("*.mzML"))
    print(f"Found {len(mzmls)} mzML files")
    
    if workers <= 1:
        for f in tqdm(mzmls, desc="mzML->voxel"):
            result = _convert_mzml_file((f, voxel_dir, cleanup_mzml, mz_bin, mz_parent_bin, rt_bin_sec))
            if "❌" in result or "🗑️" in result:
                print(f"   {result}")
    else:
        args_list = [(f, voxel_dir, cleanup_mzml, mz_bin, mz_parent_bin, rt_bin_sec) for f in mzmls]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_mzml_file, args) for args in args_list]
            # Step 2: mzML -> voxel (with mode-based folder split)
            print(f"\n🔄 Step 2/3: Converting mzML to voxels (workers={workers}, mode-based folders)")
            voxel_dir.mkdir(parents=True, exist_ok=True)
            mzmls = list(mzml_dir.rglob("*.mzML"))
            print(f"Found {len(mzmls)} mzML files")

            if workers <= 1:
                for f in tqdm(mzmls, desc="mzML->voxel (mode split)"):
                    result = _convert_mzml_file((f, voxel_dir, cleanup_mzml, mz_bin, mz_parent_bin, rt_bin_sec))
                    if "❌" in result or "🗑️" in result:
                        print(f"   {result}")
            else:
                args_list = [(f, voxel_dir, cleanup_mzml, mz_bin, mz_parent_bin, rt_bin_sec) for f in mzmls]
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_convert_mzml_file, args) for args in args_list]
                    for future in tqdm(as_completed(futures), total=len(mzmls), desc="mzML->voxel (mode split)"):
                        result = future.result()
                        if "❌" in result or "🗑️" in result:
                            print(f"   {result}")

# -----------------------------
# Main CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("batch-download", help="Download multiple datasets from config")
    p0.add_argument("--config", default="configs/datasets.yaml", help="Path to datasets.yaml")
    p0.add_argument("--base-dir", default="data", help="Base directory for downloads")
    p0.add_argument("--skip-existing", action="store_true", default=True, help="Skip existing datasets (default)")
    p0.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Don't skip existing")
    p0.add_argument("--reload", action="store_true", help="Delete and re-download existing datasets")
    p0.add_argument("--protocol", default="ftp", choices=["s3", "ftp", "aspera", "globus", "auto"])
    p0.add_argument("--source", choices=["pride", "massive"], action="append", help="Filter by source (can specify multiple)")
    p0.add_argument("--id", action="append", help="Download specific dataset IDs only (can specify multiple)")
    p0.add_argument("--max-priority", type=int, help="Only download datasets with priority <= this (1=highest, smaller size/more samples)")
    p0.add_argument("--parallel", type=int, default=1, help="Number of parallel downloads (default=1)")
    p0.add_argument("--retries", type=int, default=3, help="Retry count for PRIDE downloads")
    p0.add_argument("--retry-delay", type=int, default=10, help="Delay between retries (seconds)")

    p1 = sub.add_parser("pride-download")
    p1.add_argument("--pxd", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--protocol", default="ftp", choices=["s3", "ftp", "aspera", "globus", "auto"])
    p1.add_argument("--mode", type=str, default=None, help="Acquisition mode filter (e.g. DIA)")
    p1.add_argument("--redownload", action="store_true", help="Delete and re-download existing files")
    # p2 is initialized below for massive-download
    # New CLI: massive-download-and-voxel
    p2 = sub.add_parser("massive-download-and-voxel", help="Download and immediately convert to voxel, deleting mzML")
    p2.add_argument("--msv", required=True)
    p2.add_argument("--type", required=True, choices=["raw", "ccms_peak"], help="Type of data to download: raw or ccms_peak (required)")
    p2.add_argument("--mz-bin", type=float, default=1.0)
    p2.add_argument("--mz-parent-bin", type=float, default=1.0)
    p2.add_argument("--rt-bin-sec", type=float, default=1.0)
    p2.add_argument("--lftp-parallel", type=int, default=None, help="lftp --parallel (default: 8, overrides LFTP_PARALLEL env)")
    p2.add_argument("--lftp-pget", type=int, default=None, help="lftp --use-pget-n (default: 1, overrides LFTP_PGET env)")

    p3 = sub.add_parser("convert-raw")
    p3.add_argument("--raw-dir", required=True)
    p3.add_argument("--mzml-dir", required=True)
    p3.add_argument("--force", action="store_true", help="Re-convert RAW even if mzML exists")

    p4 = sub.add_parser("mzml-to-voxel")
    p4.add_argument("--mzml-dir", required=True)
    p4.add_argument("--voxel-dir", required=True)
    p4.add_argument("--mz-bin", type=float, default=1.0)
    p4.add_argument("--mz-parent-bin", type=float, default=1.0)
    p4.add_argument("--rt-bin-sec", type=float, default=1.0)
    p4.add_argument("--workers", type=int, default=10, help="Number of parallel workers for conversions (default=1)")

    p5 = sub.add_parser("window")
    p5.add_argument("--voxel-dir", required=True)
    p5.add_argument("--window-sec", type=int, default=30)
    p5.add_argument("--stride-sec", type=int, default=15)

    p6 = sub.add_parser("pipeline", help="Run full pipeline: raw -> mzml -> voxel -> windows")
    p6.add_argument("--dataset-dir", required=True, help="Directory containing .raw files")
    p6.add_argument("--output-base", required=True, help="Base output directory")
    p6.add_argument("--cleanup-raw", action="store_true", help="Delete .raw files after converting to mzML")
    p6.add_argument("--cleanup-mzml", action="store_true", help="Delete .mzML files after converting to voxel")
    p6.add_argument("--force-raw", action="store_true", help="Re-convert RAW even if mzML exists")
    p6.add_argument("--mz-bin", type=float, default=1.0)
    p6.add_argument("--mz-parent-bin", type=float, default=1.0)
    p6.add_argument("--rt-bin-sec", type=float, default=1.0)
    p6.add_argument("--window-sec", type=int, default=30)
    p6.add_argument("--stride-sec", type=int, default=15)
    p6.add_argument("--workers", type=int, default=1, help="Number of parallel workers for conversions (default=1)")

    args = ap.parse_args()

    if args.cmd == "batch-download":
        batch_download_datasets(
            config_file=Path(args.config),
            base_dir=Path(args.base_dir),
            skip_existing=args.skip_existing,
            reload=args.reload,
            pride_protocol=args.protocol,
            sources=args.source,
            dataset_ids=args.id,
            max_priority=args.max_priority,
            parallel=args.parallel,
            retries=args.retries,
            retry_delay_sec=args.retry_delay,
        )

    elif args.cmd == "pride-download":
        # If mode is DIA, filter for DIA files
        if args.mode and args.mode.upper() == "DIA":
            # Optionally, check datasets.yaml for acquisition mode
            import yaml
            config_path = Path("configs/datasets.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                pride_datasets = config.get("pride_datasets", [])
                ds = next((d for d in pride_datasets if d["id"] == args.pxd), None)
                if ds and ds.get("acquisition", {}).get("mode", "").upper() != "DIA":
                    print(f"[skip] Dataset {args.pxd} is not DIA. Skipping download.")
                    return
        out_path = Path(args.out)
        if args.redownload and out_path.exists():
            import shutil
            print(f"[info] --redownload specified: deleting {out_path}")
            shutil.rmtree(out_path)
        pride_download_raw(args.pxd, out_path, protocol=args.protocol)

    elif args.cmd == "massive-download":
        if args.lftp_parallel is not None:
            massive_download_dataset._cli_parallel = args.lftp_parallel
        else:
            if hasattr(massive_download_dataset, "_cli_parallel"):
                del massive_download_dataset._cli_parallel
        if args.lftp_pget is not None:
            massive_download_dataset._cli_pget = args.lftp_pget
        else:
            if hasattr(massive_download_dataset, "_cli_pget"):
                del massive_download_dataset._cli_pget
        # Construct output directory from type and msv
        if args.type == "raw":
            out_dir = Path(f"data/raw/massive/{args.msv}")
        else:
            out_dir = Path(f"data/ccms_peak/massive/{args.msv}")
        massive_download_dataset(
            args.msv,
            out_dir,
            mz_bin=args.mz_bin,
            mz_parent_bin=args.mz_parent_bin,
            rt_bin_sec=args.rt_bin_sec,
            async_voxel=False,
            type_=args.type,
        )

    elif args.cmd == "massive-download-and-voxel":
        if args.lftp_parallel is not None:
            massive_download_dataset._cli_parallel = args.lftp_parallel
        else:
            if hasattr(massive_download_dataset, "_cli_parallel"):
                del massive_download_dataset._cli_parallel
        if args.lftp_pget is not None:
            massive_download_dataset._cli_pget = args.lftp_pget
        else:
            if hasattr(massive_download_dataset, "_cli_pget"):
                del massive_download_dataset._cli_pget
        # Construct output and voxel directories from type and msv
        if args.type == "raw":
            out_dir = Path(f"data/raw/massive/{args.msv}")
        else:
            out_dir = Path(f"data/ccms_peak/massive/{args.msv}")
        voxel_dir = Path(f"data/voxel/massive/{args.msv}")
        massive_download_dataset(
            args.msv,
            out_dir,
            voxel_dir=voxel_dir,
            mz_bin=args.mz_bin,
            mz_parent_bin=args.mz_parent_bin,
            rt_bin_sec=args.rt_bin_sec,
            async_voxel=True,
            type_=args.type,
        )

    elif args.cmd == "convert-raw":
        raw_dir = Path(args.raw_dir)
        mzml_dir = Path(args.mzml_dir)
        mzml_dir.mkdir(parents=True, exist_ok=True)
        raws = list(raw_dir.rglob("*.raw")) + list(raw_dir.rglob("*.RAW"))
        for r in tqdm(raws, desc="RAW->mzML"):
            if r.is_file():
                try:
                    if r.stat().st_size == 0:
                        print(f"[warn] Skipping empty RAW file: {r.name}")
                        continue
                except OSError:
                    pass
            mzml_out = mzml_dir / (r.stem + ".mzML")
            if mzml_out.exists() and not args.force:
                continue
            if mzml_out.exists() and args.force:
                mzml_out.unlink()
            try:
                msconvert_to_mzml(r, mzml_out, centroid=True)
            except Exception as e:
                print(f"❌ {r.name}: {e}")
                continue

    elif args.cmd == "mzml-to-voxel":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        mzml_dir = Path(args.mzml_dir)
        voxel_dir = Path(args.voxel_dir)
        voxel_dir.mkdir(parents=True, exist_ok=True)
        files = list(mzml_dir.rglob("*.mzML"))

        if args.workers <= 1:
            for f in tqdm(files, desc="mzML->voxel"):
                print(convert_one(f, voxel_dir, args.mz_bin, args.mz_parent_bin, args.rt_bin_sec))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(convert_one, f, voxel_dir, args.mz_bin, args.mz_parent_bin, args.rt_bin_sec) for f in files]
                for future in tqdm(as_completed(futures), total=len(futures), desc="mzML->voxel (parallel)"):
                    result = future.result()
                    print(result)

    elif args.cmd == "window":
        make_rt_windows(Path(args.voxel_dir), window_sec=args.window_sec, stride_sec=args.stride_sec)

    elif args.cmd == "pipeline":
        run_pipeline(
            dataset_dir=Path(args.dataset_dir),
            output_base=Path(args.output_base),
            cleanup_raw=args.cleanup_raw,
            cleanup_mzml=args.cleanup_mzml,
            force_raw=args.force_raw,
            mz_bin=args.mz_bin,
            mz_parent_bin=args.mz_parent_bin,
            rt_bin_sec=args.rt_bin_sec,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            workers=args.workers,
        )

if __name__ == "__main__":
    main()
