import argparse
import ftplib
import os
import posixpath
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


_DOCKER_STATUS_CACHE: Optional[Tuple[bool, str]] = None

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

def convert_one(f, voxel_dir, mz_bin, mz_parent_bin, rt_bin_sec, force: bool = False):
    import os
    param_folder = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
    out_dir = voxel_dir / param_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f.stem + ".npz")
    worker_id = os.getpid()
    if out.exists() and not force:
        return f"⏭️  Skipped {f.name} - voxel already exists"
    if out.exists() and force:
        out.unlink()
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
    last_files = -1
    last_bytes = -1
    current_file: str = ""

    while True:
        if process.poll() is not None:
            break

        events = selector.select(timeout=1.0)
        for key, _ in events:
            line = key.fileobj.readline()
            if line:
                last_output_time = time.time()
                stripped = line.strip()
                # lftp --verbose emits lines like:
                #   `Transferring file `/remote/path/foo.mzML'`
                #   or just the bare filename on some versions
                if stripped.startswith("Transferring file"):
                    # extract filename between backtick and single-quote
                    import re as _re
                    m = _re.search(r"`(.+?)'", stripped)
                    if m:
                        current_file = m.group(1).split("/")[-1]
                        print(f"[{progress_label}] → {current_file}", flush=True)
                elif stripped.startswith("mirror:") or stripped.startswith("lftp:"):
                    print(line, end="", flush=True)
                # suppress other lftp chatter (chmod, mkdir, skipping, etc.)

        now = time.time()
        if progress_dir is not None and progress_interval_sec > 0 and now >= next_tick:
            files, total_bytes = _dir_stats(progress_dir)
            mb = total_bytes / (1024 * 1024)
            elapsed = int(now - start_time)
            # Only print if something actually changed
            if files != last_files or total_bytes != last_bytes:
                extra = f" ({current_file})" if current_file else ""
                print(
                    f"[{progress_label}] files={files} size={mb:.2f}MB elapsed={elapsed}s{extra}",
                    flush=True,
                )
                last_files = files
                last_bytes = total_bytes
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


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(max(n, 0))
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}TB"


def _remote_ftp_stats(
    host: str,
    root_path: str,
    user: str = "anonymous",
    password: str = "anonymous",
    include_exts: Optional[Tuple[str, ...]] = None,
    progress_label: str = "remote-scan",
    progress_interval_sec: int = 5,
) -> Optional[Tuple[int, int]]:
    """Recursively scan remote FTP tree and return (files, bytes).

    Prints periodic progress updates while scanning. Returns None if scan fails.
    """
    def _fallback_lftp_scan() -> Optional[Tuple[int, int]]:
        if which("lftp") is None:
            return None

        # 1) Count files via recursive `find` output.
        files = 0
        start = time.time()
        next_tick = start + max(1, int(progress_interval_sec))
        print(f"[{progress_label}] scanning ftp://{host}{root_path} via lftp find ...", flush=True)
        try:
            p = subprocess.Popen(
                [
                    "lftp",
                    "-e",
                    (
                        "set net:max-retries 1; "
                        "set net:timeout 20; "
                        "set ftp:ssl-allow no; "
                        f"open -u {user},{password} {host}; "
                        f"find {root_path}; "
                        "bye"
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert p.stdout is not None
            for line in p.stdout:
                s = line.strip()
                if not s or s.startswith("lftp"):
                    continue
                low = s.lower()
                if include_exts:
                    if any(low.endswith(ext) for ext in include_exts):
                        files += 1
                else:
                    # Directory entries typically end with '/'.
                    if not s.endswith("/"):
                        files += 1

                now = time.time()
                if now >= next_tick:
                    elapsed = int(now - start)
                    print(f"[{progress_label}] files={files} elapsed={elapsed}s", flush=True)
                    next_tick = now + max(1, int(progress_interval_sec))
            p.wait(timeout=10)
        except Exception:
            return None

        # 2) Estimate bytes via remote du (best effort).
        total_bytes = 0
        try:
            du = subprocess.run(
                [
                    "lftp",
                    "-e",
                    (
                        "set net:max-retries 1; "
                        "set net:timeout 20; "
                        "set ftp:ssl-allow no; "
                        f"open -u {user},{password} {host}; "
                        f"du -sb {root_path}; "
                        "bye"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            m = re.search(r"(\d+)", du.stdout or "")
            if m:
                total_bytes = int(m.group(1))
        except Exception:
            total_bytes = 0

        elapsed = int(time.time() - start)
        print(
            f"[{progress_label}] done files={files} size={_format_bytes(total_bytes)} elapsed={elapsed}s",
            flush=True,
        )
        return files, total_bytes

    ftp = ftplib.FTP()
    try:
        ftp.connect(host=host, timeout=20)
        ftp.login(user=user, passwd=password)

        files = 0
        total_bytes = 0
        visited_dirs = 0
        stack = [root_path.rstrip("/") or "/"]

        start = time.time()
        next_tick = start + max(1, int(progress_interval_sec))
        print(f"[{progress_label}] scanning ftp://{host}{root_path} ...", flush=True)

        while stack:
            cur = stack.pop()
            visited_dirs += 1
            try:
                entries = list(ftp.mlsd(cur, facts=["type", "size"]))
            except Exception:
                # Some FTP backends may reject selected paths/facts; skip those.
                continue

            for name, facts in entries:
                if name in (".", ".."):
                    continue
                typ = (facts or {}).get("type", "")
                full = posixpath.join(cur, name) if cur != "/" else f"/{name}"
                if typ == "dir":
                    stack.append(full)
                    continue
                if typ == "file":
                    low = name.lower()
                    if include_exts and not any(low.endswith(ext) for ext in include_exts):
                        continue
                    files += 1
                    try:
                        total_bytes += int((facts or {}).get("size", "0") or 0)
                    except Exception:
                        pass

            now = time.time()
            if now >= next_tick:
                elapsed = int(now - start)
                print(
                    f"[{progress_label}] dirs={visited_dirs} files={files} size={_format_bytes(total_bytes)} elapsed={elapsed}s",
                    flush=True,
                )
                next_tick = now + max(1, int(progress_interval_sec))

        elapsed = int(time.time() - start)
        print(
            f"[{progress_label}] done files={files} size={_format_bytes(total_bytes)} elapsed={elapsed}s",
            flush=True,
        )
        return files, total_bytes
    except Exception:
        return _fallback_lftp_scan()
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

# -----------------------------
# Downloaders
# -----------------------------
def pride_download_raw(
    project: str,
    out_dir: Path,
    protocol: str = "ftp",
    retries: int = 3,
    retry_delay_sec: int = 10,
    ftp_url: str = None,
) -> None:
    """
    Download all public files for a PRIDE project.
    If ftp_url is provided the directory is mirrored directly via lftp,
    bypassing pridepy entirely (faster, no API round-trips).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if ftp_url:
        if which("lftp") is None:
            raise RuntimeError("lftp not found. Install with: sudo apt-get install -y lftp")
        from urllib.parse import urlparse
        parsed = urlparse(ftp_url)
        host = parsed.hostname
        path = parsed.path.rstrip("/") or f"/pride/data/archive/{project}"
        print(f"[pride:{project}] Using known URL: ftp://{host}{path}")
        stats = _remote_ftp_stats(host, path, progress_label=f"pride:{project}:scan")
        if stats is not None:
            est_files, est_bytes = stats
            print(
                f"[pride:{project}] estimated transfer: {est_files} files, {_format_bytes(est_bytes)}",
                flush=True,
            )
        cmd = [
            "lftp", "-e",
            (
                "set net:max-retries 3; "
                "set net:timeout 30; "
                "set ftp:ssl-allow no; "
                f"open -u anonymous,anonymous {host}; "
                f"mirror --continue --verbose --parallel=4 {path} {out_dir}; "
                "bye"
            ),
        ]
        run(cmd, progress_dir=out_dir, progress_label=f"pride:{project}", progress_interval_sec=5)
        return

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


def massive_download_dataset(msv: str, out_dir: Path, voxel_dir: Path = None, mz_bin: float = 1.0, mz_parent_bin: float = 1.0, rt_bin_sec: float = 1.0, async_voxel: bool = False, type_: str = None, ftp_url: str = None, mzml_only: bool = False) -> None:
    """
    MassIVE exposes public datasets via anonymous FTP.
    If ftp_url is provided (e.g. 'ftp://massive-ftp.ucsd.edu/v02/MSV000083793/'),
    it is used directly without any path probing.
    If mzml_only is True, only *.mzML files are mirrored (skips RAW and other files).
    """
    if not re.match(r"^MSV\d{9}$", msv) or msv == "MSV000000000":
        raise RuntimeError(
            f"Invalid MassIVE accession: {msv}. Use a real ID like MSV000012345."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if which("lftp") is None:
        raise RuntimeError("lftp not found. Install with: sudo apt-get install -y lftp")

    # Fast path: if the caller already knows the FTP URL, use it directly.
    if ftp_url:
        from urllib.parse import urlparse
        parsed = urlparse(ftp_url)
        known_host = parsed.hostname
        known_path = parsed.path.rstrip("/") or f"/{msv}"
        parallel_files = int(os.environ.get("LFTP_PARALLEL", "1"))
        pget_n = int(os.environ.get("LFTP_PGET", "1"))
        if hasattr(massive_download_dataset, "_cli_parallel"):
            parallel_files = massive_download_dataset._cli_parallel
        if hasattr(massive_download_dataset, "_cli_pget"):
            pget_n = massive_download_dataset._cli_pget
        mirror_opts = f"--continue --verbose --parallel={parallel_files}"
        if pget_n > 1:
            mirror_opts += f" --use-pget-n={pget_n}"
        if mzml_only:
            mirror_opts += " --include-glob '*.mzML' --include-glob '*.mzml'"
            print(f"[massive] Using known URL (mzML only): ftp://{known_host}{known_path}")
        else:
            print(f"[massive] Using known URL: ftp://{known_host}{known_path}")
        stats = _remote_ftp_stats(
            known_host,
            known_path,
            include_exts=(".mzml",) if mzml_only else None,
            progress_label=f"massive:{msv}:scan",
        )
        if stats is not None:
            est_files, est_bytes = stats
            print(
                f"[massive:{msv}] estimated transfer: {est_files} files, {_format_bytes(est_bytes)}",
                flush=True,
            )
        cmd = [
            "lftp", "-e",
            (
                "set net:max-retries 3; "
                "set net:timeout 30; "
                "set ftp:ssl-allow no; "
                f"open -u anonymous,anonymous {known_host}; "
                f"mirror {mirror_opts} {known_path} {out_dir}; "
                "bye"
            ),
        ]
        run(cmd, progress_dir=out_dir, progress_label=f"massive:{msv}", progress_interval_sec=5)
        if async_voxel and voxel_dir is not None:
            from threading import Thread
            def _async_voxel_known():
                import time
                from tqdm import tqdm
                time.sleep(2)
                for f in tqdm(list(Path(out_dir).rglob("*.mzML")), desc="mzML->voxel (async)"):
                    npz_out = Path(voxel_dir) / (f.stem + ".npz")
                    try:
                        mzml_to_voxel_npz(f, npz_out, mz_bin=mz_bin, mz_parent_bin=mz_parent_bin,
                                          rt_bin_sec=rt_bin_sec, rt_range_sec=None,
                                          ms2_only=True, intensity_transform="log1p")
                        f.unlink()
                    except Exception as e:
                        print(f"[async-voxel] Failed {f}: {e}")
            Thread(target=_async_voxel_known, daemon=True).start()
        return

    versions = ["v12","v11","v10","v09","v08","v07","v06","v05","v04","v03","v02","v01","x01","z01"]
    remote_dirs = []

    if type_ is not None and type_.lower() == "raw":
        # Only look in ccms_data root, not subfolders
        product = "raw"
        for v in versions:
            remote_dirs.append(f"/{v}/{msv}")
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
            # Try bare versioned root first (e.g. ftp://massive-ftp.ucsd.edu/v02/MSV000083793/)
            remote_dirs.append(f"/{v}/{msv}")
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
    if mzml_only:
        mirror_opts += " --include-glob '*.mzML' --include-glob '*.mzml'"

    hosts = ["massive-ftp.ucsd.edu", "ccms-ftp.ucsd.edu"]
    user = "anonymous"
    password = "anonymous"

    def _probe_path(host: str, path: str, u: str, pw: str) -> bool:
        """Return True if lftp cls lists any content at host/path within 5s."""
        try:
            r = subprocess.run(
                [
                    "lftp", "-e",
                    f"set net:max-retries 0; set net:timeout 5; set ftp:ssl-allow no; "
                    f"open -u {u},{pw} {host}; cls -1 {path}; bye",
                ],
                capture_output=True, text=True, timeout=12,
            )
            return r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            return False

    def _do_mirror(host: str, remote_dir: str, u: str, pw: str) -> None:
        print(f"[massive] Mirroring ftp://{host}{remote_dir} -> {out_dir}")
        stats = _remote_ftp_stats(
            host,
            remote_dir,
            user=u,
            password=pw,
            include_exts=(".mzml",) if mzml_only else None,
            progress_label=f"massive:{msv}:scan",
        )
        if stats is not None:
            est_files, est_bytes = stats
            print(
                f"[massive:{msv}] estimated transfer: {est_files} files, {_format_bytes(est_bytes)}",
                flush=True,
            )
        cmd = [
            "lftp", "-e",
            (
                "set net:max-retries 3; "
                "set net:timeout 30; "
                "set ftp:ssl-allow no; "
                f"set net:connection-limit {parallel_files * max(1, pget_n)}; "
                f"open -u {u},{pw} {host}; "
                f"mirror {mirror_opts} {remote_dir} {out_dir}; "
                "bye"
            ),
        ]
        run(cmd, progress_dir=out_dir, progress_label=f"massive:{msv}", progress_interval_sec=5)

    def _maybe_async_voxel():
        if async_voxel and voxel_dir is not None:
            from threading import Thread
            def convert_and_cleanup():
                import time
                from tqdm import tqdm
                time.sleep(2)
                files = list(Path(out_dir).rglob("*.mzML"))
                for f in tqdm(files, desc="mzML->voxel (async)"):
                    npz_out = Path(voxel_dir) / (f.stem + ".npz")
                    try:
                        mzml_to_voxel_npz(
                            f, npz_out,
                            mz_bin=mz_bin, mz_parent_bin=mz_parent_bin,
                            rt_bin_sec=rt_bin_sec, rt_range_sec=None,
                            ms2_only=True, intensity_transform="log1p",
                        )
                        f.unlink()
                    except Exception as e:
                        print(f"[async-voxel] Failed {f}: {e}")
            Thread(target=convert_and_cleanup, daemon=True).start()

    # Phase 1: fast ls-probe to locate the right host + path (no mirror yet)
    found_host: Optional[str] = None
    found_path: Optional[str] = None
    for host in hosts:
        for remote_dir in remote_dirs:
            print(f"[massive] Probing ftp://{host}{remote_dir} ...")
            if _probe_path(host, remote_dir, user, password):
                found_host, found_path = host, remote_dir
                break
        if found_host:
            break

    # Phase 2: mirror the confirmed path (or fall back to private credentials)
    if found_host and found_path:
        try:
            _do_mirror(found_host, found_path, user, password)
            _maybe_async_voxel()
            return
        except RuntimeError as exc:
            pass  # fall through to private-creds attempt

    # Private credentials fallback
    env_user = os.environ.get("MASSIVE_FTP_USER")
    env_pass = os.environ.get("MASSIVE_FTP_PASS")
    if env_user and env_pass:
        probe_host: Optional[str] = None
        probe_path: Optional[str] = None
        for host in hosts:
            for remote_dir in remote_dirs:
                print(f"[massive] Probing (private) ftp://{host}{remote_dir} (user={env_user}) ...")
                if _probe_path(host, remote_dir, env_user, env_pass):
                    probe_host, probe_path = host, remote_dir
                    break
            if probe_host:
                break
        if probe_host and probe_path:
            try:
                _do_mirror(probe_host, probe_path, env_user, env_pass)
                _maybe_async_voxel()
                return
            except RuntimeError:
                pass

    raise RuntimeError(
        f"MassIVE download failed for {msv}. Verify the accession exists and "
        f"that massive-ftp.ucsd.edu is reachable. "
        f"Known-good URL pattern: ftp://massive-ftp.ucsd.edu/v02/{msv}/"
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
        global _DOCKER_STATUS_CACHE
        if _DOCKER_STATUS_CACHE is None:
            probe = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0:
                _DOCKER_STATUS_CACHE = (True, "")
            else:
                detail = (probe.stderr or probe.stdout or "docker info failed").strip()
                _DOCKER_STATUS_CACHE = (False, detail)

        docker_ok, docker_detail = _DOCKER_STATUS_CACHE
        if not docker_ok:
            raise RuntimeError(
                "Docker is installed but not usable for msconvert container runs. "
                f"Details: {docker_detail}. "
                "Fix options: (1) install local msconvert and set MSCONVERT_BIN, "
                "(2) enable Docker daemon access for this user (e.g. add user to docker group and relogin), "
                "or (3) run with sudo docker (not recommended in pipeline)."
            )

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
    
    def _ds_canonical_id(src: str, ds: dict) -> str:
        """Return the canonical identifier for a dataset entry.
        Pride entries with a massive_id use that (data lands in massive dir).
        """
        if src == "pride":
            return ds.get("massive_id") or ds.get("pride_id") or ds.get("id", "")
        return ds.get("massive_id") or ds.get("id", "")

    # Filter by dataset IDs if specified
    if dataset_ids:
        datasets = [(src, ds) for src, ds in datasets if _ds_canonical_id(src, ds) in dataset_ids]
    
    # Filter by priority
    if max_priority is not None:
        datasets = [(src, ds) for src, ds in datasets if ds.get("priority", 999) <= max_priority]
    
    # Sort by priority (lower number = higher priority)
    datasets = sorted(datasets, key=lambda x: x[1].get("priority", 999))
    
    print(f"Found {len(datasets)} datasets to process with {parallel} parallel downloads")
    
    def download_one(source, ds):
        canonical_id = _ds_canonical_id(source, ds)
        pride_id = ds.get("pride_id") or ds.get("id", canonical_id)
        massive_id = ds.get("massive_id")
        is_massive_mode = (source == "massive") or bool(massive_id)
        effective_massive_id = massive_id or (canonical_id if source == "massive" else None)
        mzml_only = bool(ds.get("mzml_only", True if massive_id else False))
        desc = ds.get("description", "")
        size = ds.get("size", "unknown")
        samples = ds.get("samples", "?")
        priority = ds.get("priority", "?")

        # Output dir: massive data under massive/, pride-only under pride/
        if massive_id:
            out_dir = base_dir / "raw" / "massive" / massive_id
        elif source == "pride":
            out_dir = base_dir / "raw" / "pride" / pride_id
        else:
            out_dir = base_dir / "raw" / "massive" / canonical_id

        # Check if raw-download output already exists
        exists = out_dir.exists() and any(out_dir.iterdir())

        # For mzML-only MassIVE downloads, existing mzML files already satisfy this stage.
        mzml_exists = False
        if is_massive_mode and effective_massive_id is not None and mzml_only:
            mzml_dir = base_dir / "mzml" / "massive" / effective_massive_id
            if mzml_dir.exists():
                mzml_exists = any(mzml_dir.rglob("*.mzML")) or any(mzml_dir.rglob("*.mzml"))

        if (exists or mzml_exists) and skip_existing and not reload:
            if mzml_exists and not exists:
                return f"⏭️  Skipped {canonical_id} - mzML already exists in {base_dir / 'mzml' / 'massive' / effective_massive_id}"
            return f"⏭️  Skipped {canonical_id} - already exists"

        if exists and reload:
            import shutil
            shutil.rmtree(out_dir)

        print(f"\n📥 Downloading {canonical_id} [Priority {priority}]")
        print(f"   Description: {desc}")
        print(f"   Samples: {samples}")
        print(f"   Size: {size}")
        print(f"   Output: {out_dir}")

        try:
            if source == "pride":
                # If a MassIVE mirror is known, prefer it (mzML already pre-processed).
                if massive_id:
                    try:
                        print(f"   Trying MassIVE mirror {massive_id} (mzml_only={mzml_only}) before PRIDE RAW")
                        massive_download_dataset(
                            massive_id, out_dir,
                            ftp_url=ds.get("ftp_url"),
                            mzml_only=mzml_only,
                        )
                        return f"✅ Completed {canonical_id} (via MassIVE {massive_id})"
                    except Exception as msv_exc:
                        print(f"   MassIVE attempt failed ({msv_exc}), falling back to PRIDE RAW")
                        out_dir = base_dir / "raw" / "pride" / pride_id
                        out_dir.mkdir(parents=True, exist_ok=True)
                pride_download_raw(
                    pride_id,
                    out_dir,
                    protocol=pride_protocol,
                    retries=retries,
                    retry_delay_sec=retry_delay_sec,
                    ftp_url=ds.get("ftp_url") if not massive_id else None,
                )
            else:
                massive_download_dataset(
                    canonical_id, out_dir,
                    ftp_url=ds.get("ftp_url"),
                    mzml_only=mzml_only,
                )
            return f"✅ Completed {canonical_id}"
        except Exception as e:
            return f"❌ Failed {canonical_id}: {e}"
    
    if parallel <= 1:
        # Sequential
        for source, ds in datasets:
            result = download_one(source, ds)
            print(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_one, src, ds): _ds_canonical_id(src, ds) for src, ds in datasets}
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
    f, voxel_dir, cleanup_mzml, force_voxel, mz_bin, mz_parent_bin, rt_bin_sec = args
    mode = detect_acquisition_mode(f)
    mode_folder = mode if mode in ("DIA", "DDA") else "OTHERS"
    out_dir = voxel_dir / mode_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_out = out_dir / (f.stem + ".npz")
    if voxel_out.exists() and not force_voxel:
        return f"⏭️  Skipped {f.name} - exists ({mode_folder})"
    if voxel_out.exists() and force_voxel:
        voxel_out.unlink()
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
    force_voxel: bool = False,
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
    param_folder = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
    voxel_dir = output_base / "voxel" / source / dataset_id / param_folder
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
    # Write a single metadata file for this parameter set
    import yaml
    meta = {
        "dataset_dir": str(dataset_dir),
        "mz_bin": mz_bin,
        "mz_parent_bin": mz_parent_bin,
        "rt_bin_sec": rt_bin_sec,
        "mz_range": [100.0, 2000.0],
        "mz_parent_range": [300.0, 2000.0],
        "rt_range_sec": None,
        "ms2_only": True,
        "intensity_transform": "log1p",
        "voxel_files": [str(voxel_dir / (f.stem + ".npz")) for f in mzmls],
    }
    meta_path = voxel_dir / "voxelization_params.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)
    mzmls = list(mzml_dir.rglob("*.mzML")) + list(mzml_dir.rglob("*.mzml"))
    print(f"Found {len(mzmls)} mzML files")
    
    if workers <= 1:
        for f in tqdm(mzmls, desc="mzML->voxel"):
            result = _convert_mzml_file((f, voxel_dir, cleanup_mzml, force_voxel, mz_bin, mz_parent_bin, rt_bin_sec))
            if "❌" in result or "🗑️" in result:
                print(f"   {result}")
    else:
        args_list = [(f, voxel_dir, cleanup_mzml, force_voxel, mz_bin, mz_parent_bin, rt_bin_sec) for f in mzmls]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_mzml_file, args) for args in args_list]
            # Step 2: mzML -> voxel (with mode-based folder split)
            print(f"\n🔄 Step 2/3: Converting mzML to voxels (workers={workers}, mode-based folders)")
            voxel_dir.mkdir(parents=True, exist_ok=True)
            mzmls = list(mzml_dir.rglob("*.mzML")) + list(mzml_dir.rglob("*.mzml"))
            print(f"Found {len(mzmls)} mzML files")

            if workers <= 1:
                for f in tqdm(mzmls, desc="mzML->voxel (mode split)"):
                    result = _convert_mzml_file((f, voxel_dir, cleanup_mzml, force_voxel, mz_bin, mz_parent_bin, rt_bin_sec))
                    if "❌" in result or "🗑️" in result:
                        print(f"   {result}")
            else:
                args_list = [(f, voxel_dir, cleanup_mzml, force_voxel, mz_bin, mz_parent_bin, rt_bin_sec) for f in mzmls]
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
    p4.add_argument("--force", action="store_true", help="Re-convert mzML even if voxel exists")

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
    p6.add_argument("--force-voxel", action="store_true", help="Re-convert mzML even if voxel exists")
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
        files = list(mzml_dir.rglob("*.mzML")) + list(mzml_dir.rglob("*.mzml"))

        if args.workers <= 1:
            for f in tqdm(files, desc="mzML->voxel"):
                print(convert_one(f, voxel_dir, args.mz_bin, args.mz_parent_bin, args.rt_bin_sec, force=args.force))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(convert_one, f, voxel_dir, args.mz_bin, args.mz_parent_bin, args.rt_bin_sec, args.force) for f in files]
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
            force_voxel=args.force_voxel,
            mz_bin=args.mz_bin,
            mz_parent_bin=args.mz_parent_bin,
            rt_bin_sec=args.rt_bin_sec,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            workers=args.workers,
        )

if __name__ == "__main__":
    main()
