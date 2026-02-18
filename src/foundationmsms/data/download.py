def is_dia_sample(sample: dict) -> bool:
    """
    Placeholder for sample-level DIA check.
    Adapt this function to your metadata or file naming conventions.
    """
    # Example: check sample['acquisition_mode'] or filename pattern
    return sample.get('acquisition_mode', '').upper() == 'DIA'

"""Download utilities for PRIDE/MassIVE and other sources."""

from pathlib import Path
from typing import Optional

import yaml

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


def download_priority_datasets(config_path: Path, raw_root: Path, priority: int = 1, protocol: str = "s3", massive_host: str = "massive-ftp.ucsd.edu") -> None:
    """Download all PRIDE and MassIVE datasets at or below the given priority.

    Args:
        config_path: Path to datasets.yaml.
        raw_root: Root directory under which datasets will be placed.
        priority: Priority threshold (inclusive).
        protocol: PRIDE download protocol (s3 or ftp).
        massive_host: Host for MassIVE lftp mirror.
    """

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    pride_list = []
    for d in cfg.get("pride_datasets", []) or []:
        if d.get("acquisition", {}).get("mode", "") == "DIA":
            # Filter samples if present
            if 'samples_metadata' in d:
                d['samples_metadata'] = [s for s in d['samples_metadata'] if is_dia_sample(s)]
            pride_list.append(d)
    massive_list = []
    for d in cfg.get("massive_datasets", []) or []:
        if d.get("acquisition", {}).get("mode", "") == "DIA":
            if 'samples_metadata' in d:
                d['samples_metadata'] = [s for s in d['samples_metadata'] if is_dia_sample(s)]
            massive_list.append(d)

    # Report discovered and kept DIA files
    for d in pride_list + massive_list:
        dataset_id = d.get('id', 'unknown')
        samples = d.get('samples_metadata', []) if 'samples_metadata' in d else []
        total_files = len(samples)
        dia_files = [s for s in samples if is_dia_sample(s)]
        kept_files = len(dia_files)
        print(f"Dataset {dataset_id}: {total_files} files discovered, {kept_files} DIA files will be kept.")
        for s in dia_files:
            fname = s.get('filename', 'unknown')
            print(f"  Will keep: {dataset_id}/DIA/{fname}")


    def ensure_dir(base: Path, ds_id: str) -> Path:
        target = base / ds_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    # Gather all selected datasets (priority filter)
    selected = [d for d in pride_list if int(d.get("priority", 0)) <= priority]
    selected += [d for d in massive_list if int(d.get("priority", 0)) <= priority]

    # Validation: check technology compatibility
    if selected:
        ref = selected[0]
        incompatible = [d for d in selected if not (
            d.get("instrument") == ref.get("instrument") and
            d.get("acquisition_method") == ref.get("acquisition_method") and
            d.get("platform") == ref.get("platform")
        )]
        if incompatible:
            raise ValueError(f"Incompatible datasets found: {[d['id'] for d in incompatible]}. All datasets must have matching instrument, acquisition_method, and platform.")

        # Ensure at least one DIA dataset is included
        has_dia = any(d.get("acquisition_method", "").upper() == "DIA" for d in selected)
        if not has_dia:
            raise ValueError("At least one DIA dataset must be included in the selection.")

    # Proceed with download as before
    for entry in [d for d in pride_list if int(d.get("priority", 0)) <= priority]:
        ds_id = entry["id"]
        out_dir = ensure_dir(raw_root / "pride", ds_id)
        download_pride(ds_id, out_dir, protocol=protocol)

    for entry in [d for d in massive_list if int(d.get("priority", 0)) <= priority]:
        ds_id = entry["id"]
        out_dir = ensure_dir(raw_root / "massive", ds_id)
        download_massive(ds_id, out_dir, host=massive_host)


def download_priority_missing(config_path: Path, raw_root: Path, priority: int = 1, protocol: str = "s3", massive_host: str = "massive-ftp.ucsd.edu") -> None:
    """Download priority datasets only if their target directory is empty or absent.

    Useful for debug/dry-ish runs to avoid re-pulling large datasets when already present.
    """

    def has_files(path: Path) -> bool:
        return path.exists() and any(path.rglob("*"))

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    pride_list = []
    for d in cfg.get("pride_datasets", []) or []:
        if d.get("acquisition", {}).get("mode", "") == "DIA":
            if 'samples_metadata' in d:
                d['samples_metadata'] = [s for s in d['samples_metadata'] if is_dia_sample(s)]
            pride_list.append(d)
    massive_list = []
    for d in cfg.get("massive_datasets", []) or []:
        if d.get("acquisition", {}).get("mode", "") == "DIA":
            if 'samples_metadata' in d:
                d['samples_metadata'] = [s for s in d['samples_metadata'] if is_dia_sample(s)]
            massive_list.append(d)

    def ensure_dir(base: Path, ds_id: str) -> Path:
        target = base / ds_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    for entry in pride_list:
        if int(entry.get("priority", 0)) <= priority:
            ds_id = entry["id"]
            out_dir = ensure_dir(raw_root / "pride", ds_id)
            if has_files(out_dir):
                print(f"[skip] PRIDE {ds_id} already present at {out_dir}")
                continue
            print(f"[download] PRIDE {ds_id} -> {out_dir}")
            download_pride(ds_id, out_dir, protocol=protocol)

    for entry in massive_list:
        if int(entry.get("priority", 0)) <= priority:
            ds_id = entry["id"]
            out_dir = ensure_dir(raw_root / "massive", ds_id)
            if has_files(out_dir):
                print(f"[skip] MassIVE {ds_id} already present at {out_dir}")
                continue
            print(f"[download] MassIVE {ds_id} -> {out_dir}")
            download_massive(ds_id, out_dir, host=massive_host)
