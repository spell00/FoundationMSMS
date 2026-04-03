"""mzML to voxel conversion utilities."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


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

    # Per-file metadata writing removed; now handled per parameter set folder

    if len(vals) == 0:
        np.savez_compressed(out_npz, coords=np.zeros((0, 3), np.int32), vals=np.zeros((0,), np.float32))
        return

    coords = np.asarray(coords, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float32)

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


def massive_mzml_to_voxel(
    mzml_dir: Path,
    voxel_dir: Path,
    mz_bin: float = 1.0,
    mz_parent_bin: float = 1.0,
    rt_bin_sec: float = 1.0,
    delete_mzml: bool = False,
) -> None:
    """
    Convert all mzML files in mzml_dir to voxel npz files in voxel_dir.
    Optionally delete mzML files after conversion.
    """
    from tqdm import tqdm

    mzml_dir = Path(mzml_dir)
    voxel_dir = Path(voxel_dir)
    files = list(mzml_dir.rglob("*.mzML"))
    # Write a single metadata file for this parameter set
    import yaml
    # Always write to a parameter-named subfolder based on function arguments
    param_folder = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
    out_dir = voxel_dir / param_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "mzml_dir": str(mzml_dir),
        "mz_bin": mz_bin,
        "mz_parent_bin": mz_parent_bin,
        "rt_bin_sec": rt_bin_sec,
        "mz_range": [100.0, 2000.0],
        "mz_parent_range": [300.0, 2000.0],
        "rt_range_sec": None,
        "ms2_only": True,
        "intensity_transform": "log1p",
        "voxel_files": [str(out_dir / (f.stem + ".npz")) for f in files],
    }
    meta_path = out_dir / "voxelization_params.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)

    for f in tqdm(files, desc="mzML->voxel (massive)"):
        out = out_dir / (f.stem + ".npz")
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
            if delete_mzml:
                f.unlink()
        except Exception as e:
            print(f"[massive-mzml-to-voxel] Failed {f}: {e}")
