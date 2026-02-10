"""Windowing utilities for voxel arrays."""

from pathlib import Path

import numpy as np
from tqdm import tqdm


def make_rt_windows(
    out_dir: Path,
    window_sec: int = 30,
    stride_sec: int = 15,
) -> None:
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
            c[:, 2] -= start
            v = vals[m]
            out = npz.parent / f"{base}__t{start:05d}_{end:05d}.npz"
            np.savez_compressed(out, coords=c.astype(np.int32), vals=v.astype(np.float32))
