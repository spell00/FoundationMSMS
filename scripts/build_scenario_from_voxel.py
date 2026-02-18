"""
Build a scenario .npz file from voxel files, including only selected datasets by ID.

Usage:
    python scripts/build_scenario_from_voxel.py --voxel-root data/voxel --include PXD012353 --include PXD028735 --out data/doc_scenarios/custom_scenario.npz

This script walks the voxel directory, collects parent bins and tokens for the specified datasets, and writes a scenario .npz file.
"""

import argparse
from pathlib import Path
import re
import numpy as np
from typing import List

def find_dataset_name(path: Path) -> str | None:
    for part in path.parts[::-1]:
        if re.match(r"^(PXD|MSV)\d+", part):
            return part
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel-root", type=Path, required=True)
    ap.add_argument("--include", type=str, required=True, help="Comma-separated dataset ID(s) to include, e.g. PXD012353,PXD028735")
    ap.add_argument("--out", type=Path, required=True, help="Output scenario .npz file")
    args = ap.parse_args()
    include_ids = [x.strip() for x in args.include.split(",") if x.strip()]

    from tqdm import tqdm
    parent_bins: List[int] = []
    tokens_idx: List[np.ndarray] = []
    tokens_val: List[np.ndarray] = []
    kind = "frag_only"

    npz_files = list(args.voxel_root.rglob("*.npz"))
    print(f"Found {len(npz_files)} voxel files under {args.voxel_root}")
    dataset_counts = {ds: 0 for ds in include_ids}
    for npz_path in tqdm(npz_files, desc="Processing voxel files"):
        ds = find_dataset_name(npz_path)
        if ds not in include_ids:
            continue
        npz = np.load(npz_path)
        coords = npz["coords"]
        vals = npz["vals"]
        if coords.size == 0:
            continue
        # Each parent bin is a unique value in coords[:,0]
        parents = np.unique(coords[:, 0].astype(int))
        dataset_counts[ds] += len(parents)
        for p in parents:
            mask = coords[:, 0] == p
            idx = coords[mask, 1].astype(int)  # token indices
            val = vals[mask].astype(float)
            parent_bins.append(p)
            tokens_idx.append(idx)
            tokens_val.append(val)

    print(f"Collected {len(parent_bins)} parent bins from datasets: {include_ids}")
    for ds, count in dataset_counts.items():
        print(f"  {ds}: {count} parent bins")
    np.savez(
        args.out,
        parent_bins=np.array(parent_bins, dtype=int),
        tokens_idx=np.array(tokens_idx, dtype=object),
        tokens_val=np.array(tokens_val, dtype=object),
        kind=kind,
    )
    print(f"Wrote scenario to {args.out}")

if __name__ == "__main__":
    main()
