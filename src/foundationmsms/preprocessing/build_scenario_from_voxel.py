"""
Build a scenario .npz file from voxel files, including only selected datasets by ID.

Usage:
    python -m foundationmsms.preprocessing.build_scenario_from_voxel --voxel-root data/voxel --include PXD012353,PXD028735 --out data/doc_scenarios/custom_scenario.npz

This script walks the voxel directory, collects parent bins and tokens for the specified datasets, and writes a scenario .npz file.
"""

import argparse
from pathlib import Path
import re
import numpy as np
from typing import List
import yaml
import concurrent.futures

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
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers for voxel file processing (default: 4)")
    ap.add_argument("--msv-location", type=str, default="downloaded", choices=["converted", "downloaded"], help="Location for MSV datasets: 'converted' or 'downloaded' (default: converted)")
    args = ap.parse_args()
    include_ids = [x.strip() for x in args.include.split(",") if x.strip()]

    from tqdm import tqdm
    scenario_dict = {}  # dataset -> sample_name -> parent_bin -> (idx, val)
    parent_bins: List[int] = []
    tokens_idx: List[np.ndarray] = []
    tokens_val: List[np.ndarray] = []
    dataset_ids: List[str] = []
    sample_names: List[str] = []
    sample_parents: List[str] = []
    labels: List[str] = []
    kind = "frag_only"
    dataset_counts = {ds: 0 for ds in include_ids}  # Initialize dataset_counts

    # Load label parsing config
    label_cfg = yaml.safe_load(open("configs/label_parsing.yaml"))

    def extract_label(sample_name, ds, cfg):
        rule = cfg.get(ds)
        if not rule:
            return "unknown"
        delim = rule.get("delimiter", "_")
        idx = int(rule.get("index", 0))
        trunc = rule.get("truncate", None)
        parts = sample_name.split(delim)
        if 0 <= idx < len(parts):
            part = parts[idx]
            if trunc is not None:
                try:
                    t = int(trunc)
                    part = part[:t]
                except Exception:
                    pass
            return part
        return "unknown"

    for ds in tqdm(include_ids, desc="Processing datasets"):
        if ds.startswith("PXD"):
            ds_voxel_dir = args.voxel_root / "pride" / ds
        elif ds.startswith("MSV"):
            if args.msv_location == "downloaded":
                ds_voxel_dir = args.voxel_root / "massive" / "downloaded" / ds
            elif args.msv_location == "converted":
                ds_voxel_dir = args.voxel_root / "massive" / "converted" / ds
            else:
                ds_voxel_dir = args.voxel_root / "massive" / ds
        else:
            print(f"[warn] Unknown dataset prefix for {ds}, skipping.")
            continue
        if not ds_voxel_dir.exists():
            print(f"[warn] Voxel directory not found: {ds_voxel_dir}")
            continue
        npz_files = list(ds_voxel_dir.rglob("*.npz"))

        def process_npz(npz_path):
            try:
                npz = np.load(npz_path)
                coords = npz["coords"]
                vals = npz["vals"]
                if coords.size == 0:
                    return []
                parents = np.unique(coords[:, 0].astype(int))
                results = []
                for p in parents:
                    mask = coords[:, 0] == p
                    idx = coords[mask, 1].astype(int)
                    val = vals[mask].astype(float)
                    results.append((p, idx, val))
                return results, npz_path.stem
            except Exception as e:
                print(f"[warn] Failed to load {npz_path}: {e}")
                return [], npz_path.stem

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_npz, npz_path) for npz_path in npz_files]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{ds} voxel files", leave=False):
                result, sname = f.result()
                if ds not in scenario_dict:
                    scenario_dict[ds] = {}
                if sname not in scenario_dict[ds]:
                    scenario_dict[ds][sname] = {}
                label = extract_label(sname, ds, label_cfg)
                for p, idx, val in result:
                    scenario_dict[ds][sname][p] = (idx, val)
                    parent_bins.append(p)
                    tokens_idx.append(idx)
                    tokens_val.append(val)
                    dataset_ids.append(ds)
                    sample_names.append(sname)
                    sample_parents.append(sname)  # Save sample name for each parent_bin
                    labels.append(label)
                    dataset_counts[ds] += 1

    print(f"Collected {len(parent_bins)} parent bins from datasets: {include_ids}")
    for ds, count in dataset_counts.items():
        print(f"  {ds}: {count} parent bins")
    np.savez(
        args.out,
        parent_bins=np.array(parent_bins, dtype=int),
        tokens_idx=np.array(tokens_idx, dtype=object),
        tokens_val=np.array(tokens_val, dtype=object),
        dataset_ids=np.array(dataset_ids, dtype=object),
        sample_names=np.array(sample_names, dtype=object),
        sample_parents=np.array(sample_parents, dtype=object),  # Save sample name for each parent_bin
        labels=np.array(labels, dtype=object),
        kind=kind,
    )
    print(f"Wrote scenario to {args.out}")

    # Optionally print or save scenario_dict for inspection
    # print(scenario_dict)

if __name__ == "__main__":
    main()
