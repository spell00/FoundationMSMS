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
    ap.add_argument("--voxel-param-subdir", type=str, default=None, help="Parameter subfolder for voxelization, e.g. mzbin_1.0_mzparent_10.0_rtbin_10.0")
    ap.add_argument("--delete-corrupt", action="store_true", help="Delete corrupted voxel .npz files that cannot be loaded")
    ap.add_argument("--max-corrupt-warnings", type=int, default=10, help="Max per-file corruption warnings to print per dataset (default: 10)")
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
    dataset_file_counts = {ds: 0 for ds in include_ids}
    corrupt_counts = {ds: 0 for ds in include_ids}
    deleted_corrupt_counts = {ds: 0 for ds in include_ids}

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
            base_dir = args.voxel_root / "pride" / ds
        elif ds.startswith("MSV"):
            base_dir = args.voxel_root / "massive" / ds
        else:
            print(f"[warn] Unknown dataset prefix for {ds}, skipping.")
            continue
        ds_voxel_dir = base_dir
        if args.voxel_param_subdir:
            ds_voxel_dir = base_dir / args.voxel_param_subdir
        if not ds_voxel_dir.exists():
            print(f"[warn] Voxel directory not found: {ds_voxel_dir}")
            continue
        npz_files = list(ds_voxel_dir.rglob("*.npz"))
        dataset_file_counts[ds] = len(npz_files)

        def process_npz(npz_path):
            try:
                with np.load(npz_path, allow_pickle=False) as npz:
                    if "coords" not in npz or "vals" not in npz:
                        raise ValueError("Missing required arrays: coords/vals")
                    coords = npz["coords"]
                    vals = npz["vals"]
                if coords.size == 0:
                    return [], npz_path.stem, None, False
                parents = np.unique(coords[:, 0].astype(int))
                results = []
                for p in parents:
                    mask = coords[:, 0] == p
                    idx = coords[mask, 1].astype(int)
                    val = vals[mask].astype(float)
                    results.append((p, idx, val))
                return results, npz_path.stem, None, False
            except Exception as e:
                deleted = False
                if args.delete_corrupt:
                    try:
                        npz_path.unlink(missing_ok=True)
                        deleted = True
                    except Exception:
                        deleted = False
                return [], npz_path.stem, str(e), deleted

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_npz, npz_path) for npz_path in npz_files]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{ds} voxel files", leave=False):
                result, sname, err, deleted = f.result()
                if err is not None:
                    corrupt_counts[ds] += 1
                    if deleted:
                        deleted_corrupt_counts[ds] += 1
                    if corrupt_counts[ds] <= args.max_corrupt_warnings:
                        action = "deleted" if deleted else "skipped"
                        print(f"[warn] Corrupt voxel file ({action}): {ds}/{sname}.npz -> {err}")
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
        total_files = dataset_file_counts.get(ds, 0)
        bad = corrupt_counts.get(ds, 0)
        deleted = deleted_corrupt_counts.get(ds, 0)
        if bad > args.max_corrupt_warnings:
            print(f"  {ds}: {count} parent bins, corrupt_files={bad} (showed {args.max_corrupt_warnings} warnings)")
        else:
            print(f"  {ds}: {count} parent bins, corrupt_files={bad}")
        if deleted > 0:
            print(f"      deleted_corrupt_files={deleted}")
        if total_files == 0:
            print(f"      [warn] no voxel files found")
    # Parse binning parameters from voxel_param_subdir BEFORE creating scenario_name
    mz_bin = mz_parent_bin = rt_bin_sec = None
    if args.voxel_param_subdir:
        import re
        m = re.search(r"mzbin[_-]?([\d.]+)", args.voxel_param_subdir)
        if m:
            mz_bin = float(m.group(1))
        m = re.search(r"mzparent[_-]?([\d.]+)", args.voxel_param_subdir)
        if m:
            mz_parent_bin = float(m.group(1))
        m = re.search(r"rtbin[_-]?([\d.]+)", args.voxel_param_subdir)
        if m:
            rt_bin_sec = float(m.group(1))
    # Place scenario in its own folder as:
    # <out-parent>/<voxel-param-subdir>/<scenario-name>/<scenario-name>.npz
    scenario_name = args.out.stem
    if args.voxel_param_subdir:
        scenario_dir = args.out.parent / args.voxel_param_subdir / scenario_name
    else:
        scenario_dir = args.out.parent / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_npz = scenario_dir / (scenario_name + ".npz")
    np.savez(
        scenario_npz,
        parent_bins=np.array(parent_bins, dtype=int),
        tokens_idx=np.array(tokens_idx, dtype=object),
        tokens_val=np.array(tokens_val, dtype=object),
        dataset_ids=np.array(dataset_ids, dtype=object),
        sample_names=np.array(sample_names, dtype=object),
        sample_parents=np.array(sample_parents, dtype=object),  # Save sample name for each parent_bin
        labels=np.array(labels, dtype=object),
        kind=kind,
    )
    # Save scenario metadata (binning, voxel params, included datasets, etc.)
    meta = {
        "scenario_name": scenario_name,
        "scenario_file": str(scenario_npz),
        "scenario_dir": str(scenario_dir),
        "voxel_root": str(args.voxel_root),
        "included_datasets": include_ids,
        "voxel_param_subdir": args.voxel_param_subdir,
        "binning": args.voxel_param_subdir,
        "mz_bin": mz_bin,
        "mz_parent_bin": mz_parent_bin,
        "rt_bin_sec": rt_bin_sec,
        "n_samples": len(tokens_idx),
        "n_parent_bins": len(parent_bins),
        "labels_config": "configs/label_parsing.yaml",
        "created": __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        "meta_version": 1,
        "notes": "This folder may contain additional metadata or auxiliary files in the future."
    }
    meta_path = scenario_dir / "scenario_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)
    print(f"Wrote scenario to {scenario_npz}")
    print(f"Wrote scenario metadata to {meta_path}")

    # Optionally print or save scenario_dict for inspection
    # print(scenario_dict)

if __name__ == "__main__":
    main()
