"""Attach labels to an existing docs_*.npz scenario using configs/label_parsing.yaml.

Usage:
    python -m foundationmsms.preprocessing.add_labels_to_scenario --scenario data/doc_scenarios/docs_frag_only.npz --voxel-root data/voxel

It walks voxel files under --voxel-root to find filename-derived labels and maps them to parent_bins in the scenario. Writes in-place by default (or to --out).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import yaml


def find_dataset_name(path: Path) -> str | None:
    for part in path.parts[::-1]:
        if re.match(r"^(PXD|MSV)\d+", part):
            return part
    return None


def extract_label(path: Path, cfg: dict) -> str | None:
    ds = find_dataset_name(path)
    rule = cfg.get(ds)
    if not rule:
        return None
    delim = rule.get("delimiter", "_")
    idx = int(rule.get("index", 0))
    trunc = rule.get("truncate", None)
    parts = path.stem.split(delim)
    if 0 <= idx < len(parts):
        part = parts[idx]
        if trunc is not None:
            try:
                t = int(trunc)
                part = part[:t]
            except Exception:
                pass
        return part
    return None


def get_label_for_file(path: Path, cfg: dict) -> str | None:
    return extract_label(path, cfg)

def build_parent_labels(voxel_root: Path, include: str | None, cfg: dict) -> dict[int, str]:
    from tqdm import tqdm
    import zipfile
    parent_label_map = {}
    npz_files = list(voxel_root.rglob("*.npz"))
    for npz_path in tqdm(npz_files, desc="Loading voxel files"):
        try:
            ds = find_dataset_name(npz_path)
            if ds not in list(parent_label_map.keys()):
                parent_label_map[ds] = {}
            npz = np.load(npz_path)
            sname = npz_path.stem
            parent_label_map[ds][sname] = {}
            coords = npz["coords"]
            if coords.size == 0:
                continue
            parents = np.unique(coords[:, 0].astype(int))
            for p in parents:
                if include is not None:
                    include_set = set(include.split(","))
                    if ds not in include_set:
                        continue
                parent_label_map[ds][sname][p] = get_label_for_file(npz_path, cfg)
        except zipfile.BadZipFile:
            print(f"[error] BadZipFile: {npz_path} is not a valid npz file.")
        except Exception as e:
            print(f"[error] Failed to load {npz_path}: {e}")
    return parent_label_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--voxel-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/label_parsing.yaml"))
    ap.add_argument("--include", type=str, default=None, help="Optional comma-separated dataset ID(s) to include, e.g. PXD012353,MSV000083793; if not set, will attempt to label all parents in scenario")
    ap.add_argument("--out", type=Path, default=None, help="Optional output path; defaults to overwrite scenario")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    parent_label_map = build_parent_labels(args.voxel_root, args.include, cfg)

    sc = np.load(args.scenario, allow_pickle=True)
    parent_bins = sc["parent_bins"]
    dataset_ids = sc["dataset_ids"] if "dataset_ids" in sc else None
    labels = []
    dataset_ids_out = []  # Track dataset ID for each sample
    missing = 0
    for i, p in enumerate(parent_bins):
        dsid = None
        # Find which dataset this parent_bin belongs to
        if parent_label_map:
            for ds, bin_map in parent_label_map.items():
                if p in bin_map:
                    dsid = ds
                    lab = bin_map[p]
                    break
            else:
                lab = None
        else:
            lab = None
        if lab is None:
            missing += 1
            lab = "unknown"
        labels.append(lab)
        dataset_ids_out.append(dsid if dsid is not None else "unknown")

    out_path = args.out or args.scenario
    np.savez(
        out_path,
        parent_bins=parent_bins,
        tokens_idx=sc["tokens_idx"],
        tokens_val=sc["tokens_val"],
        labels=np.array(labels, dtype=object),
        dataset_ids=np.array(dataset_ids_out, dtype=object),  # Save dataset IDs
        kind=sc.get("kind", None),
        frag_factor=sc.get("frag_factor", None),
        rt_factor=sc.get("rt_factor", None),
    )

    print(f"Wrote labels to {out_path}; missing labels for {missing} parents")

    # Print class summary per dataset
    if dataset_ids is not None:
        from collections import defaultdict
        class_summary = defaultdict(set)
        for dsid, lab in zip(dataset_ids, labels):
            class_summary[dsid].add(lab)
        print("\nClass summary per dataset:")
        supervised_datasets = set(cfg.keys())
        for dsid, classes in class_summary.items():
            if dsid in supervised_datasets:
                print(f"  {dsid}: {sorted(classes)}")
            else:
                print(f"  {dsid}: {sorted(classes)} [WARNING: not in label_parsing.yaml, will be treated as unsupervised]")
        print("\nOnly samples with known classes (not 'unknown') in supervised datasets will be used for classification.")
        # Build and print dict for easy inspection
        dataset_class_dict = {dsid: sorted(classes) for dsid, classes in class_summary.items()}
        print("\nDataset to classes dict:")
        print(dataset_class_dict)
    else:
        print("No dataset_ids found in scenario; cannot summarize classes per dataset.")


if __name__ == "__main__":
    main()
