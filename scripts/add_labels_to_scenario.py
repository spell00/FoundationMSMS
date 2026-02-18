"""Attach labels to an existing docs_*.npz scenario using configs/label_parsing.yaml.

Usage:
    python scripts/add_labels_to_scenario.py --scenario data/doc_scenarios/docs_frag_only.npz --voxel-root data/voxel

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


def build_parent_labels(voxel_root: Path, cfg: dict) -> dict[int, str]:
    parent_labels: dict[int, str] = {}
    for npz_path in voxel_root.rglob("*.npz"):
        label = extract_label(npz_path, cfg)
        if label is None:
            continue
        npz = np.load(npz_path)
        coords = npz["coords"]
        if coords.size == 0:
            continue
        parents = np.unique(coords[:, 0].astype(int))
        for p in parents:
            parent_labels.setdefault(int(p), label)
    return parent_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--voxel-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/label_parsing.yaml"))
    ap.add_argument("--out", type=Path, default=None, help="Optional output path; defaults to overwrite scenario")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    parent_label_map = build_parent_labels(args.voxel_root, cfg)

    sc = np.load(args.scenario, allow_pickle=True)
    parent_bins = sc["parent_bins"]
    labels = []
    missing = 0
    for p in parent_bins:
        lab = parent_label_map.get(int(p))
        if lab is None:
            missing += 1
            lab = "unknown"
        labels.append(lab)

    out_path = args.out or args.scenario
    np.savez(
        out_path,
        parent_bins=parent_bins,
        tokens_idx=sc["tokens_idx"],
        tokens_val=sc["tokens_val"],
        labels=np.array(labels, dtype=object),
        kind=sc.get("kind", None),
        frag_factor=sc.get("frag_factor", None),
        rt_factor=sc.get("rt_factor", None),
    )

    print(f"Wrote labels to {out_path}; missing labels for {missing} parents")


if __name__ == "__main__":
    main()
