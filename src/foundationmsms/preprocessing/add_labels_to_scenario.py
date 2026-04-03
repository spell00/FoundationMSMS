"""Attach labels to an existing docs_*.npz scenario using configs/label_parsing.yaml.

Usage:
    python -m foundationmsms.preprocessing.add_labels_to_scenario --scenario data/doc_scenarios/docs_frag_only.npz --voxel-root data/voxel

It walks voxel files under --voxel-root to find filename-derived labels and maps them to parent_bins in the scenario. Writes in-place by default (or to --out).
"""

from __future__ import annotations

import argparse
import re
import zipfile
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


def resolve_scenario_path(scenario_path: Path, voxel_param_subdir: str | None = None) -> Path:
    """Resolve scenario path when generated under voxel-param subdirectories."""
    if scenario_path.exists():
        return scenario_path

    parent = scenario_path.parent
    stem = scenario_path.stem
    if voxel_param_subdir:
        candidate = parent / voxel_param_subdir / stem / f"{stem}.npz"
        if candidate.exists():
            return candidate
    matches = sorted(parent.glob(f"mzbin_*/{stem}/{stem}.npz"))
    if matches:
        return matches[0]
    return scenario_path


def default_labeled_output_path(scenario_path: Path) -> Path:
    """Return a safe default output path that does not overwrite the source scenario."""
    return scenario_path.with_name(f"{scenario_path.stem}_labeled{scenario_path.suffix}")

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
    ap.add_argument("--voxel-param-subdir", type=str, default=None, help="Optional voxel param subfolder, e.g. mzbin_1.0_mzparent_10.0_rtbin_10.0")
    ap.add_argument("--include", type=str, default=None, help="Optional comma-separated dataset ID(s) to include, e.g. PXD012353,MSV000083793; if not set, will attempt to label all parents in scenario")
    ap.add_argument("--out", type=Path, default=None, help="Optional output path; defaults to overwrite scenario")
    args = ap.parse_args()
    args.scenario = resolve_scenario_path(args.scenario, args.voxel_param_subdir)

    cfg = yaml.safe_load(args.config.read_text())
    try:
        with np.load(args.scenario, allow_pickle=True) as sc:
            parent_bins = sc["parent_bins"]
            dataset_ids = sc["dataset_ids"] if "dataset_ids" in sc else np.array(["unknown"] * len(parent_bins), dtype=object)
            sample_names = None
            if "sample_names" in sc:
                sample_names = sc["sample_names"]
            elif "sample_parents" in sc:
                sample_names = sc["sample_parents"]
            tokens_idx = sc["tokens_idx"]
            tokens_val = sc["tokens_val"]
            kind = sc["kind"] if "kind" in sc else None
            frag_factor = sc["frag_factor"] if "frag_factor" in sc else None
            rt_factor = sc["rt_factor"] if "rt_factor" in sc else None
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Scenario file is corrupted: {args.scenario} ({exc}). "
            "Rebuild the scenario with build_scenario_from_voxel, then re-run labeling."
        ) from exc

    include_set = None
    if args.include:
        include_set = {x.strip() for x in args.include.split(",") if x.strip()}

    labels = []
    dataset_ids_out = []
    missing = 0

    for i in range(len(parent_bins)):
        dsid = str(dataset_ids[i]) if i < len(dataset_ids) else "unknown"
        lab = None

        # Respect optional dataset include filter for labeling.
        if include_set is None or dsid in include_set:
            rule = cfg.get(dsid)
            if rule:
                sample = str(sample_names[i]) if sample_names is not None and i < len(sample_names) else ""
                delim = rule.get("delimiter", "_")
                idx = int(rule.get("index", 0))
                trunc = rule.get("truncate", None)
                parts = sample.split(delim) if sample else []
                if 0 <= idx < len(parts):
                    part = parts[idx]
                    if trunc is not None:
                        try:
                            part = part[: int(trunc)]
                        except Exception:
                            pass
                    lab = part

        if lab is None:
            missing += 1
            lab = "unknown"
        labels.append(lab)
        dataset_ids_out.append(dsid)

    out_path = args.out or default_labeled_output_path(args.scenario)
    if args.out is None:
        print(f"No --out provided; writing labeled scenario to {out_path}")
    write_path = out_path
    # Avoid writing to the same file while reading it (can trigger CRC errors).
    in_place = out_path.resolve() == args.scenario.resolve()
    if in_place:
        write_path = out_path.with_suffix(out_path.suffix + ".tmp")

    np.savez(
        write_path,
        parent_bins=parent_bins,
        tokens_idx=tokens_idx,
        tokens_val=tokens_val,
        labels=np.array(labels, dtype=object),
        dataset_ids=np.array(dataset_ids_out, dtype=object),
        kind=kind,
        frag_factor=frag_factor,
        rt_factor=rt_factor,
    )

    if in_place:
        write_path.replace(out_path)

    print(f"Wrote labels to {out_path}; missing labels for {missing} parents")

    # Print class summary per dataset
    if len(dataset_ids_out) > 0:
        from collections import defaultdict
        class_summary = defaultdict(set)
        for dsid, lab in zip(dataset_ids_out, labels):
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
