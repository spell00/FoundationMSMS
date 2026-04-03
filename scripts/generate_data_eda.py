#!/usr/bin/env python3
"""Generate EDA plots for dataset dimensions used in FoundationMSMS.

Outputs summary CSV/Markdown and PNG plots for:
- files per dataset
- parents per file
- nonzero voxels per file
- token lengths per parent document (from scenario)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def _status(msg: str) -> None:
    print(f"[generate_data_eda] {msg}", flush=True)


def _find_dataset_id(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        if part.startswith("PXD") and part[3:].isdigit():
            return part
        if part.startswith("MSV") and part[3:].isdigit():
            return part
    return None


def _safe_load_npz(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            if "coords" not in d or "vals" not in d:
                return None
            return d["coords"], d["vals"]
    except Exception:
        return None


def _voxel_capacity_from_coords(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    max_bins = np.max(coords, axis=0)
    return float((int(max_bins[0]) + 1) * (int(max_bins[1]) + 1) * (int(max_bins[2]) + 1))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _style_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("Set2")


def _build_dataset_palette(keys: List[str]) -> Dict[str, tuple]:
    palette = sns.color_palette("Spectral", n_colors=max(len(keys), 1))
    return {k: palette[i] for i, k in enumerate(keys)}


def _annotate_bar_values(ax: plt.Axes) -> None:
    ymax = ax.get_ylim()[1]
    offset = ymax * 0.01 if ymax > 0 else 0.05
    for patch in ax.patches:
        h = patch.get_height()
        if h < 0:
            continue
        if abs(h - round(h)) < 1e-6:
            text = f"{int(round(h))}"
        elif h >= 1000:
            text = f"{h:,.0f}"
        else:
            text = f"{h:.1f}"
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            h + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )


def _save_bar(
    values: Dict[str, float],
    title: str,
    ylabel: str,
    out_png: Path,
    dataset_palette: Dict[str, tuple],
) -> None:
    keys = list(values.keys())
    ys = [values[k] for k in keys]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = [dataset_palette[k] for k in keys]
    sns.barplot(x=keys, y=ys, hue=keys, palette=palette, legend=False, ax=ax)
    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("dataset")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    _annotate_bar_values(ax)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_box(
    values: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_png: Path,
    dataset_palette: Dict[str, tuple],
    log_scale: bool = False,
) -> None:
    keys = [k for k, v in values.items() if v]
    data = [values[k] for k in keys]
    if not data:
        return
    melted = [(k, np.log10(v + 1.0) if log_scale else v) for k in keys for v in values[k]]
    xs = [item[0] for item in melted]
    ys = [item[1] for item in melted]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = [dataset_palette[k] for k in keys]
    sns.boxplot(x=xs, y=ys, hue=xs, palette=palette, legend=False, showfliers=False, width=0.6, ax=ax)
    sns.stripplot(x=xs, y=ys, color="#2f2f2f", alpha=0.18, size=2.5, jitter=0.24, ax=ax)
    ax.set_title(title, pad=12)
    ax.set_ylabel(f"{ylabel} (log10(1+x))" if log_scale else ylabel)
    ax.set_xlabel("dataset")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_feature_counts_bar(
    unique_counts: Dict[str, int],
    exclusive_counts: Dict[str, int],
    title: str,
    out_png: Path,
    dataset_palette: Dict[str, tuple],
) -> None:
    keys = list(unique_counts.keys())
    total = [unique_counts[k] for k in keys]
    excl = [exclusive_counts.get(k, 0) for k in keys]
    shared = [t - e for t, e in zip(total, excl)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(keys))
    bar_w = 0.55
    bars_shared = ax.bar(x, shared, bar_w, label="shared with ≥1 other dataset", color="#aec7e8")
    bars_excl = ax.bar(x, excl, bar_w, bottom=shared, label="exclusive to this dataset",
                       color=[dataset_palette[k] for k in keys])
    ax.set_title(title, pad=12)
    ax.set_ylabel("unique features (token IDs)")
    ax.set_xlabel("dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    for bar, total_v in zip(bars_excl, total):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total_v,
            f"{total_v:,}",
            ha="center", va="bottom", fontsize=8, color="#222",
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_overlap_heatmap(
    unique_tokens: Dict[str, set],
    title: str,
    out_png: Path,
    mode: str = "absolute",
) -> None:
    """mode: 'absolute' = |A∩B|, 'jaccard' = |A∩B|/|A∪B|."""
    keys = sorted(unique_tokens.keys())
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            inter = len(unique_tokens[ki] & unique_tokens[kj])
            if mode == "jaccard":
                union = len(unique_tokens[ki] | unique_tokens[kj])
                mat[i, j] = inter / union if union > 0 else 0.0
            else:
                mat[i, j] = inter
    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), max(5, n * 0.9)))
    fmt = ".2f" if mode == "jaccard" else ".0f"
    cmap = "Blues" if mode == "jaccard" else "YlOrRd"
    sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        xticklabels=keys,
        yticklabels=keys,
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1.0 if mode == "jaccard" else None,
    )
    ax.set_title(title, pad=12)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_voxel_compare_box(
    nonzero_values: Dict[str, List[float]],
    all_values: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_png: Path,
    dataset_palette: Dict[str, tuple],
    log_scale: bool = False,
) -> None:
    keys = sorted(set(nonzero_values.keys()) | set(all_values.keys()))
    melted = []
    for k in keys:
        for v in nonzero_values.get(k, []):
            melted.append((k, np.log10(v + 1.0) if log_scale else v, "nonzero"))
        for v in all_values.get(k, []):
            melted.append((k, np.log10(v + 1.0) if log_scale else v, "all"))
    if not melted:
        return

    xs = [item[0] for item in melted]
    ys = [item[1] for item in melted]
    hs = [item[2] for item in melted]
    fig, ax = plt.subplots(figsize=(12, 6))
    hue_palette = {"nonzero": "#1f77b4", "all": "#ff7f0e"}
    sns.boxplot(
        x=xs,
        y=ys,
        hue=hs,
        palette=hue_palette,
        showfliers=False,
        width=0.7,
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_ylabel(f"{ylabel} (log10(1+x))" if log_scale else ylabel)
    ax.set_xlabel("dataset")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Add subtle colored strip under each dataset tick to preserve dataset identity.
    y0, y1 = ax.get_ylim()
    tick_y = y0 + (y1 - y0) * 0.01
    for tick, k in zip(ax.get_xticks(), keys):
        ax.plot([tick - 0.28, tick + 0.28], [tick_y, tick_y], color=dataset_palette[k], linewidth=3, alpha=0.9)

    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="voxel type", loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=Path, default=Path("data/doc_scenarios/experiment_auto.npz"))
    ap.add_argument("--voxel-root", type=Path, default=Path("data/voxel"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/paper/eda"))
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated dataset IDs (optional)")
    ap.add_argument(
        "--log-scale-dists",
        action="store_true",
        help="Use log10(1+x) scale on heavy-tailed distribution boxplots (voxels/tokens)",
    )
    args = ap.parse_args()

    _style_plots()

    _status("Starting EDA generation")
    _status(f"scenario={args.scenario}")
    _status(f"voxel_root={args.voxel_root}")
    _status(f"out_dir={args.out_dir}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_meta = {
        "scenario_path": str(args.scenario),
        "scenario_name": args.scenario.stem,
        "voxel_root": str(args.voxel_root),
    }

    allow = {x.strip() for x in args.datasets.split(",") if x.strip()}

    files_per_dataset: Dict[str, int] = defaultdict(int)
    corrupt_per_dataset: Dict[str, int] = defaultdict(int)
    parents_per_file: Dict[str, List[float]] = defaultdict(list)
    voxels_per_file: Dict[str, List[float]] = defaultdict(list)
    all_voxels_per_file: Dict[str, List[float]] = defaultdict(list)
    nonzero_ratio_per_file: Dict[str, List[float]] = defaultdict(list)

    npz_files = list(args.voxel_root.rglob("*.npz")) if args.voxel_root.exists() else []
    _status(f"Found {len(npz_files)} voxel files to inspect")
    for p in tqdm(npz_files, desc="Inspecting voxel files", unit="file", dynamic_ncols=True):
        if len(npz_files) >= 20 and (files_per_dataset and sum(files_per_dataset.values()) % 250 == 0):
            _status(
                "Voxel scan progress: "
                f"ok={sum(files_per_dataset.values())} corrupt={sum(corrupt_per_dataset.values())}"
            )
        ds = _find_dataset_id(p)
        if ds is None:
            continue
        if allow and ds not in allow:
            continue

        loaded = _safe_load_npz(p)
        if loaded is None:
            corrupt_per_dataset[ds] += 1
            continue

        coords, vals = loaded
        files_per_dataset[ds] += 1
        if coords.size == 0:
            parents_per_file[ds].append(0.0)
            voxels_per_file[ds].append(0.0)
            all_voxels_per_file[ds].append(0.0)
            nonzero_ratio_per_file[ds].append(0.0)
            continue

        parents = int(np.unique(coords[:, 0]).size)
        all_voxels = _voxel_capacity_from_coords(coords)
        nonzero_voxels = float(vals.size)
        ratio = (nonzero_voxels / all_voxels) if all_voxels > 0 else 0.0
        parents_per_file[ds].append(float(parents))
        voxels_per_file[ds].append(nonzero_voxels)
        all_voxels_per_file[ds].append(all_voxels)
        nonzero_ratio_per_file[ds].append(ratio)

    docs_per_dataset: Dict[str, int] = defaultdict(int)
    tokens_per_doc: Dict[str, List[float]] = defaultdict(list)
    unique_source_samples_per_dataset: Dict[str, set] = defaultdict(set)
    token_dims_per_dataset: Dict[str, Counter] = defaultdict(Counter)
    token_repr_per_dataset: Dict[str, str] = {}
    unique_tokens_per_dataset: Dict[str, set] = defaultdict(set)
    if args.scenario.exists():
        _status("Loading scenario for token/doc statistics")
        sc = np.load(args.scenario, allow_pickle=True)
        dataset_ids = sc["dataset_ids"] if "dataset_ids" in sc else np.array([], dtype=object)
        tokens_idx = sc["tokens_idx"] if "tokens_idx" in sc else np.array([], dtype=object)
        sample_names = sc["sample_names"] if "sample_names" in sc else np.array([], dtype=object)
        scenario_kind = str(sc["kind"]) if "kind" in sc else "unknown"
        scenario_meta["scenario_kind"] = scenario_kind
        _status(f"Scenario arrays: dataset_ids={len(dataset_ids)} tokens_idx={len(tokens_idx)}")
        _status(f"Scenario kind={scenario_kind}")
        if len(dataset_ids) == len(tokens_idx):
            iter_docs = zip(dataset_ids, tokens_idx)
            if len(sample_names) == len(tokens_idx):
                iter_docs = zip(dataset_ids, tokens_idx, sample_names)
            else:
                iter_docs = ((ds, idx, "unknown") for ds, idx in iter_docs)

            for ds, idx, sample_name in tqdm(
                iter_docs,
                total=len(tokens_idx),
                desc="Inspecting scenario docs",
                unit="doc",
                dynamic_ncols=True,
            ):
                ds_s = str(ds)
                if allow and ds_s not in allow:
                    continue
                docs_per_dataset[ds_s] += 1
                unique_source_samples_per_dataset[ds_s].add(str(sample_name))

                idx_arr = np.asarray(idx)
                tokens_per_doc[ds_s].append(float(len(idx_arr)))

                if idx_arr.ndim == 1:
                    token_dims_per_dataset[ds_s]["1D"] += 1
                    token_repr_per_dataset.setdefault(ds_s, "spectral-token (fragment m/z bin id)")
                    unique_tokens_per_dataset[ds_s].update(idx_arr.tolist())
                elif idx_arr.ndim == 2 and idx_arr.shape[1] == 2:
                    token_dims_per_dataset[ds_s]["2D_pair"] += 1
                    token_repr_per_dataset.setdefault(ds_s, "spectral-token pair (fragment bin, rt bin)")
                    unique_tokens_per_dataset[ds_s].update(map(tuple, idx_arr.tolist()))
                else:
                    token_dims_per_dataset[ds_s][f"ndim={idx_arr.ndim}"] += 1
                    token_repr_per_dataset.setdefault(ds_s, "spectral-token (custom index structure)")
                    unique_tokens_per_dataset[ds_s].update(idx_arr.reshape(idx_arr.shape[0], -1).tolist())
        else:
            _status("Warning: scenario dataset_ids and tokens_idx lengths differ; skipping token/doc aggregation")
    else:
        _status("Warning: scenario file not found; token/doc stats skipped")

    meta_path = out_dir / "eda_metadata.json"
    with meta_path.open("w") as f:
        json.dump(scenario_meta, f, indent=2)
    _status(f"wrote {meta_path}")

    dataset_keys = sorted(
        set(files_per_dataset.keys())
        | set(corrupt_per_dataset.keys())
        | set(docs_per_dataset.keys())
        | set(tokens_per_doc.keys())
        | set(parents_per_file.keys())
    )

    if not dataset_keys:
        _status("No dataset stats found. Check --voxel-root and --scenario paths.")
        return

    _status(f"Datasets discovered: {', '.join(dataset_keys)}")
    dataset_palette = _build_dataset_palette(dataset_keys)
    if args.log_scale_dists:
        _status("Using log scale for heavy-tailed boxplots (voxels/tokens)")

    # Save plots
    _status("Writing plots")
    files_png = out_dir / "files_per_dataset.png"
    docs_png = out_dir / "docs_per_dataset.png"
    corrupt_png = out_dir / "corrupt_files_per_dataset.png"
    parents_png = out_dir / "parents_per_file_boxplot.png"
    voxels_png = out_dir / "voxels_per_file_boxplot.png"
    voxels_compare_png = out_dir / "voxels_nonzero_vs_all_boxplot.png"
    voxels_ratio_png = out_dir / "voxels_nonzero_ratio_per_file_boxplot.png"
    tokens_png = out_dir / "tokens_per_doc_boxplot.png"
    feature_counts_png = out_dir / "feature_counts_per_dataset.png"
    feature_overlap_png = out_dir / "feature_overlap_heatmap.png"
    feature_jaccard_png = out_dir / "feature_jaccard_heatmap.png"

    _save_bar(
        {k: files_per_dataset.get(k, 0) for k in dataset_keys},
        "Voxel Files Per Dataset",
        "files",
        files_png,
        dataset_palette,
    )
    _status(f"wrote {files_png}")
    _save_bar(
        {k: docs_per_dataset.get(k, 0) for k in dataset_keys},
        "Scenario Documents Per Dataset",
        "documents",
        docs_png,
        dataset_palette,
    )
    _status(f"wrote {docs_png}")
    _save_bar(
        {k: corrupt_per_dataset.get(k, 0) for k in dataset_keys},
        "Corrupt Voxel Files Per Dataset",
        "corrupt files",
        corrupt_png,
        dataset_palette,
    )
    _status(f"wrote {corrupt_png}")
    _save_box(
        parents_per_file,
        "Parents Per Voxel File",
        "unique parent bins per file",
        parents_png,
        dataset_palette,
    )
    _status(f"wrote {parents_png}")
    _save_box(
        voxels_per_file,
        "Nonzero Voxels Per File",
        "nonzero voxel count",
        voxels_png,
        dataset_palette,
        log_scale=args.log_scale_dists,
    )
    _status(f"wrote {voxels_png}")
    _save_voxel_compare_box(
        voxels_per_file,
        all_voxels_per_file,
        "Nonzero vs All Voxels Per File",
        "voxel count",
        voxels_compare_png,
        dataset_palette,
        log_scale=args.log_scale_dists,
    )
    _status(f"wrote {voxels_compare_png}")
    _save_box(
        nonzero_ratio_per_file,
        "Nonzero / All Voxels Ratio Per File",
        "ratio",
        voxels_ratio_png,
        dataset_palette,
        log_scale=False,
    )
    _status(f"wrote {voxels_ratio_png}")
    _save_box(
        tokens_per_doc,
        "Tokens Per Scenario Document",
        "token count",
        tokens_png,
        dataset_palette,
        log_scale=args.log_scale_dists,
    )
    _status(f"wrote {tokens_png}")

    # Feature overlap plots (requires scenario data)
    if unique_tokens_per_dataset:
        ds_with_tokens = {k: v for k, v in unique_tokens_per_dataset.items() if v}
        if ds_with_tokens:
            # Compute exclusive counts: tokens that appear in only this dataset
            all_other: Dict[str, set] = {}
            for k in ds_with_tokens:
                others = set()
                for k2, v2 in ds_with_tokens.items():
                    if k2 != k:
                        others |= v2
                all_other[k] = others
            exclusive_counts = {k: len(v - all_other[k]) for k, v in ds_with_tokens.items()}
            unique_counts = {k: len(v) for k, v in ds_with_tokens.items()}
            _save_feature_counts_bar(
                unique_counts,
                exclusive_counts,
                "Unique Feature IDs Per Dataset (exclusive vs shared)",
                feature_counts_png,
                dataset_palette,
            )
            _status(f"wrote {feature_counts_png}")
            if len(ds_with_tokens) > 1:
                _save_overlap_heatmap(
                    ds_with_tokens,
                    "Pairwise Feature Overlap (|A ∩ B|)",
                    feature_overlap_png,
                    mode="absolute",
                )
                _status(f"wrote {feature_overlap_png}")
                _save_overlap_heatmap(
                    ds_with_tokens,
                    "Pairwise Feature Jaccard Similarity (|A ∩ B| / |A ∪ B|)",
                    feature_jaccard_png,
                    mode="jaccard",
                )
                _status(f"wrote {feature_jaccard_png}")
        else:
            _status("No token data available for feature overlap plots (scenario missing or empty)")

    # Save tabular summary
    csv_path = out_dir / "dataset_dimension_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "voxel_files",
            "corrupt_voxel_files",
            "scenario_docs",
            "parents_per_file_p50",
            "parents_per_file_p90",
            "parents_per_file_p99",
            "tokens_per_doc_p50",
            "tokens_per_doc_p90",
            "tokens_per_doc_p99",
        ])
        for ds in dataset_keys:
            w.writerow([
                ds,
                files_per_dataset.get(ds, 0),
                corrupt_per_dataset.get(ds, 0),
                docs_per_dataset.get(ds, 0),
                round(_percentile(parents_per_file.get(ds, []), 50), 2),
                round(_percentile(parents_per_file.get(ds, []), 90), 2),
                round(_percentile(parents_per_file.get(ds, []), 99), 2),
                round(_percentile(tokens_per_doc.get(ds, []), 50), 2),
                round(_percentile(tokens_per_doc.get(ds, []), 90), 2),
                round(_percentile(tokens_per_doc.get(ds, []), 99), 2),
            ])
    _status(f"wrote {csv_path}")

    md_path = out_dir / "dataset_dimension_summary.md"
    with md_path.open("w") as f:
        f.write("# Dataset Dimension Summary\n\n")
        f.write("| dataset | voxel_files | corrupt_voxel_files | scenario_docs | parents_p50 | parents_p90 | parents_p99 | tokens_p50 | tokens_p90 | tokens_p99 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for ds in dataset_keys:
            f.write(
                "| "
                + " | ".join(
                    [
                        ds,
                        str(files_per_dataset.get(ds, 0)),
                        str(corrupt_per_dataset.get(ds, 0)),
                        str(docs_per_dataset.get(ds, 0)),
                        str(round(_percentile(parents_per_file.get(ds, []), 50), 2)),
                        str(round(_percentile(parents_per_file.get(ds, []), 90), 2)),
                        str(round(_percentile(parents_per_file.get(ds, []), 99), 2)),
                        str(round(_percentile(tokens_per_doc.get(ds, []), 50), 2)),
                        str(round(_percentile(tokens_per_doc.get(ds, []), 90), 2)),
                        str(round(_percentile(tokens_per_doc.get(ds, []), 99), 2)),
                    ]
                )
                + " |\n"
            )
    _status(f"wrote {md_path}")

    _status("Per-dataset summary")
    for ds in dataset_keys:
        source_samples = len(unique_source_samples_per_dataset.get(ds, set()))
        docs = docs_per_dataset.get(ds, 0)
        deconstruct_ratio = (docs / source_samples) if source_samples > 0 else 0.0
        token_mode = token_repr_per_dataset.get(ds, "unknown")
        _status(
            f"{ds}: voxel_files={files_per_dataset.get(ds, 0)} "
            f"corrupt={corrupt_per_dataset.get(ds, 0)} "
            f"source_samples={source_samples} "
            f"scenario_docs={docs} "
            f"docs_per_source_sample={deconstruct_ratio:.2f} "
            f"token_mode={token_mode}"
        )

    decomp_csv_path = out_dir / "dataset_deconstruction_summary.csv"
    with decomp_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "source_samples",
            "scenario_docs",
            "docs_per_source_sample",
            "token_representation",
            "token_dim_breakdown",
        ])
        for ds in dataset_keys:
            source_samples = len(unique_source_samples_per_dataset.get(ds, set()))
            docs = docs_per_dataset.get(ds, 0)
            ratio = (docs / source_samples) if source_samples > 0 else 0.0
            dim_breakdown = ";".join(
                f"{k}:{v}" for k, v in sorted(token_dims_per_dataset.get(ds, Counter()).items())
            )
            w.writerow([
                ds,
                source_samples,
                docs,
                round(ratio, 4),
                token_repr_per_dataset.get(ds, "unknown"),
                dim_breakdown,
            ])
    _status(f"wrote {decomp_csv_path}")

    decomp_md_path = out_dir / "dataset_deconstruction_summary.md"
    with decomp_md_path.open("w") as f:
        f.write("# Dataset Deconstruction Summary\n\n")
        f.write("This table tracks how initial source samples are deconstructed into scenario documents used for training.\n\n")
        f.write("- **Source sample**: one original LCMS run/file (word-equivalent: source document).\n")
        f.write("- **Scenario document**: one parent-bin-centered training sample.\n")
        f.write("- **Spectral token**: the replacement for NLP words in this project.\n\n")
        f.write("| dataset | source_samples | scenario_docs | docs_per_source_sample | token_representation | token_dim_breakdown |\n")
        f.write("|---|---:|---:|---:|---|---|\n")
        for ds in dataset_keys:
            source_samples = len(unique_source_samples_per_dataset.get(ds, set()))
            docs = docs_per_dataset.get(ds, 0)
            ratio = (docs / source_samples) if source_samples > 0 else 0.0
            dim_breakdown = "; ".join(
                f"{k}:{v}" for k, v in sorted(token_dims_per_dataset.get(ds, Counter()).items())
            )
            f.write(
                "| "
                + " | ".join(
                    [
                        ds,
                        str(source_samples),
                        str(docs),
                        f"{ratio:.2f}",
                        token_repr_per_dataset.get(ds, "unknown"),
                        dim_breakdown,
                    ]
                )
                + " |\n"
            )
    _status(f"wrote {decomp_md_path}")

    _status(f"EDA artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
