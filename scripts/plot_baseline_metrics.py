#!/usr/bin/env python3
"""
Plot baseline metrics from baseline_fold_metrics.csv for publication-quality figures.

Usage:
    python scripts/plot_baseline_metrics.py --csv logs/baselines/baseline_fold_metrics.csv --outfig results/baseline_metrics.png

- Plots grouped barplots for each metric (accuracy, macro_f1, etc.)
- Shows both train and validation metrics
- One subplot per dataset, grouped by model
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("mcc", "MCC"),
]

REQUIRED_COLUMNS = [
    "dataset", "model", "fold", "train_size", "val_size", "test_size", "n_classes",
    "train_accuracy", "train_balanced_accuracy", "train_macro_f1", "train_mcc",
    "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_mcc",
]


def normalize_model_name(model_name):
    model_name = str(model_name)
    if model_name.endswith("_opt"):
        return model_name[:-4]
    return model_name


def normalize_scenario_name(value):
    if pd.isna(value) or value is None or str(value).strip() == "":
        return "unspecified_scenario"
    return Path(str(value)).stem


def load_metrics_csv(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}")

    if "scenario" not in df.columns:
        df["scenario"] = "unspecified_scenario"
    else:
        df["scenario"] = df["scenario"].map(normalize_scenario_name)

    if "fold" not in df.columns:
        df["fold"] = 0

    df["model"] = df["model"].map(normalize_model_name)
    return df


def save_summary_tables(df, out_prefix):
    # Compute mean and std for each (dataset, model, metric, split)
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]
    splits = ["train", "val"]
    group_cols = ["dataset", "model"]
    agg = {}
    for split in splits:
        for metric in metrics:
            col = f"{split}_{metric}"
            agg[col] = ["mean", "std"]
    grouped = df.groupby(group_cols).agg(agg)
    # Flatten columns
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    # Save mean ± std as CSV
    csv_path = f"{out_prefix}_summary_mean_std.csv"
    grouped.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save mean ± std as Markdown
    md_path = f"{out_prefix}_summary_mean_std.md"
    with open(md_path, "w") as f:
        f.write(grouped.to_markdown(index=False, floatfmt=".3f"))
    print(f"Saved: {md_path}")

    # Save mean only (no std)
    mean_cols = [c for c in grouped.columns if c.endswith("_mean") or c in group_cols]
    mean_only = grouped[mean_cols].copy()
    mean_only.columns = [c.replace("_mean", "") for c in mean_only.columns]
    csv_path2 = f"{out_prefix}_summary_mean.csv"
    mean_only.to_csv(csv_path2, index=False)
    print(f"Saved: {csv_path2}")
    md_path2 = f"{out_prefix}_summary_mean.md"
    with open(md_path2, "w") as f:
        f.write(mean_only.to_markdown(index=False, floatfmt=".3f"))
    print(f"Saved: {md_path2}")


def plot_metrics(df, outfig_prefix):
    datasets = df["dataset"].unique()
    models = df["model"].unique()
    for metric, metric_label in METRICS:
        plt.figure(figsize=(max(7, len(models)*1.2), 4*len(datasets)))
        for i, dataset in enumerate(datasets, 1):
            plt.subplot(len(datasets), 1, i)
            sub = df[df["dataset"] == dataset]
            # Melt for seaborn
            melted = sub.melt(
                id_vars=["model", "fold"],
                value_vars=[f"train_{metric}", f"val_{metric}"],
                var_name="split",
                value_name=metric_label,
            )
            melted["split"] = melted["split"].map({f"train_{metric}": "Train", f"val_{metric}": "Validation"})
            sns.barplot(
                data=melted,
                x="model",
                y=metric_label,
                hue="split",
                errorbar="sd",
                capsize=0.1,
                palette="Set2",
                err_kws={"linewidth": 1.5},
            )
            plt.title(f"{dataset} — {metric_label}")
            plt.ylabel(metric_label)
            plt.xlabel("Model")
            plt.ylim(0, 1)
            plt.legend(title="Split")
            plt.tight_layout()
        out_png = f"{outfig_prefix}_{metric}.png"
        out_pdf = f"{outfig_prefix}_{metric}.pdf"
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()
        print(f"Saved: {out_png} and {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot baseline metrics for publication-quality figures.")
    parser.add_argument("--csv", type=Path, nargs="+", default=None, 
        help="Path to one or more baseline CSV files (auto-discovered if not provided)")
    parser.add_argument("--outfig", type=Path, default=Path("results/baseline_metrics"), help="Output figure prefix (no extension)")
    args = parser.parse_args()

    # Auto-discover CSV files from scenario subdirectories if not provided
    if args.csv is None:
        csv_paths = []
        # Discover from nested layout such as:
        # logs/baselines/<voxel-param-subdir>/<scenario_name>/baselines_fold_metrics.csv
        baselines_dir = Path("logs/baselines")
        if baselines_dir.exists():
            csv_paths.extend(sorted(baselines_dir.rglob("baselines_fold_metrics.csv")))
        # Also check logs/deep_baselines for deep baseline results
        deep_dir = Path("logs/deep_baselines")
        if deep_dir.exists():
            csv_paths.extend(sorted(deep_dir.rglob("baselines_fold_metrics.csv")))
        args.csv = csv_paths

    dfs = []
    for csv_path in args.csv:
        if csv_path.exists():
            print(f"Loading: {csv_path}")
            try:
                tdf = load_metrics_csv(csv_path)
            except Exception as e:
                print(f"[Error] Could not read {csv_path}: {e}")
                continue
            dfs.append(tdf)
        else:
            print(f"Warning: CSV file not found: {csv_path}")
            
    if not dfs:
        print("Error: No CSV files loaded.")
        return

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # DEBUG: Print all unique model names loaded for plotting
    print("[DEBUG] Models present in DataFrame:", sorted(df['model'].unique()))

    for scenario_name, scenario_df in df.groupby("scenario", dropna=False):
        scenario_outdir = args.outfig.parent / scenario_name
        os.makedirs(scenario_outdir, exist_ok=True)
        out_prefix = scenario_outdir / args.outfig.name
        print(f"[INFO] Writing results for scenario: {scenario_name} -> {out_prefix}")
        plot_metrics(scenario_df, str(out_prefix))
        save_summary_tables(scenario_df, str(out_prefix))

if __name__ == "__main__":
    main()
