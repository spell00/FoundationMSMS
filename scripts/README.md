# Scripts

Utility scripts and entry points for Foundation LCMS.

## Important: Preprocessing Code Moved

Preprocessing Python scripts have been refactored into the `src/foundationmsms/preprocessing/` module. This includes:
- `download_and_voxelize.py` (formerly `build_foundation_lcms.py`)
- `add_labels_to_scenario.py`
- `build_scenario_from_voxel.py`
- `optimize_hparams.py`
- `voxelize_with_provenance.py`

### Running Preprocessing Commands

Use the module directly:
```bash
python -m foundationmsms.preprocessing <command> [options]
```

After `pip install -e .`, you can also use installed entry points:
```bash
lcms-download-voxelize batch-download --config configs/datasets.yaml
lcms-add-labels --scenario data/doc_scenarios/docs_frag_only.npz
```

For detailed usage, see `src/foundationmsms/preprocessing/README.md`

## Shell Scripts

This directory contains utility shell scripts that invoke preprocessing commands:
- `batch_process_datasets.sh` — Batch download and process datasets
- `convert_raw_all.sh` — Convert all RAW files in directory

## Deep Baseline Reproducibility Artifacts

The deep baseline trainer now writes a full reproducibility bundle per model run.

Output root pattern:

```bash
logs/deep_baselines/<voxel-param-subdir>/<scenario-name>/models/<model-name>/run_<timestamp>/
```

Each run folder includes:

- `metadata/run_metadata.json`:
	- full CLI args and runtime environment
	- git commit/branch/dirty flag
	- binning params (`mz_bin`, `mz_parent_bin`, `rt_bin_sec`)
- `metadata/split_trace.json`:
	- train/val/test indices for each fold
- `code/train_deep_baseline.py`:
	- script snapshot used for the run
- `folds/fold_<k>/checkpoints/best.pth`:
	- best checkpoint for that fold
- `folds/fold_<k>/training_history.csv` and `best_epoch.json`:
	- epoch-wise metrics and selected best epoch
- `folds/fold_<k>/predictions/*.csv`:
	- per-sample predictions for `train`, `val`, and `test`
	- includes true/pred labels and probability for every class
- `results/all_trials.csv`:
	- Optuna trial-level rows with flattened `param_*` fields for parallel-coordinates plotting
- `results/baselines_fold_metrics.csv` and `results/baseline_summary.json`:
	- copied run outputs for self-contained review

This structure is designed to support strict reproducibility and post-hoc analysis without relying on mutable global logs.
