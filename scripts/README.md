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
