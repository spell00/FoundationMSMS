# FoundationMSMS

Clean ML project skeleton for LCMS foundation model work: data download, preprocessing, training, inference, and deployment.

## Preprocessing

Preprocessing code is organized as a Python module under `src/foundationmsms/preprocessing`.

Use the module entry point for data workflows:

```bash
python -m foundationmsms.preprocessing <command> [options]
```

Common examples:

```bash
python -m foundationmsms.preprocessing batch-download --config configs/datasets.yaml --base-dir data
python -m foundationmsms.preprocessing pipeline --dataset-dir data/raw/pride/PXD012353 --output-base data --workers 12
```

For full command reference, workflow details, and module structure, see:

- [src/foundationmsms/preprocessing/README.md](src/foundationmsms/preprocessing/README.md)

## End-To-End Experiment

Run the full workflow — download, preprocess, build scenario, train, build paper:

```bash
bash scripts/run_experiment.sh
```

Hyperparameters and paths are set via environment variables (see the script header). Datasets are read automatically from `configs/datasets.yaml`.

Build paper artifacts only (safe even when some models are not trained yet):

```bash
bash scripts/build_paper.sh
```

