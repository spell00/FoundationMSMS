# Foundation LCMS - Batch Processing Guide

## Overview

The batch processing system supports:
- **Parallel downloads** of multiple datasets
- **Automated pipeline** from RAW → mzML → voxel → windows
- **Automatic cleanup** of intermediate files to save space
- **Priority-based processing** (small files with many samples first)

## Quick Start

### 1. Download Priority 1 Datasets (parallel)
```bash
python -u build_foundation_lcms.py batch-download \
    --max-priority 1 \
    --parallel 3
```

### 2. Run Full Pipeline on One Dataset
```bash
python -u build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353 \
    --cleanup-raw \
    --cleanup-mzml
```

### 3. Batch Process All Downloaded Datasets
```bash
./scripts/batch_process_datasets.sh
```

## Available Commands

### batch-download
Download multiple datasets from config with parallel support.

**Options:**
- `--max-priority N`: Only download datasets with priority ≤ N (1=highest)
- `--parallel N`: Number of parallel downloads (default=1)
- `--skip-existing`: Skip already downloaded datasets (default)
- `--reload`: Delete and re-download existing datasets
- `--source pride/massive`: Filter by source
- `--id PXD123`: Download specific dataset(s) only
- `--protocol s3/ftp/aspera`: Protocol for PRIDE downloads

**Examples:**
```bash
# Download top 2 priority levels with 3 parallel downloads
python build_foundation_lcms.py batch-download --max-priority 2 --parallel 3

# Download only PRIDE datasets
python build_foundation_lcms.py batch-download --source pride

# Download specific datasets
python build_foundation_lcms.py batch-download --id PXD012353 --id PXD010595
```

### pipeline
Run full processing pipeline: raw → mzml → voxel → windows.

**Options:**
- `--dataset-dir`: Directory containing .raw files
- `--output-base`: Base output directory
- `--cleanup-raw`: Delete .raw files after converting to mzML (saves space!)
- `--cleanup-mzml`: Delete .mzML files after converting to voxel (saves space!)
- `--mz-bin`: m/z binning for fragments (default=1.0)
- `--mz-parent-bin`: m/z binning for precursors (default=1.0)
- `--rt-bin-sec`: RT binning in seconds (default=1.0)
- `--window-sec`: Window size for temporal slicing (default=30)
- `--stride-sec`: Stride for temporal slicing (default=15)

**Example:**
```bash
python build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353 \
    --cleanup-raw \
    --cleanup-mzml \
    --mz-bin 0.5 \
    --window-sec 60
```

### Individual Steps

**Download single PRIDE dataset:**
```bash
python build_foundation_lcms.py pride-download \
    --pxd PXD012353 \
    --out data/pride/PXD012353
```

**Convert RAW to mzML:**
```bash
python build_foundation_lcms.py convert-raw \
    --raw-dir data/pride/PXD012353 \
    --mzml-dir data/mzml/PXD012353
```

**Convert mzML to voxels:**
```bash
python build_foundation_lcms.py mzml-to-voxel \
    --mzml-dir data/mzml/PXD012353 \
    --voxel-dir data/voxel/PXD012353
```

**Create temporal windows:**
```bash
python build_foundation_lcms.py window \
    --voxel-dir data/voxel/PXD012353 \
    --window-sec 30 \
    --stride-sec 15
```

## Space-Saving Strategies

### Strategy 1: Aggressive Cleanup
Delete intermediate files immediately after processing:
```bash
python build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353 \
    --cleanup-raw \
    --cleanup-mzml
```

**Space savings:**
- RAW files: ~100% removed
- mzML files: ~100% removed
- Keep only: voxel NPZ files + windowed NPZ files

### Strategy 2: Keep mzML for Review
Delete only RAW files:
```bash
python build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353 \
    --cleanup-raw
```

### Strategy 3: Keep Everything
No cleanup flags (useful for debugging):
```bash
python build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353
```

## Workflow Examples

### Example 1: Process Top Priority Datasets
```bash
# Download priority 1 datasets (small size, many samples)
python -u build_foundation_lcms.py batch-download \
    --max-priority 1 \
    --parallel 3 \
    --skip-existing

# Process all with aggressive cleanup
./scripts/batch_process_datasets.sh
```

### Example 2: Selective Download and Process
```bash
# Download specific datasets in parallel
python build_foundation_lcms.py batch-download \
    --id PXD012353 \
    --id PXD010595 \
    --id PXD021874 \
    --parallel 3

# Process each one
for pxd in PXD012353 PXD010595 PXD021874; do
    python build_foundation_lcms.py pipeline \
        --dataset-dir data/pride/$pxd \
        --output-base data/processed/$pxd \
        --cleanup-raw --cleanup-mzml
done
```

### Example 3: Re-download Failed Dataset
```bash
# Re-download specific dataset
python build_foundation_lcms.py batch-download \
    --id PXD012353 \
    --reload

# Process it
python build_foundation_lcms.py pipeline \
    --dataset-dir data/pride/PXD012353 \
    --output-base data/processed/PXD012353 \
    --cleanup-raw --cleanup-mzml
```

## Dataset Priority Reference

**Priority 1** (Best ratio: many samples, small size):
- PXD012353: 24 samples, 600MB
- PXD010595: 96 samples, 3GB
- PXD021874: 115 samples, 5GB
- PXD028735: 200+ samples, 8GB
- MSV000082648: 40 samples, 8GB

**Priority 2-3**: Medium sizes with good sample counts

**Priority 4-5**: Large files or extensive fractionation

## Monitoring Progress

### Check download status:
```bash
find data/pride -name "*.raw" | wc -l
find data/massive -name "*.raw" | wc -l
```

### Check processing status:
```bash
find data/processed -name "*.npz" | wc -l
```

### Disk usage:
```bash
du -sh data/pride/*
du -sh data/processed/*
```
