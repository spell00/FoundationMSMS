#!/usr/bin/env bash
# run_experiment.sh – end-to-end experiment pipeline
#
# Stages:
#   1. Download datasets listed in configs/datasets.yaml
#   2. Preprocess (raw → mzML → voxel) for each dataset
#   3. Build scenario .npz from voxels
#   4. Train MSWTransformer
#   5. Build paper figures / tables
#
# All stages can be skipped individually via SKIP_* env vars.
# Training hyperparameters are read from environment or fall back to defaults.
#
# Usage:
#   bash scripts/run_experiment.sh
#   SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 bash scripts/run_experiment.sh
#   TASK=recon EPOCHS=5 DEVICE=cpu bash scripts/run_experiment.sh

set -euo pipefail

# ------------------------------------------------------------------
# Config paths  (override via env vars)
# ------------------------------------------------------------------
CONFIG="${CONFIG:-configs/datasets.yaml}"
LABEL_CONFIG="${LABEL_CONFIG:-configs/label_parsing.yaml}"

# ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
BASE_DIR="${BASE_DIR:-data}"
SCENARIO_OUT="${SCENARIO_OUT:-data/doc_scenarios/experiment_auto.npz}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
PAPER_OUT="${PAPER_OUT:-experiments/paper}"

# ------------------------------------------------------------------
# Download options
# ------------------------------------------------------------------
PARALLEL="${PARALLEL:-2}"

# ------------------------------------------------------------------
# Preprocessing options
# ------------------------------------------------------------------
WORKERS="${WORKERS:-8}"

# ------------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------------
TASK="${TASK:-clf}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
DIM="${DIM:-128}"
HEADS="${HEADS:-4}"
CV_FOLDS="${CV_FOLDS:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
WARMUP_OUT="${WARMUP_OUT:-${CHECKPOINTS_DIR}/warmup.pth}"
LOG_DIR="${LOG_DIR:-logs}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ------------------------------------------------------------------
# Stage switches  (set to 1 to skip)
# ------------------------------------------------------------------
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PAPER="${SKIP_PAPER:-0}"

# ------------------------------------------------------------------
# Python interpreter
# ------------------------------------------------------------------
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

# ------------------------------------------------------------------
echo "=========================================="
echo " FoundationMSMS – Full Experiment Run"
echo "=========================================="
echo "  Config:     $CONFIG"
echo "  Scenario:   $SCENARIO_OUT"
echo "  Task:       $TASK  |  Epochs: $EPOCHS  |  Device: $DEVICE"
echo ""

# ------------------------------------------------------------------
# Stage 1 – Download
# ------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" = "0" ]]; then
    echo "--- Stage 1: Download ---"
    $PYTHON -u -m foundationmsms.preprocessing batch-download \
        --config "$CONFIG" \
        --base-dir "$BASE_DIR" \
        --parallel "$PARALLEL" \
        --skip-existing
    echo ""
else
    echo "--- Stage 1: Download (skipped) ---"
fi

# ------------------------------------------------------------------
# Stage 2 – Preprocess
# ------------------------------------------------------------------
if [[ "$SKIP_PREPROCESS" = "0" ]]; then
    echo "--- Stage 2: Preprocess ---"
    for dataset_path in \
        "${BASE_DIR}/raw/pride/"*/ \
        "${BASE_DIR}/raw/massive/"*/; do
        [[ -d "$dataset_path" ]] || continue
        file_count=$(find "$dataset_path" \( -iname "*.raw" -o -iname "*.mzML" \) | wc -l)
        [[ "$file_count" -gt 0 ]] || continue
        dataset_name=$(basename "$dataset_path")
        echo "  Processing: $dataset_name"
        $PYTHON -u -m foundationmsms.preprocessing pipeline \
            --dataset-dir "$dataset_path" \
            --output-base "$BASE_DIR" \
            --workers "$WORKERS"
        echo "  Done: $dataset_name"
    done
    echo ""
else
    echo "--- Stage 2: Preprocess (skipped) ---"
fi

# ------------------------------------------------------------------
# Stage 3 – Build scenario (always runs; scenario info used by train and paper)
# ------------------------------------------------------------------
echo "--- Stage 3: Build scenario ---"

# Extract comma-separated dataset IDs from configs/datasets.yaml via Python
DATASET_IDS=$($PYTHON -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    ids = [str(d['id']) for d in cfg.get('datasets', [])]
    print(','.join(ids))
except Exception as e:
    print('', end='')
    sys.exit(1)
")

if [[ -z "$DATASET_IDS" ]]; then
    echo "ERROR: Could not extract dataset IDs from $CONFIG" >&2
    exit 1
fi

echo "  Datasets: $DATASET_IDS"
$PYTHON -u -m foundationmsms.preprocessing.build_scenario_from_voxel \
    --voxel-root "${BASE_DIR}/voxel" \
    --include "$DATASET_IDS" \
    --out "$SCENARIO_OUT" \
    --workers "$WORKERS"
echo ""

# ------------------------------------------------------------------
# Stage 4 – Train
# ------------------------------------------------------------------
if [[ "$SKIP_TRAIN" = "0" ]]; then
    echo "--- Stage 4: Train (task=$TASK) ---"
    train_args=(
        --scenario "$SCENARIO_OUT"
        --task      "$TASK"
        --device    "$DEVICE"
        --epochs    "$EPOCHS"
        --batch-size "$BATCH_SIZE"
        --lr        "$LR"
        --dim       "$DIM"
        --heads     "$HEADS"
        --cv-folds  "$CV_FOLDS"
        --warmup-epochs "$WARMUP_EPOCHS"
        --warmup-out    "$WARMUP_OUT"
        --log-dir   "$LOG_DIR"
    )
    [[ -n "$RESUME_CHECKPOINT" ]] && train_args+=(--resume-checkpoint "$RESUME_CHECKPOINT")
    $PYTHON -u -m foundationmsms.training.train_msw_transformer "${train_args[@]}"
    echo ""
else
    echo "--- Stage 4: Train (skipped) ---"
fi

# ------------------------------------------------------------------
# Stage 5 – Build paper artifacts
# ------------------------------------------------------------------
if [[ "$SKIP_PAPER" = "0" ]]; then
    echo "--- Stage 5: Build paper artifacts ---"
    bash "$(dirname "$0")/build_paper.sh"
    echo ""
else
    echo "--- Stage 5: Build paper (skipped) ---"
fi

echo "=========================================="
echo " Done."
echo "   Scenario:   $SCENARIO_OUT"
echo "   Paper:      $PAPER_OUT"
echo "=========================================="
