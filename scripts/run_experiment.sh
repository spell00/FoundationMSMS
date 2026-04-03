#!/usr/bin/env bash
# run_experiment.sh – end-to-end experiment pipeline
#
# Stages:
#   1. Download datasets listed in configs/download.yaml
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

# Resolve script/repo location so paths and interpreter are stable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Config paths  (override via env vars)
# ------------------------------------------------------------------
DATASETS_CONFIG="${DATASETS_CONFIG:-configs/datasets.yaml}"
DOWNLOAD_CONFIG="${DOWNLOAD_CONFIG:-configs/download.yaml}"
LABEL_CONFIG="${LABEL_CONFIG:-configs/label_parsing.yaml}"

 # ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
BASE_DIR="${BASE_DIR:-data}"
SCENARIO_OUT="${SCENARIO_OUT:-data/doc_scenarios/experiment_auto.npz}"
CUSTOM_SCENARIO_OUT="${CUSTOM_SCENARIO_OUT:-data/doc_scenarios/custom_scenario.npz}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
PAPER_OUT="${PAPER_OUT:-experiments/paper}"
# Voxel parameter subfolder (e.g. mzbin_1.0_mzparent_10.0_rtbin_10.0)
VOXEL_PARAM_SUBDIR="${VOXEL_PARAM_SUBDIR:-mzbin_1.0_mzparent_10.0_rtbin_10.0}"

# ------------------------------------------------------------------
# Download options
# ------------------------------------------------------------------
PARALLEL="${PARALLEL:-2}"

# ------------------------------------------------------------------
# Preprocessing options
# ------------------------------------------------------------------
WORKERS="${WORKERS:-8}"
FORCE_RAW="${FORCE_RAW:-0}"
FORCE_VOXEL="${FORCE_VOXEL:-0}"

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
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-0}"  # set to 1 to delete and re-download existing data
BUILD_CUSTOM_SCENARIO="${BUILD_CUSTOM_SCENARIO:-1}"  # set to 1 to also build custom scenario
CUSTOM_SCENARIO_IDS="${CUSTOM_SCENARIO_IDS:-MSV000083793,PXD010595,PXD012353,PXD021874}"
CUSTOM_MSV_LOCATION="${CUSTOM_MSV_LOCATION:-downloaded}"
AUTO_SCENARIO_NAMING="${AUTO_SCENARIO_NAMING:-1}"

# ------------------------------------------------------------------
# Python interpreter
# ------------------------------------------------------------------
DEFAULT_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
    PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
else
    PYTHON="${PYTHON:-python}"
fi
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/src}"

# ------------------------------------------------------------------
echo "=========================================="
echo " FoundationMSMS – Full Experiment Run"
echo "=========================================="
echo "  Download:   $DOWNLOAD_CONFIG"
echo "  Datasets:   $DATASETS_CONFIG"
echo "  Scenario:   $SCENARIO_OUT"
if [[ "$BUILD_CUSTOM_SCENARIO" = "1" ]]; then
    echo "  Custom:     $CUSTOM_SCENARIO_OUT  (IDs: $CUSTOM_SCENARIO_IDS, msv-location: $CUSTOM_MSV_LOCATION)"
fi
echo "  Task:       $TASK  |  Epochs: $EPOCHS  |  Device: $DEVICE"
echo "  Python:     $PYTHON ($($PYTHON --version 2>/dev/null || echo 'version unknown'))"
echo ""

# Build the active dataset list from download config.
# priority > 0 means active; priority = 0 means excluded.
ACTIVE_DATASET_IDS_CSV=$($PYTHON -c "
import sys, yaml
try:
    cfg = yaml.safe_load(open('$DOWNLOAD_CONFIG')) or {}
    ids = []
    for d in cfg.get('pride_datasets', []):
        if int(d.get('priority', 1)) > 0:
            # Use massive_id when present (data lands there), else pride_id
            canon = d.get('massive_id') or d.get('pride_id') or d.get('id')
            if canon:
                ids.append(str(canon))
    for d in cfg.get('massive_datasets', []):
        if int(d.get('priority', 1)) > 0:
            canon = d.get('massive_id') or d.get('id')
            if canon:
                ids.append(str(canon))
    print(','.join(ids))
except Exception as e:
    print(str(e), file=sys.stderr)
    print('', end='')
    sys.exit(1)
")

if [[ -z "$ACTIVE_DATASET_IDS_CSV" ]]; then
    echo "ERROR: No active datasets (priority > 0) found in $DOWNLOAD_CONFIG" >&2
    exit 1
fi

IFS=',' read -r -a ACTIVE_DATASET_IDS <<< "$ACTIVE_DATASET_IDS_CSV"

# Optional informative scenario naming: include total/labeled/unlabeled dataset counts.
if [[ "$AUTO_SCENARIO_NAMING" = "1" ]]; then
    ACTIVE_LABEL_COUNTS=$(IDS_CSV="$ACTIVE_DATASET_IDS_CSV" LABEL_CONFIG="$LABEL_CONFIG" $PYTHON - <<'PY'
import os
import yaml

ids = [x for x in os.environ.get("IDS_CSV", "").split(",") if x]
cfg_path = os.environ.get("LABEL_CONFIG", "")
label_cfg = {}
try:
    label_cfg = yaml.safe_load(open(cfg_path)) or {}
except Exception:
    label_cfg = {}
labeled = sum(1 for i in ids if i in label_cfg)
unlabeled = len(ids) - labeled
print(f"{len(ids)} {labeled} {unlabeled}")
PY
)

    ACTIVE_TOTAL_COUNT=$(echo "$ACTIVE_LABEL_COUNTS" | awk '{print $1}')
    ACTIVE_LABELED_COUNT=$(echo "$ACTIVE_LABEL_COUNTS" | awk '{print $2}')
    ACTIVE_UNLABELED_COUNT=$(echo "$ACTIVE_LABEL_COUNTS" | awk '{print $3}')

    if [[ "$SCENARIO_OUT" = "data/doc_scenarios/experiment_auto.npz" ]]; then
        SCENARIO_OUT="data/doc_scenarios/experiment_d${ACTIVE_TOTAL_COUNT}_l${ACTIVE_LABELED_COUNT}_u${ACTIVE_UNLABELED_COUNT}.npz"
    fi

    if [[ "$BUILD_CUSTOM_SCENARIO" = "1" ]]; then
        CUSTOM_LABEL_COUNTS=$(IDS_CSV="$CUSTOM_SCENARIO_IDS" LABEL_CONFIG="$LABEL_CONFIG" $PYTHON - <<'PY'
import os
import yaml

ids = [x for x in os.environ.get("IDS_CSV", "").split(",") if x]
cfg_path = os.environ.get("LABEL_CONFIG", "")
label_cfg = {}
try:
    label_cfg = yaml.safe_load(open(cfg_path)) or {}
except Exception:
    label_cfg = {}
labeled = sum(1 for i in ids if i in label_cfg)
unlabeled = len(ids) - labeled
print(f"{len(ids)} {labeled} {unlabeled}")
PY
)
        CUSTOM_TOTAL_COUNT=$(echo "$CUSTOM_LABEL_COUNTS" | awk '{print $1}')
        CUSTOM_LABELED_COUNT=$(echo "$CUSTOM_LABEL_COUNTS" | awk '{print $2}')
        CUSTOM_UNLABELED_COUNT=$(echo "$CUSTOM_LABEL_COUNTS" | awk '{print $3}')

        if [[ "$CUSTOM_SCENARIO_OUT" = "data/doc_scenarios/custom_scenario.npz" ]]; then
            CUSTOM_SCENARIO_OUT="data/doc_scenarios/custom_d${CUSTOM_TOTAL_COUNT}_l${CUSTOM_LABELED_COUNT}_u${CUSTOM_UNLABELED_COUNT}.npz"
        fi
    fi
fi

echo "  Effective scenario output: $SCENARIO_OUT"
if [[ "$BUILD_CUSTOM_SCENARIO" = "1" ]]; then
    echo "  Effective custom output:   $CUSTOM_SCENARIO_OUT"
fi
echo "  Voxel parameter subdir:    $VOXEL_PARAM_SUBDIR"
echo ""

# ------------------------------------------------------------------
# Stage 1 – Download
# ------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" = "0" ]]; then
    echo "--- Stage 1: Download ---"
    dl_args=(
        -u -m foundationmsms.preprocessing batch-download
        --config "$DOWNLOAD_CONFIG"
        --base-dir "$BASE_DIR"
        --parallel "$PARALLEL"
    )
    if [[ "$FORCE_REDOWNLOAD" = "1" ]]; then
        dl_args+=(--reload)
        echo "  (force re-download: existing data will be deleted first)"
    else
        dl_args+=(--skip-existing)
    fi
    for dsid in "${ACTIVE_DATASET_IDS[@]}"; do
        dl_args+=(--id "$dsid")
    done
    $PYTHON "${dl_args[@]}"
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
        pipe_args=(
            --dataset-dir "$dataset_path"
            --output-base "$BASE_DIR"
            --workers "$WORKERS"
        )
        [[ "$FORCE_RAW" = "1" ]] && pipe_args+=(--force-raw)
        [[ "$FORCE_VOXEL" = "1" ]] && pipe_args+=(--force-voxel)
        $PYTHON -u -m foundationmsms.preprocessing pipeline \
            "${pipe_args[@]}"
        echo "  Done: $dataset_name"
    done
    echo ""
else
    echo "--- Stage 2: Preprocess (skipped) ---"
fi

# ------------------------------------------------------------------
# Stage 3 – Build scenario (always runs; scenario info used by train and paper)
# ------------------------------------------------------------------
echo "  Datasets: $ACTIVE_DATASET_IDS_CSV"

echo "--- Stage 3: Build scenario ---"
echo "  Datasets: $ACTIVE_DATASET_IDS_CSV"
$PYTHON -u -m foundationmsms.preprocessing.build_scenario_from_voxel \
    --voxel-root "${BASE_DIR}/voxel" \
    --include "$ACTIVE_DATASET_IDS_CSV" \
    --out "$SCENARIO_OUT" \
    --workers "$WORKERS" \
    --voxel-param-subdir "$VOXEL_PARAM_SUBDIR"

SCENARIO_COUNTS=$(SCENARIO_PATH="$SCENARIO_OUT" $PYTHON - <<'PY'
import os
import numpy as np
from pathlib import Path

p = Path(os.environ["SCENARIO_PATH"])
if not p.exists():
    print("0 0")
else:
    sc = np.load(p, allow_pickle=True)
    tokens_idx = sc["tokens_idx"] if "tokens_idx" in sc else []
    total = len(tokens_idx)
    non_empty = sum(1 for x in tokens_idx if len(np.asarray(x)) > 0)
    print(f"{total} {non_empty}")
PY
)

SCENARIO_TOTAL_DOCS=$(echo "$SCENARIO_COUNTS" | awk '{print $1}')
SCENARIO_NONEMPTY_DOCS=$(echo "$SCENARIO_COUNTS" | awk '{print $2}')
echo "  Scenario docs: total=${SCENARIO_TOTAL_DOCS}, non-empty=${SCENARIO_NONEMPTY_DOCS}"
if [[ "$SCENARIO_NONEMPTY_DOCS" -eq 0 ]]; then
    echo "ERROR: Scenario has no trainable documents: $SCENARIO_OUT" >&2
    echo "       Check that preprocessing produced voxel *.npz files under ${BASE_DIR}/voxel for included datasets: $ACTIVE_DATASET_IDS_CSV" >&2
    echo "       Aborting before training." >&2
    exit 1
fi

if [[ "$BUILD_CUSTOM_SCENARIO" = "1" ]]; then
    echo "  Building custom scenario: $CUSTOM_SCENARIO_OUT"
    echo "  Custom datasets: $CUSTOM_SCENARIO_IDS"
    $PYTHON -u -m foundationmsms.preprocessing.build_scenario_from_voxel \
        --voxel-root "${BASE_DIR}/voxel" \
        --include "$CUSTOM_SCENARIO_IDS" \
        --out "$CUSTOM_SCENARIO_OUT" \
        --workers "$WORKERS" \
        --voxel-param-subdir "$VOXEL_PARAM_SUBDIR"

    CUSTOM_SCENARIO_COUNTS=$(SCENARIO_PATH="$CUSTOM_SCENARIO_OUT" $PYTHON - <<'PY'
import os
import numpy as np
from pathlib import Path

p = Path(os.environ["SCENARIO_PATH"])
if not p.exists():
    print("0 0")
else:
    sc = np.load(p, allow_pickle=True)
    tokens_idx = sc["tokens_idx"] if "tokens_idx" in sc else []
    total = len(tokens_idx)
    non_empty = sum(1 for x in tokens_idx if len(np.asarray(x)) > 0)
    print(f"{total} {non_empty}")
PY
)

    CUSTOM_SCENARIO_TOTAL_DOCS=$(echo "$CUSTOM_SCENARIO_COUNTS" | awk '{print $1}')
    CUSTOM_SCENARIO_NONEMPTY_DOCS=$(echo "$CUSTOM_SCENARIO_COUNTS" | awk '{print $2}')
    echo "  Custom scenario docs: total=${CUSTOM_SCENARIO_TOTAL_DOCS}, non-empty=${CUSTOM_SCENARIO_NONEMPTY_DOCS}"
    if [[ "$CUSTOM_SCENARIO_NONEMPTY_DOCS" -eq 0 ]]; then
        echo "ERROR: Custom scenario has no trainable documents: $CUSTOM_SCENARIO_OUT" >&2
        echo "       Check voxel files for included datasets: $CUSTOM_SCENARIO_IDS" >&2
        exit 1
    fi
fi
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
if [[ "$BUILD_CUSTOM_SCENARIO" = "1" ]]; then
    echo "   Custom:     $CUSTOM_SCENARIO_OUT"
fi
echo "   Paper:      $PAPER_OUT"
echo "=========================================="
