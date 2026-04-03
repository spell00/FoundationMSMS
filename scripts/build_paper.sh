#!/usr/bin/env bash
# build_paper.sh – build paper figures and tables from a trained experiment
#
# Works even when not all models have been trained — missing checkpoints are
# reported in the output tables rather than causing failures.
#
# Usage:
#   bash scripts/build_paper.sh
#   SCENARIO_OUT=data/doc_scenarios/my.npz PAPER_OUT=experiments/paper bash scripts/build_paper.sh

set -euo pipefail

SCENARIO_OUT="${SCENARIO_OUT:-data/doc_scenarios/experiment_auto.npz}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
PAPER_OUT="${PAPER_OUT:-experiments/paper}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "Building paper artifacts..."
echo "  Scenario:   $SCENARIO_OUT"
echo "  Checkpoints: $CHECKPOINTS_DIR"
echo "  Output:     $PAPER_OUT"
echo ""

if $PYTHON - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('foundationmsms.experiments.build_paper') else 1)
PY
then
    echo "[paper] Building model/report artifacts..."
    $PYTHON -u -m foundationmsms.experiments.build_paper \
            --scenario        "$SCENARIO_OUT" \
            --checkpoints-dir "$CHECKPOINTS_DIR" \
            --out             "$PAPER_OUT" \
            --expected-model  msw_transformer
else
    echo "[paper] Note: foundationmsms.experiments.build_paper not found; generated metrics/baseline artifacts only."
fi

echo ""
echo "[paper] Building model metrics figures..."
# Find all deep baseline metrics files from other runs or experiments
DEEP_METRICS=""
if [[ -d "logs" ]]; then
    # Individual run metrics
    DEEP_METRICS=$(find logs -name "deep_baseline_metrics.csv" 2>/dev/null | tr '\n' ' ')
    # Cumulative deep baselines CSV (if it exists)
    if [[ -f "logs/deep_baselines/baselines_fold_metrics.csv" ]]; then
        DEEP_METRICS="$DEEP_METRICS logs/deep_baselines/baselines_fold_metrics.csv"
    fi
fi

if [[ -f "logs/baselines/baseline_fold_metrics.csv" ]]; then
    $PYTHON -u scripts/plot_baseline_metrics.py \
            --csv logs/baselines/baseline_fold_metrics.csv $DEEP_METRICS \
            --outfig "$PAPER_OUT/baseline_metrics"
else
    echo "[paper] Note: logs/baselines/baseline_fold_metrics.csv not found, skipping baseline metrics figures."
fi

echo ""
echo "[paper] Building data EDA figures..."
SCENARIO_TAG="$(basename "$SCENARIO_OUT")"
SCENARIO_TAG="${SCENARIO_TAG%.npz}"
EDA_OUT="$PAPER_OUT/$SCENARIO_TAG/eda"
mkdir -p "$EDA_OUT"
echo "  EDA output:  $EDA_OUT"

$PYTHON -u scripts/generate_data_eda.py \
        --scenario "$SCENARIO_OUT" \
        --voxel-root data/voxel \
        --out-dir "$EDA_OUT"

echo ""
echo "Paper artifacts written to: $PAPER_OUT"
