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

$PYTHON -u -m foundationmsms.experiments.build_paper \
    --scenario        "$SCENARIO_OUT" \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --out             "$PAPER_OUT" \
    --expected-model  msw_transformer

echo ""
echo "Paper artifacts written to: $PAPER_OUT"
