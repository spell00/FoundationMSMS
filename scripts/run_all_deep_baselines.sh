#!/usr/bin/env bash
# run_all_deep_baselines.sh - Run all deep baselines with Optuna optimization
#
# Usage:
#   bash scripts/run_all_deep_baselines.sh data/doc_scenarios/custom_d4_l2_u0_new.npz [--force]

set -euo pipefail

SCENARIO="${1:-data/doc_scenarios/custom_d4_l2_u0_new.npz}"

TRIALS="${TRIALS:-20}" # Number of Optuna trials per model
EPOCHS="${EPOCHS:-1000}" # Number of epochs per model
OPTUNA_WARMUP="${OPTUNA_WARMUP:-0}" # Optuna pruner n_warmup_steps
RESET_OPTUNA=""
if [[ "${3:-}" == "--reset-optuna" ]]; then
    RESET_OPTUNA="--reset-optuna"
fi
PYTHON="${PYTHON:-python}"
FORCE=""
if [[ "${2:-}" == "--force" ]]; then
    FORCE="--force"
fi

# Get unique dataset IDs from the NPZ if available
DATASETS=$($PYTHON -c "import numpy as np; d = np.load('$SCENARIO', allow_pickle=True); print(' '.join(np.unique(d['dataset_ids'])))" 2>/dev/null || echo "")

if [[ -z "$DATASETS" ]]; then
    echo "Note: dataset_ids not found in NPZ. Results will be saved under scenario name."
    DATASETS="DEFAULT"
fi

for DS in $DATASETS; do

    DS_FLAG=""
    DS_NAME="$DS"
    if [[ "$DS" != "DEFAULT" ]]; then
        DS_FLAG="--dataset $DS"
    else
        DS_NAME="$(basename "$SCENARIO" .npz)"
    fi

    echo "########################################################"
    echo "### Starting optimization for Dataset: $DS_NAME"
    echo "########################################################"

    # Reordered from Fastest to Slowest
    MODELS=("mlp" "cnn" "deepmlp" "resnet" "gru" "lstm" "bigru" "bilstm" "vit")
    
    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        STEP=$((i+1))
        TOTAL=${#MODELS[@]}

        # Optuna study name is unique per model/dataset
        STUDY_NAME="study_${DS_NAME}_${MODEL}"

        echo ""
        echo "[$STEP/$TOTAL] Optimizing $MODEL on $DS_NAME..."
        echo "[INFO] Optuna study name: $STUDY_NAME (reset for each model/dataset)"
        $PYTHON scripts/train_deep_baseline.py \
            --scenario "$SCENARIO" \
            --model "$MODEL" \
            --optuna-trials "$TRIALS" \
            --epochs "$EPOCHS" \
            --optuna-warmup "$OPTUNA_WARMUP" \
            --outdir logs/deep_baselines \
            $DS_FLAG \
            $FORCE \
            $RESET_OPTUNA
    done
done

echo ""
echo "All runs complete for all datasets."
echo "Metrics and summary saved in logs/deep_baselines/"
echo "You can now run 'bash scripts/build_paper.sh' to update figures."
