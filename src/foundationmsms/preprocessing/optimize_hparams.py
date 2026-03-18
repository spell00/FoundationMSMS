"""
Hyperparameter optimization script using Optuna for MSWTransformer, BERT, and other models.

Usage:
    python -m foundationmsms.preprocessing.optimize_hparams --model msw_transformer --scenario data/doc_scenarios/docs_frag_only.npz --voxel-root data/voxel --n_trials 50

This script runs Optuna to optimize hyperparameters for the selected model. It is designed to be extensible for BERT and other architectures.
"""

import argparse
from pathlib import Path
import optuna
import numpy as np
import yaml

# Import model-specific training functions
from foundationmsms.training.train_msw_transformer import run_training, TrainConfig
# For BERT and other models, import their training entry points here
# from train_bert import run_training as run_bert_training, BertConfig


def objective(trial, args):
    # Example: define search space for MSWTransformer
    if args.model == "msw_transformer":
        dim = trial.suggest_int("dim", 32, 256)
        heads = trial.suggest_int("heads", 1, 8)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 64)
        windows = tuple(trial.suggest_categorical("windows", [(5,10,20), (10,20,40), (20,40,80)]))
        epochs = trial.suggest_int("epochs", 1, 10)
        recon_weight = trial.suggest_float("recon_weight", 0.1, 2.0)
        clf_weight = trial.suggest_float("clf_weight", 0.1, 2.0)
        embedding_size = trial.suggest_int("embedding_size", 64, 512)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        num_layers = trial.suggest_int("num_layers", 1, 8)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        cfg = TrainConfig(
            scenario_path=Path(args.scenario),
            voxel_dir=Path(args.voxel_root) if args.voxel_root else None,
            npz_limit=None,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            dim=dim,
            heads=heads,
            windows=windows,
            device=args.device,
            task=args.task,
            recon_weight=recon_weight,
            clf_weight=clf_weight,
            test_frac=args.test_frac,
            cv_folds=args.cv_folds,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            n_steps_per_epoch=args.n_steps_per_epoch,
            embedding_size=embedding_size,
            weight_decay=weight_decay,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Run training and return validation score for Optuna
        result = run_training(cfg, args)
        # Assume run_training returns best validation score (lower is better)
        return result
    elif args.model == "bert":
        # Example for BERT (to be implemented)
        pass
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["msw_transformer", "bert"], help="Model to optimize")
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--voxel-root", type=str, default=None)
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--task", type=str, default="joint")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--n-steps-per-epoch", type=int, default=None)
    args = ap.parse_args()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("Best trial:")
    print(study.best_trial)
    print("Best params:")
    print(study.best_params)

if __name__ == "__main__":
    main()
