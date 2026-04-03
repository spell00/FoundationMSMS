#!/usr/bin/env python3
"""
Train deep learning baselines (CNN, ViT, etc.) for LC-MS/MS scenario classification.

 Hyperparameters via CLI
- Logs to terminal, TensorBoard, and saves learning curves
- Outputs metrics CSV compatible with plot_baseline_metrics.py
- Easily extensible for new architectures
"""

import argparse
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Model zoo ---
class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, channels, n_classes):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(channels * 2, n_classes)

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        encoded = (token_emb + value_emb).transpose(1, 2)
        features = self.encoder(encoded)
        avg_pool = torch.mean(features, dim=-1)
        max_pool = torch.amax(features, dim=-1)
        pooled = self.dropout(torch.cat([avg_pool, max_pool], dim=-1))
        return self.classifier(pooled)

class SimpleViT(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, token_ids, values):
        seq_len = token_ids.size(1)
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = token_emb + value_emb + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        mask = (token_ids == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        weights = (~mask).float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return self.classifier(pooled)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, rnn_type="lstm", n_layers=2, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        rnn_init = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_init(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = token_emb + value_emb
        x, _ = self.rnn(x)
        mask = (token_ids == 0)
        weights = (~mask).float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return self.classifier(pooled)

class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = token_emb + value_emb
        mask = (token_ids == 0)
        temp_weights = (~mask).float().unsqueeze(-1)
        pooled = (x * temp_weights).sum(dim=1) / (temp_weights.sum(dim=1) + 1e-6)
        return self.mlp(pooled)

class ResNetCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, channels, n_classes, n_blocks=6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.init_conv = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
            ) for _ in range(n_blocks)
        ])
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, n_classes)
        )

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = (token_emb + value_emb).transpose(1, 2)
        x = self.init_conv(x)
        for block in self.blocks:
            x = x + block(x) # Skip connection
            x = torch.relu(x)
        avg_p = torch.mean(x, dim=-1)
        max_p = torch.amax(x, dim=-1)
        pooled = self.dropout(torch.cat([avg_p, max_p], dim=-1))
        return self.classifier(pooled)

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, rnn_type="lstm", n_layers=3, dropout=0.3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = token_emb + value_emb
        x, _ = self.rnn(x)
        mask = (token_ids == 0)
        weights = (~mask).float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        return self.classifier(pooled)

class DeepMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=5, dropout=0.3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.value_projection = nn.Linear(1, embed_dim, bias=False)
        self.init_proj = nn.Linear(embed_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, token_ids, values):
        token_emb = self.token_embedding(token_ids)
        value_emb = self.value_projection(values.unsqueeze(-1))
        x = token_emb + value_emb
        mask = (token_ids == 0)
        temp_w = (~mask).float().unsqueeze(-1)
        x = (x * temp_w).sum(dim=1) / (temp_w.sum(dim=1) + 1e-6)
        x = self.init_proj(x)
        for layer in self.layers:
            x = x + layer(x) # Residual
        return self.classifier(x)

MODEL_ZOO = {
    "cnn": SimpleCNN,
    "vit": SimpleViT,
    "lstm": lambda v, e, h, n: SimpleRNN(v, e, h, n, "lstm"),
    "gru": lambda v, e, h, n: SimpleRNN(v, e, h, n, "gru"),
    "mlp": SimpleMLP,
    "resnet": ResNetCNN,
    "bilstm": lambda v, e, h, n: BiRNN(v, e, h, n, "lstm"),
    "bigru": lambda v, e, h, n: BiRNN(v, e, h, n, "gru"),
    "deepmlp": DeepMLP,
}

# --- Dataset ---
class ScenarioDataset(Dataset):
    def __init__(self, npz_path, max_seq_len=2048, filter_dataset=None):
        sc = np.load(npz_path, allow_pickle=True)
        tokens_idx = sc["tokens_idx"]
        tokens_val = sc["tokens_val"]
        raw_labels = sc["labels"]
        dataset_ids = sc.get("dataset_ids") if "dataset_ids" in sc.files else None
        sample_names = None
        if "sample_names" in sc.files:
            sample_names = sc["sample_names"]
        elif "sample_parents" in sc.files:
            sample_names = sc["sample_parents"]
        
        # Apply filtering if requested
        if filter_dataset and dataset_ids is not None:
            mask = (dataset_ids == filter_dataset)
            self.token_ids = tokens_idx[mask]
            self.token_vals = tokens_val[mask]
            raw_labels = raw_labels[mask]
            self.dataset_ids = dataset_ids[mask]
            self.sample_names = sample_names[mask] if sample_names is not None else np.array(["unknown"] * len(raw_labels), dtype=object)
            print(f"Filtered dataset to '{filter_dataset}': {len(raw_labels)} samples remaining.")
        else:
            self.token_ids = tokens_idx
            self.token_vals = tokens_val
            self.dataset_ids = dataset_ids if dataset_ids is not None else np.array(["unknown"] * len(raw_labels), dtype=object)
            self.sample_names = sample_names if sample_names is not None else np.array(["unknown"] * len(raw_labels), dtype=object)

        # Map labels (which might be strings) to integer indices
        self.raw_labels = np.asarray(raw_labels, dtype=object)
        self.unique_labels, self.labels = np.unique(self.raw_labels, return_inverse=True)
        
        self.max_seq_len = max_seq_len
        self.n_classes = len(self.unique_labels)
        self.vocab_size = int(np.max([np.max(x) if len(x) > 0 else 0 for x in self.token_ids])) + 2 if len(self.token_ids) > 0 else 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.token_ids[idx][:self.max_seq_len] + 1
        vals = self.token_vals[idx][:self.max_seq_len]
        label = int(self.labels[idx])
        return torch.as_tensor(ids, dtype=torch.long), torch.as_tensor(vals, dtype=torch.float32), label


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = int(self.indices[idx])
        token_ids, token_vals, label = self.dataset[original_idx]
        return token_ids, token_vals, label, original_idx

def collate_fn(batch):
    ids, vals, labels = zip(*batch)
    max_len = max(x.shape[0] for x in ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_vals = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, (id_, val) in enumerate(zip(ids, vals)):
        l = id_.shape[0]
        padded_ids[i, :l] = id_
        padded_vals[i, :l] = val
    return padded_ids, padded_vals, torch.as_tensor(labels, dtype=torch.long)


def collate_fn_with_index(batch):
    ids, vals, labels, original_indices = zip(*batch)
    max_len = max(x.shape[0] for x in ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_vals = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, (id_, val) in enumerate(zip(ids, vals)):
        l = id_.shape[0]
        padded_ids[i, :l] = id_
        padded_vals[i, :l] = val
    return (
        padded_ids,
        padded_vals,
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(original_indices, dtype=torch.long),
    )


def get_git_info():
    info = {"commit": None, "branch": None, "dirty": None}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def write_run_metadata(run_dir, args, scenario_path, dataset_name, model_name, mz_bin, mz_parent_bin, rt_bin_sec):
    metadata_dir = run_dir / "metadata"
    code_dir = run_dir / "code"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "scenario": str(scenario_path),
        "dataset": dataset_name,
        "model": model_name,
        "binning": {
            "mz_bin": mz_bin,
            "mz_parent_bin": mz_parent_bin,
            "rt_bin_sec": rt_bin_sec,
        },
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "git": get_git_info(),
    }
    with (metadata_dir / "run_metadata.json").open("w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    this_script = Path(__file__).resolve()
    shutil.copy2(this_script, code_dir / this_script.name)

    scenario_meta = scenario_path.parent / "scenario_meta.yaml"
    if scenario_meta.exists():
        shutil.copy2(scenario_meta, metadata_dir / "scenario_meta.yaml")


def export_split_predictions(model, ds, indices, split_name, fold, batch_size, device, out_csv):
    subset = IndexedSubset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_index)
    model.eval()

    rows = []
    label_names = [str(x) for x in ds.unique_labels]
    with torch.no_grad():
        for token_ids, values, labels, original_indices in loader:
            token_ids = token_ids.to(device)
            values = values.to(device)
            logits = model(token_ids, values)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            labels_np = labels.numpy()
            orig_np = original_indices.numpy()
            for i in range(len(orig_np)):
                original_idx = int(orig_np[i])
                row = {
                    "fold": int(fold),
                    "split": split_name,
                    "sample_index": original_idx,
                    "dataset_id": str(ds.dataset_ids[original_idx]) if original_idx < len(ds.dataset_ids) else "unknown",
                    "sample_name": str(ds.sample_names[original_idx]) if original_idx < len(ds.sample_names) else "unknown",
                    "true_label_idx": int(labels_np[i]),
                    "true_label": label_names[int(labels_np[i])] if int(labels_np[i]) < len(label_names) else "unknown",
                    "pred_label_idx": int(preds[i]),
                    "pred_label": label_names[int(preds[i])] if int(preds[i]) < len(label_names) else "unknown",
                }
                for class_idx, class_name in enumerate(label_names):
                    row[f"prob_class_{class_name}"] = float(probs[i, class_idx])
                rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)

# --- Training ---
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch=0, epochs=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
    for token_ids, values, labels in pbar:
        token_ids, values, labels = token_ids.to(device), values.to(device), labels.to(device)
        logits = model(token_ids, values)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / (total + 1e-9), correct / (total + 1e-9)

def evaluate(model, loader, loss_fn, device, desc="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    pbar = tqdm(loader, desc=f"Evaluating [{desc}]", leave=False)
    with torch.no_grad():
        for token_ids, values, labels in pbar:
            token_ids, values, labels = token_ids.to(device), values.to(device), labels.to(device)
            logits = model(token_ids, values)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.set_postfix(acc=f"{correct/total:.4f}")
    acc = correct / (total + 1e-9)
    from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    return total_loss / (total + 1e-9), acc, macro_f1, bal_acc, mcc

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def run_training(
    args,
    train_idx,
    val_idx,
    ds,
    vocab_size,
    n_classes,
    device,
    trial=None,
    mz_bin=None,
    mz_parent_bin=None,
    rt_bin_sec=None,
    fold=None,
    fold_dir=None,
):
    # Hyperparams from args or trial
    lr = args.lr
    embed_dim = args.embed_dim
    channels = args.channels
    batch_size = args.batch_size
    n_heads = 8
    n_layers = 4
    
    if trial:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        if args.model in ["cnn", "resnet"]:
            channels = trial.suggest_categorical("channels", [64, 128, 256])
        elif args.model == "vit":
            n_heads = trial.suggest_categorical("n_heads", [4, 8])
            n_layers = trial.suggest_int("n_layers", 2, 6)
        elif args.model in ["lstm", "gru", "mlp", "bilstm", "bigru", "deepmlp"]:
            channels = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    train_loader = DataLoader(ds, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
    val_loader = DataLoader(ds, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), collate_fn=collate_fn)

    if args.model in ["cnn", "resnet", "lstm", "gru", "mlp", "bilstm", "bigru", "deepmlp"]:
        model = MODEL_ZOO[args.model](vocab_size, embed_dim, channels, n_classes).to(device)
    elif args.model == "vit":
        model = SimpleViT(vocab_size, embed_dim, n_classes, n_heads=n_heads, n_layers=n_layers).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=15)
    
    best_v_f1 = 0
    best_epoch = 0
    best_state_dict = None
    history_rows = []
    if trial:
        print(f"\n[Dataset: {args.scenario.stem}] [Trial {trial.number}] Parameters: {trial.params}")

    scenario_id = str(args.scenario) if hasattr(args, 'scenario') else ''
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, args.epochs)
        v_loss, v_acc, v_f1, v_bal, v_mcc = evaluate(model, val_loader, loss_fn, device, desc="Val")

        if v_f1 > best_v_f1:
            best_v_f1 = v_f1
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Periodic update for trials or always for normal runs
        if not trial or (epoch % 5 == 0) or epoch == 1:
            log_str = f"Epoch {epoch:03d}/{args.epochs:03d} | Train: loss={t_loss:.4f}, acc={t_acc:.4f} | Val: loss={v_loss:.4f}, f1={v_f1:.4f}, acc={v_acc:.4f}, mcc={v_mcc:.4f} (best_f1={best_v_f1:.4f})"
            print(log_str)

        history_rows.append({
            "epoch": int(epoch),
            "train_loss": float(t_loss),
            "train_accuracy": float(t_acc),
            "val_loss": float(v_loss),
            "val_accuracy": float(v_acc),
            "val_macro_f1": float(v_f1),
            "val_balanced_accuracy": float(v_bal),
            "val_mcc": float(v_mcc),
        })

        early_stopper(v_f1)
        if early_stopper.early_stop:
            if not trial or (epoch % 5 != 0): # Final print if we stop early
                print(f"Early stopping at epoch {epoch} | Best Val F1: {best_v_f1:.4f}")
            break

        if trial:
            trial.report(v_f1, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}.")
                import optuna
                raise optuna.exceptions.TrialPruned()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if fold_dir is not None and not trial:
        checkpoints_dir = fold_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoints_dir / "best.pth")
        with (fold_dir / "training_history.csv").open("w") as f:
            pd.DataFrame(history_rows).to_csv(f, index=False)
        with (fold_dir / "best_epoch.json").open("w") as f:
            json.dump({"best_epoch": int(best_epoch), "best_val_macro_f1": float(best_v_f1)}, f, indent=2)

    # Log all metrics for this trial if using Optuna
    if trial:
        # Save metrics for this trial to baselines_all_trials.csv
        import pandas as pd
        from datetime import datetime
        all_trials_csv = Path("logs/deep_baselines/baselines_all_trials.csv")
        row = {
            "scenario": scenario_id,
            "dataset": args.dataset if args.dataset else args.scenario.stem,
            "model": args.model,
            "fold": int(fold) if fold is not None else 0,
            "trial_number": trial.number,
            "params": str(trial.params),
            "train_loss": t_loss,
            "train_accuracy": t_acc,
            "val_loss": v_loss,
            "val_accuracy": v_acc,
            "val_macro_f1": v_f1,
            "val_balanced_accuracy": v_bal,
            "val_mcc": v_mcc,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "mz_bin": mz_bin,
            "mz_parent_bin": mz_parent_bin,
            "rt_bin_sec": rt_bin_sec,
        }
        for param_name, param_value in trial.params.items():
            row[f"param_{param_name}"] = param_value
        df = pd.DataFrame([row])
        if not all_trials_csv.parent.exists():
            all_trials_csv.parent.mkdir(parents=True, exist_ok=True)
        if all_trials_csv.exists():
            df.to_csv(all_trials_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(all_trials_csv, index=False)
        run_trials_csv = getattr(args, "run_trials_csv", None)
        if run_trials_csv is not None:
            run_trials_csv = Path(run_trials_csv)
            run_trials_csv.parent.mkdir(parents=True, exist_ok=True)
            if run_trials_csv.exists():
                df.to_csv(run_trials_csv, mode='a', header=False, index=False)
            else:
                df.to_csv(run_trials_csv, index=False)

    return best_v_f1, model

def load_scenario_binning_params(scenario_path):
    """Load mz_bin, mz_parent_bin, rt_bin_sec from scenario_meta.yaml."""
    import yaml
    scenario_dir = scenario_path.parent
    meta_path = scenario_dir / "scenario_meta.yaml"
    
    if not meta_path.exists():
        return None, None, None
    
    try:
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        return (
            meta.get("mz_bin"),
            meta.get("mz_parent_bin"),
            meta.get("rt_bin_sec")
        )
    except Exception:
        return None, None, None


def resolve_scenario_path(scenario_path):
    """Resolve scenario path from a simple --scenario path.

    If the path does not exist, try:
      <parent>/mzbin_*/<stem>/<stem>.npz
    """
    scenario_path = Path(scenario_path)
    if scenario_path.exists():
        return scenario_path

    parent = scenario_path.parent
    stem = scenario_path.stem
    matches = sorted(parent.glob(f"mzbin_*/{stem}/{stem}.npz"))
    if matches:
        return matches[0]
    return scenario_path

# --- Main ---
def main():

    parser = argparse.ArgumentParser(description="Train deep learning baselines (CNN, ViT, etc.)")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to scenario NPZ")
    parser.add_argument("--model", type=str, default="cnn", choices=list(MODEL_ZOO.keys()), help="Model type")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--outdir", type=Path, default=Path("logs/deep_baseline"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--folds", type=int, default=1, help="Number of CV folds for deep baselines (1 disables CV)")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Number of Optuna trials (0 = skip)")
    parser.add_argument("--force", action="store_true", help="Force saving metrics even if not better than previous runs")
    parser.add_argument("--dataset", type=str, default=None, help="Filter to a specific dataset ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--optuna-warmup", type=int, default=15, help="Optuna pruner n_warmup_steps (patience before pruning, 0 disables pruning)")
    parser.add_argument("--reset-optuna", action="store_true", help="If set, delete Optuna study for this model/dataset before running (start from trial 0)")
    args = parser.parse_args()
    args.scenario = resolve_scenario_path(args.scenario)

    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # Data
    ds = ScenarioDataset(args.scenario, max_seq_len=args.max_seq_len, filter_dataset=args.dataset)
    n_classes = ds.n_classes
    vocab_size = ds.vocab_size
    
    if n_classes < 2:
        print(f"Skipping training for dataset '{args.dataset}': only {n_classes} classes.")
        return

    from sklearn.model_selection import StratifiedKFold
    idx = np.arange(len(ds))
    y = ds.labels if hasattr(ds, 'labels') else np.zeros(len(ds))
    n_folds = max(1, int(args.folds))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_name = args.dataset if args.dataset else args.scenario.stem
    model_name = args.model if args.optuna_trials == 0 else f"{args.model}_opt"
    
    # Load binning parameters and organize output by scenario with binning info
    mz_bin, mz_parent_bin, rt_bin_sec = load_scenario_binning_params(args.scenario)
    
    # Build output path as:
    # logs/deep_baselines/mzbin_*_mzparent_*_rtbin_*/<scenario-name>/...
    if mz_bin is not None and mz_parent_bin is not None and rt_bin_sec is not None:
        voxel_param_subdir = f"mzbin_{mz_bin}_mzparent_{mz_parent_bin}_rtbin_{rt_bin_sec}"
        scenario_outdir = Path("logs/deep_baselines") / voxel_param_subdir / dataset_name
    else:
        scenario_outdir = Path("logs/deep_baselines") / dataset_name
    scenario_outdir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = scenario_outdir / "models" / model_name / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = str(run_dir)
    args.run_trials_csv = str(run_dir / "results" / "all_trials.csv")
    write_run_metadata(run_dir, args, args.scenario, dataset_name, model_name, mz_bin, mz_parent_bin, rt_bin_sec)
    
    out_csv = scenario_outdir / "baselines_fold_metrics.csv"
    out_db = out_csv.with_suffix(".db")
    optuna_db = f"sqlite:///{out_csv.parent}/optuna.db"

    # Step 4: Skip non-optuna runs if already exists
    if args.optuna_trials == 0 and not args.force and out_csv.exists():
        try:
            old_df = pd.read_csv(out_csv)
            if ((old_df["dataset"] == dataset_name) & (old_df["model"] == model_name)).any():
                print(f"Skipping: {model_name} on {dataset_name} already computed. Use --force to re-run.")
                return
        except Exception as e:
            print(f"Warning checking CSV: {e}")

    final_params = {
        "lr": args.lr, "embed_dim": args.embed_dim, "batch_size": args.batch_size,
        "channels": args.channels, "n_heads": 8, "n_layers": 4, "max_seq_len": args.max_seq_len
    }

    split_trace = []
    if n_folds < 2:
        # Manual 80/10/10 split
        np.random.shuffle(idx)
        n = len(idx)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]
        split_trace.append({
            "fold": 0,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist()
        })
        folds = [(0, train_idx, val_idx, test_idx)]
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []
        for fold, (trainval_idx, test_idx) in enumerate(skf.split(idx, y)):
            np.random.shuffle(trainval_idx)
            split = int(0.8 * len(trainval_idx))
            train_idx, val_idx = trainval_idx[:split], trainval_idx[split:]
            split_trace.append({
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist()
            })
            folds.append((fold, train_idx, val_idx, test_idx))

    # Save split trace to file
    trace_path = out_csv.parent / f"split_trace_{dataset_name}_{model_name}.json"
    import json
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as f:
        json.dump(split_trace, f, indent=2)
    print(f"Saved split trace to {trace_path}")
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    with (run_dir / "metadata" / "split_trace.json").open("w") as f:
        json.dump(split_trace, f, indent=2)

    # Now run training/eval for each fold
    for fold, train_idx, val_idx, test_idx in folds:
        fold_dir = run_dir / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "split_indices.json").open("w") as f:
            json.dump(
                {
                    "fold": int(fold),
                    "train_idx": np.asarray(train_idx).tolist(),
                    "val_idx": np.asarray(val_idx).tolist(),
                    "test_idx": np.asarray(test_idx).tolist(),
                },
                f,
                indent=2,
            )
        # ...existing code for Optuna and non-Optuna runs, using train_idx, val_idx, test_idx, and fold...

        if args.optuna_trials > 0:
            import optuna
            if args.optuna_warmup == 0:
                pruner = optuna.pruners.NopPruner()
                print("[INFO] Optuna pruning is disabled (NopPruner)")
            else:
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.optuna_warmup)
            study_name = f"study_{dataset_name}_{args.model}"
            # If --reset-optuna, delete the study from the DB before running
            if args.reset_optuna:
                from sqlalchemy import create_engine, text
                engine = create_engine(optuna_db)
                try:
                    with engine.connect() as conn:
                        # Delete study and all associated trials
                        conn.execute(text("DELETE FROM studies WHERE study_name=:name"), {"name": study_name})
                        conn.execute(text("DELETE FROM trials WHERE study_id NOT IN (SELECT study_id FROM studies)"))
                        conn.commit()
                    print(f"[INFO] Deleted Optuna study '{study_name}' from DB (reset)")
                except Exception as e:
                    print(f"[WARN] Could not reset Optuna study '{study_name}': {e}")
            study = optuna.create_study(study_name=study_name, storage=optuna_db, direction="maximize", load_if_exists=True, pruner=pruner)

            # Step 5: Quota logic
            n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if not args.force and n_existing >= args.optuna_trials:
                print(f"Skipping: {args.model} on {dataset_name} already has {n_existing} trials (quota: {args.optuna_trials}).")
                best_f1 = study.best_value
                final_params.update(study.best_params)
            else:
                n_to_run = args.optuna_trials - n_existing if not args.force else args.optuna_trials
                print(f"Starting/Resuming Optuna for {args.model} on {dataset_name}. Quota: {args.optuna_trials}, Existing: {n_existing}. Running {n_to_run} more.")
                study.optimize(lambda t: run_training(args, train_idx, val_idx, ds, vocab_size, n_classes, device, t, mz_bin=mz_bin, mz_parent_bin=mz_parent_bin, rt_bin_sec=rt_bin_sec, fold=fold, fold_dir=fold_dir)[0], 
                            n_trials=n_to_run)
                print("\nOptimization Complete!")
                print(f"Best trial value for {dataset_name}/{args.model}: {study.best_value:.4f}")
                print(f"Best params: {study.best_params}")
                best_f1 = study.best_value
                final_params.update(study.best_params)

            # After Optuna, evaluate the best model with all metrics and log a single row
            # Re-train model with best params
            # Update args with best params for reproducibility
            for k, v in study.best_params.items():
                if hasattr(args, k):
                    setattr(args, k, v)
            # Update final_params with actual values used for retraining
            final_params.update({
                "lr": args.lr,
                "embed_dim": args.embed_dim,
                "batch_size": args.batch_size,
                "channels": args.channels,
                "n_heads": getattr(args, "n_heads", 8),
                "n_layers": getattr(args, "n_layers", 4),
                "max_seq_len": args.max_seq_len
            })
            best_f1, model = run_training(args, train_idx, val_idx, ds, vocab_size, n_classes, device, mz_bin=mz_bin, mz_parent_bin=mz_parent_bin, rt_bin_sec=rt_bin_sec, fold=fold, fold_dir=fold_dir)
            # Evaluate on train and val sets
            _, t_acc, t_f1, t_bal, t_mcc = evaluate(model, DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn), nn.CrossEntropyLoss(), device, desc="Final Train")
            _, v_acc, v_f1, v_bal, v_mcc = evaluate(model, DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), collate_fn=collate_fn), nn.CrossEntropyLoss(), device, desc="Final Val")
            metrics_df = pd.DataFrame([{**final_params,
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold if 'fold' in locals() else 0,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "n_classes": n_classes,
                "train_accuracy": t_acc,
                "train_balanced_accuracy": t_bal,
                "train_macro_f1": t_f1,
                "train_mcc": t_mcc,
                "val_accuracy": v_acc,
                "val_balanced_accuracy": v_bal,
                "val_macro_f1": v_f1,
                "val_mcc": v_mcc,
            }])
            # Export per-sample predictions with class probabilities for reproducibility.
            preds_dir = fold_dir / "predictions"
            preds_dir.mkdir(parents=True, exist_ok=True)
            export_split_predictions(model, ds, train_idx, "train", fold, args.batch_size, device, preds_dir / "train_predictions.csv")
            export_split_predictions(model, ds, val_idx, "val", fold, args.batch_size, device, preds_dir / "val_predictions.csv")
            export_split_predictions(model, ds, test_idx, "test", fold, args.batch_size, device, preds_dir / "test_predictions.csv")

            with (fold_dir / "fold_result.json").open("w") as f:
                json.dump(
                    {
                        "fold": int(fold),
                        "best_val_macro_f1": float(best_f1),
                        "final_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in final_params.items()},
                        "metrics": {
                            "train_accuracy": float(t_acc),
                            "train_balanced_accuracy": float(t_bal),
                            "train_macro_f1": float(t_f1),
                            "train_mcc": float(t_mcc),
                            "val_accuracy": float(v_acc),
                            "val_balanced_accuracy": float(v_bal),
                            "val_macro_f1": float(v_f1),
                            "val_mcc": float(v_mcc),
                        },
                    },
                    f,
                    indent=2,
                )

            log_metrics(metrics_df, out_csv, out_db)
        else:
            best_f1, model = run_training(args, train_idx, val_idx, ds, vocab_size, n_classes, device, mz_bin=mz_bin, mz_parent_bin=mz_parent_bin, rt_bin_sec=rt_bin_sec, fold=fold, fold_dir=fold_dir)

            # Evaluate and log this fold (non-Optuna path) to support full CV tracking.
            _, t_acc, t_f1, t_bal, t_mcc = evaluate(
                model,
                DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn),
                nn.CrossEntropyLoss(),
                device,
                desc="Final Train",
            )
            _, v_acc, v_f1, v_bal, v_mcc = evaluate(
                model,
                DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), collate_fn=collate_fn),
                nn.CrossEntropyLoss(),
                device,
                desc="Final Val",
            )
            metrics_df = pd.DataFrame([{
                **final_params,
                "dataset": dataset_name,
                "model": model_name,
                "fold": int(fold),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "n_classes": n_classes,
                "train_accuracy": t_acc,
                "train_balanced_accuracy": t_bal,
                "train_macro_f1": t_f1,
                "train_mcc": t_mcc,
                "val_accuracy": v_acc,
                "val_balanced_accuracy": v_bal,
                "val_macro_f1": v_f1,
                "val_mcc": v_mcc,
                "mz_bin": mz_bin,
                "mz_parent_bin": mz_parent_bin,
                "rt_bin_sec": rt_bin_sec,
            }])
            log_metrics(metrics_df, out_csv, out_db, check_better=False)

            with (fold_dir / "fold_result.json").open("w") as f:
                json.dump(
                    {
                        "fold": int(fold),
                        "best_val_macro_f1": float(best_f1),
                        "final_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in final_params.items()},
                        "metrics": {
                            "train_accuracy": float(t_acc),
                            "train_balanced_accuracy": float(t_bal),
                            "train_macro_f1": float(t_f1),
                            "train_mcc": float(t_mcc),
                            "val_accuracy": float(v_acc),
                            "val_balanced_accuracy": float(v_bal),
                            "val_macro_f1": float(v_f1),
                            "val_mcc": float(v_mcc),
                        },
                    },
                    f,
                    indent=2,
                )

            preds_dir = fold_dir / "predictions"
            preds_dir.mkdir(parents=True, exist_ok=True)
            export_split_predictions(model, ds, train_idx, "train", fold, args.batch_size, device, preds_dir / "train_predictions.csv")
            export_split_predictions(model, ds, val_idx, "val", fold, args.batch_size, device, preds_dir / "val_predictions.csv")
            export_split_predictions(model, ds, test_idx, "test", fold, args.batch_size, device, preds_dir / "test_predictions.csv")

    # Build global summary from all fold rows logged above.
    update_summary_json(out_csv, out_csv.parent / "baseline_summary.json")

    # Capture final run-level outputs for reproducibility and analysis tooling.
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        shutil.copy2(out_csv, results_dir / "baselines_fold_metrics.csv")
    summary_json = out_csv.parent / "baseline_summary.json"
    if summary_json.exists():
        shutil.copy2(summary_json, results_dir / "baseline_summary.json")
    with (results_dir / "run_pointers.json").open("w") as f:
        json.dump(
            {
                "global_metrics_csv": str(out_csv),
                "global_metrics_db": str(out_db),
                "run_dir": str(run_dir),
                "scenario_outdir": str(scenario_outdir),
            },
            f,
            indent=2,
        )

def log_metrics(metrics_df, csv_path, db_path, check_better=False):
    import sqlite3
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add scenario column if missing
    if "scenario" not in metrics_df.columns:
        scenario_id = ""
        if hasattr(metrics_df, 'scenario'):
            scenario_id = metrics_df.scenario
        elif hasattr(metrics_df, 'scenario_id'):
            scenario_id = metrics_df.scenario_id
        elif hasattr(metrics_df, 'scenario_path'):
            scenario_id = metrics_df.scenario_path
        else:
            # Try to get from args if available
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'args' in frame.f_locals and hasattr(frame.f_locals['args'], 'scenario'):
                    scenario_id = str(frame.f_locals['args'].scenario)
                    break
                frame = frame.f_back
        metrics_df["scenario"] = scenario_id
    should_save = True
    if check_better and csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path)
            model = metrics_df["model"].iloc[0]
            dataset = metrics_df["dataset"].iloc[0]
            mask = (old_df["dataset"] == dataset) & (old_df["model"] == model)
            if mask.any():
                best_existing = old_df.loc[mask, "val_macro_f1"].max()
                if metrics_df["val_macro_f1"].iloc[0] <= best_existing:
                    print(f"Skipping log: {model} on {dataset} not better than {best_existing:.4f}")
                    should_save = False
        except Exception as e:
            print(f"Error checking metrics: {e}")

    if should_save:
        # Only keep standard metrics columns for CSV output, plus timestamp and scenario
        standard_cols = [
            "scenario", "dataset", "model", "fold", "mz_bin", "mz_parent_bin", "rt_bin_sec",
            "train_size", "val_size", "test_size", "n_classes",
            "train_accuracy", "train_balanced_accuracy", "train_macro_f1", "train_mcc",
            "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_mcc",
            "timestamp"
        ]
        # Add timestamp if missing
        if "timestamp" not in metrics_df.columns:
            metrics_df["timestamp"] = datetime.now().isoformat(timespec='seconds')
        # If any standard column is missing, fill with 0 or empty
        for col in standard_cols:
            if col not in metrics_df.columns:
                metrics_df[col] = 0
        # Select and order columns
        metrics_out = metrics_df[standard_cols]
        if csv_path.exists():
            metrics_out.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            metrics_out.to_csv(csv_path, index=False)
        # SQL
        try:
            conn = sqlite3.connect(db_path)
            metrics_df.to_sql("baselines", conn, if_exists="append", index=False)
            conn.close()
        except Exception as e:
            print(f"Error saving to SQL: {e}")
        print(f"Logged metrics to CSV/SQL for {metrics_df['model'].iloc[0]}")

def log_all_trials(trials, csv_path, extra_info=None):
    # trials: list of optuna.trial.FrozenTrial
    import pandas as pd
    from datetime import datetime
    rows = []
    for t in trials:
        row = dict(t.params)
        row.update({
            "trial_number": t.number,
            "state": str(t.state),
            "value": t.value,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "timestamp": datetime.now().isoformat(timespec='seconds')
        })
        if extra_info:
            row.update(extra_info)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

def update_summary_json(csv_path, json_path):
    import json
    try:
        df = pd.read_csv(csv_path)
        metrics = [
            "train_accuracy", "train_balanced_accuracy", "train_macro_f1", "train_mcc",
            "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_mcc"
        ]
        summary = {}
        for dataset in df["dataset"].unique():
            summary[str(dataset)] = {}
            sdf = df[df["dataset"] == dataset]
            for model in sdf["model"].unique():
                summary[str(dataset)][str(model)] = {}
                mdf = sdf[sdf["model"] == model]
                for m in metrics:
                    if m in mdf.columns:
                        summary[str(dataset)][str(model)][m] = {
                            "mean": float(mdf[m].mean()) if not pd.isna(mdf[m].mean()) else 0.0,
                            "std": float(mdf[m].std()) if len(mdf) > 1 and not pd.isna(mdf[m].std()) else 0.0
                        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary JSON: {json_path}")
    except Exception as e:
        print(f"Warning: Could not update summary JSON: {e}")

# Restore main entry point
if __name__ == "__main__":
    main()
