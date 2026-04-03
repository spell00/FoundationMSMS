#!/usr/bin/env python3
"""
train_deep.py - State-of-the-art deep learning training for LC-MS/MS scenarios

Features:
- CNN, ViT, (optionally MLP) architectures
- Early stopping, learning rate scheduling, mixed precision
- Data augmentation (random masking, dropout, etc.)
- Bayesian hyperparameter optimization (Optuna)
- Extensive logging and debugging

Usage:
  python scripts/train_deep.py --scenario data/doc_scenarios/custom_d4_l2_u0_new.npz --model cnn --epochs 1000 --outdir logs/deep_debug
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import optuna
import random

# --- Data Augmentation ---
def random_mask(tokens, values, mask_prob=0.15):
    mask = np.random.rand(*tokens.shape) < mask_prob
    tokens = tokens.copy()
    values = values.copy()
    tokens[mask] = 0  # or special MASK_ID
    values[mask] = 0.0
    return tokens, values

# --- Dataset ---
class ScenarioDataset(Dataset):
    def __init__(self, npz_path, max_seq_len=2048, augment=False, mask_prob=0.15):
        sc = np.load(npz_path, allow_pickle=True)
        self.token_ids = sc["tokens_idx"]
        self.token_vals = sc["tokens_val"]
        self.raw_labels = sc["labels"]
        # Map string labels to integer indices if needed
        unique_labels, label_indices = np.unique(self.raw_labels, return_inverse=True)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.inv_label_map = {idx: label for idx, label in enumerate(unique_labels)}
        self.labels = label_indices
        self.max_seq_len = max_seq_len
        self.n_classes = len(unique_labels)
        self.vocab_size = int(np.max([np.max(x) if len(x) > 0 else 0 for x in self.token_ids])) + 2
        self.augment = augment
        self.mask_prob = mask_prob
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        ids = self.token_ids[idx][:self.max_seq_len] + 1
        vals = self.token_vals[idx][:self.max_seq_len]
        label = int(self.labels[idx])
        if self.augment and random.random() < 0.5:
            ids, vals = random_mask(ids, vals, self.mask_prob)
        return torch.as_tensor(ids, dtype=torch.long), torch.as_tensor(vals, dtype=torch.float32), label

def collate_fn(batch):
    ids, vals, labels = zip(*batch)
    max_len = max(x.shape[0] for x in ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_vals = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (id_, val) in enumerate(zip(ids, vals)):
        l = id_.shape[0]
        padded_ids[i, :l] = id_
        padded_vals[i, :l] = val
        mask[i, :l] = True
    return padded_ids, padded_vals, torch.as_tensor(labels, dtype=torch.long)

# --- Models ---
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
# TODO: Add ViT and optionally MLP
MODEL_ZOO = {
    "cnn": SimpleCNN,
    # "vit": ViT,
    # "mlp": MLP,
}

# --- Training Loop ---
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for token_ids, values, labels in loader:
        token_ids, values, labels = token_ids.to(device), values.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            logits = model(token_ids, values)
            loss = loss_fn(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for token_ids, values, labels in loader:
            token_ids, values, labels = token_ids.to(device), values.to(device), labels.to(device)
            logits = model(token_ids, values)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, macro_f1, bal_acc, mcc

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.best_state = None
    def step(self, metric, model):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Train deep learning models on LC-MS/MS scenarios")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to scenario NPZ")
    parser.add_argument("--model", type=str, default="cnn", choices=list(MODEL_ZOO.keys()), help="Model type")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--outdir", type=Path, default=Path("logs/deep_debug"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="Masking probability for augmentation")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Number of Optuna trials for Bayesian optimization (0=disable)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.outdir)

    # Data
    ds = ScenarioDataset(args.scenario, max_seq_len=args.max_seq_len, augment=args.augment, mask_prob=args.mask_prob)
    n_classes = ds.n_classes
    vocab_size = ds.vocab_size
    idx = np.arange(len(ds))
    np.random.shuffle(idx)
    split = int(0.8 * len(ds))
    train_idx, val_idx = idx[:split], idx[split:]
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
    val_loader = DataLoader(ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MODEL_ZOO[args.model](vocab_size, args.embed_dim, args.channels, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    early_stopper = EarlyStopping(patience=30)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        v_loss, v_acc, v_f1, v_bal, v_mcc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(v_f1)
        writer.add_scalar("Loss/train", t_loss, epoch)
        writer.add_scalar("Loss/val", v_loss, epoch)
        writer.add_scalar("Acc/train", t_acc, epoch)
        writer.add_scalar("Acc/val", v_acc, epoch)
        writer.add_scalar("F1/val", v_f1, epoch)
        writer.add_scalar("BalancedAcc/val", v_bal, epoch)
        writer.add_scalar("MCC/val", v_mcc, epoch)
        print(f"Epoch {epoch}: train_loss={t_loss:.4f} val_loss={v_loss:.4f} val_f1={v_f1:.4f} val_bal={v_bal:.4f} val_mcc={v_mcc:.4f}")
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), args.outdir / "best_model.pth")
        if early_stopper.step(v_f1, model):
            print(f"Early stopping at epoch {epoch}")
            break
    early_stopper.restore(model)
    print(f"Best val F1: {best_val_f1:.4f}")
    writer.close()

    # Optionally: Optuna Bayesian optimization
    if args.optuna_trials > 0:
        def objective(trial):
            embed_dim = trial.suggest_categorical("embed_dim", [64, 128, 256])
            channels = trial.suggest_categorical("channels", [64, 128, 256])
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            model = MODEL_ZOO[args.model](vocab_size, embed_dim, channels, n_classes).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
            train_loader = DataLoader(ds, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), collate_fn=collate_fn)
            val_loader = DataLoader(ds, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), collate_fn=collate_fn)
            best_f1 = 0.0
            early_stopper = EarlyStopping(patience=15)
            for epoch in range(1, 200):
                t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
                v_loss, v_acc, v_f1, v_bal, v_mcc = evaluate(model, val_loader, loss_fn, device)
                if v_f1 > best_f1:
                    best_f1 = v_f1
                if early_stopper.step(v_f1, model):
                    break
            return best_f1
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)
        print("Optuna best trial:", study.best_trial.params)

if __name__ == "__main__":
    main()
